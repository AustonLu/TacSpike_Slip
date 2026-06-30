from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.train.evaluate_checkpoint_ensemble import load_model, predict_scores
from scripts.train.train_lite_scnn import order_indices_for_io
from tacspike.data import sample_epoch_indices
from tacspike.training import binary_classification_metrics


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def transform_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    transformed = scores.astype(np.float64, copy=True)
    if mode == "raw":
        return transformed
    if mode == "zscore":
        mean = transformed.mean(axis=1, keepdims=True)
        std = transformed.std(axis=1, keepdims=True)
        return (transformed - mean) / np.maximum(std, 1e-12)
    if mode == "minmax":
        lo = transformed.min(axis=1, keepdims=True)
        hi = transformed.max(axis=1, keepdims=True)
        return (transformed - lo) / np.maximum(hi - lo, 1e-12)
    if mode == "rank_centered":
        ranked = np.empty_like(transformed)
        n = transformed.shape[1]
        denom = max(n - 1, 1)
        for idx, row in enumerate(transformed):
            order = np.argsort(row, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64) / denom
            ranked[idx] = ranks - 0.5
        return ranked
    raise ValueError(f"Unsupported transform mode: {mode}")


def best_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    is_pos = sorted_labels == 1
    is_neg = ~is_pos
    total_pos = int(is_pos.sum())
    total_neg = int(is_neg.sum())
    cum_pos = np.cumsum(is_pos)
    cum_neg = np.cumsum(is_neg)

    change = np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
    split_positions = np.concatenate([change, np.array([labels.shape[0]], dtype=np.int64)])
    pos_left = cum_pos[split_positions - 1]
    neg_left = cum_neg[split_positions - 1]
    tp = total_pos - pos_left
    tn = neg_left
    accuracy = (tp + tn) / max(labels.shape[0], 1)

    best_idx = int(np.argmax(accuracy))
    threshold = float(sorted_scores[split_positions[best_idx] - 1])
    metrics = binary_classification_metrics(labels, scores, threshold=threshold)
    return {"threshold": threshold, **metrics}


def candidate_weights(num_models: int, rng: np.random.Generator, trials: int) -> List[np.ndarray]:
    candidates: List[np.ndarray] = []
    candidates.append(np.full(num_models, 1.0 / num_models, dtype=np.float64))
    for idx in range(num_models):
        one_hot = np.zeros(num_models, dtype=np.float64)
        one_hot[idx] = 1.0
        candidates.append(one_hot)
    for size in range(2, num_models + 1):
        for combo in itertools.combinations(range(num_models), size):
            weights = np.zeros(num_models, dtype=np.float64)
            weights[list(combo)] = 1.0 / size
            candidates.append(weights)
    for alpha in (0.2, 0.5, 1.0, 2.0, 5.0):
        for _ in range(max(trials // 5, 1)):
            candidates.append(rng.dirichlet(np.full(num_models, alpha, dtype=np.float64)))
    return candidates


def summarize_candidate(
    labels: np.ndarray,
    score_matrix: np.ndarray,
    weights: np.ndarray,
    transform: str,
    checkpoint_names: List[str],
) -> Dict[str, Any]:
    scores = weights @ score_matrix
    tuned = best_accuracy_threshold(labels, scores)
    default_metrics = binary_classification_metrics(labels, scores, threshold=0.0)
    return {
        "transform": transform,
        "weights": [float(x) for x in weights.tolist()],
        "members": checkpoint_names,
        "default_metrics": default_metrics,
        "best_threshold_metrics": tuned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search weighted score ensembles over TacSpike checkpoints.")
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling", choices=("balanced", "random"), default="random")
    parser.add_argument("--search-trials", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models_and_args = [
        load_model(path, args.data_root, args.batch_size, args.num_workers, device)
        for path in args.checkpoints
    ]
    reference_args = models_and_args[0][1]
    indices = sample_epoch_indices(
        data_root=args.data_root,
        split=args.split,
        cache_dir=reference_args.cache_dir,
        num_samples=args.samples,
        seed=args.seed,
        sampling=args.sampling,
    )
    ordered_indices = order_indices_for_io(indices, args.batch_size, args.seed + 99)

    score_parts = []
    labels = None
    for model, train_args in models_and_args:
        scores, current_labels = predict_scores(model, train_args, args.split, ordered_indices, args.batch_size, device)
        score_parts.append(scores.astype(np.float64, copy=False))
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise RuntimeError("Ensemble checkpoints produced labels in different order.")

    assert labels is not None
    raw_score_matrix = np.stack(score_parts, axis=0)
    checkpoint_names = [path.parent.name for path in args.checkpoints]
    rng = np.random.default_rng(args.seed)
    weights_to_try = candidate_weights(len(args.checkpoints), rng, args.search_trials)

    best_records: List[Dict[str, Any]] = []
    for transform in ("raw", "zscore", "minmax", "rank_centered"):
        score_matrix = transform_scores(raw_score_matrix, transform)
        for weights in weights_to_try:
            record = summarize_candidate(labels, score_matrix, weights, transform, checkpoint_names)
            best_records.append(record)

    best_records.sort(key=lambda item: item["best_threshold_metrics"]["accuracy"], reverse=True)
    result = {
        "checkpoints": [str(path) for path in args.checkpoints],
        "checkpoint_names": checkpoint_names,
        "split": args.split,
        "sampling": args.sampling,
        "samples": int(labels.shape[0]),
        "search_trials": int(args.search_trials),
        "top_k": best_records[: args.top_k],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(result["top_k"][0], sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
