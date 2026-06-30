from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from tacspike.training import binary_classification_metrics


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def load_score_caches(paths: Iterable[Path]) -> dict[str, Any]:
    labels = None
    ordered_indices = None
    original_indices = None
    scores = []
    names = []
    cache_paths = []
    for path in paths:
        with np.load(path) as cache:
            current_labels = cache["labels"].astype(np.int64, copy=False)
            current_ordered_indices = cache["ordered_indices"].astype(np.int64, copy=False)
            current_original_indices = cache["original_indices"].astype(np.int64, copy=False)
            current_scores = cache["scores"].astype(np.float64, copy=False)
            current_name = str(cache["checkpoint_name"].item())
        if labels is None:
            labels = current_labels
            ordered_indices = current_ordered_indices
            original_indices = current_original_indices
        else:
            if not np.array_equal(labels, current_labels):
                raise ValueError(f"Label mismatch in {path}")
            if not np.array_equal(ordered_indices, current_ordered_indices):
                raise ValueError(f"Ordered index mismatch in {path}")
            if not np.array_equal(original_indices, current_original_indices):
                raise ValueError(f"Original index mismatch in {path}")
        scores.append(current_scores)
        names.append(current_name)
        cache_paths.append(str(path))
    if labels is None or ordered_indices is None or original_indices is None:
        raise ValueError("No score caches provided")
    return {
        "labels": labels,
        "ordered_indices": ordered_indices,
        "original_indices": original_indices,
        "score_matrix": np.stack(scores, axis=0),
        "checkpoint_names": names,
        "cache_paths": cache_paths,
    }


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


def best_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    is_pos = sorted_labels == 1
    total_pos = int(is_pos.sum())
    total_neg = int(labels.shape[0] - total_pos)
    cum_pos = np.cumsum(is_pos)
    cum_neg = np.cumsum(~is_pos)

    change = np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
    splits = np.concatenate([change, np.array([labels.shape[0]], dtype=np.int64)])
    pos_left = cum_pos[splits - 1]
    neg_left = cum_neg[splits - 1]
    tp = total_pos - pos_left
    tn = neg_left
    accuracy = (tp + tn) / max(labels.shape[0], 1)

    best_idx = int(np.argmax(accuracy))
    threshold = float(sorted_scores[splits[best_idx] - 1])
    return {"threshold": threshold, **binary_classification_metrics(labels, scores, threshold=threshold)}


def parse_float_csv(text: str | None) -> list[float]:
    if text is None or not text.strip():
        return []
    return [float(part) for part in text.split(",") if part.strip()]


def normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return weights / total


def candidate_weights(num_models: int, rng: np.random.Generator, trials: int, max_subset_size: int) -> List[np.ndarray]:
    candidates: List[np.ndarray] = []
    candidates.append(np.full(num_models, 1.0 / num_models, dtype=np.float64))
    for idx in range(num_models):
        weights = np.zeros(num_models, dtype=np.float64)
        weights[idx] = 1.0
        candidates.append(weights)

    subset_limit = min(max_subset_size, num_models)
    for size in range(2, subset_limit + 1):
        for combo in itertools.combinations(range(num_models), size):
            weights = np.zeros(num_models, dtype=np.float64)
            weights[list(combo)] = 1.0 / size
            candidates.append(weights)

    alphas = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    for alpha in alphas:
        for _ in range(max(trials // len(alphas), 1)):
            candidates.append(rng.dirichlet(np.full(num_models, alpha, dtype=np.float64)))
    return candidates


def quick_best_accuracy(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    is_pos = sorted_labels == 1
    total_pos = int(is_pos.sum())
    cum_pos = np.cumsum(is_pos)
    cum_neg = np.cumsum(~is_pos)
    change = np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
    splits = np.concatenate([change, np.array([labels.shape[0]], dtype=np.int64)])
    pos_left = cum_pos[splits - 1]
    neg_left = cum_neg[splits - 1]
    tp = total_pos - pos_left
    tn = neg_left
    accuracy = (tp + tn) / max(labels.shape[0], 1)

    best_idx = int(np.argmax(accuracy))
    return {
        "accuracy": float(accuracy[best_idx]),
        "threshold": float(sorted_scores[splits[best_idx] - 1]),
    }


def active_members(weights: np.ndarray, checkpoint_names: list[str]) -> list[dict[str, Any]]:
    return [
        {"name": checkpoint_names[idx], "weight": float(value)}
        for idx, value in enumerate(weights.tolist())
        if value > 1e-6
    ]


def summarize(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    transform: str,
    checkpoint_names: list[str],
) -> dict[str, Any]:
    tuned = best_accuracy_threshold(labels, scores)
    default_metrics = binary_classification_metrics(labels, scores, threshold=0.0)
    active = active_members(weights, checkpoint_names)
    return {
        "transform": transform,
        "weights": [float(x) for x in weights.tolist()],
        "active_members": active,
        "num_active": int(len(active)),
        "default_metrics": default_metrics,
        "best_threshold_metrics": tuned,
    }


def quick_record(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    transform: str,
    checkpoint_names: list[str],
) -> dict[str, Any]:
    best = quick_best_accuracy(labels, scores)
    active = [
        {"name": checkpoint_names[idx], "weight": float(value)}
        for idx, value in enumerate(weights.tolist())
        if value > 1e-6
    ]
    return {
        "transform": transform,
        "weights": [float(x) for x in weights.tolist()],
        "active_members": active,
        "num_active": int(len(active)),
        "quick_best_accuracy": best["accuracy"],
        "quick_best_threshold": best["threshold"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search weighted ensembles from cached TacSpike scores.")
    parser.add_argument("--score-caches", required=True, nargs="+", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--search-trials", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-subset-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--transforms", default="raw,zscore,minmax,rank_centered")
    parser.add_argument("--fixed-weights", default=None)
    parser.add_argument("--fixed-transform", choices=("raw", "zscore", "minmax", "rank_centered"), default="raw")
    parser.add_argument("--detail-candidates", type=int, default=500)
    args = parser.parse_args()

    cache = load_score_caches(args.score_caches)
    labels = cache["labels"]
    raw_score_matrix = cache["score_matrix"]
    checkpoint_names = cache["checkpoint_names"]

    fixed_weights = parse_float_csv(args.fixed_weights)
    records: list[dict[str, Any]] = []
    if fixed_weights:
        weights = normalize(np.asarray(fixed_weights, dtype=np.float64))
        if weights.shape[0] != raw_score_matrix.shape[0]:
            raise ValueError(f"Expected {raw_score_matrix.shape[0]} fixed weights, got {weights.shape[0]}")
        score_matrix = transform_scores(raw_score_matrix, args.fixed_transform)
        scores = weights @ score_matrix
        records.append(summarize(labels, scores, weights, args.fixed_transform, checkpoint_names))
    else:
        rng = np.random.default_rng(args.seed)
        candidates = candidate_weights(
            num_models=raw_score_matrix.shape[0],
            rng=rng,
            trials=args.search_trials,
            max_subset_size=args.max_subset_size,
        )
        quick_records: list[dict[str, Any]] = []
        for transform in [part.strip() for part in args.transforms.split(",") if part.strip()]:
            score_matrix = transform_scores(raw_score_matrix, transform)
            for weights in candidates:
                scores = weights @ score_matrix
                quick_records.append(quick_record(labels, scores, weights, transform, checkpoint_names))

        quick_records.sort(key=lambda item: item["quick_best_accuracy"], reverse=True)
        for item in quick_records[: args.detail_candidates]:
            score_matrix = transform_scores(raw_score_matrix, item["transform"])
            weights = np.asarray(item["weights"], dtype=np.float64)
            scores = weights @ score_matrix
            records.append(summarize(labels, scores, weights, item["transform"], checkpoint_names))

    records.sort(key=lambda item: item["best_threshold_metrics"]["accuracy"], reverse=True)
    result = {
        "score_caches": cache["cache_paths"],
        "checkpoint_names": checkpoint_names,
        "samples": int(labels.shape[0]),
        "positive_fraction": float(labels.mean()) if labels.size else 0.0,
        "search_trials": int(args.search_trials),
        "max_subset_size": int(args.max_subset_size),
        "top_k": records[: args.top_k],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(result["top_k"][0], sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
