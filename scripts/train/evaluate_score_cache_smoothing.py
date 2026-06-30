from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from tacspike.training import binary_classification_metrics


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def parse_ints(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def load_caches(paths: Iterable[Path]) -> Dict[str, Any]:
    labels = None
    seq_offsets = None
    sequence_ids = None
    score_parts = []
    cache_paths = []
    for path in paths:
        with np.load(path) as cache:
            current_labels = cache["labels"].astype(np.int64, copy=False)
            current_offsets = cache["seq_offsets"].astype(np.int64, copy=False)
            current_ids = cache["sequence_ids"]
            score_parts.append(cache["scores"].astype(np.float64, copy=False))
            cache_paths.append(str(path))
            if labels is None:
                labels = current_labels
                seq_offsets = current_offsets
                sequence_ids = current_ids
            else:
                if not np.array_equal(labels, current_labels):
                    raise ValueError(f"Label mismatch in {path}")
                if not np.array_equal(seq_offsets, current_offsets):
                    raise ValueError(f"Sequence offset mismatch in {path}")
                if not np.array_equal(sequence_ids, current_ids):
                    raise ValueError(f"Sequence id mismatch in {path}")
    if labels is None or seq_offsets is None or sequence_ids is None:
        raise ValueError("No score caches provided")
    return {
        "cache_paths": cache_paths,
        "labels": labels,
        "seq_offsets": seq_offsets,
        "sequence_ids": sequence_ids,
        "score_matrix": np.stack(score_parts, axis=0),
    }


def transform_score_matrix(score_matrix: np.ndarray, mode: str) -> np.ndarray:
    scores = score_matrix.astype(np.float64, copy=True)
    if mode == "raw":
        return scores
    if mode == "zscore":
        mean = scores.mean(axis=1, keepdims=True)
        std = scores.std(axis=1, keepdims=True)
        return (scores - mean) / np.maximum(std, 1e-12)
    if mode == "rank_centered":
        ranked = np.empty_like(scores)
        n = scores.shape[1]
        denom = max(n - 1, 1)
        for idx, row in enumerate(scores):
            order = np.argsort(row, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64) / denom
            ranked[idx] = ranks - 0.5
        return ranked
    raise ValueError(f"Unsupported score transform: {mode}")


def normalize_weights(weights_text: str | None, num_models: int) -> np.ndarray:
    if weights_text is None:
        return np.full(num_models, 1.0 / num_models, dtype=np.float64)
    weights = np.asarray(parse_floats(weights_text), dtype=np.float64)
    if weights.shape[0] != num_models:
        raise ValueError(f"Expected {num_models} weights, got {weights.shape[0]}")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return weights / total


def per_sequence_causal_ma(scores: np.ndarray, seq_offsets: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores.astype(np.float64, copy=True)
    out = np.empty_like(scores, dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        seq = scores[int(start) : int(stop)].astype(np.float64, copy=False)
        cumsum = np.cumsum(np.concatenate([[0.0], seq]))
        positions = np.arange(seq.shape[0])
        left = np.maximum(0, positions + 1 - window)
        out[int(start) : int(stop)] = (cumsum[positions + 1] - cumsum[left]) / (positions + 1 - left)
    return out


def per_sequence_ema(scores: np.ndarray, seq_offsets: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(scores, dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq = scores[start_i:stop_i].astype(np.float64, copy=False)
        if seq.size == 0:
            continue
        out[start_i] = seq[0]
        for idx in range(start_i + 1, stop_i):
            out[idx] = alpha * scores[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def transition_distances_per_sequence(labels: np.ndarray, seq_offsets: np.ndarray) -> np.ndarray:
    distances = np.empty(labels.shape[0], dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq = labels[start_i:stop_i]
        if seq.size == 0:
            continue
        transitions = np.flatnonzero(seq[1:] != seq[:-1]) + 1
        if transitions.size == 0:
            distances[start_i:stop_i] = np.inf
            continue
        positions = np.arange(seq.shape[0], dtype=np.int64)
        idx = np.searchsorted(transitions, positions)
        right = np.where(idx < transitions.size, transitions[np.minimum(idx, transitions.size - 1)], np.inf)
        left_idx = np.maximum(idx - 1, 0)
        left = np.where(idx > 0, transitions[left_idx], -np.inf)
        distances[start_i:stop_i] = np.minimum(np.abs(positions - left), np.abs(positions - right))
    return distances


def best_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
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
    metrics = binary_classification_metrics(labels, scores, threshold=threshold)
    return {"threshold": threshold, **metrics}


def threshold_candidates(scores: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.float64)
    quantiles = np.linspace(0.05, 0.95, count)
    return np.unique(np.quantile(scores, quantiles).astype(np.float64))


def debounce_predictions(
    scores: np.ndarray,
    seq_offsets: np.ndarray,
    threshold: float,
    on_k: int,
    off_k: int,
) -> np.ndarray:
    out = np.zeros(scores.shape[0], dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        state = False
        pos_run = 0
        neg_run = 0
        for idx in range(int(start), int(stop)):
            positive = scores[idx] > threshold
            if positive:
                pos_run += 1
                neg_run = 0
            else:
                neg_run += 1
                pos_run = 0
            if not state and pos_run >= on_k:
                state = True
            elif state and neg_run >= off_k:
                state = False
            out[idx] = 1.0 if state else 0.0
    return out


def hysteresis_predictions(
    scores: np.ndarray,
    seq_offsets: np.ndarray,
    on_threshold: float,
    off_threshold: float,
    on_k: int,
    off_k: int,
) -> np.ndarray:
    out = np.zeros(scores.shape[0], dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        state = False
        pos_run = 0
        neg_run = 0
        for idx in range(int(start), int(stop)):
            value = scores[idx]
            if not state:
                if value > on_threshold:
                    pos_run += 1
                else:
                    pos_run = 0
                if pos_run >= on_k:
                    state = True
                    neg_run = 0
            else:
                if value <= off_threshold:
                    neg_run += 1
                else:
                    neg_run = 0
                if neg_run >= off_k:
                    state = False
                    pos_run = 0
            out[idx] = 1.0 if state else 0.0
    return out


def metrics_record(labels: np.ndarray, scores: np.ndarray, distances: np.ndarray | None = None) -> Dict[str, Any]:
    best = best_accuracy_threshold(labels, scores)
    record: Dict[str, Any] = {
        "default_metrics": binary_classification_metrics(labels, scores, threshold=0.0),
        "best_threshold_metrics": best,
    }
    if distances is not None:
        filtered: Dict[str, Any] = {}
        for min_distance in (50, 100, 150):
            mask = distances >= float(min_distance)
            if not bool(mask.any()):
                filtered[f"gt_{min_distance}_ms"] = {"count": 0}
            else:
                filtered[f"gt_{min_distance}_ms"] = {
                    "count": int(mask.sum()),
                    "best_threshold_metrics": best_accuracy_threshold(labels[mask], scores[mask]),
                }
        record["filtered_transition_metrics"] = filtered
    return record


def binary_metrics_record(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    return {"metrics": binary_classification_metrics(labels, predictions, threshold=0.5)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Search smoothing and hysteresis on cached TacSpike sequence scores.")
    parser.add_argument("--score-caches", required=True, nargs="+", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--transform", choices=("raw", "zscore", "rank_centered"), default="raw")
    parser.add_argument("--ma-windows", default="3,5,10,20,50")
    parser.add_argument("--ema-alphas", default="0.1,0.2,0.3,0.5,0.7")
    parser.add_argument("--debounce-threshold-count", type=int, default=21)
    parser.add_argument("--debounce-on-k", default="1,2,3,5")
    parser.add_argument("--debounce-off-k", default="1,2,3,5,10")
    parser.add_argument("--hysteresis-threshold-count", type=int, default=9)
    args = parser.parse_args()

    cache = load_caches(args.score_caches)
    labels = cache["labels"]
    seq_offsets = cache["seq_offsets"]
    sequence_ids = cache["sequence_ids"]
    score_matrix = transform_score_matrix(cache["score_matrix"], args.transform)
    weights = normalize_weights(args.weights, score_matrix.shape[0])
    scores = weights @ score_matrix
    distances = transition_distances_per_sequence(labels, seq_offsets)

    results: List[Dict[str, Any]] = []

    raw_record = {
        "method": "raw",
        "params": {},
        **metrics_record(labels, scores, distances),
    }
    results.append(raw_record)

    for window in parse_ints(args.ma_windows):
        smoothed = per_sequence_causal_ma(scores, seq_offsets, window)
        results.append(
            {
                "method": "causal_ma",
                "params": {"window": window},
                **metrics_record(labels, smoothed, distances),
            }
        )

    for alpha in parse_floats(args.ema_alphas):
        smoothed = per_sequence_ema(scores, seq_offsets, alpha)
        results.append(
            {
                "method": "ema",
                "params": {"alpha": alpha},
                **metrics_record(labels, smoothed, distances),
            }
        )

    candidates = threshold_candidates(scores, args.debounce_threshold_count)
    for threshold, on_k, off_k in itertools.product(
        candidates,
        parse_ints(args.debounce_on_k),
        parse_ints(args.debounce_off_k),
    ):
        pred = debounce_predictions(scores, seq_offsets, float(threshold), on_k, off_k)
        results.append(
            {
                "method": "debounce",
                "params": {"threshold": float(threshold), "on_k": on_k, "off_k": off_k},
                **binary_metrics_record(labels, pred),
            }
        )

    h_candidates = threshold_candidates(scores, args.hysteresis_threshold_count)
    for on_threshold in h_candidates:
        for off_threshold in h_candidates:
            if off_threshold > on_threshold:
                continue
            for on_k, off_k in itertools.product((1, 2, 3), (1, 2, 3, 5)):
                pred = hysteresis_predictions(
                    scores,
                    seq_offsets,
                    float(on_threshold),
                    float(off_threshold),
                    on_k,
                    off_k,
                )
                results.append(
                    {
                        "method": "hysteresis",
                        "params": {
                            "on_threshold": float(on_threshold),
                            "off_threshold": float(off_threshold),
                            "on_k": on_k,
                            "off_k": off_k,
                        },
                        **binary_metrics_record(labels, pred),
                    }
                )

    def accuracy_of(record: Dict[str, Any]) -> float:
        if "best_threshold_metrics" in record:
            return float(record["best_threshold_metrics"]["accuracy"])
        return float(record["metrics"]["accuracy"])

    results.sort(key=accuracy_of, reverse=True)
    top = results[:50]
    result = {
        "score_caches": cache["cache_paths"],
        "num_models": int(score_matrix.shape[0]),
        "weights": [float(x) for x in weights.tolist()],
        "transform": args.transform,
        "windows": int(labels.shape[0]),
        "sequences": int(sequence_ids.shape[0]),
        "positive_fraction": float(labels.mean()) if labels.size else 0.0,
        "top_results": top,
        "raw": raw_record,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({"best": top[0], "windows": result["windows"], "sequences": result["sequences"]}, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
