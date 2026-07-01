from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_state_decoder import json_default


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def transition_distances(labels: np.ndarray, seq_offsets: np.ndarray) -> np.ndarray:
    distances = np.empty(labels.shape[0], dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq = labels[start_i:stop_i]
        transitions = np.flatnonzero(seq[1:] != seq[:-1]) + 1 if seq.size > 1 else np.empty((0,), dtype=np.int64)
        if transitions.size == 0:
            distances[start_i:stop_i] = np.inf
            continue
        positions = np.arange(seq.shape[0], dtype=np.int64)
        right_idx = np.searchsorted(transitions, positions)
        right = np.where(right_idx < transitions.size, transitions[np.minimum(right_idx, transitions.size - 1)], np.inf)
        left_idx = np.maximum(right_idx - 1, 0)
        left = np.where(right_idx > 0, transitions[left_idx], -np.inf)
        distances[start_i:stop_i] = np.minimum(np.abs(positions - left), np.abs(positions - right))
    return distances


def segment_lengths(labels: np.ndarray, seq_offsets: np.ndarray) -> dict[str, Any]:
    slip_lengths: list[int] = []
    no_slip_lengths: list[int] = []
    transitions = 0
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        seq = labels[int(start) : int(stop)]
        if seq.size == 0:
            continue
        change = np.flatnonzero(seq[1:] != seq[:-1]) + 1
        transitions += int(change.shape[0])
        bounds = np.concatenate([[0], change, [seq.shape[0]]])
        for left, right in zip(bounds[:-1], bounds[1:]):
            length = int(right - left)
            if int(seq[int(left)]) == 1:
                slip_lengths.append(length)
            else:
                no_slip_lengths.append(length)

    def stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.shape[0]),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p05_ms": float(np.percentile(arr, 5)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
        }

    return {
        "label_transitions": transitions,
        "slip_segments": stats(slip_lengths),
        "no_slip_segments": stats(no_slip_lengths),
    }


def delayed_label_prediction(labels: np.ndarray, seq_offsets: np.ndarray, delay: int) -> np.ndarray:
    pred = np.zeros_like(labels, dtype=np.int64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq = labels[start_i:stop_i]
        if delay == 0:
            pred[start_i:stop_i] = seq
        else:
            pred[start_i + delay : stop_i] = seq[: max(seq.shape[0] - delay, 0)]
            pred[start_i : min(start_i + delay, stop_i)] = seq[0]
    return pred


def symmetric_boundary_tolerant_accuracy(labels: np.ndarray, predictions: np.ndarray, distances: np.ndarray, radius: int) -> dict[str, Any]:
    keep = distances > radius
    return {
        "radius_ms": int(radius),
        "kept_windows": int(keep.sum()),
        "ignored_windows": int((~keep).sum()),
        "ignored_fraction": safe_div(float((~keep).sum()), float(labels.shape[0])),
        "accuracy": safe_div(float((labels[keep] == predictions[keep]).sum()), float(keep.sum())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sequence-state labels and transition-limited upper bounds.")
    parser.add_argument("--score-cache", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--transition-radii", default="10,20,50,100,200,500")
    parser.add_argument("--oracle-delays", default="0,10,20,50,100,200,300,500")
    args = parser.parse_args()

    cache = np.load(args.score_cache, allow_pickle=False)
    labels = cache["labels"].astype(np.int64, copy=False)
    seq_offsets = cache["seq_offsets"].astype(np.int64, copy=False)
    distances = transition_distances(labels, seq_offsets)

    majority_label = int(labels.mean() >= 0.5)
    majority_pred = np.full(labels.shape, majority_label, dtype=np.int64)
    delay_records = []
    for delay in parse_csv_ints(args.oracle_delays):
        pred = delayed_label_prediction(labels, seq_offsets, int(delay))
        delay_records.append(
            {
                "delay_ms": int(delay),
                "strict_accuracy_if_perfect_state_is_delayed": safe_div(float((pred == labels).sum()), float(labels.shape[0])),
            }
        )

    result = {
        "score_cache": str(args.score_cache),
        "windows": int(labels.shape[0]),
        "sequences": int(seq_offsets.shape[0] - 1),
        "positive_fraction": float(labels.mean()) if labels.size else 0.0,
        "majority_baseline": {
            "label": majority_label,
            "accuracy": safe_div(float((majority_pred == labels).sum()), float(labels.shape[0])),
        },
        "segments": segment_lengths(labels, seq_offsets),
        "transition_window_fraction": [
            {
                "radius_ms": int(radius),
                "fraction": safe_div(float((distances <= radius).sum()), float(labels.shape[0])),
                "windows": int((distances <= radius).sum()),
            }
            for radius in parse_csv_ints(args.transition_radii)
        ],
        "perfect_state_delay_upper_bounds": delay_records,
        "majority_tolerant_accuracy": [
            symmetric_boundary_tolerant_accuracy(labels, majority_pred, distances, int(radius))
            for radius in parse_csv_ints(args.transition_radii)
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({"windows": result["windows"], "positive_fraction": result["positive_fraction"]}, sort_keys=True))


if __name__ == "__main__":
    main()
