from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_state_decoder import (
    aggregate_event_metrics,
    basic_binary_metrics,
    best_accuracy_threshold,
    json_default,
    moving_average_per_sequence,
)


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def debounce(scores: np.ndarray, seq_offsets: np.ndarray, threshold: float, on_k: int, off_k: int) -> np.ndarray:
    out = np.zeros(scores.shape[0], dtype=bool)
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
            out[idx] = state
    return out


def zscore(scores: np.ndarray) -> np.ndarray:
    return (scores - float(scores.mean())) / max(float(scores.std()), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a two-score-cache ensemble without torch.")
    parser.add_argument("--score-cache-a", required=True, type=Path)
    parser.add_argument("--score-cache-b", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--ma-windows", default="1,10,20,50,100")
    parser.add_argument("--debounce-on-k", default="1,2,3,5")
    parser.add_argument("--debounce-off-k", default="2,5,10,20")
    args = parser.parse_args()

    a = np.load(args.score_cache_a, allow_pickle=False)
    b = np.load(args.score_cache_b, allow_pickle=False)
    labels = a["labels"].astype(np.int64, copy=False)
    if not np.array_equal(labels, b["labels"]):
        raise RuntimeError("Label arrays differ")
    if not np.array_equal(a["global_indices"], b["global_indices"]):
        raise RuntimeError("Global indices differ")
    seq_offsets = a["seq_offsets"].astype(np.int64, copy=False)
    if not np.array_equal(seq_offsets, b["seq_offsets"]):
        raise RuntimeError("Sequence offsets differ")
    score_a = zscore(a["raw_score_matrix"][0].astype(np.float64, copy=False))
    score_b = zscore(b["raw_score_matrix"][0].astype(np.float64, copy=False))

    methods: list[dict[str, Any]] = []
    for weight in parse_csv_floats(args.weights):
        base_scores = (1.0 - weight) * score_a + weight * score_b
        for ma_window in parse_csv_ints(args.ma_windows):
            scores = moving_average_per_sequence(base_scores, seq_offsets, ma_window)
            tuned = best_accuracy_threshold(labels, scores)
            threshold = float(tuned["threshold"])
            pred = scores > threshold
            methods.append(
                {
                    "method": f"w{weight:g}_ma{ma_window}",
                    "weight_b": float(weight),
                    "ma_window": int(ma_window),
                    "threshold": threshold,
                    "kind": "score",
                    "metrics": tuned,
                    "event_metrics": aggregate_event_metrics(labels, pred, seq_offsets),
                }
            )
            for on_k in parse_csv_ints(args.debounce_on_k):
                for off_k in parse_csv_ints(args.debounce_off_k):
                    pred = debounce(scores, seq_offsets, threshold, on_k, off_k)
                    methods.append(
                        {
                            "method": f"w{weight:g}_ma{ma_window}_debounce_on{on_k}_off{off_k}",
                            "weight_b": float(weight),
                            "ma_window": int(ma_window),
                            "threshold": threshold,
                            "on_k": int(on_k),
                            "off_k": int(off_k),
                            "kind": "binary_state",
                            "metrics": basic_binary_metrics(labels, pred),
                            "event_metrics": aggregate_event_metrics(labels, pred, seq_offsets),
                        }
                    )
    methods.sort(key=lambda item: float(item["metrics"]["accuracy"]), reverse=True)
    result = {
        "score_cache_a": str(args.score_cache_a),
        "score_cache_b": str(args.score_cache_b),
        "total_windows": int(labels.shape[0]),
        "best_method": methods[0],
        "methods_top": methods[:50],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({"best_method": methods[0]["method"], "accuracy": methods[0]["metrics"]["accuracy"]}, sort_keys=True))


if __name__ == "__main__":
    main()
