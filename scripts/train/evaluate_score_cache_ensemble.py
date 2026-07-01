from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_score_cache_pair import debounce
from evaluate_state_decoder import (
    aggregate_event_metrics,
    basic_binary_metrics,
    best_accuracy_threshold,
    json_default,
    moving_average_per_sequence,
)


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def zscore_row(scores: np.ndarray) -> np.ndarray:
    return (scores - float(scores.mean())) / max(float(scores.std()), 1e-12)


def load_score_rows(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    labels = None
    indices = None
    offsets = None
    rows = []
    records: list[dict[str, Any]] = []
    for path in paths:
        cache = np.load(path, allow_pickle=False)
        cache_labels = cache["labels"].astype(np.int64, copy=False)
        cache_indices = cache["global_indices"].astype(np.int64, copy=False)
        cache_offsets = cache["seq_offsets"].astype(np.int64, copy=False)
        if labels is None:
            labels = cache_labels
            indices = cache_indices
            offsets = cache_offsets
        elif not np.array_equal(labels, cache_labels):
            raise RuntimeError(f"Label mismatch for {path}")
        elif not np.array_equal(indices, cache_indices):
            raise RuntimeError(f"Index mismatch for {path}")
        elif not np.array_equal(offsets, cache_offsets):
            raise RuntimeError(f"Sequence offset mismatch for {path}")
        matrix = cache["raw_score_matrix"].astype(np.float64, copy=False)
        for row_idx in range(matrix.shape[0]):
            rows.append(zscore_row(matrix[row_idx]))
            records.append({"cache": str(path), "row": int(row_idx)})
    if labels is None or offsets is None:
        raise RuntimeError("No score caches loaded")
    return np.stack(rows, axis=0), labels, offsets, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Search zscore-mean ensembles over cached sequence scores.")
    parser.add_argument("--score-caches", required=True, nargs="+", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-subset-size", type=int, default=5)
    parser.add_argument("--ma-windows", default="1,10,20,50,100")
    parser.add_argument("--debounce-on-k", default="2,3,5")
    parser.add_argument("--debounce-off-k", default="2,5,10,20")
    args = parser.parse_args()

    score_rows, labels, seq_offsets, records = load_score_rows(args.score_caches)
    methods: list[dict[str, Any]] = []
    candidate_subsets = []
    row_count = score_rows.shape[0]
    for size in range(1, min(args.max_subset_size, row_count) + 1):
        candidate_subsets.extend(itertools.combinations(range(row_count), size))

    for subset in candidate_subsets:
        base_scores = score_rows[list(subset)].mean(axis=0)
        members = [records[idx] for idx in subset]
        for ma_window in parse_csv_ints(args.ma_windows):
            scores = moving_average_per_sequence(base_scores, seq_offsets, ma_window)
            tuned = best_accuracy_threshold(labels, scores)
            threshold = float(tuned["threshold"])
            pred = scores > threshold
            methods.append(
                {
                    "method": f"subset_{'_'.join(map(str, subset))}_ma{ma_window}",
                    "members": members,
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
                            "method": f"subset_{'_'.join(map(str, subset))}_ma{ma_window}_debounce_on{on_k}_off{off_k}",
                            "members": members,
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
        "score_caches": [str(path) for path in args.score_caches],
        "row_records": records,
        "total_windows": int(labels.shape[0]),
        "searched_subsets": int(len(candidate_subsets)),
        "best_method": methods[0],
        "methods_top": methods[:100],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({"best_method": methods[0]["method"], "accuracy": methods[0]["metrics"]["accuracy"]}, sort_keys=True))


if __name__ == "__main__":
    main()
