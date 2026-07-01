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


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


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
            raise RuntimeError(f"Global-index mismatch for {path}")
        elif not np.array_equal(offsets, cache_offsets):
            raise RuntimeError(f"Sequence-offset mismatch for {path}")
        matrix = cache["raw_score_matrix"].astype(np.float64, copy=False)
        for row_idx in range(matrix.shape[0]):
            rows.append(zscore_row(matrix[row_idx]))
            records.append({"cache": str(path), "row": int(row_idx)})
    if labels is None or offsets is None:
        raise RuntimeError("No score caches loaded")
    return np.stack(rows, axis=0), labels, offsets, records


def causal_ema(scores: np.ndarray, seq_offsets: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty(scores.shape[0], dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        if start_i >= stop_i:
            continue
        out[start_i] = scores[start_i]
        for idx in range(start_i + 1, stop_i):
            out[idx] = float(alpha) * scores[idx] + (1.0 - float(alpha)) * out[idx - 1]
    return out


def centered_ma(scores: np.ndarray, seq_offsets: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores.astype(np.float64, copy=True)
    out = np.empty(scores.shape[0], dtype=np.float64)
    left = int(window) // 2
    right = int(window) - left
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq = scores[start_i:stop_i].astype(np.float64, copy=False)
        cumsum = np.cumsum(np.concatenate([[0.0], seq]))
        idx = np.arange(seq.shape[0])
        lo = np.maximum(0, idx - left)
        hi = np.minimum(seq.shape[0], idx + right)
        out[start_i:stop_i] = (cumsum[hi] - cumsum[lo]) / np.maximum(hi - lo, 1)
    return out


def fill_short_gaps(pred: np.ndarray, seq_offsets: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return pred.astype(bool, copy=True)
    out = pred.astype(bool, copy=True)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq = out[start_i:stop_i]
        n = seq.shape[0]
        idx = 0
        while idx < n:
            if seq[idx]:
                idx += 1
                continue
            gap_start = idx
            while idx < n and not seq[idx]:
                idx += 1
            gap_stop = idx
            if gap_start > 0 and gap_stop < n and gap_stop - gap_start <= max_gap:
                seq[gap_start:gap_stop] = True
    return out


def remove_short_runs(pred: np.ndarray, seq_offsets: np.ndarray, min_on: int, min_off: int) -> np.ndarray:
    out = pred.astype(bool, copy=True)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq = out[start_i:stop_i]
        n = seq.shape[0]
        idx = 0
        while idx < n:
            value = bool(seq[idx])
            run_start = idx
            while idx < n and bool(seq[idx]) == value:
                idx += 1
            run_stop = idx
            run_len = run_stop - run_start
            if value and min_on > 0 and run_len < min_on:
                seq[run_start:run_stop] = False
            elif (not value) and min_off > 0 and run_start > 0 and run_stop < n and run_len < min_off:
                seq[run_start:run_stop] = True
    return out


def threshold_grid(scores: np.ndarray, base_threshold: float, grid_size: int) -> np.ndarray:
    if grid_size <= 1:
        return np.asarray([base_threshold], dtype=np.float64)
    quantiles = np.linspace(0.002, 0.998, int(grid_size), dtype=np.float64)
    return np.unique(np.concatenate([np.quantile(scores, quantiles), np.asarray([base_threshold])]))


def evaluate_binary(labels: np.ndarray, pred: np.ndarray, seq_offsets: np.ndarray, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "params": params,
        "metrics": basic_binary_metrics(labels, pred),
        "event_metrics": aggregate_event_metrics(labels, pred, seq_offsets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search sequence-state post-processing on cached TacSpike scores.")
    parser.add_argument("--score-caches", required=True, nargs="+", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--ma-windows", default="1,20,50,80,100,150,200")
    parser.add_argument("--centered-ma-windows", default="")
    parser.add_argument("--ema-alphas", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--debounce-on-k", default="1,2,3,5,8,10")
    parser.add_argument("--debounce-off-k", default="2,5,10,20,30,50")
    parser.add_argument("--gap-fill", default="0,10,20,50,100")
    parser.add_argument("--min-on", default="0,10,20,50")
    parser.add_argument("--min-off", default="0")
    parser.add_argument("--threshold-grid-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    score_rows, labels, seq_offsets, row_records = load_score_rows(args.score_caches)
    row_count = int(score_rows.shape[0])
    subsets: list[tuple[int, ...]] = []
    for size in range(1, min(int(args.max_subset_size), row_count) + 1):
        subsets.extend(itertools.combinations(range(row_count), size))

    methods: list[dict[str, Any]] = []
    for subset in subsets:
        base_scores = score_rows[list(subset)].mean(axis=0)
        members = [row_records[idx] for idx in subset]
        series: dict[str, np.ndarray] = {"raw": base_scores}
        for window in parse_csv_ints(args.ma_windows):
            series[f"causal_ma{window}"] = moving_average_per_sequence(base_scores, seq_offsets, int(window))
        for window in parse_csv_ints(args.centered_ma_windows):
            series[f"centered_ma{window}"] = centered_ma(base_scores, seq_offsets, int(window))
        for alpha in parse_csv_floats(args.ema_alphas):
            series[f"ema{alpha:g}"] = causal_ema(base_scores, seq_offsets, float(alpha))

        for series_name, scores in series.items():
            tuned = best_accuracy_threshold(labels, scores)
            thresholds = threshold_grid(scores, float(tuned["threshold"]), int(args.threshold_grid_size))
            for threshold in thresholds:
                raw_pred = scores > float(threshold)
                methods.append(
                    evaluate_binary(
                        labels,
                        raw_pred,
                        seq_offsets,
                        f"subset_{'_'.join(map(str, subset))}_{series_name}_thr",
                        {
                            "members": members,
                            "series": series_name,
                            "threshold": float(threshold),
                        },
                    )
                )
                for on_k in parse_csv_ints(args.debounce_on_k):
                    for off_k in parse_csv_ints(args.debounce_off_k):
                        pred = debounce(scores, seq_offsets, float(threshold), int(on_k), int(off_k))
                        for gap in parse_csv_ints(args.gap_fill):
                            gap_pred = fill_short_gaps(pred, seq_offsets, int(gap))
                            for min_on in parse_csv_ints(args.min_on):
                                for min_off in parse_csv_ints(args.min_off):
                                    final_pred = remove_short_runs(gap_pred, seq_offsets, int(min_on), int(min_off))
                                    methods.append(
                                        evaluate_binary(
                                            labels,
                                            final_pred,
                                            seq_offsets,
                                            (
                                                f"subset_{'_'.join(map(str, subset))}_{series_name}"
                                                f"_thr_deb_on{on_k}_off{off_k}_gap{gap}_minon{min_on}_minoff{min_off}"
                                            ),
                                            {
                                                "members": members,
                                                "series": series_name,
                                                "threshold": float(threshold),
                                                "on_k": int(on_k),
                                                "off_k": int(off_k),
                                                "gap_fill": int(gap),
                                                "min_on": int(min_on),
                                                "min_off": int(min_off),
                                            },
                                        )
                                    )

    methods.sort(key=lambda item: float(item["metrics"]["accuracy"]), reverse=True)
    result = {
        "score_caches": [str(path) for path in args.score_caches],
        "row_records": row_records,
        "total_windows": int(labels.shape[0]),
        "label_positive_fraction": float(labels.mean()) if labels.size else 0.0,
        "searched_methods": int(len(methods)),
        "best_method": methods[0],
        "methods_top": methods[: int(args.top_k)],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "searched_methods": result["searched_methods"],
                "best_method": result["best_method"]["method"],
                "accuracy": result["best_method"]["metrics"]["accuracy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
