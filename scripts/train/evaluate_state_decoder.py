from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def parse_csv_floats(text: str) -> List[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    rank = 1.0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1.0) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += j - i
        i = j
    pos_rank_sum = float(ranks[y_true == 1].sum())
    return float((pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int((y_true == 1).sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    precision = tp / (np.arange(len(sorted_true)) + 1)
    return float((precision * (sorted_true == 1)).sum() / positives)


def binary_classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    y_pred = scores > threshold
    tp = int(((y_true == 1) & y_pred).sum())
    tn = int(((y_true == 0) & ~y_pred).sum())
    fp = int(((y_true == 0) & y_pred).sum())
    fn = int(((y_true == 1) & ~y_pred).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": safe_div(tp + tn, y_true.shape[0]),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "roc_auc": roc_auc(y_true, scores),
        "pr_auc": average_precision(y_true, scores),
    }


def basic_binary_metrics(y_true: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(predictions) > 0
    tp = int(((y_true == 1) & y_pred).sum())
    tn = int(((y_true == 0) & ~y_pred).sum())
    fp = int(((y_true == 0) & y_pred).sum())
    fn = int(((y_true == 1) & ~y_pred).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": safe_div(tp + tn, y_true.shape[0]),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def event_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions) > 0
    if labels.size == 0:
        return {"true_slip_segments": 0, "detected_slip_segments": 0, "missed_slip_segments": 0}
    slip = labels == 1
    starts = np.flatnonzero(slip & np.concatenate([[True], ~slip[:-1]]))
    stops = np.flatnonzero(slip & np.concatenate([~slip[1:], [True]])) + 1
    delays = []
    missed = 0
    for start, stop in zip(starts, stops):
        hits = np.flatnonzero(pred[int(start) : int(stop)])
        if hits.size:
            delays.append(float(hits[0]))
        else:
            missed += 1
    pred_starts = np.flatnonzero(pred & np.concatenate([[True], ~pred[:-1]]))
    pred_stops = np.flatnonzero(pred & np.concatenate([~pred[1:], [True]])) + 1
    false_alarm_runs = 0
    early_or_false_runs = 0
    for start, stop in zip(pred_starts, pred_stops):
        if not bool(slip[int(start) : int(stop)].any()):
            false_alarm_runs += 1
        if not bool(slip[int(start)]):
            early_or_false_runs += 1
    no_slip = ~slip
    return {
        "true_slip_segments": int(starts.shape[0]),
        "detected_slip_segments": int(starts.shape[0] - missed),
        "missed_slip_segments": int(missed),
        "segment_recall": safe_div(starts.shape[0] - missed, starts.shape[0]),
        "delay_mean_ms": float(np.mean(delays)) if delays else None,
        "delay_median_ms": float(np.median(delays)) if delays else None,
        "delay_p95_ms": float(np.percentile(delays, 95)) if delays else None,
        "delay_max_ms": float(np.max(delays)) if delays else None,
        "false_alarm_runs": int(false_alarm_runs),
        "predicted_runs_starting_in_no_slip": int(early_or_false_runs),
        "false_positive_windows": int((pred & no_slip).sum()),
        "slip_positive_windows": int((pred & slip).sum()),
        "no_slip_windows": int(no_slip.sum()),
        "slip_windows": int(slip.sum()),
        "prediction_switches": int(np.count_nonzero(pred[1:] != pred[:-1])) if pred.size > 1 else 0,
        "label_transitions": int(np.count_nonzero(labels[1:] != labels[:-1])) if labels.size > 1 else 0,
    }


def aggregate_event_metrics(labels: np.ndarray, predictions: np.ndarray, seq_offsets: np.ndarray) -> Dict[str, Any]:
    totals = {
        "true_slip_segments": 0,
        "detected_slip_segments": 0,
        "missed_slip_segments": 0,
        "false_alarm_runs": 0,
        "predicted_runs_starting_in_no_slip": 0,
        "false_positive_windows": 0,
        "slip_positive_windows": 0,
        "no_slip_windows": 0,
        "slip_windows": 0,
        "prediction_switches": 0,
        "label_transitions": 0,
    }
    delays: list[float] = []
    total_windows = 0
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        metrics = event_metrics(labels[start_i:stop_i], predictions[start_i:stop_i])
        for key in totals:
            totals[key] += int(metrics.get(key, 0))
        if metrics.get("delay_mean_ms") is not None:
            slip = labels[start_i:stop_i] == 1
            pred = np.asarray(predictions[start_i:stop_i]) > 0
            starts = np.flatnonzero(slip & np.concatenate([[True], ~slip[:-1]]))
            stops = np.flatnonzero(slip & np.concatenate([~slip[1:], [True]])) + 1
            for seg_start, seg_stop in zip(starts, stops):
                hits = np.flatnonzero(pred[int(seg_start) : int(seg_stop)])
                if hits.size:
                    delays.append(float(hits[0]))
        total_windows += stop_i - start_i
    duration_min = total_windows / 60000.0
    return {
        **totals,
        "segment_recall": safe_div(totals["detected_slip_segments"], totals["true_slip_segments"]),
        "delay_mean_ms": float(np.mean(delays)) if delays else None,
        "delay_median_ms": float(np.median(delays)) if delays else None,
        "delay_p95_ms": float(np.percentile(delays, 95)) if delays else None,
        "delay_max_ms": float(np.max(delays)) if delays else None,
        "false_alarm_runs_per_min": safe_div(totals["false_alarm_runs"], duration_min),
        "false_positive_window_fraction": safe_div(totals["false_positive_windows"], totals["no_slip_windows"]),
        "slip_positive_window_fraction": safe_div(totals["slip_positive_windows"], totals["slip_windows"]),
    }


def moving_average_per_sequence(scores: np.ndarray, seq_offsets: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores.astype(np.float64, copy=True)
    out = np.empty(scores.shape[0], dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq = scores[start_i:stop_i].astype(np.float64, copy=False)
        cumsum = np.cumsum(np.concatenate([[0.0], seq]))
        idx = np.arange(seq.shape[0])
        left = np.maximum(0, idx + 1 - window)
        out[start_i:stop_i] = (cumsum[idx + 1] - cumsum[left]) / (idx + 1 - left)
    return out


def best_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
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
    threshold = float(sorted_scores[splits[best_idx] - 1])
    return {"threshold": threshold, **binary_classification_metrics(labels, scores, threshold=threshold)}


def viterbi_binary(
    scores: np.ndarray,
    threshold: float,
    positive_scale: float,
    switch_on_cost: float,
    switch_off_cost: float,
    prior_logit: float,
) -> np.ndarray:
    evidence = (scores.astype(np.float64, copy=False) - float(threshold)) * float(positive_scale) + float(prior_logit)
    n = evidence.shape[0]
    if n == 0:
        return np.empty((0,), dtype=bool)
    dp0 = np.empty(n, dtype=np.float64)
    dp1 = np.empty(n, dtype=np.float64)
    prev0 = np.zeros(n, dtype=np.int8)
    prev1 = np.zeros(n, dtype=np.int8)
    dp0[0] = 0.0
    dp1[0] = evidence[0] - switch_on_cost
    for idx in range(1, n):
        stay0 = dp0[idx - 1]
        off = dp1[idx - 1] - switch_off_cost
        if stay0 >= off:
            dp0[idx] = stay0
            prev0[idx] = 0
        else:
            dp0[idx] = off
            prev0[idx] = 1
        on = dp0[idx - 1] - switch_on_cost
        stay1 = dp1[idx - 1]
        if stay1 >= on:
            dp1[idx] = stay1 + evidence[idx]
            prev1[idx] = 1
        else:
            dp1[idx] = on + evidence[idx]
            prev1[idx] = 0
    state = 1 if dp1[-1] > dp0[-1] else 0
    pred = np.empty(n, dtype=bool)
    for idx in range(n - 1, -1, -1):
        pred[idx] = bool(state)
        state = int(prev1[idx] if state else prev0[idx])
    return pred


def decode_per_sequence(
    scores: np.ndarray,
    seq_offsets: np.ndarray,
    threshold: float,
    positive_scale: float,
    switch_on_cost: float,
    switch_off_cost: float,
    prior_logit: float,
) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=bool)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        pred[start_i:stop_i] = viterbi_binary(
            scores[start_i:stop_i],
            threshold=threshold,
            positive_scale=positive_scale,
            switch_on_cost=switch_on_cost,
            switch_off_cost=switch_off_cost,
            prior_logit=prior_logit,
        )
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode TacSpike sequence scores with a binary state model.")
    parser.add_argument("--score-cache", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--score-index", type=int, default=0)
    parser.add_argument("--ma-windows", default="1,3,5,10,20,50,100")
    parser.add_argument("--scales", default="0.25,0.5,1,2,4")
    parser.add_argument("--switch-on-costs", default="0,0.5,1,2,4,8,16,32")
    parser.add_argument("--switch-off-costs", default="0,0.5,1,2,4,8,16,32")
    parser.add_argument("--prior-logits", default="-2,-1,0,1")
    args = parser.parse_args()

    cache = np.load(args.score_cache, allow_pickle=False)
    raw_score_matrix = cache["raw_score_matrix"].astype(np.float64, copy=False)
    labels = cache["labels"].astype(np.int64, copy=False)
    seq_offsets = cache["seq_offsets"].astype(np.int64, copy=False)
    scores_raw = raw_score_matrix[int(args.score_index)]

    methods: list[Dict[str, Any]] = []
    for ma_window in parse_csv_ints(args.ma_windows):
        scores = moving_average_per_sequence(scores_raw, seq_offsets, ma_window)
        tuned = best_accuracy_threshold(labels, scores)
        threshold = float(tuned["threshold"])
        for scale in parse_csv_floats(args.scales):
            for on_cost in parse_csv_floats(args.switch_on_costs):
                for off_cost in parse_csv_floats(args.switch_off_costs):
                    for prior in parse_csv_floats(args.prior_logits):
                        pred = decode_per_sequence(
                            scores,
                            seq_offsets,
                            threshold=threshold,
                            positive_scale=scale,
                            switch_on_cost=on_cost,
                            switch_off_cost=off_cost,
                            prior_logit=prior,
                        )
                        metrics = basic_binary_metrics(labels, pred)
                        event = aggregate_event_metrics(labels, pred, seq_offsets)
                        methods.append(
                            {
                                "method": f"viterbi_ma{ma_window}_scale{scale:g}_on{on_cost:g}_off{off_cost:g}_prior{prior:g}",
                                "ma_window": int(ma_window),
                                "threshold": threshold,
                                "scale": float(scale),
                                "switch_on_cost": float(on_cost),
                                "switch_off_cost": float(off_cost),
                                "prior_logit": float(prior),
                                "metrics": metrics,
                                "event_metrics": event,
                            }
                        )

    methods.sort(key=lambda item: float(item["metrics"]["accuracy"]), reverse=True)
    result = {
        "score_cache": str(args.score_cache),
        "total_windows": int(labels.shape[0]),
        "label_positive_fraction": float(labels.mean()) if labels.size else 0.0,
        "best_method": methods[0],
        "methods_top": methods[:100],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({"best_method": methods[0]["method"], "accuracy": methods[0]["metrics"]["accuracy"]}, sort_keys=True))


if __name__ == "__main__":
    main()
