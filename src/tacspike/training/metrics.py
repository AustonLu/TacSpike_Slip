from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def binary_classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> Dict[str, float]:
    """Compute dependency-light binary metrics.

    scores are slip-vs-no-slip margins. Prediction is scores >= threshold.
    """

    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    y_pred = (scores > threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(y_true))
    balanced_accuracy = 0.5 * (recall + specificity)

    result = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }

    result["roc_auc"] = _roc_auc(y_true, scores)
    result["pr_auc"] = _average_precision(y_true, scores)
    return result


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int((y_true == 1).sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    precision = tp / (np.arange(len(sorted_true)) + 1)
    return float((precision * (sorted_true == 1)).sum() / positives)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
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
