from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.train.evaluate_checkpoint_ensemble import fill_legacy_args
from scripts.train.train_lite_scnn import build_model
from tacspike.data import IndexedTacSpikeDataset, TacSpikeH5Dataset
from tacspike.training import binary_classification_metrics


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


def parse_csv_ints(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_weights(text: str | None, n: int) -> np.ndarray:
    if text is None or not text.strip():
        return np.full(n, 1.0 / n, dtype=np.float64)
    weights = np.asarray(parse_csv_floats(text), dtype=np.float64)
    if weights.shape[0] != n:
        raise ValueError(f"Expected {n} weights, got {weights.shape[0]}")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return weights / total


def checkpoint_name(path: Path) -> str:
    return path.parent.name if path.name == "best.pt" else path.stem


def select_sequence_indices(total: int, max_sequences: int, seed: int) -> np.ndarray:
    if max_sequences <= 0 or max_sequences >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(total, dtype=np.int64), size=max_sequences, replace=False))


def sequence_global_indices(base: TacSpikeH5Dataset, seq_idx: int, max_windows: int) -> np.ndarray:
    start = int(base.offsets[seq_idx])
    stop = int(base.offsets[seq_idx + 1])
    if max_windows > 0:
        stop = min(stop, start + max_windows)
    return np.arange(start, stop, dtype=np.int64)


def load_model_and_args(
    checkpoint: Path,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.nn.Module, argparse.Namespace]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    train_args = fill_legacy_args(argparse.Namespace(**ckpt["args"]))
    train_args.data_root = data_root
    train_args.batch_size = batch_size
    train_args.num_workers = num_workers
    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, train_args


def predict_scores(
    model: torch.nn.Module,
    train_args: argparse.Namespace,
    data_root: Path,
    split: str,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = IndexedTacSpikeDataset(
        data_root=data_root,
        split=split,
        indices=indices,
        polarity_mode=train_args.polarity_mode,
        clip_max=train_args.clip_max,
        spatial_pool=train_args.spatial_pool,
        context_ms=getattr(train_args, "context_ms", None),
        time_bins=getattr(train_args, "time_bins", None),
    )
    dataset.input_scale = getattr(train_args, "input_scale", 1.0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    scores = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            input_scale = float(getattr(loader.dataset, "input_scale", 1.0))
            if input_scale != 1.0:
                x = x * input_scale
            logits, _ = model(x)
            scores.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
            labels.append(y.numpy())
    return (
        np.concatenate(scores) if scores else np.empty((0,), dtype=np.float32),
        np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64),
    )


def transform_score_matrix(score_matrix: np.ndarray, mode: str) -> np.ndarray:
    scores = score_matrix.astype(np.float64, copy=True)
    if mode == "raw":
        return scores
    if mode == "zscore":
        mean = scores.mean(axis=1, keepdims=True)
        std = scores.std(axis=1, keepdims=True)
        return (scores - mean) / np.maximum(std, 1e-12)
    if mode == "minmax":
        lo = scores.min(axis=1, keepdims=True)
        hi = scores.max(axis=1, keepdims=True)
        return (scores - lo) / np.maximum(hi - lo, 1e-12)
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


def per_sequence_causal_ma(scores: np.ndarray, seq_offsets: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores.astype(np.float64, copy=True)
    out = np.empty_like(scores, dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq = scores[start_i:stop_i].astype(np.float64, copy=False)
        cumsum = np.cumsum(np.concatenate([[0.0], seq]))
        positions = np.arange(seq.shape[0])
        left = np.maximum(0, positions + 1 - window)
        out[start_i:stop_i] = (cumsum[positions + 1] - cumsum[left]) / (positions + 1 - left)
    return out


def per_sequence_ema(scores: np.ndarray, seq_offsets: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(scores, dtype=np.float64)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        if start_i >= stop_i:
            continue
        out[start_i] = scores[start_i]
        for idx in range(start_i + 1, stop_i):
            out[idx] = alpha * scores[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def per_sequence_debounce(
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


def per_sequence_debounce_many(
    scores: np.ndarray,
    seq_offsets: np.ndarray,
    thresholds: np.ndarray,
    on_k: int,
    off_k: int,
) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    out = np.zeros((scores.shape[0], thresholds.shape[0]), dtype=bool)
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        state = np.zeros(thresholds.shape[0], dtype=bool)
        pos_run = np.zeros(thresholds.shape[0], dtype=np.int16)
        neg_run = np.zeros(thresholds.shape[0], dtype=np.int16)
        for idx in range(int(start), int(stop)):
            positive = scores[idx] > thresholds
            pos_run = np.where(positive, pos_run + 1, 0)
            neg_run = np.where(positive, 0, neg_run + 1)
            state = np.where((~state) & (pos_run >= on_k), True, state)
            state = np.where(state & (neg_run >= off_k), False, state)
            out[idx] = state
    return out


def threshold_candidates(scores: np.ndarray, base_threshold: float, grid_size: int) -> np.ndarray:
    if grid_size <= 1:
        return np.asarray([base_threshold], dtype=np.float64)
    quantiles = np.linspace(0.005, 0.995, int(grid_size), dtype=np.float64)
    values = np.quantile(scores.astype(np.float64, copy=False), quantiles)
    values = np.concatenate([values, np.asarray([base_threshold], dtype=np.float64)])
    return np.unique(values)


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


def event_metrics(labels: np.ndarray, predictions: np.ndarray, time_per_window_ms: float = 1.0) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions) > 0
    if labels.size == 0:
        return {
            "true_slip_segments": 0,
            "detected_slip_segments": 0,
            "missed_slip_segments": 0,
            "false_alarm_runs": 0,
        }

    slip = labels == 1
    starts = np.flatnonzero(slip & np.concatenate([[True], ~slip[:-1]]))
    stops = np.flatnonzero(slip & np.concatenate([~slip[1:], [True]])) + 1
    delays = []
    missed = 0
    for start, stop in zip(starts, stops):
        hits = np.flatnonzero(pred[int(start) : int(stop)])
        if hits.size:
            delays.append(float(hits[0]) * time_per_window_ms)
        else:
            missed += 1

    pred_starts = np.flatnonzero(pred & np.concatenate([[True], ~pred[:-1]]))
    pred_stops = np.flatnonzero(pred & np.concatenate([~pred[1:], [True]])) + 1
    false_alarm_runs = 0
    early_or_false_runs = 0
    for start, stop in zip(pred_starts, pred_stops):
        segment_labels = slip[int(start) : int(stop)]
        if not bool(segment_labels.any()):
            false_alarm_runs += 1
        if not bool(slip[int(start)]):
            early_or_false_runs += 1

    no_slip = ~slip
    duration_min = labels.shape[0] * time_per_window_ms / 60000.0
    false_positive_windows = int((pred & no_slip).sum())
    true_positive_windows = int((pred & slip).sum())
    return {
        "true_slip_segments": int(starts.shape[0]),
        "detected_slip_segments": int(starts.shape[0] - missed),
        "missed_slip_segments": int(missed),
        "segment_recall": float((starts.shape[0] - missed) / max(starts.shape[0], 1)),
        "delay_mean_ms": float(np.mean(delays)) if delays else None,
        "delay_median_ms": float(np.median(delays)) if delays else None,
        "delay_p95_ms": float(np.percentile(delays, 95)) if delays else None,
        "delay_max_ms": float(np.max(delays)) if delays else None,
        "false_alarm_runs": int(false_alarm_runs),
        "predicted_runs_starting_in_no_slip": int(early_or_false_runs),
        "false_alarm_runs_per_min": float(false_alarm_runs / max(duration_min, 1e-12)),
        "false_positive_windows": false_positive_windows,
        "false_positive_window_fraction": float(false_positive_windows / max(int(no_slip.sum()), 1)),
        "slip_positive_window_fraction": float(true_positive_windows / max(int(slip.sum()), 1)),
        "prediction_switches": int(np.count_nonzero(pred[1:] != pred[:-1])) if pred.size > 1 else 0,
        "label_transitions": int(np.count_nonzero(labels[1:] != labels[:-1])) if labels.size > 1 else 0,
    }


def transition_bucket_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    distances: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    buckets = {
        "lt_20_ms": distances < 20,
        "20_50_ms": (distances >= 20) & (distances < 50),
        "50_100_ms": (distances >= 50) & (distances < 100),
        "gt_100_ms": distances >= 100,
    }
    result: Dict[str, Any] = {}
    for name, mask in buckets.items():
        count = int(mask.sum())
        if count == 0:
            result[name] = {"count": 0}
            continue
        result[name] = {
            "count": count,
            "fraction": float(count / max(labels.shape[0], 1)),
            "metrics": binary_classification_metrics(labels[mask], scores[mask], threshold=threshold),
        }
    return result


def score_method_record(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    seq_offsets: np.ndarray,
    distances: np.ndarray,
) -> Dict[str, Any]:
    tuned = best_accuracy_threshold(labels, scores)
    threshold = float(tuned["threshold"])
    predictions = scores > threshold
    return {
        "method": name,
        "kind": "score",
        "threshold": threshold,
        "default_metrics": binary_classification_metrics(labels, scores, threshold=0.0),
        "best_threshold_metrics": tuned,
        "event_metrics": aggregate_event_metrics(labels, predictions, seq_offsets),
        "transition_buckets": transition_bucket_metrics(labels, scores, distances, threshold),
    }


def binary_method_record(
    name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    seq_offsets: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "method": name,
        "kind": "binary_state",
        "params": params,
        "metrics": binary_classification_metrics(labels, predictions, threshold=0.5),
        "event_metrics": aggregate_event_metrics(labels, predictions > 0.5, seq_offsets),
    }


def best_debounce_record(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    seq_offsets: np.ndarray,
    base_threshold: float,
    on_k: int,
    off_k: int,
    grid_size: int,
) -> Dict[str, Any]:
    candidates = threshold_candidates(scores, base_threshold, grid_size)
    if candidates.shape[0] == 1:
        pred = per_sequence_debounce(scores, seq_offsets, float(candidates[0]), on_k, off_k)
        return binary_method_record(
            name,
            labels,
            pred,
            seq_offsets,
            {"base": name.rsplit("_debounce_", 1)[0], "threshold": float(candidates[0]), "on_k": on_k, "off_k": off_k},
        )

    pred_many = per_sequence_debounce_many(scores, seq_offsets, candidates, on_k, off_k)
    labels_bool = labels.astype(bool)
    accuracy = (pred_many == labels_bool[:, None]).mean(axis=0)
    best_idx = int(np.argmax(accuracy))
    pred = pred_many[:, best_idx].astype(np.float64)
    record = binary_method_record(
        name,
        labels,
        pred,
        seq_offsets,
        {
            "base": name.rsplit("_debounce_", 1)[0],
            "threshold": float(candidates[best_idx]),
            "on_k": on_k,
            "off_k": off_k,
            "threshold_grid_size": int(candidates.shape[0]),
            "threshold_selection_metric": "accuracy",
        },
    )
    record["params"]["base_threshold"] = float(base_threshold)
    record["params"]["candidate_accuracy"] = float(accuracy[best_idx])
    return record


def aggregate_event_metrics(labels: np.ndarray, predictions: np.ndarray, seq_offsets: np.ndarray) -> Dict[str, Any]:
    totals = {
        "true_slip_segments": 0,
        "detected_slip_segments": 0,
        "missed_slip_segments": 0,
        "false_alarm_runs": 0,
        "predicted_runs_starting_in_no_slip": 0,
        "false_positive_windows": 0,
        "prediction_switches": 0,
        "label_transitions": 0,
    }
    delays: list[float] = []
    no_slip_windows = 0
    slip_windows = 0
    true_positive_windows = 0
    false_positive_windows = 0
    total_windows = 0
    for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
        start_i = int(start)
        stop_i = int(stop)
        seq_labels = labels[start_i:stop_i]
        seq_pred = predictions[start_i:stop_i]
        metrics = event_metrics(seq_labels, seq_pred)
        for key in totals:
            totals[key] += int(metrics.get(key, 0))
        if metrics.get("delay_mean_ms") is not None:
            # Recompute exact delays for aggregation instead of averaging means.
            slip = seq_labels == 1
            pred = np.asarray(seq_pred) > 0
            starts = np.flatnonzero(slip & np.concatenate([[True], ~slip[:-1]]))
            stops = np.flatnonzero(slip & np.concatenate([~slip[1:], [True]])) + 1
            for seg_start, seg_stop in zip(starts, stops):
                hits = np.flatnonzero(pred[int(seg_start) : int(seg_stop)])
                if hits.size:
                    delays.append(float(hits[0]))
        no_slip = seq_labels == 0
        slip = seq_labels == 1
        no_slip_windows += int(no_slip.sum())
        slip_windows += int(slip.sum())
        true_positive_windows += int((seq_pred.astype(bool) & slip).sum())
        false_positive_windows += int((seq_pred.astype(bool) & no_slip).sum())
        total_windows += int(seq_labels.shape[0])

    duration_min = total_windows / 60000.0
    return {
        **totals,
        "segment_recall": float(totals["detected_slip_segments"] / max(totals["true_slip_segments"], 1)),
        "delay_mean_ms": float(np.mean(delays)) if delays else None,
        "delay_median_ms": float(np.median(delays)) if delays else None,
        "delay_p95_ms": float(np.percentile(delays, 95)) if delays else None,
        "delay_max_ms": float(np.max(delays)) if delays else None,
        "false_alarm_runs_per_min": float(totals["false_alarm_runs"] / max(duration_min, 1e-12)),
        "false_positive_window_fraction": float(false_positive_windows / max(no_slip_windows, 1)),
        "slip_positive_window_fraction": float(true_positive_windows / max(slip_windows, 1)),
    }


def per_sequence_summary(
    sequence_records: list[Dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    seq_offsets: np.ndarray,
) -> list[Dict[str, Any]]:
    records = []
    for idx, record in enumerate(sequence_records):
        start = int(seq_offsets[idx])
        stop = int(seq_offsets[idx + 1])
        seq_labels = labels[start:stop]
        seq_scores = scores[start:stop]
        seq_pred = predictions[start:stop]
        records.append(
            {
                **record,
                "window_metrics": binary_classification_metrics(seq_labels, seq_scores, threshold=0.0),
                "event_metrics": event_metrics(seq_labels, seq_pred),
            }
        )
    records.sort(key=lambda item: item["window_metrics"]["accuracy"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate continuous sliding/sequence TacSpike detection.")
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-windows-per-sequence", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-transform", choices=("raw", "zscore", "minmax", "rank_centered"), default="raw")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--ma-windows", default="3,5,10,20,50")
    parser.add_argument("--ema-alphas", default="0.1,0.2,0.4")
    parser.add_argument("--debounce-on-k", default="2,3,5")
    parser.add_argument("--debounce-off-k", default="2,3,5,10")
    parser.add_argument("--debounce-threshold-grid", type=int, default=1)
    parser.add_argument("--score-cache", type=Path, default=None)
    parser.add_argument("--output-score-cache", type=Path, default=None)
    parser.add_argument("--top-sequences", type=int, default=30)
    args = parser.parse_args()
    if args.score_cache is None and not args.checkpoints:
        parser.error("--checkpoints is required unless --score-cache is provided")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base = TacSpikeH5Dataset(data_root=args.data_root, split=args.split)
    selected = select_sequence_indices(len(base.sequences), args.max_sequences, args.seed)
    seq_offsets = [0]
    sequence_records: list[Dict[str, Any]] = []
    indices_parts = []
    for seq_idx in selected:
        indices = sequence_global_indices(base, int(seq_idx), args.max_windows_per_sequence)
        info = base.sequences[int(seq_idx)]
        indices_parts.append(indices)
        seq_offsets.append(seq_offsets[-1] + int(indices.shape[0]))
        sequence_records.append(
            {
                "sequence_id": info.sequence_id,
                "sequence_index": int(seq_idx),
                "windows": int(indices.shape[0]),
            }
        )
    all_indices = np.concatenate(indices_parts) if indices_parts else np.empty((0,), dtype=np.int64)
    seq_offsets_np = np.asarray(seq_offsets, dtype=np.int64)
    base.close()

    if args.score_cache is not None:
        cache = np.load(args.score_cache, allow_pickle=False)
        raw_score_matrix = cache["raw_score_matrix"].astype(np.float64, copy=False)
        labels = cache["labels"].astype(np.int64, copy=False)
        cached_indices = cache["global_indices"].astype(np.int64, copy=False)
        cached_offsets = cache["seq_offsets"].astype(np.int64, copy=False)
        if not np.array_equal(cached_indices, all_indices):
            raise RuntimeError("Score cache global indices do not match current sequence selection")
        if not np.array_equal(cached_offsets, seq_offsets_np):
            raise RuntimeError("Score cache sequence offsets do not match current sequence selection")
        model_records = json.loads(str(cache["model_records_json"].item()))
    else:
        score_parts = []
        labels = None
        model_records = []
        for checkpoint in args.checkpoints:
            model, train_args = load_model_and_args(
                checkpoint=checkpoint,
                data_root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            scores, current_labels = predict_scores(
                model=model,
                train_args=train_args,
                data_root=args.data_root,
                split=args.split,
                indices=all_indices,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            score_parts.append(scores.astype(np.float64, copy=False))
            if labels is None:
                labels = current_labels.astype(np.int64, copy=False)
            elif not np.array_equal(labels, current_labels):
                raise RuntimeError(f"Label mismatch for checkpoint {checkpoint}")
            model_records.append(
                {
                    "checkpoint": str(checkpoint),
                    "name": checkpoint_name(checkpoint),
                    "args": vars(train_args),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if labels is None:
            raise RuntimeError("No labels were produced")
        raw_score_matrix = np.stack(score_parts, axis=0)
        if args.output_score_cache is not None:
            args.output_score_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output_score_cache,
                raw_score_matrix=raw_score_matrix.astype(np.float32),
                labels=labels.astype(np.int64),
                global_indices=all_indices.astype(np.int64),
                seq_offsets=seq_offsets_np.astype(np.int64),
                selected_sequences=selected.astype(np.int64),
                model_records_json=np.asarray(json.dumps(model_records, sort_keys=True, default=json_default)),
            )

    weights = parse_weights(args.weights, raw_score_matrix.shape[0])
    transformed = transform_score_matrix(raw_score_matrix, args.score_transform)
    scores = weights @ transformed
    distances = transition_distances_per_sequence(labels, seq_offsets_np)

    methods: list[Dict[str, Any]] = []
    score_series = {"raw": scores}
    for window in parse_csv_ints(args.ma_windows):
        score_series[f"ma_{window}"] = per_sequence_causal_ma(scores, seq_offsets_np, window)
    for alpha in parse_csv_floats(args.ema_alphas):
        score_series[f"ema_{alpha:g}"] = per_sequence_ema(scores, seq_offsets_np, alpha)

    for name, method_scores in score_series.items():
        record = score_method_record(name, labels, method_scores, seq_offsets_np, distances)
        methods.append(record)
        threshold = float(record["threshold"])
        for on_k in parse_csv_ints(args.debounce_on_k):
            for off_k in parse_csv_ints(args.debounce_off_k):
                methods.append(
                    best_debounce_record(
                        f"{name}_debounce_on{on_k}_off{off_k}",
                        labels,
                        method_scores,
                        seq_offsets_np,
                        threshold,
                        on_k,
                        off_k,
                        args.debounce_threshold_grid,
                    )
                )

    def method_accuracy(record: Dict[str, Any]) -> float:
        if record["kind"] == "score":
            return float(record["best_threshold_metrics"]["accuracy"])
        return float(record["metrics"]["accuracy"])

    methods.sort(key=method_accuracy, reverse=True)
    best_method = methods[0]
    if best_method["kind"] == "score":
        best_scores = score_series[best_method["method"]]
        best_predictions = best_scores > float(best_method["threshold"])
    else:
        params = best_method["params"]
        best_scores = score_series[params["base"]]
        best_predictions = per_sequence_debounce(
            best_scores,
            seq_offsets_np,
            float(params["threshold"]),
            int(params["on_k"]),
            int(params["off_k"]),
        ) > 0.5

    result = {
        "split": args.split,
        "selected_sequences": int(selected.shape[0]),
        "total_windows": int(labels.shape[0]),
        "checkpoints": model_records,
        "weights": [float(x) for x in weights.tolist()],
        "score_transform": args.score_transform,
        "score_cache": str(args.score_cache) if args.score_cache is not None else None,
        "output_score_cache": str(args.output_score_cache) if args.output_score_cache is not None else None,
        "debounce_threshold_grid": int(args.debounce_threshold_grid),
        "label_positive_fraction": float(labels.mean()) if labels.size else 0.0,
        "methods_top": methods[:50],
        "best_method": best_method,
        "worst_sequences_by_best_method": per_sequence_summary(
            sequence_records,
            labels,
            best_scores,
            best_predictions,
            seq_offsets_np,
        )[: args.top_sequences],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_sequences": result["selected_sequences"],
                "total_windows": result["total_windows"],
                "best_method": best_method["method"],
                "best_accuracy": method_accuracy(best_method),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
