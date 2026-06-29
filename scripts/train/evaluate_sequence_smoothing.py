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

from scripts.train.evaluate_lite_scnn import best_threshold
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


def causal_moving_average(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores.astype(np.float64, copy=True)
    scores64 = scores.astype(np.float64, copy=False)
    cumsum = np.cumsum(np.concatenate([[0.0], scores64]))
    out = np.empty_like(scores64)
    for i in range(scores64.shape[0]):
        start = max(0, i + 1 - window)
        out[i] = (cumsum[i + 1] - cumsum[start]) / (i + 1 - start)
    return out


def ema(scores: np.ndarray, alpha: float) -> np.ndarray:
    scores64 = scores.astype(np.float64, copy=False)
    if scores64.size == 0:
        return scores64.copy()
    out = np.empty_like(scores64)
    out[0] = scores64[0]
    for i in range(1, scores64.shape[0]):
        out[i] = alpha * scores64[i] + (1.0 - alpha) * out[i - 1]
    return out


def consecutive_trigger_scores(scores: np.ndarray, threshold: float, k: int) -> np.ndarray:
    binary = scores > threshold
    out = np.zeros(scores.shape[0], dtype=np.float64)
    run = 0
    active = False
    for i, value in enumerate(binary):
        if value:
            run += 1
            if run >= k:
                active = True
        else:
            run = 0
            active = False
        out[i] = 1.0 if active else 0.0
    return out


def parse_csv_ints(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def sequence_indices(base: TacSpikeH5Dataset, seq_idx: int, max_windows: int | None) -> np.ndarray:
    start = int(base.offsets[seq_idx])
    stop = int(base.offsets[seq_idx + 1])
    if max_windows is not None:
        stop = min(stop, start + max_windows)
    return np.arange(start, stop, dtype=np.int64)


def select_sequence_indices(total: int, max_sequences: int | None, seed: int) -> np.ndarray:
    if max_sequences is None or max_sequences >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(total, dtype=np.int64), size=max_sequences, replace=False))


def predict_sequence(
    model: torch.nn.Module,
    train_args: argparse.Namespace,
    data_root: Path,
    split: str,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
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

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            input_scale = float(getattr(loader.dataset, "input_scale", 1.0))
            if input_scale != 1.0:
                x = x * input_scale
            logits, _ = model(x)
            all_scores.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
            all_labels.append(y.numpy())
    return {
        "scores": np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float32),
        "labels": np.concatenate(all_labels) if all_labels else np.empty((0,), dtype=np.int64),
    }


def metrics_with_tuning(labels: np.ndarray, scores: np.ndarray, metric_name: str) -> Dict[str, Any]:
    return {
        "default": binary_classification_metrics(labels, scores, threshold=0.0),
        "best_threshold_metric": metric_name,
        "best": best_threshold(labels, scores, metric_name=metric_name),
    }


def transition_distances(labels: np.ndarray) -> np.ndarray:
    if labels.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    transitions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if transitions.size == 0:
        return np.full(labels.shape[0], np.inf, dtype=np.float64)
    positions = np.arange(labels.shape[0], dtype=np.int64)
    idx = np.searchsorted(transitions, positions)
    right = np.where(idx < transitions.size, transitions[np.minimum(idx, transitions.size - 1)], np.inf)
    left_idx = np.maximum(idx - 1, 0)
    left = np.where(idx > 0, transitions[left_idx], -np.inf)
    return np.minimum(np.abs(positions - left), np.abs(positions - right)).astype(np.float64)


def onset_bucket_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    distances = transition_distances(labels)
    buckets = {
        "0_10_ms": distances < 10,
        "10_20_ms": (distances >= 10) & (distances < 20),
        "20_50_ms": (distances >= 20) & (distances < 50),
        "gt_50_ms": distances >= 50,
        "no_transition": np.isinf(distances),
    }
    result: Dict[str, Any] = {}
    for name, mask in buckets.items():
        count = int(mask.sum())
        if count == 0:
            result[name] = {"count": 0}
            continue
        result[name] = {
            "count": count,
            "metrics": binary_classification_metrics(labels[mask], scores[mask], threshold=threshold),
        }
    return result


def detection_delays(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    predictions = scores > threshold
    onsets = np.flatnonzero((labels[1:] == 1) & (labels[:-1] == 0)) + 1
    offsets = np.flatnonzero((labels[1:] == 0) & (labels[:-1] == 1)) + 1
    delays = []
    missed = 0
    for onset in onsets:
        next_offsets = offsets[offsets > onset]
        end = int(next_offsets[0]) if next_offsets.size else labels.shape[0]
        hits = np.flatnonzero(predictions[onset:end])
        if hits.size:
            delays.append(float(hits[0]))
        else:
            missed += 1
    return {
        "num_onsets": int(onsets.size),
        "missed_onsets": int(missed),
        "mean_delay_ms": float(np.mean(delays)) if delays else None,
        "median_delay_ms": float(np.median(delays)) if delays else None,
        "max_delay_ms": float(np.max(delays)) if delays else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TacSpike checkpoints on complete sequences with smoothing.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--max-windows-per-sequence", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--best-threshold-metric", choices=("accuracy", "balanced_accuracy", "f1"), default="accuracy")
    parser.add_argument("--ma-windows", default="5,10,20,50")
    parser.add_argument("--ema-alphas", default="0.1,0.2,0.4")
    parser.add_argument("--trigger-ks", default="3,5,10")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ckpt["args"])
    train_args.data_root = args.data_root
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers
    for name, value in (
        ("model", "lite_scnn"),
        ("model_width", 32),
        ("hidden_dim", 128),
        ("time_steps", None),
        ("temporal_mode", "time_channels"),
        ("dropout", 0.1),
        ("context_ms", None),
        ("time_bins", None),
    ):
        if not hasattr(train_args, name):
            setattr(train_args, name, value)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    base = TacSpikeH5Dataset(data_root=args.data_root, split=args.split)
    selected = select_sequence_indices(len(base.sequences), args.max_sequences, args.seed)

    sequence_records = []
    all_scores = []
    all_labels = []
    for seq_idx in selected:
        indices = sequence_indices(base, int(seq_idx), args.max_windows_per_sequence)
        pred = predict_sequence(
            model=model,
            train_args=train_args,
            data_root=args.data_root,
            split=args.split,
            indices=indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        scores = pred["scores"]
        labels = pred["labels"]
        info = base.sequences[int(seq_idx)]
        all_scores.append(scores)
        all_labels.append(labels)
        sequence_records.append(
            {
                "sequence_id": info.sequence_id,
                "windows": int(labels.shape[0]),
                "label_positive_fraction": float(labels.mean()) if labels.size else 0.0,
                "raw_default": binary_classification_metrics(labels, scores, threshold=0.0),
                "num_transitions": int(np.count_nonzero(labels[1:] != labels[:-1])) if labels.size > 1 else 0,
            }
        )

    labels = np.concatenate(all_labels) if all_labels else np.empty((0,), dtype=np.int64)
    scores = np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float32)
    raw = metrics_with_tuning(labels, scores, args.best_threshold_metric)
    best_threshold_value = float(raw["best"]["threshold"])

    smoothing: Dict[str, Any] = {}
    for window in parse_csv_ints(args.ma_windows):
        smoothed = causal_moving_average(scores, window)
        smoothing[f"ma_{window}"] = metrics_with_tuning(labels, smoothed, args.best_threshold_metric)
    for alpha in parse_csv_floats(args.ema_alphas):
        smoothed = ema(scores, alpha)
        smoothing[f"ema_{alpha:g}"] = metrics_with_tuning(labels, smoothed, args.best_threshold_metric)
    for k in parse_csv_ints(args.trigger_ks):
        triggered = consecutive_trigger_scores(scores, best_threshold_value, k)
        smoothing[f"trigger_k{k}"] = {
            "threshold_source": "raw_best_threshold",
            "raw_threshold": best_threshold_value,
            "metrics": binary_classification_metrics(labels, triggered, threshold=0.5),
        }

    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "selected_sequences": int(len(selected)),
        "total_windows": int(labels.shape[0]),
        "train_args": vars(train_args),
        "raw": raw,
        "smoothing": smoothing,
        "onset_buckets": onset_bucket_metrics(labels, scores, best_threshold_value),
        "detection_delays": detection_delays(labels, scores, best_threshold_value),
        "sequences": sequence_records,
    }

    base.close()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("checkpoint", "split", "selected_sequences", "total_windows")}, sort_keys=True))


if __name__ == "__main__":
    main()
