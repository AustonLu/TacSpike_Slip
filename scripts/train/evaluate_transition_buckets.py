from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.train.evaluate_lite_scnn import best_threshold
from scripts.train.train_lite_scnn import build_model, make_loader, order_indices_for_io
from tacspike.data import TacSpikeH5Dataset, sample_epoch_indices
from tacspike.training import binary_classification_metrics


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def transition_distances(labels: np.ndarray) -> np.ndarray:
    if labels.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    transitions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if transitions.size == 0:
        return np.full(labels.shape[0], np.inf, dtype=np.float32)
    positions = np.arange(labels.shape[0], dtype=np.int64)
    idx = np.searchsorted(transitions, positions)
    right = np.where(idx < transitions.size, transitions[np.minimum(idx, transitions.size - 1)], np.inf)
    left_idx = np.maximum(idx - 1, 0)
    left = np.where(idx > 0, transitions[left_idx], -np.inf)
    return np.minimum(np.abs(positions - left), np.abs(positions - right)).astype(np.float32)


def build_distance_lookup(data_root: Path, split: str) -> tuple[np.ndarray, list[Dict[str, Any]]]:
    base = TacSpikeH5Dataset(data_root=data_root, split=split)
    distances = np.empty((len(base),), dtype=np.float32)
    sequence_records = []
    for seq_idx, info in enumerate(base.sequences):
        start = int(base.offsets[seq_idx])
        stop = int(base.offsets[seq_idx + 1])
        with h5py.File(info.path, "r") as h5:
            labels = h5["label/slip"][:].astype(np.int8, copy=False)
        seq_dist = transition_distances(labels)
        distances[start:stop] = seq_dist
        sequence_records.append(
            {
                "sequence_id": info.sequence_id,
                "windows": int(labels.shape[0]),
                "positive_fraction": float(labels.mean()) if labels.size else 0.0,
                "num_transitions": int(np.count_nonzero(labels[1:] != labels[:-1])) if labels.size > 1 else 0,
            }
        )
    base.close()
    return distances, sequence_records


def fill_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in (
        ("model", "lite_scnn"),
        ("model_width", 32),
        ("hidden_dim", 128),
        ("scnn_hidden_dim", 64),
        ("scnn_conv1_channels", 16),
        ("scnn_conv2_channels", 32),
        ("readout_start_frac", 0.0),
        ("time_steps", None),
        ("temporal_mode", "time_channels"),
        ("dropout", 0.1),
        ("context_ms", None),
        ("time_bins", None),
        ("label_smoothing", 0.0),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def bucket_metrics(labels: np.ndarray, scores: np.ndarray, distances: np.ndarray, threshold: float) -> Dict[str, Any]:
    bucket_masks = {
        "0_10_ms": distances < 10,
        "10_20_ms": (distances >= 10) & (distances < 20),
        "20_50_ms": (distances >= 20) & (distances < 50),
        "50_100_ms": (distances >= 50) & (distances < 100),
        "gt_100_ms": distances >= 100,
        "gt_50_ms": distances >= 50,
    }
    result: Dict[str, Any] = {}
    for name, mask in bucket_masks.items():
        count = int(mask.sum())
        if count == 0:
            result[name] = {"count": 0}
            continue
        result[name] = {
            "count": count,
            "fraction": float(count / max(labels.shape[0], 1)),
            "positive_fraction": float(labels[mask].mean()) if count else 0.0,
            "metrics": binary_classification_metrics(labels[mask], scores[mask], threshold=threshold),
        }
    return result


def filtered_metrics(labels: np.ndarray, scores: np.ndarray, distances: np.ndarray) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for min_distance in (10, 20, 50, 100, 150, 200):
        mask = distances >= float(min_distance)
        count = int(mask.sum())
        if count == 0:
            result[f"gt_{min_distance}_ms"] = {"count": 0}
            continue
        tuned = best_threshold(labels[mask], scores[mask], metric_name="accuracy")
        result[f"gt_{min_distance}_ms"] = {
            "count": count,
            "fraction": float(count / max(labels.shape[0], 1)),
            "default_metrics": binary_classification_metrics(labels[mask], scores[mask], threshold=0.0),
            "best_threshold_metrics": tuned,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sampled TacSpike windows by transition-distance buckets.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--sampling", choices=("balanced", "random"), default="random")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = fill_legacy_args(argparse.Namespace(**ckpt["args"]))
    train_args.data_root = args.data_root
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    indices = sample_epoch_indices(
        data_root=args.data_root,
        split=args.split,
        cache_dir=train_args.cache_dir,
        num_samples=args.samples,
        seed=args.seed,
        sampling=args.sampling,
    )
    ordered_indices = order_indices_for_io(indices, args.batch_size, args.seed + 99)
    loader = make_loader(train_args, args.split, ordered_indices, shuffle=False)
    loader.dataset.input_scale = getattr(train_args, "input_scale", 1.0)

    scores_parts = []
    labels_parts = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            input_scale = float(getattr(loader.dataset, "input_scale", 1.0))
            if input_scale != 1.0:
                x = x * input_scale
            logits, _ = model(x)
            scores_parts.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
            labels_parts.append(y.numpy())

    scores = np.concatenate(scores_parts) if scores_parts else np.empty((0,), dtype=np.float32)
    labels = np.concatenate(labels_parts) if labels_parts else np.empty((0,), dtype=np.int64)
    distance_lookup, sequence_records = build_distance_lookup(args.data_root, args.split)
    distances = distance_lookup[ordered_indices]

    tuned = best_threshold(labels, scores, metric_name="accuracy")
    threshold = float(tuned["threshold"])
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sampling": args.sampling,
        "samples": int(labels.shape[0]),
        "default_metrics": binary_classification_metrics(labels, scores, threshold=0.0),
        "best_threshold_metrics": tuned,
        "filtered_transition_metrics": filtered_metrics(labels, scores, distances),
        "transition_buckets_at_best_threshold": bucket_metrics(labels, scores, distances, threshold),
        "transition_buckets_at_default_threshold": bucket_metrics(labels, scores, distances, 0.0),
        "sequence_summary": sequence_records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("split", "sampling", "samples")}, sort_keys=True))


if __name__ == "__main__":
    main()
