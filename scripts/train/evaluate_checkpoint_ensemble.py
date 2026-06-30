from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

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
from tacspike.data import sample_epoch_indices
from tacspike.training import binary_classification_metrics


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


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


def load_model(checkpoint: Path, data_root: Path, batch_size: int, num_workers: int, device: torch.device):
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
    split: str,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ordered_indices = order_indices_for_io(indices, batch_size, seed=99)
    loader = make_loader(train_args, split, ordered_indices, shuffle=False)
    loader.dataset.input_scale = getattr(train_args, "input_scale", 1.0)
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            input_scale = float(getattr(loader.dataset, "input_scale", 1.0))
            if input_scale != 1.0:
                x = x * input_scale
            logits, _ = model(x)
            all_scores.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
            all_labels.append(y.numpy())
    scores = np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.empty((0,), dtype=np.int64)
    return scores, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate score averaging over TacSpike checkpoints.")
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling", choices=("balanced", "random"), default="random")
    parser.add_argument("--best-threshold-metric", choices=("accuracy", "balanced_accuracy", "f1"), default="accuracy")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models_and_args = [
        load_model(path, args.data_root, args.batch_size, args.num_workers, device)
        for path in args.checkpoints
    ]
    reference_args = models_and_args[0][1]
    indices = sample_epoch_indices(
        data_root=args.data_root,
        split=args.split,
        cache_dir=reference_args.cache_dir,
        num_samples=args.samples,
        seed=args.seed,
        sampling=args.sampling,
    )

    score_parts = []
    labels = None
    for model, train_args in models_and_args:
        scores, current_labels = predict_scores(model, train_args, args.split, indices, args.batch_size, device)
        score_parts.append(scores.astype(np.float64, copy=False))
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise RuntimeError("Ensemble checkpoints produced labels in different order.")

    assert labels is not None
    ensemble_scores = np.mean(np.stack(score_parts, axis=0), axis=0)
    default_metrics = binary_classification_metrics(labels, ensemble_scores, threshold=0.0)
    tuned_metrics = best_threshold(labels, ensemble_scores, metric_name=args.best_threshold_metric)
    result = {
        "checkpoints": [str(path) for path in args.checkpoints],
        "split": args.split,
        "sampling": args.sampling,
        "samples": int(labels.shape[0]),
        "default_threshold": 0.0,
        "default_metrics": default_metrics,
        "best_threshold_metric": args.best_threshold_metric,
        "best_threshold_metrics": tuned_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
