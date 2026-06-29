from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.train.train_lite_scnn import build_model, make_loader, order_indices_for_io
from tacspike.data import sample_epoch_indices
from tacspike.training import binary_classification_metrics


def best_threshold(y_true: np.ndarray, scores: np.ndarray, metric_name: str) -> Dict[str, Any]:
    thresholds = np.unique(scores.astype(np.float64, copy=False))
    if thresholds.size > 5000:
        thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 5000)))

    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    best = {"threshold": 0.0, metric_name: -1.0}
    for threshold in thresholds:
        y_pred = scores > threshold
        tp = int(((y_true == 1) & y_pred).sum())
        tn = int(((y_true == 0) & ~y_pred).sum())
        fp = negatives - tn
        fn = positives - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = (2.0 * precision * recall / max(precision + recall, 1e-12))
        accuracy = (tp + tn) / max(len(y_true), 1)
        metrics = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(0.5 * (recall + specificity)),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
        }
        if metrics[metric_name] > best[metric_name]:
            best = {"threshold": float(threshold), **metrics}

    auc_metrics = binary_classification_metrics(y_true, scores, threshold=float(best["threshold"]))
    best["roc_auc"] = auc_metrics["roc_auc"]
    best["pr_auc"] = auc_metrics["pr_auc"]
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TacSpike-Lite-SCNN checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling", choices=("checkpoint", "balanced", "random"), default="checkpoint")
    parser.add_argument("--best-threshold-metric", choices=("accuracy", "balanced_accuracy", "f1"), default="f1")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ckpt["args"])
    train_args.data_root = args.data_root
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if not hasattr(train_args, "model"):
        train_args.model = "lite_scnn"
    if not hasattr(train_args, "model_width"):
        train_args.model_width = 32
    if not hasattr(train_args, "hidden_dim"):
        train_args.hidden_dim = 128
    if not hasattr(train_args, "time_steps"):
        train_args.time_steps = None
    if not hasattr(train_args, "temporal_mode"):
        train_args.temporal_mode = "time_channels"
    if not hasattr(train_args, "dropout"):
        train_args.dropout = 0.1
    if not hasattr(train_args, "context_ms"):
        train_args.context_ms = None
    if not hasattr(train_args, "time_bins"):
        train_args.time_bins = None
    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sampling = train_args.sampling if args.sampling == "checkpoint" else args.sampling
    indices = sample_epoch_indices(
        data_root=args.data_root,
        split=args.split,
        cache_dir=train_args.cache_dir,
        num_samples=args.samples,
        seed=args.seed,
        sampling=sampling,
    )
    indices = order_indices_for_io(indices, args.batch_size, args.seed + 99)
    loader = make_loader(train_args, args.split, indices, shuffle=False)
    loader.dataset.input_scale = getattr(train_args, "input_scale", 1.0)

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

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    default_metrics = binary_classification_metrics(labels, scores, threshold=0.0)
    tuned_metrics = best_threshold(labels, scores, metric_name=args.best_threshold_metric)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sampling": sampling,
        "samples": int(labels.shape[0]),
        "default_threshold": 0.0,
        "default_metrics": default_metrics,
        "best_threshold_metric": args.best_threshold_metric,
        "best_threshold_metrics": tuned_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
