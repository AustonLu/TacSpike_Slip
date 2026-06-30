from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.train.evaluate_checkpoint_ensemble import fill_legacy_args
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


def best_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    is_pos = sorted_labels == 1
    total_pos = int(is_pos.sum())
    total_neg = int(labels.shape[0] - total_pos)
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


def checkpoint_name(path: Path) -> str:
    if path.name == "best.pt":
        return path.parent.name
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache fixed random-window scores for one TacSpike checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--sampling", choices=("balanced", "random"), default="random")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    start = time.time()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = fill_legacy_args(argparse.Namespace(**ckpt["args"]))
    train_args.data_root = args.data_root
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers

    original_indices = sample_epoch_indices(
        data_root=args.data_root,
        split=args.split,
        cache_dir=train_args.cache_dir,
        num_samples=args.samples,
        seed=args.seed,
        sampling=args.sampling,
    )
    ordered_indices = order_indices_for_io(original_indices, args.batch_size, args.seed + 99)
    loader = make_loader(train_args, args.split, ordered_indices, shuffle=False)
    loader.dataset.input_scale = getattr(train_args, "input_scale", 1.0)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            input_scale = float(getattr(loader.dataset, "input_scale", 1.0))
            if input_scale != 1.0:
                x = x * input_scale
            logits, _ = model(x)
            all_scores.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy().astype(np.float32, copy=False))
            all_labels.append(y.numpy().astype(np.int64, copy=False))

    scores = np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.empty((0,), dtype=np.int64)
    name = checkpoint_name(args.checkpoint)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        scores=scores,
        labels=labels,
        ordered_indices=ordered_indices.astype(np.int64, copy=False),
        original_indices=original_indices.astype(np.int64, copy=False),
        checkpoint=str(args.checkpoint),
        checkpoint_name=name,
        split=args.split,
        sampling=args.sampling,
        seed=int(args.seed),
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_name": name,
        "output_npz": str(args.output_npz),
        "split": args.split,
        "sampling": args.sampling,
        "seed": int(args.seed),
        "samples": int(labels.shape[0]),
        "positive_fraction": float(labels.mean()) if labels.size else 0.0,
        "default_metrics": binary_classification_metrics(labels, scores, threshold=0.0),
        "best_threshold_metrics": best_accuracy_threshold(labels, scores),
        "seconds": time.time() - start,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "train_args": vars(train_args),
    }
    args.output_npz.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in ("checkpoint_name", "samples", "seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
