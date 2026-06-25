from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tacspike.data import IndexedTacSpikeDataset, build_label_index_cache, sample_epoch_indices
from tacspike.models import TacSpikeLiteSCNN, count_parameters
from tacspike.training import binary_classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def class_weights(data_root: Path, split: str, cache_dir: Path, device: torch.device) -> torch.Tensor:
    cache_path = build_label_index_cache(data_root=data_root, split=split, cache_dir=cache_dir)
    with np.load(cache_path) as cache:
        n_slip = int(cache["slip"].shape[0])
        n_no_slip = int(cache["no_slip"].shape[0])
    total = n_slip + n_no_slip
    weights = torch.tensor(
        [total / max(2 * n_no_slip, 1), total / max(2 * n_slip, 1)],
        dtype=torch.float32,
        device=device,
    )
    return weights


def make_class_weight_tensor(args: argparse.Namespace, device: torch.device) -> torch.Tensor | None:
    if args.class_weight == "none":
        return None
    if args.class_weight == "inverse_frequency":
        return class_weights(args.data_root, "train", args.cache_dir, device)
    raise ValueError(f"Unsupported class_weight={args.class_weight!r}")


def make_loader(
    args: argparse.Namespace,
    split: str,
    indices: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = IndexedTacSpikeDataset(
        data_root=args.data_root,
        split=split,
        indices=indices,
        polarity_mode=args.polarity_mode,
        clip_max=args.clip_max,
        spatial_pool=args.spatial_pool,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


def order_indices_for_io(indices: np.ndarray, batch_size: int, seed: int) -> np.ndarray:
    """Reduce HDF5 file thrashing while retaining batch-level randomness."""

    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    chunks = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
    rng = np.random.default_rng(seed)
    rng.shuffle(chunks)
    return np.concatenate(chunks).astype(np.int64, copy=False)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    all_scores = []
    all_labels = []
    firing_totals: Dict[str, float] = {}
    steps = 0
    start = time.time()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            input_scale = float(getattr(loader.dataset, "input_scale", 1.0))
            if input_scale != 1.0:
                x = x * input_scale
            y = y.to(device=device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits, stats = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = int(y.shape[0])
            total_loss += float(loss.detach().item()) * bs
            total_samples += bs
            score = (logits[:, 1] - logits[:, 0]).detach().float().cpu().numpy()
            all_scores.append(score)
            all_labels.append(y.detach().cpu().numpy())
            for key, value in stats.items():
                firing_totals[key] = firing_totals.get(key, 0.0) + float(value.detach().item())
            steps += 1

    labels = np.concatenate(all_labels) if all_labels else np.empty((0,), dtype=np.int64)
    scores = np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float32)
    metrics = binary_classification_metrics(labels, scores)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["num_samples"] = float(total_samples)
    metrics["num_batches"] = float(steps)
    metrics["seconds"] = time.time() - start
    metrics["samples_per_second"] = total_samples / max(metrics["seconds"], 1e-9)
    for key, value in firing_totals.items():
        metrics[key] = value / max(steps, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TacSpike-Lite-SCNN-v1 main model.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-samples-per-epoch", type=int, default=200000)
    parser.add_argument("--val-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--surrogate-alpha", type=float, default=2.0)
    parser.add_argument("--readout", choices=("spike_count", "membrane", "logit_mean", "logit_sum"), default="spike_count")
    parser.add_argument("--spatial-pool", type=int, default=4)
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--polarity-mode", choices=("both", "positive", "negative", "sum"), default="both")
    parser.add_argument("--sampling", choices=("balanced", "random"), default="balanced")
    parser.add_argument("--class-weight", choices=("none", "inverse_frequency"), default="inverse_frequency")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--target-accuracy", type=float, default=0.94)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    args.cache_dir = cache_dir
    log_path = args.output_dir / "metrics.jsonl"
    summary_path = args.output_dir / "summary.json"

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = TacSpikeLiteSCNN(
        input_channels=2 if args.polarity_mode == "both" else 1,
        num_classes=2,
        beta=args.beta,
        threshold=args.threshold,
        surrogate_alpha=args.surrogate_alpha,
        readout=args.readout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = make_class_weight_tensor(args, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    config_record = {
        "event": "config",
        "args": vars(args),
        "device": str(device),
        "parameter_count": count_parameters(model),
        "class_weights": None if weights is None else [float(x) for x in weights.detach().cpu().tolist()],
    }
    write_jsonl(log_path, config_record)
    best_metric = -math.inf
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}

    val_indices = sample_epoch_indices(
        data_root=args.data_root,
        split="val",
        cache_dir=cache_dir,
        num_samples=args.val_samples,
        seed=args.seed + 10_000,
        sampling=args.sampling,
    )
    val_indices = order_indices_for_io(val_indices, args.batch_size, args.seed + 20_000)
    val_loader = make_loader(args, "val", val_indices, shuffle=False)
    val_loader.dataset.input_scale = args.input_scale

    for epoch in range(1, args.epochs + 1):
        train_indices = sample_epoch_indices(
            data_root=args.data_root,
            split="train",
            cache_dir=cache_dir,
            num_samples=args.train_samples_per_epoch,
            seed=args.seed + epoch,
            sampling=args.sampling,
        )
        train_indices = order_indices_for_io(train_indices, args.batch_size, args.seed + epoch + 30_000)
        train_loader = make_loader(args, "train", train_indices, shuffle=False)
        train_loader.dataset.input_scale = args.input_scale
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
            grad_clip=args.grad_clip,
        )

        record = {
            "event": "epoch",
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        write_jsonl(log_path, record)
        metric = float(val_metrics.get("accuracy", 0.0))
        save_checkpoint(args.output_dir / "latest.pt", model, optimizer, epoch, best_metric, args, val_metrics)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_metrics = val_metrics
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, best_metric, args, val_metrics)

        summary = {
            "best_epoch": best_epoch,
            "best_val_accuracy": best_metric,
            "best_val_metrics": best_metrics,
            "latest_epoch": epoch,
            "target_accuracy": args.target_accuracy,
            "target_met": best_metric >= args.target_accuracy,
            "parameter_count": count_parameters(model),
            "output_dir": str(args.output_dir),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True, default=json_default), flush=True)


if __name__ == "__main__":
    main()
