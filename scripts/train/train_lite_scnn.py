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
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from tacspike.data import (
    IndexedTacSpikeDataset,
    build_label_index_cache,
    sample_epoch_indices,
    transition_distance_for_indices,
)
from tacspike.models import (
    TacSpikeDeepSCNN,
    TacSpikeFrameCNN,
    TacSpikeLiteSCNN,
    TacSpikeTemporalConvSCNN,
    TacSpikeTimeChannelSCNN,
    count_parameters,
)
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


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        prob = F.softmax(logits, dim=1)
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        return ((1.0 - pt).pow(self.gamma) * ce).mean()


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def per_sample_classification_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if isinstance(criterion, FocalCrossEntropyLoss):
        ce = F.cross_entropy(
            logits,
            target,
            weight=criterion.weight,
            reduction="none",
            label_smoothing=criterion.label_smoothing,
        )
        prob = F.softmax(logits, dim=1)
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        return (1.0 - pt).pow(criterion.gamma) * ce
    if isinstance(criterion, nn.CrossEntropyLoss):
        return F.cross_entropy(
            logits,
            target,
            weight=criterion.weight,
            reduction="none",
            label_smoothing=criterion.label_smoothing,
        )
    raise TypeError(f"Unsupported criterion type: {type(criterion)!r}")


def margin_regularization(logits: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    if margin <= 0:
        return logits.new_tensor(0.0)
    signed_target = target.to(logits.dtype) * 2.0 - 1.0
    score = logits[:, 1] - logits[:, 0]
    return F.relu(float(margin) - signed_target * score).mean()


def margin_regularization_values(logits: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    if margin <= 0:
        return logits.new_zeros((logits.shape[0],))
    signed_target = target.to(logits.dtype) * 2.0 - 1.0
    score = logits[:, 1] - logits[:, 0]
    return F.relu(float(margin) - signed_target * score)


def sample_weights_from_transition_distance(
    distances: np.ndarray,
    near_ms: float,
    mid_ms: float,
    near_weight: float,
    mid_weight: float,
) -> np.ndarray | None:
    if near_ms <= 0 and mid_ms <= 0:
        return None
    distances = np.asarray(distances, dtype=np.float32)
    weights = np.ones(distances.shape[0], dtype=np.float32)
    if mid_ms > 0:
        weights[distances < float(mid_ms)] = float(mid_weight)
    if near_ms > 0:
        weights[distances < float(near_ms)] = float(near_weight)
    return weights


def build_model(args: argparse.Namespace) -> nn.Module:
    input_channels = 2 if args.polarity_mode == "both" else 1
    time_steps = args.time_steps if args.time_steps is not None else int(args.time_bins or args.context_ms or 20)
    if args.model == "lite_scnn":
        return TacSpikeLiteSCNN(
            input_channels=input_channels,
            num_classes=2,
            beta=args.beta,
            threshold=args.threshold,
            surrogate_alpha=args.surrogate_alpha,
            readout=args.readout,
            conv1_channels=args.scnn_conv1_channels,
            conv2_channels=args.scnn_conv2_channels,
            hidden=args.scnn_hidden_dim,
            readout_start_frac=args.readout_start_frac,
        )
    if args.model == "deep_scnn":
        return TacSpikeDeepSCNN(
            input_channels=input_channels,
            num_classes=2,
            beta=args.beta,
            threshold=args.threshold,
            surrogate_alpha=args.surrogate_alpha,
            width=args.model_width,
            hidden=args.hidden_dim,
            readout=args.readout if args.readout in {"logit_mean", "logit_sum"} else "logit_mean",
        )
    if args.model == "frame_cnn":
        return TacSpikeFrameCNN(
            input_channels=input_channels,
            time_steps=time_steps,
            num_classes=2,
            width=args.model_width,
            temporal_mode=args.temporal_mode,
            dropout=args.dropout,
        )
    if args.model == "time_channel_scnn":
        return TacSpikeTimeChannelSCNN(
            input_channels=input_channels,
            time_steps=time_steps,
            num_classes=2,
            beta=args.beta,
            threshold=args.threshold,
            surrogate_alpha=args.surrogate_alpha,
            width=args.model_width,
            hidden=args.hidden_dim,
            dropout=args.dropout,
        )
    if args.model == "temporal_conv_scnn":
        return TacSpikeTemporalConvSCNN(
            input_channels=input_channels,
            num_classes=2,
            beta=args.beta,
            threshold=args.threshold,
            surrogate_alpha=args.surrogate_alpha,
            width=args.model_width,
            hidden=args.hidden_dim,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model={args.model!r}")


def make_loader(
    args: argparse.Namespace,
    split: str,
    indices: np.ndarray,
    shuffle: bool,
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    dataset = IndexedTacSpikeDataset(
        data_root=args.data_root,
        split=split,
        indices=indices,
        polarity_mode=args.polarity_mode,
        clip_max=args.clip_max,
        spatial_pool=args.spatial_pool,
        context_ms=getattr(args, "context_ms", None),
        time_bins=getattr(args, "time_bins", None),
        sample_weights=sample_weights,
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
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
    teacher_model: nn.Module | None = None,
    distill_alpha: float = 0.0,
    distill_temperature: float = 2.0,
    margin_loss_weight: float = 0.0,
    margin_value: float = 1.0,
) -> Dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    all_scores = []
    all_labels = []
    firing_totals: Dict[str, float] = {}
    sample_weight_sum = 0.0
    sample_weight_count = 0
    steps = 0
    start = time.time()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if len(batch) == 3:
                x, y, sample_weight = batch
                sample_weight = sample_weight.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                x, y = batch
                sample_weight = None
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            input_scale = float(getattr(loader.dataset, "input_scale", 1.0))
            if input_scale != 1.0:
                x = x * input_scale
            y = y.to(device=device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits, stats = model(x)
                if sample_weight is None:
                    ce_loss = criterion(logits, y)
                else:
                    ce_loss = weighted_mean(per_sample_classification_loss(criterion, logits, y), sample_weight)
                if teacher_model is not None and distill_alpha > 0.0:
                    with torch.no_grad():
                        teacher_logits, _ = teacher_model(x)
                    temperature = float(distill_temperature)
                    student_log_prob = F.log_softmax(logits / temperature, dim=1)
                    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
                    distill_values = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1)
                    distill_loss = weighted_mean(distill_values, sample_weight) * temperature * temperature
                    loss = (1.0 - distill_alpha) * ce_loss + distill_alpha * distill_loss
                else:
                    loss = ce_loss
                if margin_loss_weight > 0.0:
                    if sample_weight is None:
                        margin_loss = margin_regularization(logits, y, margin_value)
                    else:
                        margin_loss = weighted_mean(margin_regularization_values(logits, y, margin_value), sample_weight)
                    loss = loss + float(margin_loss_weight) * margin_loss
            if train:
                if scaler is not None and amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
            if sample_weight is not None:
                sample_weight_sum += float(sample_weight.detach().sum().cpu())
                sample_weight_count += int(sample_weight.numel())
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
    if sample_weight_count:
        metrics["sample_weight_mean"] = sample_weight_sum / max(sample_weight_count, 1)
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
    parser.add_argument(
        "--model",
        choices=("lite_scnn", "deep_scnn", "frame_cnn", "time_channel_scnn", "temporal_conv_scnn"),
        default="lite_scnn",
    )
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
    parser.add_argument("--readout-start-frac", type=float, default=0.0)
    parser.add_argument("--scnn-conv1-channels", type=int, default=16)
    parser.add_argument("--scnn-conv2-channels", type=int, default=32)
    parser.add_argument("--scnn-hidden-dim", type=int, default=64)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--temporal-mode", choices=("time_channels", "sum"), default="time_channels")
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--context-ms", type=float, default=None)
    parser.add_argument("--time-bins", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spatial-pool", type=int, default=4)
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--polarity-mode", choices=("both", "positive", "negative", "sum"), default="both")
    parser.add_argument("--sampling", choices=("balanced", "random"), default="balanced")
    parser.add_argument("--class-weight", choices=("none", "inverse_frequency"), default="inverse_frequency")
    parser.add_argument("--loss-type", choices=("ce", "focal"), default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--margin-value", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--ignore-transition-ms", type=float, default=0.0)
    parser.add_argument("--transition-weight-near-ms", type=float, default=0.0)
    parser.add_argument("--transition-weight-mid-ms", type=float, default=0.0)
    parser.add_argument("--transition-near-weight", type=float, default=1.0)
    parser.add_argument("--transition-mid-weight", type=float, default=1.0)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--distill-alpha", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
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
    if args.time_bins is None and args.context_ms is not None:
        args.time_bins = int(round(args.context_ms))
    if args.time_steps is None and args.time_bins is None:
        args.time_steps = 20
        args.time_bins = 20
    elif args.time_steps is None and args.time_bins is not None:
        args.time_steps = int(args.time_bins)
    elif args.time_bins is None and args.time_steps is not None:
        args.time_bins = int(args.time_steps)
    model = build_model(args).to(device)
    teacher_model = None
    if args.teacher_checkpoint is not None and args.distill_alpha > 0.0:
        teacher_ckpt = torch.load(args.teacher_checkpoint, map_location="cpu")
        teacher_args = argparse.Namespace(**teacher_ckpt["args"])
        teacher_args.data_root = args.data_root
        teacher_args.batch_size = args.batch_size
        teacher_args.num_workers = args.num_workers
        for name, value in (
            ("model", "frame_cnn"),
            ("model_width", 32),
            ("hidden_dim", 64),
            ("scnn_hidden_dim", 64),
            ("scnn_conv1_channels", 16),
            ("scnn_conv2_channels", 32),
            ("readout_start_frac", 0.0),
            ("time_steps", None),
            ("temporal_mode", "time_channels"),
            ("dropout", 0.1),
            ("context_ms", None),
            ("time_bins", None),
        ):
            if not hasattr(teacher_args, name):
                setattr(teacher_args, name, value)
        teacher_model = build_model(teacher_args).to(device)
        teacher_model.load_state_dict(teacher_ckpt["model"])
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
        if args.scheduler == "cosine"
        else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    weights = make_class_weight_tensor(args, device)
    if args.loss_type == "focal":
        criterion = FocalCrossEntropyLoss(
            gamma=args.focal_gamma,
            weight=weights,
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

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
            ignore_transition_ms=args.ignore_transition_ms,
        )
        train_indices = order_indices_for_io(train_indices, args.batch_size, args.seed + epoch + 30_000)
        train_sample_weights = None
        if args.transition_weight_near_ms > 0 or args.transition_weight_mid_ms > 0:
            train_distances = transition_distance_for_indices(args.data_root, "train", cache_dir, train_indices)
            train_sample_weights = sample_weights_from_transition_distance(
                train_distances,
                near_ms=args.transition_weight_near_ms,
                mid_ms=args.transition_weight_mid_ms,
                near_weight=args.transition_near_weight,
                mid_weight=args.transition_mid_weight,
            )
        train_loader = make_loader(args, "train", train_indices, shuffle=False, sample_weights=train_sample_weights)
        train_loader.dataset.input_scale = args.input_scale
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=args.amp,
            max_batches=args.max_train_batches,
            grad_clip=args.grad_clip,
            teacher_model=teacher_model,
            distill_alpha=args.distill_alpha,
            distill_temperature=args.distill_temperature,
            margin_loss_weight=args.margin_loss_weight,
            margin_value=args.margin_value,
        )
        if scheduler is not None:
            scheduler.step()
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            amp=args.amp,
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
