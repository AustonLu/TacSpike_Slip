from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tacspike.data import (
    load_stream_cache_manifest,
    onset_valid_mask,
    parse_feature_windows,
    read_stream_feature_slice,
    sample_stream_segments,
)
from tacspike.models import TacSpikeMultiTauStreamingSCNN, TacSpikeStreamingLiteSCNN, count_parameters
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


class StreamCacheSegmentDataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        split: str,
        seq_indices: np.ndarray,
        starts: np.ndarray,
        segment_steps: int,
        spatial_pool: int = 4,
        polarity_mode: str = "both",
        clip_max: float | None = None,
        cache_dtype: str = "float16",
        cache_format: str = "dense",
        transition_ignore_steps: int = 0,
        input_scale: float = 1.0,
        feature_mode: str = "raw",
        feature_windows: str = "1,20,50,100,200,400",
        multiscale_normalization: str = "sqrt",
    ) -> None:
        self.sequences = load_stream_cache_manifest(
            cache_root=cache_root,
            split=split,
            spatial_pool=spatial_pool,
            polarity_mode=polarity_mode,
            clip_max=clip_max,
            dtype=cache_dtype,
            cache_format=cache_format,
        )
        self.seq_indices = np.asarray(seq_indices, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.segment_steps = int(segment_steps)
        self.transition_ignore_steps = int(transition_ignore_steps)
        self.input_scale = float(input_scale)
        self.feature_mode = feature_mode
        self.feature_windows = parse_feature_windows(feature_windows)
        self.multiscale_normalization = multiscale_normalization
        self._open_path: Path | None = None
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_open_path"] = None
        state["_h5"] = None
        return state

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._open_path = None

    def _open_h5(self, path: Path) -> h5py.File:
        if self._h5 is not None and self._open_path == path:
            return self._h5
        if self._h5 is not None:
            self._h5.close()
        self._h5 = h5py.File(path, "r")
        self._open_path = path
        return self._h5

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_idx = int(self.seq_indices[index])
        start = int(self.starts[index])
        stop = start + self.segment_steps
        h5 = self._open_h5(self.sequences[seq_idx].path)
        x = read_stream_feature_slice(
            h5,
            start,
            stop,
            feature_mode=self.feature_mode,
            feature_windows=self.feature_windows,
            multiscale_normalization=self.multiscale_normalization,
        )
        if self.input_scale != 1.0:
            x = x * self.input_scale
        y = h5["labels"][start:stop].astype(np.float32, copy=False)
        valid = h5["valid_mask"][start:stop].astype(np.bool_, copy=False)
        if self.transition_ignore_steps > 0:
            valid &= onset_valid_mask(y.astype(np.int64, copy=False), self.transition_ignore_steps)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(valid.astype(np.float32, copy=False))


def make_loader(
    args: argparse.Namespace,
    split: str,
    seq_indices: np.ndarray,
    starts: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = StreamCacheSegmentDataset(
        cache_root=args.stream_cache_root,
        split=split,
        seq_indices=seq_indices,
        starts=starts,
        segment_steps=args.segment_steps,
        spatial_pool=args.spatial_pool,
        polarity_mode=args.polarity_mode,
        clip_max=args.clip_max,
        cache_dtype=args.cache_dtype,
        cache_format=args.cache_format,
        transition_ignore_steps=args.transition_ignore_steps,
        input_scale=args.input_scale,
        feature_mode=args.feature_mode,
        feature_windows=args.feature_windows,
        multiscale_normalization=args.multiscale_normalization,
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


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def sequence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float,
    smoothness_weight: float,
    loss_mode: str,
    warmup_steps: int,
    supervise_tail_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    score = logits[..., 1] - logits[..., 0]
    mask = valid > 0
    if loss_mode == "last":
        mask[:, :-1] = False
    if warmup_steps > 0:
        mask[:, : int(warmup_steps)] = False
    if supervise_tail_steps > 0 and supervise_tail_steps < labels.shape[1]:
        mask[:, : labels.shape[1] - int(supervise_tail_steps)] = False
    mask_f = mask.to(dtype=score.dtype)
    weight = torch.where(
        labels > 0.5,
        torch.as_tensor(float(pos_weight), device=labels.device, dtype=score.dtype),
        torch.ones((), device=labels.device, dtype=score.dtype),
    )
    loss_raw = F.binary_cross_entropy_with_logits(score, labels, weight=weight, reduction="none")
    loss = masked_mean(loss_raw, mask_f)
    smooth_loss = score.new_tensor(0.0)
    if smoothness_weight > 0.0 and labels.shape[1] > 1:
        pair_mask = mask[:, 1:] & mask[:, :-1] & ((labels[:, 1:] > 0.5) == (labels[:, :-1] > 0.5))
        if pair_mask.any():
            prob = torch.sigmoid(score)
            smooth_loss = masked_mean((prob[:, 1:] - prob[:, :-1]).abs(), pair_mask.to(dtype=score.dtype))
            loss = loss + float(smoothness_weight) * smooth_loss
    diagnostics = {
        "valid_fraction": float(mask_f.mean().detach().cpu()),
        "bce_loss": float(loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
    }
    return loss, mask, diagnostics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    valid_scores: List[np.ndarray] = []
    valid_labels: List[np.ndarray] = []
    firing_totals: Dict[str, float] = {}
    diagnostic_totals: Dict[str, float] = {}
    batches = 0
    start_time = time.time()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, valid in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            valid = valid.to(device=device, dtype=torch.float32, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits, stats = model(x)
                loss, valid_mask, diagnostics = sequence_loss(
                    logits=logits,
                    labels=y,
                    valid=valid,
                    pos_weight=args.positive_weight,
                    smoothness_weight=args.smoothness_weight,
                    loss_mode=args.loss_mode,
                    warmup_steps=args.warmup_steps,
                    supervise_tail_steps=args.supervise_tail_steps,
                )
            if training:
                if scaler is not None and amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
            score = (logits[..., 1] - logits[..., 0]).detach().float().cpu().numpy().reshape(-1)
            labels = y.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
            valid_np = valid_mask.detach().cpu().numpy().reshape(-1)
            all_scores.append(score)
            all_labels.append(labels)
            valid_scores.append(score[valid_np])
            valid_labels.append(labels[valid_np])
            total_loss += float(loss.detach().item()) * int(labels.shape[0])
            total_items += int(labels.shape[0])
            for key, value in stats.items():
                firing_totals[key] = firing_totals.get(key, 0.0) + float(value.detach().cpu())
            for key, value in diagnostics.items():
                diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value)
            batches += 1

    labels_np = np.concatenate(all_labels) if all_labels else np.empty((0,), dtype=np.int64)
    scores_np = np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float32)
    valid_labels_np = np.concatenate(valid_labels) if valid_labels else np.empty((0,), dtype=np.int64)
    valid_scores_np = np.concatenate(valid_scores) if valid_scores else np.empty((0,), dtype=np.float32)
    metrics = binary_classification_metrics(labels_np, scores_np)
    valid_metrics = binary_classification_metrics(valid_labels_np, valid_scores_np) if valid_labels_np.size else {}
    for key, value in valid_metrics.items():
        metrics[f"valid_{key}"] = value
    metrics["loss"] = total_loss / max(total_items, 1)
    metrics["num_items"] = float(total_items)
    metrics["num_batches"] = float(batches)
    metrics["seconds"] = time.time() - start_time
    metrics["items_per_second"] = total_items / max(metrics["seconds"], 1e-9)
    for key, value in firing_totals.items():
        metrics[key] = value / max(batches, 1)
    for key, value in diagnostic_totals.items():
        metrics[key] = value / max(batches, 1)
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


def parse_csv_floats(text: str) -> Tuple[float, ...]:
    values = tuple(float(part) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one float")
    return values


def input_channels_from_args(args: argparse.Namespace) -> int:
    base_channels = 2 if args.polarity_mode == "both" else 1
    if args.feature_mode == "raw":
        return base_channels
    if args.feature_mode == "multiscale":
        return base_channels * len(parse_feature_windows(args.feature_windows))
    raise ValueError(f"Unsupported feature_mode={args.feature_mode!r}")


def build_stream_model(args: argparse.Namespace, input_channels: int) -> nn.Module:
    if args.model_type == "lite":
        return TacSpikeStreamingLiteSCNN(
            input_channels=input_channels,
            beta=args.beta,
            threshold=args.threshold,
            surrogate_alpha=args.surrogate_alpha,
            hidden=args.hidden_dim,
            conv1_channels=args.conv1_channels,
            conv2_channels=args.conv2_channels,
            dropout=args.dropout,
        )
    if args.model_type == "multitau":
        return TacSpikeMultiTauStreamingSCNN(
            input_channels=input_channels,
            betas=parse_csv_floats(args.multi_tau_betas),
            threshold=args.threshold,
            surrogate_alpha=args.surrogate_alpha,
            hidden=args.hidden_dim,
            conv1_channels=args.conv1_channels,
            conv2_channels=args.conv2_channels,
            dropout=args.dropout,
            fusion=args.multi_tau_fusion,
        )
    raise ValueError(f"Unsupported model_type={args.model_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stateful SNN from TacSpike stream cache.")
    parser.add_argument("--stream-cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-segments-per-epoch", type=int, default=20000)
    parser.add_argument("--val-segments", type=int, default=5000)
    parser.add_argument("--segment-steps", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--surrogate-alpha", type=float, default=2.0)
    parser.add_argument("--model-type", choices=("lite", "multitau"), default="lite")
    parser.add_argument("--multi-tau-betas", default="0.65,0.85,0.95")
    parser.add_argument("--multi-tau-fusion", choices=("mean", "linear"), default="mean")
    parser.add_argument("--conv1-channels", type=int, default=32)
    parser.add_argument("--conv2-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--spatial-pool", type=int, default=4)
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--cache-dtype", default="float16")
    parser.add_argument("--cache-format", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--polarity-mode", choices=("both", "positive", "negative", "sum"), default="both")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--feature-mode", choices=("raw", "multiscale"), default="raw")
    parser.add_argument("--feature-windows", default="1,20,50,100,200,400")
    parser.add_argument("--multiscale-normalization", choices=("sqrt", "mean", "none"), default="sqrt")
    parser.add_argument("--sampling", choices=("balanced", "random", "end_balanced", "state_balanced", "transition_mix"), default="transition_mix")
    parser.add_argument("--loss-mode", choices=("all", "last"), default="all")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--supervise-tail-steps", type=int, default=0)
    parser.add_argument("--transition-ignore-steps", type=int, default=0)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--best-metric", choices=("accuracy", "valid_accuracy", "balanced_accuracy", "valid_balanced_accuracy", "f1", "valid_f1"), default="valid_balanced_accuracy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir = args.index_dir or (args.output_dir / "index")
    log_path = args.output_dir / "metrics.jsonl"
    summary_path = args.output_dir / "summary.json"

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    input_channels = input_channels_from_args(args)
    model = build_stream_model(args, input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
        if args.scheduler == "cosine"
        else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    write_jsonl(
        log_path,
        {
            "event": "config",
            "args": vars(args),
            "device": str(device),
            "parameter_count": count_parameters(model),
        },
    )

    val_seq, val_start = sample_stream_segments(
        cache_root=args.stream_cache_root,
        split="val",
        index_dir=args.index_dir,
        segment_steps=args.segment_steps,
        count=args.val_segments,
        seed=args.seed + 10_000,
        sampling=args.sampling,
        spatial_pool=args.spatial_pool,
        polarity_mode=args.polarity_mode,
        clip_max=args.clip_max,
        dtype=args.cache_dtype,
        cache_format=args.cache_format,
    )
    val_loader = make_loader(args, "val", val_seq, val_start, shuffle=False)
    best_metric = -math.inf
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        train_seq, train_start = sample_stream_segments(
            cache_root=args.stream_cache_root,
            split="train",
            index_dir=args.index_dir,
            segment_steps=args.segment_steps,
            count=args.train_segments_per_epoch,
            seed=args.seed + epoch,
            sampling=args.sampling,
            spatial_pool=args.spatial_pool,
            polarity_mode=args.polarity_mode,
            clip_max=args.clip_max,
        dtype=args.cache_dtype,
        cache_format=args.cache_format,
        )
        train_loader = make_loader(args, "train", train_seq, train_start, shuffle=False)
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.amp, scaler, args)
        if scheduler is not None:
            scheduler.step()
        val_metrics = run_epoch(model, val_loader, device, None, args.amp, None, args)
        record = {"event": "epoch", "epoch": epoch, "train": train_metrics, "val": val_metrics}
        write_jsonl(log_path, record)
        metric = float(val_metrics.get(args.best_metric, val_metrics.get("accuracy", 0.0)))
        save_checkpoint(args.output_dir / "latest.pt", model, optimizer, epoch, best_metric, args, val_metrics)
        if args.save_epoch_checkpoints:
            save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, best_metric, args, val_metrics)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_metrics = val_metrics
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, best_metric, args, val_metrics)
        summary = {
            "best_epoch": int(best_epoch),
            "best_metric_value": float(best_metric),
            "best_val_metrics": best_metrics,
            "latest_epoch": int(epoch),
            "target_accuracy": float(args.target_accuracy),
            "target_met": bool(best_metric >= args.target_accuracy),
            "best_metric": args.best_metric,
            "parameter_count": count_parameters(model),
            "output_dir": str(args.output_dir),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True, default=json_default), flush=True)


if __name__ == "__main__":
    main()
