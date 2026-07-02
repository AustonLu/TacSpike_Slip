from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.train.train_lite_scnn import build_model, class_weights
from tacspike.data import IndexedTacSpikeDataset, TacSpikeH5Dataset
from tacspike.models import count_parameters
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


def build_sequence_segment_index(
    data_root: Path,
    split: str,
    cache_dir: Path,
    segment_windows: int,
    force: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_window_segments_s{segment_windows}.npz"
    if cache_path.exists() and not force:
        return cache_path

    base = TacSpikeH5Dataset(data_root=data_root, split=split)
    seq_parts = []
    start_parts = []
    end_label_parts = []
    pos_fraction_parts = []
    transition_parts = []
    for seq_idx, info in enumerate(base.sequences):
        with h5py.File(info.path, "r") as h5:
            labels = h5["label/slip"][:].astype(np.int8, copy=False)
        max_start = int(labels.shape[0] - segment_windows + 1)
        if max_start <= 0:
            continue
        starts = np.arange(max_start, dtype=np.int64)
        local_labels = np.lib.stride_tricks.sliding_window_view(labels, segment_windows)
        seq_parts.append(np.full(max_start, seq_idx, dtype=np.int64))
        start_parts.append(starts)
        end_label_parts.append(local_labels[:, -1].astype(np.int8, copy=False))
        pos_fraction_parts.append(local_labels.mean(axis=1).astype(np.float32, copy=False))
        transition_parts.append((np.count_nonzero(local_labels[:, 1:] != local_labels[:, :-1], axis=1) > 0))
    base.close()

    np.savez_compressed(
        cache_path,
        seq=np.concatenate(seq_parts) if seq_parts else np.empty((0,), dtype=np.int64),
        start=np.concatenate(start_parts) if start_parts else np.empty((0,), dtype=np.int64),
        end_label=np.concatenate(end_label_parts) if end_label_parts else np.empty((0,), dtype=np.int8),
        pos_fraction=np.concatenate(pos_fraction_parts) if pos_fraction_parts else np.empty((0,), dtype=np.float32),
        has_transition=np.concatenate(transition_parts) if transition_parts else np.empty((0,), dtype=bool),
    )
    return cache_path


def sample_sequence_segments(
    data_root: Path,
    split: str,
    cache_dir: Path,
    segment_windows: int,
    count: int,
    seed: int,
    sampling: str,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cache_path = build_sequence_segment_index(data_root, split, cache_dir, segment_windows)
    with np.load(cache_path) as cache:
        seq = cache["seq"]
        start = cache["start"]
        end_label = cache["end_label"]
        pos_fraction = cache["pos_fraction"]
        has_transition = cache["has_transition"]

    if seq.size == 0:
        raise RuntimeError(f"No sequence segments found for split={split!r}, segment_windows={segment_windows}")

    if sampling == "random":
        eligible = np.arange(seq.shape[0])
        pick = rng.choice(eligible, size=count, replace=count > eligible.shape[0])
    elif sampling == "end_balanced":
        pos = np.flatnonzero(end_label == 1)
        neg = np.flatnonzero(end_label == 0)
        n_pos = count // 2
        n_neg = count - n_pos
        pick = np.concatenate(
            [
                rng.choice(pos, size=n_pos, replace=n_pos > pos.shape[0]),
                rng.choice(neg, size=n_neg, replace=n_neg > neg.shape[0]),
            ]
        )
    elif sampling == "state_balanced":
        slip = np.flatnonzero(pos_fraction >= 0.5)
        no_slip = np.flatnonzero(pos_fraction < 0.5)
        n_slip = count // 2
        n_no = count - n_slip
        pick = np.concatenate(
            [
                rng.choice(slip, size=n_slip, replace=n_slip > slip.shape[0]),
                rng.choice(no_slip, size=n_no, replace=n_no > no_slip.shape[0]),
            ]
        )
    elif sampling == "transition_mix":
        trans = np.flatnonzero(has_transition)
        stable_pos = np.flatnonzero((~has_transition) & (end_label == 1))
        stable_neg = np.flatnonzero((~has_transition) & (end_label == 0))
        n_trans = count // 3
        n_pos = (count - n_trans) // 2
        n_neg = count - n_trans - n_pos
        pick = np.concatenate(
            [
                rng.choice(trans, size=n_trans, replace=n_trans > trans.shape[0]),
                rng.choice(stable_pos, size=n_pos, replace=n_pos > stable_pos.shape[0]),
                rng.choice(stable_neg, size=n_neg, replace=n_neg > stable_neg.shape[0]),
            ]
        )
    else:
        raise ValueError(f"Unsupported sampling={sampling!r}")

    rng.shuffle(pick)
    return seq[pick].astype(np.int64, copy=False), start[pick].astype(np.int64, copy=False)


class WindowSequenceDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        seq_indices: np.ndarray,
        starts: np.ndarray,
        segment_windows: int,
        polarity_mode: str = "both",
        clip_max: float | None = None,
        spatial_pool: int = 4,
        context_ms: float | None = None,
        time_bins: int | None = None,
        sample_stride: int = 1,
    ) -> None:
        self.base = TacSpikeH5Dataset(
            data_root=data_root,
            split=split,
            polarity_mode=polarity_mode,
            clip_max=clip_max,
            spatial_pool=spatial_pool,
            context_ms=context_ms,
            time_bins=time_bins,
        )
        self.seq_indices = np.asarray(seq_indices, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.segment_windows = int(segment_windows)
        self.sample_stride = int(sample_stride)
        if self.sample_stride <= 0:
            raise ValueError(f"sample_stride must be positive, got {sample_stride}")
        if self.seq_indices.shape[0] != self.starts.shape[0]:
            raise ValueError("seq_indices and starts must have the same length")

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["base"].close()
        return state

    def close(self) -> None:
        self.base.close()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx = int(self.seq_indices[index])
        local_start = int(self.starts[index])
        seq_offset = int(self.base.offsets[seq_idx])
        local_positions = local_start + np.arange(
            0,
            self.segment_windows,
            self.sample_stride,
            dtype=np.int64,
        )
        global_indices = seq_offset + local_positions
        xs = []
        labels = []
        for global_index in global_indices:
            x, y = self.base[int(global_index)]
            xs.append(x)
            labels.append(y)
        return torch.from_numpy(np.stack(xs, axis=0)), torch.as_tensor(labels, dtype=torch.long)


def make_sequence_loader(
    args: argparse.Namespace,
    split: str,
    seq_indices: np.ndarray,
    starts: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = WindowSequenceDataset(
        data_root=args.data_root,
        split=split,
        seq_indices=seq_indices,
        starts=starts,
        segment_windows=args.segment_windows,
        polarity_mode=args.polarity_mode,
        clip_max=args.clip_max,
        spatial_pool=args.spatial_pool,
        context_ms=args.context_ms,
        time_bins=args.time_bins,
        sample_stride=args.sequence_stride,
    )
    dataset.input_scale = args.input_scale
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


def expand_transition_mask(labels: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0 or labels.shape[1] <= 1:
        return torch.zeros_like(labels, dtype=torch.bool)
    transitions = torch.zeros_like(labels, dtype=torch.bool)
    transitions[:, 1:] = labels[:, 1:] != labels[:, :-1]
    expanded = transitions.clone()
    for offset in range(1, int(radius) + 1):
        expanded[:, offset:] |= transitions[:, :-offset]
        expanded[:, :-offset] |= transitions[:, offset:]
    return expanded


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def sequence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: torch.Tensor | None,
    transition_ignore_steps: int,
    warmup_windows: int,
    tail_windows: int,
    smoothness_weight: float,
    flip_penalty_weight: float,
    transition_weight: float,
    label_smoothing: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    batch, steps = labels.shape
    valid = torch.ones((batch, steps), device=labels.device, dtype=torch.bool)
    if warmup_windows > 0:
        valid[:, : int(warmup_windows)] = False
    if tail_windows > 0 and tail_windows < steps:
        valid[:, : steps - int(tail_windows)] = False
    transition_mask = expand_transition_mask(labels, transition_ignore_steps)
    valid &= ~transition_mask

    flat_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        weight=class_weight,
        reduction="none",
        label_smoothing=float(label_smoothing),
    ).reshape(batch, steps)

    sample_weight = torch.ones_like(flat_ce)
    if transition_weight != 1.0 and transition_ignore_steps <= 0:
        sample_weight = torch.where(
            transition_mask,
            torch.as_tensor(float(transition_weight), device=labels.device, dtype=flat_ce.dtype),
            sample_weight,
        )
    ce_loss = masked_mean(flat_ce * sample_weight, valid)
    loss = ce_loss

    score = logits[..., 1] - logits[..., 0]
    same_label_pair = labels[:, 1:] == labels[:, :-1]
    pair_valid = valid[:, 1:] & valid[:, :-1] & same_label_pair
    smooth_loss = score.new_tensor(0.0)
    flip_loss = score.new_tensor(0.0)
    if smoothness_weight > 0.0 and pair_valid.any():
        score_delta = score[:, 1:] - score[:, :-1]
        smooth_loss = masked_mean(score_delta.pow(2), pair_valid)
        loss = loss + float(smoothness_weight) * smooth_loss
    if flip_penalty_weight > 0.0 and pair_valid.any():
        prob = torch.softmax(logits, dim=-1)[..., 1]
        prob_delta = prob[:, 1:] - prob[:, :-1]
        flip_loss = masked_mean(prob_delta.abs(), pair_valid)
        loss = loss + float(flip_penalty_weight) * flip_loss

    diagnostics = {
        "ce_loss": float(ce_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "flip_loss": float(flip_loss.detach().cpu()),
        "valid_fraction": float(valid.float().mean().detach().cpu()),
    }
    return loss, valid, diagnostics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp: bool,
    grad_clip: float,
    class_weight: torch.Tensor | None,
    transition_ignore_steps: int,
    warmup_windows: int,
    tail_windows: int,
    smoothness_weight: float,
    flip_penalty_weight: float,
    transition_weight: float,
    label_smoothing: float,
    max_batches: int | None = None,
) -> Dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    steps = 0
    all_scores = []
    all_labels = []
    valid_scores = []
    valid_labels = []
    diagnostic_totals: Dict[str, float] = {}
    firing_totals: Dict[str, float] = {}
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
            y = y.to(device=device, dtype=torch.long, non_blocking=True)
            batch, steps_per_segment = x.shape[:2]
            x_flat = x.reshape(batch * steps_per_segment, *x.shape[2:])

            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits_flat, stats = model(x_flat)
                logits = logits_flat.reshape(batch, steps_per_segment, -1)
                loss, valid_mask, diagnostics = sequence_loss(
                    logits=logits,
                    labels=y,
                    class_weight=class_weight,
                    transition_ignore_steps=transition_ignore_steps,
                    warmup_windows=warmup_windows,
                    tail_windows=tail_windows,
                    smoothness_weight=smoothness_weight,
                    flip_penalty_weight=flip_penalty_weight,
                    transition_weight=transition_weight,
                    label_smoothing=label_smoothing,
                )
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

            score_np = (logits[..., 1] - logits[..., 0]).detach().float().cpu().numpy().reshape(-1)
            label_np = y.detach().cpu().numpy().reshape(-1)
            valid_np = valid_mask.detach().cpu().numpy().reshape(-1)
            all_scores.append(score_np)
            all_labels.append(label_np)
            valid_scores.append(score_np[valid_np])
            valid_labels.append(label_np[valid_np])
            sample_count = int(label_np.shape[0])
            total_loss += float(loss.detach().item()) * sample_count
            total_samples += sample_count
            for key, value in diagnostics.items():
                diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value)
            for key, value in stats.items():
                firing_totals[key] = firing_totals.get(key, 0.0) + float(value.detach().item())
            steps += 1

    labels_np = np.concatenate(all_labels) if all_labels else np.empty((0,), dtype=np.int64)
    scores_np = np.concatenate(all_scores) if all_scores else np.empty((0,), dtype=np.float32)
    valid_labels_np = np.concatenate(valid_labels) if valid_labels else np.empty((0,), dtype=np.int64)
    valid_scores_np = np.concatenate(valid_scores) if valid_scores else np.empty((0,), dtype=np.float32)
    metrics = binary_classification_metrics(labels_np, scores_np)
    valid_metrics = binary_classification_metrics(valid_labels_np, valid_scores_np) if valid_labels_np.size else {}
    for key, value in valid_metrics.items():
        metrics[f"valid_{key}"] = value
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["num_samples"] = float(total_samples)
    metrics["num_batches"] = float(steps)
    metrics["seconds"] = time.time() - start
    metrics["samples_per_second"] = total_samples / max(metrics["seconds"], 1e-9)
    for key, value in diagnostic_totals.items():
        metrics[key] = value / max(steps, 1)
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
    parser = argparse.ArgumentParser(description="Train a sliding-window SCNN with sequence-state loss.")
    parser.add_argument("--model", choices=("lite_scnn", "deep_scnn", "frame_cnn", "time_channel_scnn", "temporal_conv_scnn"), default="time_channel_scnn")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-segments-per-epoch", type=int, default=3000)
    parser.add_argument("--val-segments", type=int, default=1000)
    parser.add_argument("--segment-windows", type=int, default=32)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temporal-mode", choices=("time_channels", "sum"), default="time_channels")
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--context-ms", type=float, default=400.0)
    parser.add_argument("--time-bins", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spatial-pool", type=int, default=4)
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--polarity-mode", choices=("both", "positive", "negative", "sum"), default="both")
    parser.add_argument("--sampling", choices=("random", "end_balanced", "state_balanced", "transition_mix"), default="transition_mix")
    parser.add_argument("--class-weight", choices=("none", "inverse_frequency"), default="none")
    parser.add_argument("--transition-ignore-steps", type=int, default=0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--warmup-windows", type=int, default=0)
    parser.add_argument("--tail-windows", type=int, default=0)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--flip-penalty-weight", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--best-metric", choices=("accuracy", "valid_accuracy", "balanced_accuracy", "valid_balanced_accuracy", "f1", "valid_f1"), default="valid_accuracy")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = args.cache_dir or (args.output_dir / "cache")
    if args.time_bins is None and args.context_ms is not None:
        args.time_bins = int(round(args.context_ms))
    if args.time_steps is None and args.time_bins is None:
        args.time_steps = 20
        args.time_bins = 20
    elif args.time_steps is None and args.time_bins is not None:
        args.time_steps = int(args.time_bins)
    elif args.time_bins is None and args.time_steps is not None:
        args.time_bins = int(args.time_steps)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(args).to(device)
    init_record: Dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        init_record = {
            "checkpoint": str(args.init_checkpoint),
            "epoch": int(checkpoint.get("epoch", -1)),
            "best_metric": float(checkpoint.get("best_metric", float("nan"))),
        }
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith(("fc2", "head", "classifier")):
                parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
        if args.scheduler == "cosine"
        else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    weight_tensor = None if args.class_weight == "none" else class_weights(args.data_root, "train", args.cache_dir, device)

    log_path = args.output_dir / "metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    write_jsonl(
        log_path,
        {
            "event": "config",
            "args": vars(args),
            "device": str(device),
            "parameter_count": count_parameters(model),
            "class_weights": None if weight_tensor is None else [float(x) for x in weight_tensor.detach().cpu().tolist()],
            "init": init_record,
        },
    )

    val_seq, val_starts = sample_sequence_segments(
        data_root=args.data_root,
        split="val",
        cache_dir=args.cache_dir,
        segment_windows=args.segment_windows,
        count=args.val_segments,
        seed=args.seed + 10_000,
        sampling=args.sampling,
    )
    val_loader = make_sequence_loader(args, "val", val_seq, val_starts, shuffle=False)
    best_metric = -math.inf
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        train_seq, train_starts = sample_sequence_segments(
            data_root=args.data_root,
            split="train",
            cache_dir=args.cache_dir,
            segment_windows=args.segment_windows,
            count=args.train_segments_per_epoch,
            seed=args.seed + epoch,
            sampling=args.sampling,
        )
        train_loader = make_sequence_loader(args, "train", train_seq, train_starts, shuffle=False)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=args.amp,
            grad_clip=args.grad_clip,
            class_weight=weight_tensor,
            transition_ignore_steps=args.transition_ignore_steps,
            warmup_windows=args.warmup_windows,
            tail_windows=args.tail_windows,
            smoothness_weight=args.smoothness_weight,
            flip_penalty_weight=args.flip_penalty_weight,
            transition_weight=args.transition_weight,
            label_smoothing=args.label_smoothing,
            max_batches=args.max_train_batches,
        )
        if scheduler is not None:
            scheduler.step()
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            scaler=None,
            amp=args.amp,
            grad_clip=args.grad_clip,
            class_weight=weight_tensor,
            transition_ignore_steps=args.transition_ignore_steps,
            warmup_windows=args.warmup_windows,
            tail_windows=args.tail_windows,
            smoothness_weight=args.smoothness_weight,
            flip_penalty_weight=args.flip_penalty_weight,
            transition_weight=args.transition_weight,
            label_smoothing=args.label_smoothing,
            max_batches=args.max_val_batches,
        )
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
            "best_epoch": best_epoch,
            "best_val_accuracy": best_metric,
            "best_val_metrics": best_metrics,
            "latest_epoch": epoch,
            "target_accuracy": args.target_accuracy,
            "target_met": best_metric >= args.target_accuracy,
            "best_metric": args.best_metric,
            "parameter_count": count_parameters(model),
            "output_dir": str(args.output_dir),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True, default=json_default), flush=True)


if __name__ == "__main__":
    main()
