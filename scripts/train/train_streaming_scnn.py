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

from tacspike.data import TacSpikeH5Dataset, voxelize_events_pooled
from tacspike.models import TacSpikeStreamingLiteSCNN, count_parameters
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


def build_segment_index(data_root: Path, split: str, cache_dir: Path, segment_steps: int, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_segments_t{segment_steps}.npz"
    if cache_path.exists() and not force:
        return cache_path
    base = TacSpikeH5Dataset(data_root=data_root, split=split)
    slip_seq = []
    no_slip_seq = []
    starts_slip = []
    starts_no_slip = []
    for seq_idx, info in enumerate(base.sequences):
        with h5py.File(info.path, "r") as h5:
            labels = h5["label/slip"][:].astype(np.int8, copy=False)
        max_start = max(0, labels.shape[0] - segment_steps)
        if max_start <= 0:
            continue
        # Use segment end label as the balancing target.
        end_labels = labels[segment_steps - 1 : segment_steps - 1 + max_start]
        local_starts = np.arange(max_start, dtype=np.int64)
        slip_mask = end_labels == 1
        no_slip_mask = ~slip_mask
        if slip_mask.any():
            slip_seq.append(np.full(int(slip_mask.sum()), seq_idx, dtype=np.int64))
            starts_slip.append(local_starts[slip_mask])
        if no_slip_mask.any():
            no_slip_seq.append(np.full(int(no_slip_mask.sum()), seq_idx, dtype=np.int64))
            starts_no_slip.append(local_starts[no_slip_mask])
    base.close()
    np.savez_compressed(
        cache_path,
        slip_seq=np.concatenate(slip_seq) if slip_seq else np.empty((0,), dtype=np.int64),
        slip_start=np.concatenate(starts_slip) if starts_slip else np.empty((0,), dtype=np.int64),
        no_slip_seq=np.concatenate(no_slip_seq) if no_slip_seq else np.empty((0,), dtype=np.int64),
        no_slip_start=np.concatenate(starts_no_slip) if starts_no_slip else np.empty((0,), dtype=np.int64),
    )
    return cache_path


def sample_segments(
    data_root: Path,
    split: str,
    cache_dir: Path,
    num_segments: int,
    segment_steps: int,
    seed: int,
    sampling: str,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cache_path = build_segment_index(data_root, split, cache_dir, segment_steps)
    with np.load(cache_path) as cache:
        slip_seq = cache["slip_seq"]
        slip_start = cache["slip_start"]
        no_seq = cache["no_slip_seq"]
        no_start = cache["no_slip_start"]

    if sampling == "balanced":
        n_slip = num_segments // 2
        n_no = num_segments - n_slip
        slip_pick = rng.choice(np.arange(len(slip_seq)), size=n_slip, replace=n_slip > len(slip_seq))
        no_pick = rng.choice(np.arange(len(no_seq)), size=n_no, replace=n_no > len(no_seq))
        seq = np.concatenate([slip_seq[slip_pick], no_seq[no_pick]])
        start = np.concatenate([slip_start[slip_pick], no_start[no_pick]])
    elif sampling == "random":
        seq_all = np.concatenate([slip_seq, no_seq])
        start_all = np.concatenate([slip_start, no_start])
        pick = rng.choice(np.arange(len(seq_all)), size=num_segments, replace=num_segments > len(seq_all))
        seq = seq_all[pick]
        start = start_all[pick]
    else:
        raise ValueError(f"Unsupported sampling={sampling!r}")

    order = rng.permutation(seq.shape[0])
    return seq[order].astype(np.int64, copy=False), start[order].astype(np.int64, copy=False)


class StreamingSegmentDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        seq_indices: np.ndarray,
        starts: np.ndarray,
        segment_steps: int,
        polarity_mode: str = "both",
        clip_max: float | None = None,
        spatial_pool: int = 4,
    ) -> None:
        self.base = TacSpikeH5Dataset(data_root=data_root, split=split)
        self.seq_indices = np.asarray(seq_indices, dtype=np.int64)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.segment_steps = int(segment_steps)
        self.polarity_mode = polarity_mode
        self.clip_max = clip_max
        self.spatial_pool = int(spatial_pool)
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
        self.base.close()

    def _open_h5(self, path: Path) -> h5py.File:
        if self._h5 is not None and self._open_path == path:
            return self._h5
        if self._h5 is not None:
            self._h5.close()
        self._h5 = h5py.File(path, "r")
        self._open_path = path
        return self._h5

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx = int(self.seq_indices[index])
        start = int(self.starts[index])
        info = self.base.sequences[seq_idx]
        h5 = self._open_h5(info.path)
        stop = start + self.segment_steps
        labels = h5["label/slip"][start:stop].astype(np.int64, copy=False)
        t_start = float(h5["windows/t_label"][start] - 0.001)
        t_end = float(h5["windows/t_label"][stop - 1])
        t_dataset = h5["events/t"]
        left = int(np.searchsorted(t_dataset, t_start, side="left"))
        right = int(np.searchsorted(t_dataset, t_end, side="right"))
        events = {key: h5[f"events/{key}"][left:right] for key in ("t", "x", "y", "p")}
        voxel = voxelize_events_pooled(
            events=events,
            t_start=t_start,
            t_end=t_end,
            bins=self.segment_steps,
            height=int(h5.attrs["height"]),
            width=int(h5.attrs["width"]),
            pool=self.spatial_pool,
            polarity_mode=self.polarity_mode,
            clip_max=self.clip_max,
        )
        return torch.from_numpy(voxel), torch.from_numpy(labels)


def make_loader(
    args: argparse.Namespace,
    split: str,
    seq_indices: np.ndarray,
    starts: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = StreamingSegmentDataset(
        data_root=args.data_root,
        split=split,
        seq_indices=seq_indices,
        starts=starts,
        segment_steps=args.segment_steps,
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


def sequence_detection_loss(
    logits_seq: torch.Tensor,
    labels: torch.Tensor,
    loss_mode: str,
    warmup_steps: int,
    supervise_tail_steps: int,
    transition_ignore_steps: int,
    positive_weight: float,
    negative_weight: float,
    smoothness_weight: float,
    flip_penalty_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    batch, steps = labels.shape
    valid = torch.ones((batch, steps), device=labels.device, dtype=torch.bool)
    if loss_mode == "last":
        valid[:, :-1] = False
    if warmup_steps > 0:
        valid[:, : int(warmup_steps)] = False
    if supervise_tail_steps > 0 and supervise_tail_steps < steps:
        valid[:, : steps - int(supervise_tail_steps)] = False
    if transition_ignore_steps > 0:
        valid &= ~expand_transition_mask(labels, int(transition_ignore_steps))

    ce = F.cross_entropy(
        logits_seq.reshape(-1, logits_seq.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).reshape(batch, steps)
    sample_weight = torch.where(
        labels == 1,
        torch.as_tensor(float(positive_weight), device=labels.device, dtype=ce.dtype),
        torch.as_tensor(float(negative_weight), device=labels.device, dtype=ce.dtype),
    )
    ce_loss = masked_mean(ce * sample_weight, valid)
    loss = ce_loss

    score = logits_seq[..., 1] - logits_seq[..., 0]
    pair_valid = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] == labels[:, :-1])
    smooth_loss = score.new_tensor(0.0)
    flip_loss = score.new_tensor(0.0)
    if smoothness_weight > 0.0 and pair_valid.any():
        score_delta = score[:, 1:] - score[:, :-1]
        smooth_loss = masked_mean(score_delta.pow(2), pair_valid)
        loss = loss + float(smoothness_weight) * smooth_loss
    if flip_penalty_weight > 0.0 and pair_valid.any():
        prob = torch.softmax(logits_seq, dim=-1)[..., 1]
        prob_delta = prob[:, 1:] - prob[:, :-1]
        flip_loss = masked_mean(prob_delta.abs(), pair_valid)
        loss = loss + float(flip_penalty_weight) * flip_loss

    diagnostics = {
        "valid_fraction": float(valid.float().mean().detach().cpu()),
        "ce_loss": float(ce_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "flip_loss": float(flip_loss.detach().cpu()),
    }
    return loss, valid, diagnostics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    loss_mode: str = "all",
    grad_clip: float = 1.0,
    warmup_steps: int = 0,
    supervise_tail_steps: int = 0,
    transition_ignore_steps: int = 0,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    smoothness_weight: float = 0.0,
    flip_penalty_weight: float = 0.0,
) -> Dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    valid_scores: List[np.ndarray] = []
    valid_labels: List[np.ndarray] = []
    firing_totals: Dict[str, float] = {}
    diagnostic_totals: Dict[str, float] = {}
    steps = 0
    start = time.time()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits_seq, stats = model(x)
                loss, valid_mask, diagnostics = sequence_detection_loss(
                    logits_seq=logits_seq,
                    labels=y,
                    loss_mode=loss_mode,
                    warmup_steps=warmup_steps,
                    supervise_tail_steps=supervise_tail_steps,
                    transition_ignore_steps=transition_ignore_steps,
                    positive_weight=positive_weight,
                    negative_weight=negative_weight,
                    smoothness_weight=smoothness_weight,
                    flip_penalty_weight=flip_penalty_weight,
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

            score = (logits_seq[..., 1] - logits_seq[..., 0]).detach().float().cpu().numpy().reshape(-1)
            labels = y.detach().cpu().numpy().reshape(-1)
            valid_np = valid_mask.detach().cpu().numpy().reshape(-1)
            all_scores.append(score)
            all_labels.append(labels)
            valid_scores.append(score[valid_np])
            valid_labels.append(labels[valid_np])
            bs = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * bs
            total_samples += bs
            for key, value in stats.items():
                firing_totals[key] = firing_totals.get(key, 0.0) + float(value.detach().item())
            for key, value in diagnostics.items():
                diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value)
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
    for key, value in firing_totals.items():
        metrics[key] = value / max(steps, 1)
    for key, value in diagnostic_totals.items():
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
    parser = argparse.ArgumentParser(description="Train stateful streaming Lite-SCNN with truncated BPTT.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-segments-per-epoch", type=int, default=50000)
    parser.add_argument("--val-segments", type=int, default=20000)
    parser.add_argument("--segment-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--surrogate-alpha", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--spatial-pool", type=int, default=4)
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--polarity-mode", choices=("both", "positive", "negative", "sum"), default="both")
    parser.add_argument("--sampling", choices=("balanced", "random"), default="balanced")
    parser.add_argument("--loss-mode", choices=("all", "last"), default="all")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--supervise-tail-steps", type=int, default=0)
    parser.add_argument("--transition-ignore-steps", type=int, default=0)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--flip-penalty-weight", type=float, default=0.0)
    parser.add_argument("--best-metric", choices=("accuracy", "valid_accuracy", "balanced_accuracy", "valid_balanced_accuracy"), default="valid_accuracy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = args.cache_dir or (args.output_dir / "cache")
    log_path = args.output_dir / "metrics.jsonl"
    summary_path = args.output_dir / "summary.json"

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    input_channels = 2 if args.polarity_mode == "both" else 1
    model = TacSpikeStreamingLiteSCNN(
        input_channels=input_channels,
        beta=args.beta,
        threshold=args.threshold,
        surrogate_alpha=args.surrogate_alpha,
        hidden=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
        if args.scheduler == "cosine"
        else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    write_jsonl(
        log_path,
        {
            "event": "config",
            "args": vars(args),
            "device": str(device),
            "parameter_count": count_parameters(model),
        },
    )

    val_seq, val_start = sample_segments(
        args.data_root,
        "val",
        args.cache_dir,
        args.val_segments,
        args.segment_steps,
        args.seed + 10_000,
        args.sampling,
    )
    val_loader = make_loader(args, "val", val_seq, val_start, shuffle=False)
    best_metric = -math.inf
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        train_seq, train_start = sample_segments(
            args.data_root,
            "train",
            args.cache_dir,
            args.train_segments_per_epoch,
            args.segment_steps,
            args.seed + epoch,
            args.sampling,
        )
        train_loader = make_loader(args, "train", train_seq, train_start, shuffle=False)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            amp=args.amp,
            scaler=scaler,
            loss_mode=args.loss_mode,
            grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps,
            supervise_tail_steps=args.supervise_tail_steps,
            transition_ignore_steps=args.transition_ignore_steps,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
            smoothness_weight=args.smoothness_weight,
            flip_penalty_weight=args.flip_penalty_weight,
        )
        if scheduler is not None:
            scheduler.step()
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            amp=args.amp,
            scaler=None,
            loss_mode=args.loss_mode,
            grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps,
            supervise_tail_steps=args.supervise_tail_steps,
            transition_ignore_steps=args.transition_ignore_steps,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
            smoothness_weight=args.smoothness_weight,
            flip_penalty_weight=args.flip_penalty_weight,
        )
        record = {"event": "epoch", "epoch": epoch, "train": train_metrics, "val": val_metrics}
        write_jsonl(log_path, record)
        metric = float(val_metrics.get(args.best_metric, val_metrics.get("accuracy", 0.0)))
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
            "best_metric": args.best_metric,
            "parameter_count": count_parameters(model),
            "output_dir": str(args.output_dir),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True, default=json_default), flush=True)


if __name__ == "__main__":
    main()
