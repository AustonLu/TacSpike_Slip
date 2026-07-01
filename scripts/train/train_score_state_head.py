from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from evaluate_score_cache_pair import debounce
from evaluate_state_decoder import (
    aggregate_event_metrics,
    basic_binary_metrics,
    best_accuracy_threshold,
    json_default,
    moving_average_per_sequence,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def zscore_row(scores: np.ndarray, mean: float | None = None, std: float | None = None) -> tuple[np.ndarray, float, float]:
    row_mean = float(scores.mean()) if mean is None else float(mean)
    row_std = max(float(scores.std()) if std is None else float(std), 1e-12)
    return (scores.astype(np.float64, copy=False) - row_mean) / row_std, row_mean, row_std


def load_score_rows(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    labels = None
    indices = None
    offsets = None
    rows = []
    records: list[dict[str, Any]] = []
    for path in paths:
        cache = np.load(path, allow_pickle=False)
        cache_labels = cache["labels"].astype(np.int64, copy=False)
        cache_indices = cache["global_indices"].astype(np.int64, copy=False)
        cache_offsets = cache["seq_offsets"].astype(np.int64, copy=False)
        if labels is None:
            labels = cache_labels
            indices = cache_indices
            offsets = cache_offsets
        elif not np.array_equal(labels, cache_labels):
            raise RuntimeError(f"Label mismatch for {path}")
        elif not np.array_equal(indices, cache_indices):
            raise RuntimeError(f"Global-index mismatch for {path}")
        elif not np.array_equal(offsets, cache_offsets):
            raise RuntimeError(f"Sequence-offset mismatch for {path}")
        matrix = cache["raw_score_matrix"].astype(np.float64, copy=False)
        for row_idx in range(matrix.shape[0]):
            rows.append(matrix[row_idx])
            records.append({"cache": str(path), "row": int(row_idx)})
    if labels is None or offsets is None:
        raise RuntimeError("No score caches loaded")
    return np.stack(rows, axis=0), labels, offsets, records


def normalize_val_rows(val_rows: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = train_rows.mean(axis=1)
    stds = np.maximum(train_rows.std(axis=1), 1e-12)
    return (val_rows - means[:, None]) / stds[:, None], means, stds


def normalize_rows(rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (rows - mean[:, None]) / std[:, None]


def build_features(
    score_rows: np.ndarray,
    seq_offsets: np.ndarray,
    ma_windows: list[int],
    ema_alphas: list[float],
) -> np.ndarray:
    features = [score_rows]
    for window in ma_windows:
        if window <= 1:
            continue
        features.append(np.stack([moving_average_per_sequence(row, seq_offsets, window) for row in score_rows], axis=0))
    for alpha in ema_alphas:
        ema_rows = []
        for row in score_rows:
            out = np.empty_like(row, dtype=np.float64)
            for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
                start_i = int(start)
                stop_i = int(stop)
                if start_i >= stop_i:
                    continue
                out[start_i] = row[start_i]
                for idx in range(start_i + 1, stop_i):
                    out[idx] = float(alpha) * row[idx] + (1.0 - float(alpha)) * out[idx - 1]
            ema_rows.append(out)
        features.append(np.stack(ema_rows, axis=0))
    stacked = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    return stacked.T


def build_features_from_rows(
    raw_rows: np.ndarray,
    seq_offsets: np.ndarray,
    feature_specs: list[tuple[str, Any]],
) -> np.ndarray:
    features = []
    for kind, value in feature_specs:
        if kind == "raw":
            features.append(raw_rows)
        elif kind == "ma":
            window = int(value)
            if window <= 1:
                features.append(raw_rows)
            else:
                features.append(np.stack([moving_average_per_sequence(row, seq_offsets, window) for row in raw_rows], axis=0))
        elif kind == "ema":
            alpha = float(value)
            ema_rows = []
            for row in raw_rows:
                out = np.empty_like(row, dtype=np.float64)
                for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
                    start_i = int(start)
                    stop_i = int(stop)
                    if start_i >= stop_i:
                        continue
                    out[start_i] = row[start_i]
                    for idx in range(start_i + 1, stop_i):
                        out[idx] = float(alpha) * row[idx] + (1.0 - float(alpha)) * out[idx - 1]
                ema_rows.append(out)
            features.append(np.stack(ema_rows, axis=0))
        else:
            raise ValueError(f"Unsupported feature kind={kind!r}")
    stacked = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    return stacked.T


def sequence_chunk_starts(seq_offsets: np.ndarray, chunk_len: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    seq_parts = []
    start_parts = []
    for seq_idx, (start, stop) in enumerate(zip(seq_offsets[:-1], seq_offsets[1:])):
        start_i = int(start)
        stop_i = int(stop)
        if stop_i - start_i <= chunk_len:
            seq_parts.append(seq_idx)
            start_parts.append(start_i)
            continue
        local = np.arange(start_i, stop_i - chunk_len + 1, int(stride), dtype=np.int64)
        if local[-1] != stop_i - chunk_len:
            local = np.concatenate([local, np.asarray([stop_i - chunk_len], dtype=np.int64)])
        seq_parts.extend([seq_idx] * int(local.shape[0]))
        start_parts.extend(local.tolist())
    return np.asarray(seq_parts, dtype=np.int64), np.asarray(start_parts, dtype=np.int64)


class ScoreChunkDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        starts: np.ndarray,
        chunk_len: int,
        random_jitter: int = 0,
    ) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.starts = starts.astype(np.int64, copy=False)
        self.chunk_len = int(chunk_len)
        self.random_jitter = int(random_jitter)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        if self.random_jitter > 0:
            jitter = random.randint(-self.random_jitter, self.random_jitter)
            start = max(0, min(start + jitter, self.features.shape[0] - self.chunk_len))
        stop = start + self.chunk_len
        return torch.from_numpy(self.features[start:stop]), torch.from_numpy(self.labels[start:stop])


class CausalTemporalStateHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        channels = int(input_dim)
        for layer_idx in range(int(layers)):
            dilation = 2**layer_idx
            conv = nn.Conv1d(channels, int(hidden_dim), kernel_size=int(kernel_size), padding=0, dilation=dilation)
            modules.append(conv)
            modules.append(nn.GELU())
            modules.append(nn.Dropout(float(dropout)))
            channels = int(hidden_dim)
        self.net = nn.ModuleList(modules)
        self.out = nn.Conv1d(channels, 1, kernel_size=1)
        self.kernel_size = int(kernel_size)
        self.layers = int(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        z = x.transpose(1, 2)
        layer_idx = 0
        for module in self.net:
            if isinstance(module, nn.Conv1d):
                dilation = module.dilation[0]
                left_pad = (module.kernel_size[0] - 1) * dilation
                z = F.pad(z, (left_pad, 0))
                z = module(z)
                layer_idx += 1
            else:
                z = module(z)
        logits = self.out(z).squeeze(1)
        return logits


def transition_mask(labels: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0 or labels.shape[1] <= 1:
        return torch.zeros_like(labels, dtype=torch.bool)
    trans = torch.zeros_like(labels, dtype=torch.bool)
    trans[:, 1:] = labels[:, 1:] != labels[:, :-1]
    out = trans.clone()
    for offset in range(1, int(radius) + 1):
        out[:, offset:] |= trans[:, :-offset]
        out[:, :-offset] |= trans[:, offset:]
    return out


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor | None,
    transition_ignore: int,
    smoothness_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    scores = []
    labels_all = []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        valid = ~transition_mask(y, transition_ignore)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss_raw = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
            loss = (loss_raw * valid.to(loss_raw.dtype)).sum() / valid.to(loss_raw.dtype).sum().clamp_min(1.0)
            if smoothness_weight > 0 and logits.shape[1] > 1:
                probs = torch.sigmoid(logits)
                smooth = (probs[:, 1:] - probs[:, :-1]).abs().mean()
                loss = loss + float(smoothness_weight) * smooth
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
        total_loss += float(loss.detach().item()) * int(y.numel())
        total_count += int(y.numel())
        scores.append(logits.detach().cpu().numpy().reshape(-1))
        labels_all.append(y.detach().cpu().numpy().reshape(-1).astype(np.int64))
    label_np = np.concatenate(labels_all) if labels_all else np.empty((0,), dtype=np.int64)
    score_np = np.concatenate(scores) if scores else np.empty((0,), dtype=np.float32)
    metrics = best_accuracy_threshold(label_np, score_np)
    metrics["loss"] = total_loss / max(total_count, 1)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float, np.integer, np.floating))}


def predict_full_sequences(model: nn.Module, features: np.ndarray, seq_offsets: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = np.empty(features.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start, stop in zip(seq_offsets[:-1], seq_offsets[1:]):
            start_i = int(start)
            stop_i = int(stop)
            seq = features[start_i:stop_i]
            tensor = torch.from_numpy(seq[None, ...]).to(device=device, dtype=torch.float32)
            logits = model(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            out[start_i:stop_i] = logits
    return out


def evaluate_sequence_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    seq_offsets: np.ndarray,
    ma_windows: list[int],
    debounce_on: list[int],
    debounce_off: list[int],
) -> dict[str, Any]:
    methods: list[dict[str, Any]] = []
    series = {"raw": scores.astype(np.float64, copy=False)}
    for window in ma_windows:
        series[f"ma{window}"] = moving_average_per_sequence(scores, seq_offsets, int(window))
    for name, seq_scores in series.items():
        tuned = best_accuracy_threshold(labels, seq_scores)
        threshold = float(tuned["threshold"])
        pred = seq_scores > threshold
        methods.append(
            {
                "method": f"{name}_threshold",
                "threshold": threshold,
                "metrics": tuned,
                "event_metrics": aggregate_event_metrics(labels, pred, seq_offsets),
            }
        )
        for on_k in debounce_on:
            for off_k in debounce_off:
                pred = debounce(seq_scores, seq_offsets, threshold, int(on_k), int(off_k))
                methods.append(
                    {
                        "method": f"{name}_debounce_on{on_k}_off{off_k}",
                        "threshold": threshold,
                        "on_k": int(on_k),
                        "off_k": int(off_k),
                        "metrics": basic_binary_metrics(labels, pred),
                        "event_metrics": aggregate_event_metrics(labels, pred, seq_offsets),
                    }
                )
    methods.sort(key=lambda item: float(item["metrics"]["accuracy"]), reverse=True)
    return {"best_method": methods[0], "methods_top": methods[:50]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight causal temporal state head on cached SNN scores.")
    parser.add_argument("--train-score-caches", required=True, nargs="+", type=Path)
    parser.add_argument("--val-score-caches", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--chunk-len", type=int, default=2048)
    parser.add_argument("--chunk-stride", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ma-feature-windows", default="20,50,100")
    parser.add_argument("--ema-feature-alphas", default="0.02,0.05,0.1")
    parser.add_argument("--extra-train-score-caches", nargs="*", type=Path, default=[])
    parser.add_argument("--extra-val-score-caches", nargs="*", type=Path, default=[])
    parser.add_argument("--extra-feature-specs", default="")
    parser.add_argument("--eval-ma-windows", default="1,20,50,100")
    parser.add_argument("--debounce-on-k", default="1,2,3,5")
    parser.add_argument("--debounce-off-k", default="2,5,10,20")
    parser.add_argument("--transition-ignore", type=int, default=0)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows_raw, train_labels, train_offsets, row_records = load_score_rows(args.train_score_caches)
    val_rows_raw, val_labels, val_offsets, val_records = load_score_rows(args.val_score_caches)
    if len(row_records) != len(val_records):
        raise RuntimeError("Train and val cache row counts differ")
    train_rows, means, stds = normalize_val_rows(train_rows_raw, train_rows_raw)
    val_rows = (val_rows_raw - means[:, None]) / stds[:, None]

    ma_features = parse_csv_ints(args.ma_feature_windows)
    ema_features = parse_csv_floats(args.ema_feature_alphas)
    train_features = build_features(train_rows, train_offsets, ma_features, ema_features)
    val_features = build_features(val_rows, val_offsets, ma_features, ema_features)

    extra_specs: list[tuple[str, Any]] = []
    if args.extra_feature_specs.strip():
        for part in args.extra_feature_specs.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                kind, value = part.split(":", 1)
            else:
                kind, value = part, None
            extra_specs.append((kind, value))

    if args.extra_train_score_caches or args.extra_val_score_caches:
        extra_train_rows_raw, extra_train_labels, extra_train_offsets, extra_row_records = load_score_rows(args.extra_train_score_caches)
        extra_val_rows_raw, extra_val_labels, extra_val_offsets, extra_val_records = load_score_rows(args.extra_val_score_caches)
        if not np.array_equal(train_labels, extra_train_labels) or not np.array_equal(train_offsets, extra_train_offsets):
            raise RuntimeError("Extra train score cache does not align with the main train cache")
        if not np.array_equal(val_labels, extra_val_labels) or not np.array_equal(val_offsets, extra_val_offsets):
            raise RuntimeError("Extra val score cache does not align with the main val cache")
        extra_train_rows, extra_means, extra_stds = normalize_val_rows(extra_train_rows_raw, extra_train_rows_raw)
        extra_val_rows = (extra_val_rows_raw - extra_means[:, None]) / extra_stds[:, None]
        train_features = np.concatenate(
            [
                train_features,
                build_features_from_rows(extra_train_rows, train_offsets, extra_specs or [("raw", None)]),
            ],
            axis=1,
        )
        val_features = np.concatenate(
            [
                val_features,
                build_features_from_rows(extra_val_rows, val_offsets, extra_specs or [("raw", None)]),
            ],
            axis=1,
        )
        row_records.extend(extra_row_records)
        val_records.extend(extra_val_records)

    _, starts = sequence_chunk_starts(train_offsets, args.chunk_len, args.chunk_stride)
    dataset = ScoreChunkDataset(train_features, train_labels, starts, args.chunk_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = CausalTemporalStateHead(
        input_dim=train_features.shape[1],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    if args.pos_weight > 0:
        pos_weight = torch.tensor(float(args.pos_weight), dtype=torch.float32, device=device)
    else:
        pos_frac = float(train_labels.mean())
        pos_weight = torch.tensor((1.0 - pos_frac) / max(pos_frac, 1e-6), dtype=torch.float32, device=device)

    best_acc = -math.inf
    best_eval: dict[str, Any] = {}
    log_path = args.output_dir / "metrics.jsonl"
    write_jsonl(
        log_path,
        {
            "event": "config",
            "args": vars(args),
            "row_records": row_records,
            "train_windows": int(train_labels.shape[0]),
            "val_windows": int(val_labels.shape[0]),
            "feature_dim": int(train_features.shape[1]),
            "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "pos_weight": float(pos_weight.detach().cpu().item()),
        },
    )

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=loader,
            device=device,
            optimizer=optimizer,
            pos_weight=pos_weight,
            transition_ignore=args.transition_ignore,
            smoothness_weight=args.smoothness_weight,
            grad_clip=args.grad_clip,
        )
        scheduler.step()
        val_scores = predict_full_sequences(model, val_features, val_offsets, device=device)
        val_eval = evaluate_sequence_predictions(
            val_labels,
            val_scores,
            val_offsets,
            ma_windows=parse_csv_ints(args.eval_ma_windows),
            debounce_on=parse_csv_ints(args.debounce_on_k),
            debounce_off=parse_csv_ints(args.debounce_off_k),
        )
        val_acc = float(val_eval["best_method"]["metrics"]["accuracy"])
        record = {"event": "epoch", "epoch": epoch, "train": train_metrics, "val_sequence": val_eval["best_method"]}
        write_jsonl(log_path, record)
        if val_acc > best_acc:
            best_acc = val_acc
            best_eval = val_eval
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "feature_means": means.astype(np.float32),
                    "feature_stds": stds.astype(np.float32),
                    "row_records": row_records,
                    "epoch": epoch,
                    "best_accuracy": best_acc,
                    "best_eval": best_eval,
                },
                args.output_dir / "best.pt",
            )
        summary = {
            "best_accuracy": best_acc,
            "best_eval": best_eval,
            "latest_epoch": epoch,
            "seconds": time.time() - start_time,
            "target_accuracy": 0.95,
            "target_met": best_acc >= 0.95,
            "output_dir": str(args.output_dir),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"epoch": epoch, "best_accuracy": best_acc, "val_accuracy": val_acc}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
