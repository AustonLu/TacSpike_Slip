from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.train.evaluate_sliding_detection import (
    aggregate_event_metrics,
    best_debounce_record,
    per_sequence_causal_ma,
    per_sequence_ema,
    score_method_record,
    transition_distances_per_sequence,
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def parse_csv_ints(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def zscore_scores(scores: np.ndarray) -> np.ndarray:
    scores64 = scores.astype(np.float64, copy=False)
    return ((scores64 - float(scores64.mean())) / max(float(scores64.std()), 1e-12)).astype(np.float32, copy=False)


def load_score_cache(path: Path, score_index: int = 0, score_reduce: str = "mean_zscore") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    labels = data["labels"].astype(np.int64, copy=False)
    seq_offsets = data["seq_offsets"].astype(np.int64, copy=False)
    if "scores" in data:
        scores = data["scores"].astype(np.float32, copy=False)
    elif "raw_score_matrix" in data:
        matrix = data["raw_score_matrix"].astype(np.float64, copy=False)
        if matrix.ndim != 2:
            raise ValueError(f"Expected raw_score_matrix shape [N,S], got {matrix.shape}")
        if score_reduce == "row":
            scores = matrix[int(score_index)].astype(np.float32, copy=False)
        elif score_reduce == "mean":
            scores = matrix.mean(axis=0).astype(np.float32, copy=False)
        elif score_reduce == "mean_zscore":
            scores = np.stack([zscore_scores(row) for row in matrix], axis=0).mean(axis=0).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported score_reduce={score_reduce!r}")
    else:
        keys = ", ".join(data.files)
        raise KeyError(f"{path} does not contain scores or raw_score_matrix. Available keys: {keys}")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"Score/label length mismatch in {path}: {scores.shape[0]} vs {labels.shape[0]}")
    return scores, labels, seq_offsets


def build_score_feature_matrix(scores: np.ndarray, seq_offsets: np.ndarray, ma_windows: List[int]) -> np.ndarray:
    parts = [scores.astype(np.float64, copy=False)]
    for window in ma_windows:
        ma = per_sequence_causal_ma(scores, seq_offsets, window)
        parts.append(ma)
        parts.append(scores.astype(np.float64, copy=False) - ma)
    for short, long in ((20, 100), (50, 200), (100, 400)):
        if short in ma_windows and long in ma_windows:
            parts.append(per_sequence_causal_ma(scores, seq_offsets, short) - per_sequence_causal_ma(scores, seq_offsets, long))
    return np.stack(parts, axis=1).astype(np.float32, copy=False)


def fit_feature_normalizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)


def apply_feature_normalizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32, copy=False)


def transition_valid_mask(labels: np.ndarray, seq_offsets: np.ndarray, ignore_steps: int) -> np.ndarray:
    if ignore_steps <= 0:
        return np.ones(labels.shape[0], dtype=bool)
    distances = transition_distances_per_sequence(labels, seq_offsets)
    return distances > float(ignore_steps)


class ScoreMLPAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_adapter(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    y_t = torch.from_numpy(y.astype(np.float32, copy=False)).to(device=device)
    valid_t = torch.from_numpy(valid.astype(np.float32, copy=False)).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_state = None
    history = []
    indices = np.arange(x.shape[0])
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        model.train()
        total_loss = 0.0
        total_count = 0
        for start in range(0, indices.shape[0], args.batch_size):
            batch = indices[start : start + args.batch_size]
            xb = x_t[batch]
            yb = y_t[batch]
            vb = valid_t[batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_raw = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (loss_raw * vb).sum() / vb.sum().clamp_min(1.0)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch.shape[0])
            total_count += int(batch.shape[0])
        model.eval()
        with torch.no_grad():
            logits_all = model(x_t)
            loss_raw = F.binary_cross_entropy_with_logits(logits_all, y_t, reduction="none")
            valid_loss = float(((loss_raw * valid_t).sum() / valid_t.sum().clamp_min(1.0)).detach().cpu())
            scores = logits_all.detach().cpu().numpy()
        valid_metrics = binary_classification_metrics(y[valid], scores[valid])
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "valid_loss": valid_loss,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
            "valid_f1": valid_metrics["f1"],
            "valid_roc_auc": valid_metrics["roc_auc"],
            "valid_pr_auc": valid_metrics["pr_auc"],
        }
        history.append(record)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_loss": best_loss}


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    seq_offsets: np.ndarray,
    ma_windows: List[int],
    ema_alphas: List[float],
    debounce_on_k: List[int],
    debounce_off_k: List[int],
    debounce_threshold_grid: int,
) -> Dict[str, Any]:
    distances = transition_distances_per_sequence(labels, seq_offsets)
    methods: List[Dict[str, Any]] = []
    score_series: Dict[str, np.ndarray] = {"adapter_raw": scores.astype(np.float64)}
    for window in ma_windows:
        score_series[f"adapter_ma_{window}"] = per_sequence_causal_ma(scores, seq_offsets, window)
    for alpha in ema_alphas:
        score_series[f"adapter_ema_{alpha:g}"] = per_sequence_ema(scores, seq_offsets, alpha)

    for name, method_scores in score_series.items():
        record = score_method_record(name, labels, method_scores, seq_offsets, distances)
        methods.append(record)
        threshold = float(record["threshold"])
        for on_k in debounce_on_k:
            for off_k in debounce_off_k:
                methods.append(
                    best_debounce_record(
                        f"{name}_debounce_on{on_k}_off{off_k}",
                        labels,
                        method_scores,
                        seq_offsets,
                        threshold,
                        on_k,
                        off_k,
                        debounce_threshold_grid,
                    )
                )

    def method_accuracy(record: Dict[str, Any]) -> float:
        if record["kind"] == "score":
            return float(record["best_threshold_metrics"]["accuracy"])
        return float(record["metrics"]["accuracy"])

    methods.sort(key=method_accuracy, reverse=True)
    best_pred = scores > float(methods[0].get("threshold", 0.0)) if methods and methods[0]["kind"] == "score" else None
    return {
        "methods_top": methods[:50],
        "best_method": methods[0] if methods else None,
        "raw_event_metrics": None if best_pred is None else aggregate_event_metrics(labels, best_pred, seq_offsets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight temporal adapter on sequence score cache.")
    parser.add_argument("--train-score-cache", required=True, type=Path)
    parser.add_argument("--val-score-cache", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--score-index", type=int, default=0)
    parser.add_argument("--score-reduce", choices=("row", "mean", "mean_zscore"), default="mean_zscore")
    parser.add_argument("--ma-windows", default="20,50,100,200,400")
    parser.add_argument("--eval-ma-windows", default="20,50,80,100,150,200")
    parser.add_argument("--ema-alphas", default="0.02,0.05,0.1")
    parser.add_argument("--debounce-on-k", default="2,3,5,8")
    parser.add_argument("--debounce-off-k", default="10,20,30,50")
    parser.add_argument("--debounce-threshold-grid", type=int, default=24)
    parser.add_argument("--transition-ignore-steps", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_scores, train_labels, train_offsets = load_score_cache(args.train_score_cache, args.score_index, args.score_reduce)
    val_scores, val_labels, val_offsets = load_score_cache(args.val_score_cache, args.score_index, args.score_reduce)
    feature_windows = parse_csv_ints(args.ma_windows)
    x_train_raw = build_score_feature_matrix(train_scores, train_offsets, feature_windows)
    x_val_raw = build_score_feature_matrix(val_scores, val_offsets, feature_windows)
    feature_mean, feature_std = fit_feature_normalizer(x_train_raw)
    x_train = apply_feature_normalizer(x_train_raw, feature_mean, feature_std)
    x_val = apply_feature_normalizer(x_val_raw, feature_mean, feature_std)
    train_valid = transition_valid_mask(train_labels, train_offsets, args.transition_ignore_steps)
    val_valid = transition_valid_mask(val_labels, val_offsets, args.transition_ignore_steps)

    model = ScoreMLPAdapter(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    train_info = train_adapter(model, x_train, train_labels, train_valid, args, device)
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_val).to(device=device, dtype=torch.float32)).detach().cpu().numpy()

    valid_metrics = binary_classification_metrics(val_labels[val_valid], val_logits[val_valid])
    sequence_eval = evaluate_scores(
        labels=val_labels,
        scores=val_logits,
        seq_offsets=val_offsets,
        ma_windows=parse_csv_ints(args.eval_ma_windows),
        ema_alphas=parse_csv_floats(args.ema_alphas),
        debounce_on_k=parse_csv_ints(args.debounce_on_k),
        debounce_off_k=parse_csv_ints(args.debounce_off_k),
        debounce_threshold_grid=args.debounce_threshold_grid,
    )
    result = {
        "train_score_cache": str(args.train_score_cache),
        "val_score_cache": str(args.val_score_cache),
        "score_reduce": args.score_reduce,
        "score_index": int(args.score_index),
        "feature_dim": int(x_train.shape[1]),
        "feature_mean": feature_mean.reshape(-1),
        "feature_std": feature_std.reshape(-1),
        "train_windows": int(train_labels.shape[0]),
        "val_windows": int(val_labels.shape[0]),
        "transition_ignore_steps": int(args.transition_ignore_steps),
        "valid_metrics": valid_metrics,
        "train_info": train_info,
        "sequence_eval": sequence_eval,
        "args": vars(args),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    if args.output_checkpoint is not None:
        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "result": result}, args.output_checkpoint)
    print(
        json.dumps(
            {
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
                "best_method": None if sequence_eval["best_method"] is None else sequence_eval["best_method"]["method"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
