from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.train.evaluate_sliding_detection import (
    best_debounce_record,
    per_sequence_causal_ma,
    per_sequence_ema,
    score_method_record,
    transition_distances_per_sequence,
)
from tacspike.data import load_stream_cache_manifest, read_event_bins_slice
from tacspike.models import TacSpikeStreamingLiteSCNN


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


def select_sequence_indices(total: int, max_sequences: int, seed: int) -> np.ndarray:
    if max_sequences <= 0 or max_sequences >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(total, dtype=np.int64), size=max_sequences, replace=False))


def build_model_from_checkpoint(checkpoint: Path, device: torch.device) -> Tuple[TacSpikeStreamingLiteSCNN, argparse.Namespace]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ckpt["args"])
    input_channels = 2 if getattr(train_args, "polarity_mode", "both") == "both" else 1
    model = TacSpikeStreamingLiteSCNN(
        input_channels=input_channels,
        beta=getattr(train_args, "beta", 0.85),
        threshold=getattr(train_args, "threshold", 0.1),
        surrogate_alpha=getattr(train_args, "surrogate_alpha", 2.0),
        hidden=getattr(train_args, "hidden_dim", 64),
        conv1_channels=getattr(train_args, "conv1_channels", 16),
        conv2_channels=getattr(train_args, "conv2_channels", 32),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, train_args


def sequence_scores_from_cache(
    model: TacSpikeStreamingLiteSCNN,
    cache_path: Path,
    device: torch.device,
    chunk_steps: int,
    input_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(cache_path, "r") as h5:
        labels = h5["labels"][:].astype(np.int64, copy=False)
        length = int(labels.shape[0])
        scores: List[np.ndarray] = []
        state = None
        with torch.no_grad():
            for start in range(0, length, int(chunk_steps)):
                stop = min(start + int(chunk_steps), length)
                x_np = read_event_bins_slice(h5, start, stop)
                if input_scale != 1.0:
                    x_np = x_np * float(input_scale)
                x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
                logits, _, state = model(x.unsqueeze(0), state=state, return_state=True)
                scores.append((logits[..., 1] - logits[..., 0]).squeeze(0).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32, copy=False), labels


def method_accuracy(record: Dict[str, Any]) -> float:
    if record["kind"] == "score":
        return float(record["best_threshold_metrics"]["accuracy"])
    return float(record["metrics"]["accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stream-cache-trained stateful SNN on full sequences.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stream-cache-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-score-cache", type=Path, default=None)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-steps", type=int, default=2048)
    parser.add_argument("--ma-windows", default="20,50,80,100")
    parser.add_argument("--ema-alphas", default="0.02,0.05,0.1")
    parser.add_argument("--debounce-on-k", default="2,3,5")
    parser.add_argument("--debounce-off-k", default="10,20,30")
    parser.add_argument("--debounce-threshold-grid", type=int, default=24)
    parser.add_argument("--top-sequences", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, train_args = build_model_from_checkpoint(args.checkpoint, device)
    sequences = load_stream_cache_manifest(
        cache_root=args.stream_cache_root,
        split=args.split,
        spatial_pool=getattr(train_args, "spatial_pool", 4),
        polarity_mode=getattr(train_args, "polarity_mode", "both"),
        clip_max=getattr(train_args, "clip_max", None),
        dtype=getattr(train_args, "cache_dtype", "float16"),
        cache_format=getattr(train_args, "cache_format", "dense"),
    )
    selected = select_sequence_indices(len(sequences), args.max_sequences, args.seed)

    scores_parts = []
    labels_parts = []
    seq_offsets = [0]
    sequence_records: List[Dict[str, Any]] = []
    for seq_idx in selected:
        info = sequences[int(seq_idx)]
        scores, labels = sequence_scores_from_cache(
            model=model,
            cache_path=info.path,
            device=device,
            chunk_steps=args.chunk_steps,
            input_scale=float(getattr(train_args, "input_scale", 1.0)),
        )
        scores_parts.append(scores)
        labels_parts.append(labels)
        seq_offsets.append(seq_offsets[-1] + int(labels.shape[0]))
        sequence_records.append(
            {
                "sequence_id": info.sequence_id,
                "sequence_index": int(seq_idx),
                "windows": int(labels.shape[0]),
                "positive_fraction": float(labels.mean()) if labels.size else 0.0,
                "score_mean": float(scores.mean()) if scores.size else 0.0,
                "score_std": float(scores.std()) if scores.size else 0.0,
            }
        )

    scores_all = np.concatenate(scores_parts) if scores_parts else np.empty((0,), dtype=np.float32)
    labels_all = np.concatenate(labels_parts) if labels_parts else np.empty((0,), dtype=np.int64)
    seq_offsets_np = np.asarray(seq_offsets, dtype=np.int64)
    distances = transition_distances_per_sequence(labels_all, seq_offsets_np)

    methods: List[Dict[str, Any]] = []
    score_series: Dict[str, np.ndarray] = {"raw": scores_all.astype(np.float64)}
    for window in parse_csv_ints(args.ma_windows):
        score_series[f"ma_{window}"] = per_sequence_causal_ma(scores_all, seq_offsets_np, window)
    for alpha in parse_csv_floats(args.ema_alphas):
        score_series[f"ema_{alpha:g}"] = per_sequence_ema(scores_all, seq_offsets_np, alpha)

    for name, method_scores in score_series.items():
        record = score_method_record(name, labels_all, method_scores, seq_offsets_np, distances)
        methods.append(record)
        threshold = float(record["threshold"])
        for on_k in parse_csv_ints(args.debounce_on_k):
            for off_k in parse_csv_ints(args.debounce_off_k):
                methods.append(
                    best_debounce_record(
                        f"{name}_debounce_on{on_k}_off{off_k}",
                        labels_all,
                        method_scores,
                        seq_offsets_np,
                        threshold,
                        on_k,
                        off_k,
                        args.debounce_threshold_grid,
                    )
                )

    methods.sort(key=method_accuracy, reverse=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "stream_cache_root": str(args.stream_cache_root),
        "split": args.split,
        "selected_sequences": int(selected.shape[0]),
        "total_windows": int(labels_all.shape[0]),
        "label_positive_fraction": float(labels_all.mean()) if labels_all.size else 0.0,
        "sequence_records": sequence_records[: args.top_sequences],
        "methods_top": methods[:50],
        "best_method": methods[0] if methods else None,
        "train_args": vars(train_args),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    if args.output_score_cache is not None:
        args.output_score_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_score_cache,
            scores=scores_all.astype(np.float32, copy=False),
            labels=labels_all.astype(np.int64, copy=False),
            seq_offsets=seq_offsets_np,
            selected_sequence_indices=selected,
        )
    print(
        json.dumps(
            {
                "selected_sequences": result["selected_sequences"],
                "total_windows": result["total_windows"],
                "best_method": None if result["best_method"] is None else result["best_method"]["method"],
                "best_accuracy": 0.0 if result["best_method"] is None else method_accuracy(result["best_method"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
