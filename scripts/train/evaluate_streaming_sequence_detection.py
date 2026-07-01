from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
    binary_method_record,
    per_sequence_causal_ma,
    per_sequence_debounce,
    per_sequence_ema,
    score_method_record,
    transition_distances_per_sequence,
)
from tacspike.data import TacSpikeH5Dataset, voxelize_events_pooled
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


def build_model_from_checkpoint(checkpoint: Path, device: torch.device) -> tuple[TacSpikeStreamingLiteSCNN, argparse.Namespace]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = argparse.Namespace(**ckpt["args"])
    input_channels = 2 if getattr(args, "polarity_mode", "both") == "both" else 1
    model = TacSpikeStreamingLiteSCNN(
        input_channels=input_channels,
        beta=getattr(args, "beta", 0.85),
        threshold=getattr(args, "threshold", 0.1),
        surrogate_alpha=getattr(args, "surrogate_alpha", 2.0),
        hidden=getattr(args, "hidden_dim", 64),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, args


def stream_sequence_scores(
    model: TacSpikeStreamingLiteSCNN,
    h5: h5py.File,
    args: argparse.Namespace,
    device: torch.device,
    chunk_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = h5["label/slip"][:].astype(np.int64, copy=False)
    if labels.size == 0:
        return np.empty((0,), dtype=np.float32), labels

    t_start = float(h5["windows/t_label"][0] - 0.001)
    t_end = float(h5["windows/t_label"][labels.shape[0] - 1])
    t_dataset = h5["events/t"]
    left = int(np.searchsorted(t_dataset, t_start, side="left"))
    right = int(np.searchsorted(t_dataset, t_end, side="right"))
    events = {key: h5[f"events/{key}"][left:right] for key in ("t", "x", "y", "p")}
    voxel = voxelize_events_pooled(
        events=events,
        t_start=t_start,
        t_end=t_end,
        bins=int(labels.shape[0]),
        height=int(h5.attrs["height"]),
        width=int(h5.attrs["width"]),
        pool=int(getattr(args, "spatial_pool", 4)),
        polarity_mode=getattr(args, "polarity_mode", "both"),
        clip_max=getattr(args, "clip_max", None),
    )

    scores: list[np.ndarray] = []
    state = None
    with torch.no_grad():
        for start in range(0, voxel.shape[0], chunk_steps):
            stop = min(start + chunk_steps, voxel.shape[0])
            x = torch.from_numpy(voxel[start:stop]).to(device=device, dtype=torch.float32)
            for step in range(x.shape[0]):
                logits, state, _ = model.step(x[step : step + 1], state)
                scores.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32, copy=False), labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stateful streaming SNN on full TacSpike sequences.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-steps", type=int, default=2048)
    parser.add_argument("--ma-windows", default="3,5,10,20,50")
    parser.add_argument("--ema-alphas", default="0.1,0.2,0.4")
    parser.add_argument("--debounce-on-k", default="2,3,5")
    parser.add_argument("--debounce-off-k", default="2,3,5,10")
    parser.add_argument("--debounce-threshold-grid", type=int, default=1)
    parser.add_argument("--top-sequences", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, train_args = build_model_from_checkpoint(args.checkpoint, device)
    base = TacSpikeH5Dataset(data_root=args.data_root, split=args.split)
    selected = select_sequence_indices(len(base.sequences), args.max_sequences, args.seed)

    scores_parts = []
    labels_parts = []
    seq_offsets = [0]
    sequence_records: list[Dict[str, Any]] = []
    for seq_idx in selected:
        info = base.sequences[int(seq_idx)]
        with h5py.File(info.path, "r") as h5:
            scores, labels = stream_sequence_scores(model, h5, train_args, device, args.chunk_steps)
        scores_parts.append(scores)
        labels_parts.append(labels)
        seq_offsets.append(seq_offsets[-1] + int(labels.shape[0]))
        sequence_records.append(
            {
                "sequence_id": info.sequence_id,
                "sequence_index": int(seq_idx),
                "windows": int(labels.shape[0]),
            }
        )
    base.close()

    scores_all = np.concatenate(scores_parts) if scores_parts else np.empty((0,), dtype=np.float32)
    labels_all = np.concatenate(labels_parts) if labels_parts else np.empty((0,), dtype=np.int64)
    seq_offsets_np = np.asarray(seq_offsets, dtype=np.int64)
    distances = transition_distances_per_sequence(labels_all, seq_offsets_np)

    methods: list[Dict[str, Any]] = []
    score_series = {"raw": scores_all.astype(np.float64)}
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

    def method_accuracy(record: Dict[str, Any]) -> float:
        if record["kind"] == "score":
            return float(record["best_threshold_metrics"]["accuracy"])
        return float(record["metrics"]["accuracy"])

    methods.sort(key=method_accuracy, reverse=True)
    best_method = methods[0]
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "selected_sequences": int(selected.shape[0]),
        "total_windows": int(labels_all.shape[0]),
        "label_positive_fraction": float(labels_all.mean()) if labels_all.size else 0.0,
        "sequence_records": sequence_records[: args.top_sequences],
        "methods_top": methods[:50],
        "best_method": best_method,
        "train_args": vars(train_args),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_sequences": result["selected_sequences"],
                "total_windows": result["total_windows"],
                "best_method": best_method["method"],
                "best_accuracy": method_accuracy(best_method),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
