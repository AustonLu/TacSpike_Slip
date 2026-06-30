from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.train.evaluate_transition_buckets import fill_legacy_args
from scripts.train.train_lite_scnn import build_model
from tacspike.data import IndexedTacSpikeDataset, TacSpikeH5Dataset


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache ordered per-window scores for complete TacSpike sequences.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = fill_legacy_args(argparse.Namespace(**ckpt["args"]))
    train_args.data_root = args.data_root
    train_args.batch_size = args.batch_size
    train_args.num_workers = args.num_workers

    base = TacSpikeH5Dataset(data_root=args.data_root, split=args.split)
    indices = np.arange(len(base), dtype=np.int64)
    sequence_ids = np.asarray([info.sequence_id for info in base.sequences])
    sequence_paths = np.asarray([str(info.path) for info in base.sequences])
    sequence_windows = np.asarray([info.num_windows for info in base.sequences], dtype=np.int64)
    seq_offsets = np.asarray(base.offsets, dtype=np.int64)
    base.close()

    dataset = IndexedTacSpikeDataset(
        data_root=args.data_root,
        split=args.split,
        indices=indices,
        polarity_mode=train_args.polarity_mode,
        clip_max=train_args.clip_max,
        spatial_pool=train_args.spatial_pool,
        context_ms=getattr(train_args, "context_ms", None),
        time_bins=getattr(train_args, "time_bins", None),
    )
    dataset.input_scale = getattr(train_args, "input_scale", 1.0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scores = np.empty((len(indices),), dtype=np.float32)
    labels = np.empty((len(indices),), dtype=np.int64)
    cursor = 0
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            input_scale = float(getattr(loader.dataset, "input_scale", 1.0))
            if input_scale != 1.0:
                x = x * input_scale
            logits, _ = model(x)
            batch_scores = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy().astype(np.float32, copy=False)
            batch_labels = y.numpy().astype(np.int64, copy=False)
            stop = cursor + batch_scores.shape[0]
            scores[cursor:stop] = batch_scores
            labels[cursor:stop] = batch_labels
            cursor = stop

    if cursor != len(indices):
        raise RuntimeError(f"Cached {cursor} scores, expected {len(indices)}")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        scores=scores,
        labels=labels,
        seq_offsets=seq_offsets,
        sequence_ids=sequence_ids,
        sequence_paths=sequence_paths,
        sequence_windows=sequence_windows,
        global_indices=indices,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "output_npz": str(args.output_npz),
        "split": args.split,
        "windows": int(labels.shape[0]),
        "sequences": int(sequence_ids.shape[0]),
        "positive_fraction": float(labels.mean()) if labels.size else 0.0,
        "seconds": time.time() - start,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "model_args": vars(train_args),
    }
    args.output_npz.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in ("output_npz", "windows", "sequences", "seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
