from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tacspike.data.h5_dataset import TacSpikeH5Dataset, sample_summary


def _find_target_label_index(
    dataset: TacSpikeH5Dataset,
    target_label: int,
    seed: int,
    max_attempts: int,
) -> int:
    rng = random.Random(seed)
    for _ in range(max_attempts):
        idx = rng.randrange(len(dataset))
        sample = dataset.get_sample(idx, return_events=False)
        if int(sample["label"]) == target_label:
            return idx
    raise RuntimeError(f"Could not find target label {target_label} in {max_attempts} random attempts")


def _plot_sample(sample: Dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    voxel = sample["voxel"]
    pooled = sample["x"]
    per_tc = voxel.sum(axis=(2, 3))
    event_map = voxel.sum(axis=(0, 1))
    pooled_map = pooled.sum(axis=(0, 1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        f"{sample['sequence_id']} window={sample['window_index']} "
        f"label={sample['label']} events={sample['event_count']}"
    )

    ax = axes[0, 0]
    for c in range(per_tc.shape[1]):
        ax.plot(per_tc[:, c], marker="o", linewidth=1.2, label=f"p={c}")
    ax.set_title("Events per 1 ms bin")
    ax.set_xlabel("time bin")
    ax.set_ylabel("event count")
    ax.legend()

    ax = axes[0, 1]
    im = ax.imshow(event_map, cmap="magma", interpolation="nearest")
    ax.set_title("Full-resolution event map")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    im = ax.imshow(pooled_map, cmap="magma", interpolation="nearest")
    ax.set_title(f"Pooled event map, pool={sample['spatial_pool']}")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    events = sample.get("events")
    if events is not None and len(events["t"]) > 0:
        t0 = float(sample["t_start"])
        xs = events["x"].astype(np.float32)
        ys = events["y"].astype(np.float32)
        ts_ms = (events["t"].astype(np.float64) - t0) * 1000.0
        colors = events["p"].astype(np.int64)
        sc = ax.scatter(ts_ms, ys * sample["width"] + xs, c=colors, s=10, cmap="coolwarm", alpha=0.8)
        fig.colorbar(sc, ax=ax, fraction=0.046, label="polarity")
    ax.set_title("Raw events raster")
    ax.set_xlabel("time since window start (ms)")
    ax.set_ylabel("flattened pixel index")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one TacSpike slip HDF5 window.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--index", type=int, default=None, help="Global index within split.")
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--target-label", type=int, choices=(0, 1), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--polarity-mode", default="both", choices=("both", "positive", "negative", "sum"))
    parser.add_argument("--clip-max", type=float, default=1.0)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sanity"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    clip_max: Optional[float] = None if args.no_clip else args.clip_max
    dataset = TacSpikeH5Dataset(
        data_root=args.data_root,
        split=args.split,
        polarity_mode=args.polarity_mode,
        clip_max=clip_max,
        spatial_pool=args.pool,
    )

    if args.sequence_id is not None:
        global_index = dataset.global_index(args.sequence_id, args.window_index)
    elif args.index is not None:
        global_index = args.index
    elif args.target_label is not None:
        global_index = _find_target_label_index(dataset, args.target_label, args.seed, args.max_attempts)
    elif args.random:
        global_index = random.Random(args.seed).randrange(len(dataset))
    else:
        global_index = 0

    sample = dataset.get_sample(global_index, return_events=True)
    summary = sample_summary(sample)
    summary["dataset_length"] = len(dataset)
    summary["num_sequences"] = len(dataset.sequences)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.no_plot:
        output_path = args.output_dir / (
            f"{args.split}_{sample['sequence_id']}_w{sample['window_index']}_label{sample['label']}.png"
        )
        _plot_sample(sample, output_path)
        print(f"plot_path={output_path}")

    dataset.close()


if __name__ == "__main__":
    main()
