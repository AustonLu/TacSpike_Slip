from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tacspike.data.h5_dataset import TacSpikeH5Dataset, sample_summary


def check_split(args: argparse.Namespace, split: str) -> Dict[str, Any]:
    dataset = TacSpikeH5Dataset(
        data_root=args.data_root,
        split=split,
        polarity_mode=args.polarity_mode,
        clip_max=None,
        spatial_pool=args.pool,
    )
    rng = random.Random(args.seed)
    samples: List[Dict[str, Any]] = []
    mismatched_event_count = 0

    for _ in range(args.samples_per_split):
        idx = rng.randrange(len(dataset))
        sample = dataset.get_sample(idx, return_events=False)
        summary = sample_summary(sample)
        if int(sample["event_count"]) != int(sample["h5_event_count"]):
            mismatched_event_count += 1
        samples.append(summary)

    split_summary = {
        "split": split,
        "num_sequences": len(dataset.sequences),
        "num_windows": len(dataset),
        "first_sequence": dataset.sequences[0].sequence_id,
        "last_sequence": dataset.sequences[-1].sequence_id,
        "samples_checked": len(samples),
        "mismatched_event_count": mismatched_event_count,
        "sample_summaries": samples,
    }
    dataset.close()
    return split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight TacSpike HDF5 schema and sample checks.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--samples-per-split", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", type=int, default=4)
    parser.add_argument("--polarity-mode", default="both", choices=("both", "positive", "negative", "sum"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = {
        "data_root": str(args.data_root),
        "splits": [check_split(args, split) for split in args.splits],
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
