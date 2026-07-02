from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tacspike.data import build_stream_cache_for_split, load_stream_cache_manifest


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def parse_splits(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def summarize_manifest(
    cache_root: Path,
    split: str,
    spatial_pool: int,
    polarity_mode: str,
    clip_max: float | None,
    dtype: str,
    cache_format: str,
) -> Dict[str, Any]:
    sequences = load_stream_cache_manifest(
        cache_root=cache_root,
        split=split,
        spatial_pool=spatial_pool,
        polarity_mode=polarity_mode,
        clip_max=clip_max,
        dtype=dtype,
        cache_format=cache_format,
    )
    lengths = np.asarray([item.length for item in sequences], dtype=np.int64)
    pos = np.asarray([item.positive_fraction for item in sequences], dtype=np.float64)
    event_sum = np.asarray([item.event_sum for item in sequences], dtype=np.float64)
    return {
        "split": split,
        "num_sequences": int(len(sequences)),
        "total_bins": int(lengths.sum()),
        "length_min": int(lengths.min()) if lengths.size else 0,
        "length_max": int(lengths.max()) if lengths.size else 0,
        "length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "positive_fraction_mean": float(pos.mean()) if pos.size else 0.0,
        "positive_fraction_weighted": float(np.sum(pos * lengths) / max(int(lengths.sum()), 1)) if lengths.size else 0.0,
        "event_sum": float(event_sum.sum()),
        "event_sum_per_bin": float(event_sum.sum() / max(int(lengths.sum()), 1)),
        "first_sequences": [
            {
                "sequence_id": item.sequence_id,
                "path": str(item.path),
                "length": int(item.length),
                "positive_fraction": float(item.positive_fraction),
                "event_sum": float(item.event_sum),
            }
            for item in sequences[:10]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TacSpike 1 ms sequence-level stream cache.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--spatial-pool", type=int, default=4)
    parser.add_argument("--polarity-mode", choices=("both", "positive", "negative", "sum"), default="both")
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--dtype", choices=("float32", "float16", "uint16", "uint8"), default="float16")
    parser.add_argument("--cache-format", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--chunk-steps", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summaries = []
    for split in parse_splits(args.splits):
        build_summary = build_stream_cache_for_split(
            data_root=args.data_root,
            cache_root=args.cache_root,
            split=split,
            spatial_pool=args.spatial_pool,
            polarity_mode=args.polarity_mode,
            clip_max=args.clip_max,
            dtype=args.dtype,
            cache_format=args.cache_format,
            compression=args.compression,
            chunk_steps=args.chunk_steps,
            force=args.force,
            max_sequences=args.max_sequences,
        )
        manifest_summary = summarize_manifest(
            cache_root=args.cache_root,
            split=split,
            spatial_pool=args.spatial_pool,
            polarity_mode=args.polarity_mode,
            clip_max=args.clip_max,
            dtype=args.dtype,
            cache_format=args.cache_format,
        )
        summaries.append({"build": build_summary, "manifest": manifest_summary})

    result = {
        "data_root": str(args.data_root),
        "cache_root": str(args.cache_root),
        "spatial_pool": int(args.spatial_pool),
        "polarity_mode": args.polarity_mode,
        "clip_max": None if args.clip_max is None else float(args.clip_max),
        "dtype": args.dtype,
        "cache_format": args.cache_format,
        "chunk_steps": int(args.chunk_steps),
        "splits": summaries,
    }
    text = json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
