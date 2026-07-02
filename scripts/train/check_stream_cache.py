from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

REPO_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tacspike.data import (
    TacSpikeH5Dataset,
    load_stream_cache_manifest,
    read_event_bins_slice,
    voxelize_events_pooled_edges,
)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def compare_one(
    source_path: Path,
    cache_path: Path,
    start: int,
    length: int,
    spatial_pool: int,
    polarity_mode: str,
    clip_max: float | None,
) -> Dict[str, Any]:
    with h5py.File(source_path, "r") as src, h5py.File(cache_path, "r") as cache:
        stop = min(start + length, int(cache["labels"].shape[0]))
        start = max(0, stop - length)
        labels_cache = cache["labels"][start:stop].astype(np.int64, copy=False)
        labels_src = src["label/slip"][start:stop].astype(np.int64, copy=False)
        all_t_labels = src["windows/t_label"][:].astype(np.float64, copy=False)
        bin_edges = np.concatenate([np.asarray([all_t_labels[0] - 0.001], dtype=np.float64), all_t_labels])
        t_labels = all_t_labels[start:stop]
        t_start = float(bin_edges[start])
        t_end = float(bin_edges[stop])
        t_dataset = src["events/t"]
        left = int(np.searchsorted(t_dataset, t_start, side="left"))
        right = int(np.searchsorted(t_dataset, t_end, side="right"))
        events = {key: src[f"events/{key}"][left:right] for key in ("t", "x", "y", "p")}
        rebuilt = voxelize_events_pooled_edges(
            events=events,
            bin_edges=bin_edges,
            height=int(src.attrs["height"]),
            width=int(src.attrs["width"]),
            pool=int(spatial_pool),
            polarity_mode=polarity_mode,
            clip_max=clip_max,
            start_bin=start,
            stop_bin=stop,
        ).astype(np.float32, copy=False)
        cached = read_event_bins_slice(cache, start, stop)
        diff = np.abs(rebuilt - cached)
        t_cache = cache["t_label"][start:stop] if "t_label" in cache else np.empty((0,), dtype=np.float64)
        t_diff = float(np.max(np.abs(t_cache - t_labels))) if t_cache.shape[0] == t_labels.shape[0] else None
    return {
        "source_path": str(source_path),
        "cache_path": str(cache_path),
        "start": int(start),
        "stop": int(stop),
        "label_match": bool(np.array_equal(labels_src, labels_cache)),
        "t_label_max_abs_diff": t_diff,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "sum_abs_diff": float(diff.sum()) if diff.size else 0.0,
        "rebuilt_sum": float(rebuilt.sum()),
        "cached_sum": float(cached.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check TacSpike stream cache against original HDF5.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--stream-cache-root", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--spatial-pool", type=int, default=4)
    parser.add_argument("--polarity-mode", choices=("both", "positive", "negative", "sum"), default="both")
    parser.add_argument("--clip-max", type=float, default=None)
    parser.add_argument("--cache-dtype", default="float16")
    parser.add_argument("--cache-format", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    base = TacSpikeH5Dataset(data_root=args.data_root, split=args.split)
    source_by_id = {info.sequence_id: info.path for info in base.sequences}
    base.close()
    caches = load_stream_cache_manifest(
        cache_root=args.stream_cache_root,
        split=args.split,
        spatial_pool=args.spatial_pool,
        polarity_mode=args.polarity_mode,
        clip_max=args.clip_max,
        dtype=args.cache_dtype,
        cache_format=args.cache_format,
    )
    rng = np.random.default_rng(args.seed)
    records: List[Dict[str, Any]] = []
    for _ in range(int(args.samples)):
        info = caches[int(rng.integers(0, len(caches)))]
        max_start = max(0, info.length - int(args.length))
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        records.append(
            compare_one(
                source_path=source_by_id[info.sequence_id],
                cache_path=info.path,
                start=start,
                length=args.length,
                spatial_pool=args.spatial_pool,
                polarity_mode=args.polarity_mode,
                clip_max=args.clip_max,
            )
        )
    result = {
        "split": args.split,
        "samples": int(args.samples),
        "length": int(args.length),
        "all_label_match": bool(all(item["label_match"] for item in records)),
        "max_abs_diff": float(max((item["max_abs_diff"] for item in records), default=0.0)),
        "sum_abs_diff": float(sum(item["sum_abs_diff"] for item in records)),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
