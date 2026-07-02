from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    h5py = None  # type: ignore[assignment]

from .h5_dataset import TacSpikeH5Dataset, load_manifest
from .torch_dataset import transition_distances


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required for TacSpike stream cache loading. Install with `pip install h5py`.")


def stream_cache_config_name(
    spatial_pool: int,
    polarity_mode: str,
    clip_max: Optional[float],
    dtype: str,
    cache_format: str = "dense",
) -> str:
    clip_name = "none" if clip_max is None else str(float(clip_max)).replace(".", "p")
    return f"{cache_format}_sp{int(spatial_pool)}_{polarity_mode}_clip{clip_name}_{dtype}"


def split_cache_dir(
    cache_root: Path,
    split: str,
    spatial_pool: int,
    polarity_mode: str,
    clip_max: Optional[float],
    dtype: str,
    cache_format: str = "dense",
) -> Path:
    return Path(cache_root) / stream_cache_config_name(spatial_pool, polarity_mode, clip_max, dtype, cache_format) / split


def sequence_cache_path(
    cache_root: Path,
    split: str,
    sequence_id: str,
    spatial_pool: int,
    polarity_mode: str,
    clip_max: Optional[float],
    dtype: str,
    cache_format: str = "dense",
) -> Path:
    return split_cache_dir(cache_root, split, spatial_pool, polarity_mode, clip_max, dtype, cache_format) / f"{sequence_id}.h5"


@dataclass(frozen=True)
class StreamCacheSequence:
    path: Path
    split: str
    sequence_id: str
    length: int
    positive_fraction: float
    event_sum: float


def _dataset_dtype(dtype: str) -> np.dtype:
    if dtype == "float32":
        return np.dtype(np.float32)
    if dtype == "float16":
        return np.dtype(np.float16)
    if dtype == "uint16":
        return np.dtype(np.uint16)
    if dtype == "uint8":
        return np.dtype(np.uint8)
    raise ValueError(f"Unsupported stream cache dtype={dtype!r}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def voxelize_events_pooled_edges(
    events: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    height: int,
    width: int,
    pool: int,
    polarity_mode: str = "both",
    clip_max: Optional[float] = None,
    start_bin: int = 0,
    stop_bin: Optional[int] = None,
) -> np.ndarray:
    """Voxelize events into pooled 1 ms bins using explicit global bin edges.

    Events exactly on an internal edge are assigned to the bin starting at that
    edge. The final right edge is included in the last bin.
    """

    if polarity_mode not in {"both", "positive", "negative", "sum"}:
        raise ValueError(f"Unsupported polarity_mode={polarity_mode!r}")
    if pool <= 0:
        raise ValueError(f"pool must be positive, got {pool}")
    if height % pool != 0 or width % pool != 0:
        raise ValueError(f"Spatial shape {(height, width)} is not divisible by pool={pool}")

    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.shape[0] < 2:
        raise ValueError("bin_edges must be a 1D array with at least two elements")
    total_bins = edges.shape[0] - 1
    start_bin = int(start_bin)
    stop_bin = total_bins if stop_bin is None else int(stop_bin)
    if start_bin < 0 or stop_bin > total_bins or stop_bin <= start_bin:
        raise ValueError(f"Invalid bin range [{start_bin}, {stop_bin}) for total_bins={total_bins}")

    bins = stop_bin - start_bin
    channels = 2 if polarity_mode == "both" else 1
    pooled_h = height // pool
    pooled_w = width // pool
    voxel = np.zeros((bins, channels, pooled_h, pooled_w), dtype=np.float32)
    if len(events["t"]) == 0:
        return voxel

    x = events["x"].astype(np.int64, copy=False)
    y = events["y"].astype(np.int64, copy=False)
    p = events["p"].astype(np.int64, copy=False)
    t = events["t"].astype(np.float64, copy=False)

    valid = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
        & ((p == 0) | (p == 1))
        & (t >= edges[0])
        & (t <= edges[-1])
    )
    if polarity_mode == "positive":
        valid &= p == 1
    elif polarity_mode == "negative":
        valid &= p == 0
    if not bool(valid.any()):
        return voxel

    global_bin = np.searchsorted(edges, t[valid], side="right") - 1
    global_bin = np.clip(global_bin, 0, total_bins - 1)
    in_range = (global_bin >= start_bin) & (global_bin < stop_bin)
    if not bool(in_range.any()):
        return voxel

    valid_indices = np.flatnonzero(valid)[in_range]
    local_bin = global_bin[in_range] - start_bin
    pooled_x = x[valid_indices] // pool
    pooled_y = y[valid_indices] // pool
    if polarity_mode == "both":
        channel_idx = p[valid_indices]
    else:
        channel_idx = np.zeros_like(local_bin)

    np.add.at(voxel, (local_bin, channel_idx, pooled_y, pooled_x), 1.0)
    if clip_max is not None:
        np.minimum(voxel, float(clip_max), out=voxel)
    return voxel


def build_stream_cache_for_split(
    data_root: Path,
    cache_root: Path,
    split: str,
    spatial_pool: int = 4,
    polarity_mode: str = "both",
    clip_max: Optional[float] = None,
    dtype: str = "float16",
    cache_format: str = "dense",
    compression: str = "gzip",
    chunk_steps: int = 512,
    force: bool = False,
    max_sequences: int = 0,
) -> Dict[str, Any]:
    """Build per-sequence 1 ms pooled event cache for one split."""

    _require_h5py()
    data_root = Path(data_root)
    cache_root = Path(cache_root)
    if cache_format not in {"dense", "sparse"}:
        raise ValueError(f"Unsupported cache_format={cache_format!r}")
    out_dir = split_cache_dir(cache_root, split, spatial_pool, polarity_mode, clip_max, dtype, cache_format)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_manifest(data_root, split)
    if max_sequences > 0:
        sequences = sequences[: int(max_sequences)]

    built: List[Dict[str, Any]] = []
    reused: List[Dict[str, Any]] = []
    total_bins = 0
    total_positive = 0
    total_event_sum = 0.0
    target_dtype = _dataset_dtype(dtype)

    for info in sequences:
        out_path = out_dir / f"{info.sequence_id}.h5"
        if out_path.exists() and not force:
            with h5py.File(out_path, "r") as cached:
                length = int(cached["labels"].shape[0])
                positive = int(cached["labels"][:].sum())
                if "event_sum" in cached.attrs:
                    event_sum = float(cached.attrs["event_sum"])
                elif "event_bins" in cached:
                    event_sum = float(cached["event_bins"][:].sum())
                else:
                    event_sum = float(cached["sparse/v"][:].sum())
            total_bins += length
            total_positive += positive
            total_event_sum += event_sum
            reused.append({"sequence_id": info.sequence_id, "path": str(out_path), "length": length})
            continue

        with h5py.File(info.path, "r") as h5:
            labels = h5["label/slip"][:].astype(np.int8, copy=False)
            if labels.size == 0:
                continue
            t_labels = h5["windows/t_label"][:].astype(np.float64, copy=False)
            bin_edges = np.concatenate([np.asarray([t_labels[0] - 0.001], dtype=np.float64), t_labels])
            t_start = float(bin_edges[0])
            t_end = float(bin_edges[-1])
            t_dataset = h5["events/t"]
            left = int(np.searchsorted(t_dataset, t_start, side="left"))
            right = int(np.searchsorted(t_dataset, t_end, side="right"))
            events = {key: h5[f"events/{key}"][left:right] for key in ("t", "x", "y", "p")}
            event_bins = voxelize_events_pooled_edges(
                events=events,
                bin_edges=bin_edges,
                height=int(h5.attrs["height"]),
                width=int(h5.attrs["width"]),
                pool=int(spatial_pool),
                polarity_mode=polarity_mode,
                clip_max=clip_max,
            )
            event_bins = event_bins.astype(target_dtype, copy=False)
            nz = np.nonzero(event_bins)
            sparse_values = event_bins[nz].astype(np.float32, copy=False)
            sparse_t = nz[0].astype(np.int32, copy=False)
            sparse_c = nz[1].astype(np.int8, copy=False)
            sparse_y = nz[2].astype(np.int16, copy=False)
            sparse_x = nz[3].astype(np.int16, copy=False)
            distances = transition_distances(labels).astype(np.float32, copy=False)
            valid_mask = np.ones(labels.shape[0], dtype=np.bool_)
            attrs = {
                "sequence_id": info.sequence_id,
                "split": split,
                "source_path": str(info.path),
                "length": int(labels.shape[0]),
                "height": int(h5.attrs["height"]),
                "width": int(h5.attrs["width"]),
                "pooled_height": int(event_bins.shape[2]),
                "pooled_width": int(event_bins.shape[3]),
                "channels": int(event_bins.shape[1]),
                "spatial_pool": int(spatial_pool),
                "polarity_mode": polarity_mode,
                "clip_max": "none" if clip_max is None else float(clip_max),
                "dtype": dtype,
                "cache_format": cache_format,
                "t_start": t_start,
                "t_end": t_end,
                "event_sum": float(event_bins.sum()),
                "nonzero_events": int(sparse_values.shape[0]),
                "positive_fraction": float(labels.mean()) if labels.size else 0.0,
                "cache_version": 1,
                "chunk_steps": int(chunk_steps),
            }

        tmp_path = out_path.with_suffix(".tmp.h5")
        if tmp_path.exists():
            tmp_path.unlink()
        with h5py.File(tmp_path, "w") as out:
            vector_chunks = (min(int(chunk_steps), int(labels.shape[0])),)
            if cache_format == "dense":
                event_chunks = (
                    min(int(chunk_steps), int(event_bins.shape[0])),
                    int(event_bins.shape[1]),
                    int(event_bins.shape[2]),
                    int(event_bins.shape[3]),
                )
                out.create_dataset("event_bins", data=event_bins, compression=compression, chunks=event_chunks)
            else:
                sparse_group = out.create_group("sparse")
                sparse_chunks = (min(max(int(chunk_steps) * 4, 1024), max(int(sparse_values.shape[0]), 1)),)
                sparse_group.create_dataset("t", data=sparse_t, compression=compression, chunks=sparse_chunks)
                sparse_group.create_dataset("c", data=sparse_c, compression=compression, chunks=sparse_chunks)
                sparse_group.create_dataset("y", data=sparse_y, compression=compression, chunks=sparse_chunks)
                sparse_group.create_dataset("x", data=sparse_x, compression=compression, chunks=sparse_chunks)
                sparse_group.create_dataset("v", data=sparse_values, compression=compression, chunks=sparse_chunks)
            out.create_dataset("labels", data=labels, compression=compression, chunks=vector_chunks)
            out.create_dataset(
                "t_label",
                data=t_labels.astype(np.float64, copy=False),
                compression=compression,
                chunks=vector_chunks,
            )
            out.create_dataset("valid_mask", data=valid_mask, compression=compression, chunks=vector_chunks)
            out.create_dataset("transition_distance", data=distances, compression=compression, chunks=vector_chunks)
            for key, value in attrs.items():
                out.attrs[key] = value
        tmp_path.replace(out_path)

        length = int(labels.shape[0])
        positive = int(labels.sum())
        event_sum = float(event_bins.sum())
        total_bins += length
        total_positive += positive
        total_event_sum += event_sum
        built.append(
            {
                "sequence_id": info.sequence_id,
                "path": str(out_path),
                "length": length,
                "positive_fraction": float(labels.mean()) if length else 0.0,
                "event_sum": event_sum,
            }
        )

    summary = {
        "data_root": str(data_root),
        "cache_root": str(cache_root),
        "cache_dir": str(out_dir),
        "split": split,
        "spatial_pool": int(spatial_pool),
        "polarity_mode": polarity_mode,
        "clip_max": None if clip_max is None else float(clip_max),
        "dtype": dtype,
        "cache_format": cache_format,
        "compression": compression,
        "chunk_steps": int(chunk_steps),
        "num_sequences": len(sequences),
        "built_sequences": len(built),
        "reused_sequences": len(reused),
        "total_bins": int(total_bins),
        "positive_fraction": float(total_positive / max(total_bins, 1)),
        "event_sum": float(total_event_sum),
        "built": built,
        "reused": reused[:20],
    }
    summary_path = out_dir / "stream_cache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return summary


def load_stream_cache_manifest(
    cache_root: Path,
    split: str,
    spatial_pool: int = 4,
    polarity_mode: str = "both",
    clip_max: Optional[float] = None,
    dtype: str = "float16",
    cache_format: str = "dense",
) -> List[StreamCacheSequence]:
    _require_h5py()
    cache_dir = split_cache_dir(cache_root, split, spatial_pool, polarity_mode, clip_max, dtype, cache_format)
    paths = sorted(cache_dir.glob("*.h5"))
    sequences: List[StreamCacheSequence] = []
    for path in paths:
        if path.name.endswith(".tmp.h5"):
            continue
        with h5py.File(path, "r") as h5:
            sequence_id = str(h5.attrs.get("sequence_id", path.stem))
            length = int(h5["labels"].shape[0])
            positive_fraction = (
                float(h5.attrs["positive_fraction"])
                if "positive_fraction" in h5.attrs
                else float(h5["labels"][:].mean())
            )
            if "event_sum" in h5.attrs:
                event_sum = float(h5.attrs["event_sum"])
            elif "event_bins" in h5:
                event_sum = float(h5["event_bins"][:].sum())
            else:
                event_sum = float(h5["sparse/v"][:].sum())
        sequences.append(
            StreamCacheSequence(
                path=path,
                split=split,
                sequence_id=sequence_id,
                length=length,
                positive_fraction=positive_fraction,
                event_sum=event_sum,
            )
        )
    if not sequences:
        raise FileNotFoundError(f"No stream cache files found in {cache_dir}")
    return sequences


def build_stream_segment_index(
    cache_root: Path,
    split: str,
    index_dir: Path,
    segment_steps: int,
    spatial_pool: int = 4,
    polarity_mode: str = "both",
    clip_max: Optional[float] = None,
    dtype: str = "float16",
    cache_format: str = "dense",
    force: bool = False,
) -> Path:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    config = stream_cache_config_name(spatial_pool, polarity_mode, clip_max, dtype, cache_format)
    index_path = index_dir / f"{split}_{config}_segments_t{int(segment_steps)}.npz"
    if index_path.exists() and not force:
        return index_path

    sequences = load_stream_cache_manifest(cache_root, split, spatial_pool, polarity_mode, clip_max, dtype, cache_format)
    seq_parts = []
    start_parts = []
    end_label_parts = []
    pos_fraction_parts = []
    transition_parts = []
    segment_steps = int(segment_steps)
    for seq_idx, info in enumerate(sequences):
        with h5py.File(info.path, "r") as h5:
            labels = h5["labels"][:].astype(np.int8, copy=False)
        max_start = int(labels.shape[0] - segment_steps + 1)
        if max_start <= 0:
            continue
        starts = np.arange(max_start, dtype=np.int64)
        label_i32 = labels.astype(np.int32, copy=False)
        cumsum = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(label_i32, dtype=np.int64)])
        pos_count = cumsum[segment_steps:] - cumsum[:-segment_steps]
        transition = (labels[1:] != labels[:-1]).astype(np.int32, copy=False)
        transition_cumsum = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(transition, dtype=np.int64)]
        )
        transition_count = transition_cumsum[starts + segment_steps - 1] - transition_cumsum[starts]
        seq_parts.append(np.full(max_start, seq_idx, dtype=np.int64))
        start_parts.append(starts)
        end_label_parts.append(labels[segment_steps - 1 :].astype(np.int8, copy=False))
        pos_fraction_parts.append((pos_count / float(segment_steps)).astype(np.float32, copy=False))
        transition_parts.append(transition_count > 0)

    np.savez_compressed(
        index_path,
        seq=np.concatenate(seq_parts) if seq_parts else np.empty((0,), dtype=np.int64),
        start=np.concatenate(start_parts) if start_parts else np.empty((0,), dtype=np.int64),
        end_label=np.concatenate(end_label_parts) if end_label_parts else np.empty((0,), dtype=np.int8),
        pos_fraction=np.concatenate(pos_fraction_parts) if pos_fraction_parts else np.empty((0,), dtype=np.float32),
        has_transition=np.concatenate(transition_parts) if transition_parts else np.empty((0,), dtype=bool),
        sequence_paths=np.asarray([str(info.path) for info in sequences]),
        sequence_ids=np.asarray([info.sequence_id for info in sequences]),
        segment_steps=np.asarray([int(segment_steps)], dtype=np.int64),
    )
    return index_path


def sample_stream_segments(
    cache_root: Path,
    split: str,
    index_dir: Path,
    segment_steps: int,
    count: int,
    seed: int,
    sampling: str,
    spatial_pool: int = 4,
    polarity_mode: str = "both",
    clip_max: Optional[float] = None,
    dtype: str = "float16",
    cache_format: str = "dense",
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    index_path = build_stream_segment_index(
        cache_root=cache_root,
        split=split,
        index_dir=index_dir,
        segment_steps=segment_steps,
        spatial_pool=spatial_pool,
        polarity_mode=polarity_mode,
        clip_max=clip_max,
        dtype=dtype,
        cache_format=cache_format,
    )
    with np.load(index_path) as cache:
        seq = cache["seq"]
        start = cache["start"]
        end_label = cache["end_label"]
        pos_fraction = cache["pos_fraction"]
        has_transition = cache["has_transition"]

    if seq.size == 0:
        raise RuntimeError(f"No stream segments found for split={split!r}, segment_steps={segment_steps}")

    if sampling == "balanced":
        sampling = "end_balanced"

    if sampling == "random":
        eligible = np.arange(seq.shape[0])
        pick = rng.choice(eligible, size=count, replace=count > eligible.shape[0])
    elif sampling == "end_balanced":
        pos = np.flatnonzero(end_label == 1)
        neg = np.flatnonzero(end_label == 0)
        n_pos = count // 2
        n_neg = count - n_pos
        pick = np.concatenate(
            [
                rng.choice(pos, size=n_pos, replace=n_pos > pos.shape[0]),
                rng.choice(neg, size=n_neg, replace=n_neg > neg.shape[0]),
            ]
        )
    elif sampling == "state_balanced":
        slip = np.flatnonzero(pos_fraction >= 0.5)
        no_slip = np.flatnonzero(pos_fraction < 0.5)
        n_slip = count // 2
        n_no = count - n_slip
        pick = np.concatenate(
            [
                rng.choice(slip, size=n_slip, replace=n_slip > slip.shape[0]),
                rng.choice(no_slip, size=n_no, replace=n_no > no_slip.shape[0]),
            ]
        )
    elif sampling == "transition_mix":
        trans = np.flatnonzero(has_transition)
        stable_pos = np.flatnonzero((~has_transition) & (end_label == 1))
        stable_neg = np.flatnonzero((~has_transition) & (end_label == 0))
        n_trans = count // 3
        n_pos = (count - n_trans) // 2
        n_neg = count - n_trans - n_pos
        parts = []
        if trans.size:
            parts.append(rng.choice(trans, size=n_trans, replace=n_trans > trans.shape[0]))
        if stable_pos.size:
            parts.append(rng.choice(stable_pos, size=n_pos, replace=n_pos > stable_pos.shape[0]))
        if stable_neg.size:
            parts.append(rng.choice(stable_neg, size=n_neg, replace=n_neg > stable_neg.shape[0]))
        if not parts:
            raise RuntimeError("transition_mix has no eligible stream segments")
        pick = np.concatenate(parts)
        if pick.shape[0] < count:
            extra = rng.choice(np.arange(seq.shape[0]), size=count - pick.shape[0], replace=True)
            pick = np.concatenate([pick, extra])
    else:
        raise ValueError(f"Unsupported sampling={sampling!r}")

    rng.shuffle(pick)
    return seq[pick].astype(np.int64, copy=False), start[pick].astype(np.int64, copy=False)


def onset_valid_mask(labels: np.ndarray, ignore_steps: int) -> np.ndarray:
    labels = np.asarray(labels)
    if ignore_steps <= 0 or labels.shape[0] <= 1:
        return np.ones(labels.shape[0], dtype=np.bool_)
    invalid = transition_distances(labels) <= float(ignore_steps)
    return ~invalid


def read_event_bins_slice(h5: Any, start: int, stop: int) -> np.ndarray:
    """Read [start:stop] event bins from dense or sparse stream cache."""

    start = int(start)
    stop = int(stop)
    if "event_bins" in h5:
        return h5["event_bins"][start:stop].astype(np.float32, copy=False)

    sparse = h5["sparse"]
    t_all = sparse["t"]
    left = int(np.searchsorted(t_all, start, side="left"))
    right = int(np.searchsorted(t_all, stop, side="left"))
    channels = int(h5.attrs["channels"])
    pooled_h = int(h5.attrs["pooled_height"])
    pooled_w = int(h5.attrs["pooled_width"])
    out = np.zeros((stop - start, channels, pooled_h, pooled_w), dtype=np.float32)
    if right <= left:
        return out
    t = t_all[left:right].astype(np.int64, copy=False) - start
    c = sparse["c"][left:right].astype(np.int64, copy=False)
    y = sparse["y"][left:right].astype(np.int64, copy=False)
    x = sparse["x"][left:right].astype(np.int64, copy=False)
    v = sparse["v"][left:right].astype(np.float32, copy=False)
    np.add.at(out, (t, c, y, x), v)
    return out


def iter_batches(items: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
