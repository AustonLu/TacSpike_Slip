from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    h5py = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SequenceInfo:
    path: Path
    split: str
    sequence_id: str
    num_windows: int


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required for TacSpike HDF5 loading. Install with `pip install h5py`.")


def _int_from_csv(value: str) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(float(value))


def _h5_num_windows(path: Path) -> int:
    _require_h5py()
    with h5py.File(path, "r") as h5:
        return int(h5["windows/t_label"].shape[0])


def load_manifest(data_root: Path, split: str) -> List[SequenceInfo]:
    """Load sequence metadata for one split.

    The Hugging Face release includes manifest_sequences.csv. If it is absent,
    this falls back to scanning sequences/{split}/*.h5.
    """

    data_root = Path(data_root)
    manifest_path = data_root / "manifest_sequences.csv"
    sequence_infos: List[SequenceInfo] = []

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = row.get("split", "")
                if row_split != split:
                    continue
                rel_path = row.get("sequence_path", "")
                if not rel_path:
                    continue
                path = data_root / Path(rel_path)
                sequence_id = row.get("sequence_id") or path.stem
                num_windows = _int_from_csv(row.get("num_windows", ""))
                if num_windows is None:
                    num_windows = _h5_num_windows(path)
                sequence_infos.append(
                    SequenceInfo(
                        path=path,
                        split=split,
                        sequence_id=sequence_id,
                        num_windows=num_windows,
                    )
                )
    else:
        split_dir = data_root / "sequences" / split
        for path in sorted(split_dir.glob("*.h5")):
            sequence_infos.append(
                SequenceInfo(
                    path=path,
                    split=split,
                    sequence_id=path.stem,
                    num_windows=_h5_num_windows(path),
                )
            )

    if not sequence_infos:
        raise FileNotFoundError(f"No HDF5 sequences found for split={split!r} under {data_root}")
    return sequence_infos


def select_events(h5: Any, window_index: int) -> Dict[str, np.ndarray]:
    """Select events within one window using the release's inclusive t_end convention."""

    t_dataset = h5["events/t"]
    t_start = float(h5["windows/t_start"][window_index])
    t_end = float(h5["windows/t_end"][window_index])
    left = int(np.searchsorted(t_dataset, t_start, side="left"))
    right = int(np.searchsorted(t_dataset, t_end, side="right"))
    return {key: h5[f"events/{key}"][left:right] for key in ("t", "x", "y", "p")}


def voxelize_events(
    events: Dict[str, np.ndarray],
    t_start: float,
    t_end: float,
    bins: int,
    height: int,
    width: int,
    polarity_mode: str = "both",
    clip_max: Optional[float] = None,
) -> np.ndarray:
    """Voxelize events into [T, C, H, W].

    polarity_mode:
      - "both": two channels, p=0 then p=1.
      - "positive": one channel, p=1 only.
      - "negative": one channel, p=0 only.
      - "sum": one channel, both polarities merged.
    """

    if polarity_mode not in {"both", "positive", "negative", "sum"}:
        raise ValueError(f"Unsupported polarity_mode={polarity_mode!r}")

    channels = 2 if polarity_mode == "both" else 1
    voxel = np.zeros((bins, channels, height, width), dtype=np.float32)
    if t_end <= t_start or len(events["t"]) == 0:
        return voxel

    x = events["x"].astype(np.int64, copy=False)
    y = events["y"].astype(np.int64, copy=False)
    p = events["p"].astype(np.int64, copy=False)
    t = events["t"].astype(np.float64, copy=False)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & ((p == 0) | (p == 1))
    if polarity_mode == "positive":
        valid &= p == 1
    elif polarity_mode == "negative":
        valid &= p == 0

    if not bool(valid.any()):
        return voxel

    t_valid = t[valid]
    duration = t_end - t_start
    bin_idx = np.floor((t_valid - t_start) / duration * bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, bins - 1)

    if polarity_mode == "both":
        channel_idx = p[valid]
    else:
        channel_idx = np.zeros_like(bin_idx)

    np.add.at(voxel, (bin_idx, channel_idx, y[valid], x[valid]), 1.0)
    if clip_max is not None:
        np.minimum(voxel, float(clip_max), out=voxel)
    return voxel


def spatial_sum_pool(voxel: np.ndarray, pool: int) -> np.ndarray:
    """Non-overlapping spatial sum pooling for [T, C, H, W]."""

    if pool <= 1:
        return voxel
    if voxel.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W], got shape={voxel.shape}")
    t, c, h, w = voxel.shape
    if h % pool != 0 or w % pool != 0:
        raise ValueError(f"Spatial shape {(h, w)} is not divisible by pool={pool}")
    return voxel.reshape(t, c, h // pool, pool, w // pool, pool).sum(axis=(3, 5))


class TacSpikeH5Dataset:
    """Index TacSpike sequence HDF5 files without materializing all windows.

    This class intentionally does not depend on torch. Training code can wrap it
    in a torch Dataset later, while stage-1 checks can run in a minimal env.
    """

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        polarity_mode: str = "both",
        clip_max: Optional[float] = 1.0,
        spatial_pool: int = 4,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.polarity_mode = polarity_mode
        self.clip_max = clip_max
        self.spatial_pool = spatial_pool
        self.sequences = load_manifest(self.data_root, split)
        self.offsets = self._build_offsets(self.sequences)
        self._open_path: Optional[Path] = None
        self._h5: Any = None

    @staticmethod
    def _build_offsets(sequences: Sequence[SequenceInfo]) -> List[int]:
        offsets = [0]
        total = 0
        for info in sequences:
            total += int(info.num_windows)
            offsets.append(total)
        return offsets

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_open_path"] = None
        state["_h5"] = None
        return state

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._open_path = None

    def __del__(self) -> None:
        self.close()

    def sequence_for_index(self, global_index: int) -> Tuple[int, int, SequenceInfo]:
        if global_index < 0:
            global_index += len(self)
        if global_index < 0 or global_index >= len(self):
            raise IndexError(f"global_index={global_index} out of range for dataset length {len(self)}")
        seq_idx = bisect.bisect_right(self.offsets, global_index) - 1
        window_index = global_index - self.offsets[seq_idx]
        return seq_idx, window_index, self.sequences[seq_idx]

    def find_sequence(self, sequence_id: str) -> Tuple[int, SequenceInfo]:
        for idx, info in enumerate(self.sequences):
            if info.sequence_id == sequence_id or info.path.stem == sequence_id:
                return idx, info
        raise KeyError(f"sequence_id={sequence_id!r} not found in split={self.split!r}")

    def global_index(self, sequence_id: str, window_index: int) -> int:
        seq_idx, info = self.find_sequence(sequence_id)
        if window_index < 0:
            window_index += info.num_windows
        if window_index < 0 or window_index >= info.num_windows:
            raise IndexError(f"window_index={window_index} out of range for {sequence_id}")
        return self.offsets[seq_idx] + window_index

    def _open_h5(self, path: Path) -> Any:
        _require_h5py()
        if self._h5 is not None and self._open_path == path:
            return self._h5
        self.close()
        self._h5 = h5py.File(path, "r")
        self._open_path = path
        return self._h5

    def get_sample(self, global_index: int, return_events: bool = False) -> Dict[str, Any]:
        _, window_index, info = self.sequence_for_index(global_index)
        h5 = self._open_h5(info.path)

        events = select_events(h5, window_index)
        t_start = float(h5["windows/t_start"][window_index])
        t_end = float(h5["windows/t_end"][window_index])
        t_label = float(h5["windows/t_label"][window_index])
        bins = int(h5.attrs["bins"])
        height = int(h5.attrs["height"])
        width = int(h5.attrs["width"])

        voxel = voxelize_events(
            events,
            t_start=t_start,
            t_end=t_end,
            bins=bins,
            height=height,
            width=width,
            polarity_mode=self.polarity_mode,
            clip_max=self.clip_max,
        )
        pooled = spatial_sum_pool(voxel, self.spatial_pool)
        label = int(h5["label/slip"][window_index])
        h5_event_count = int(h5["windows/event_count"][window_index])

        sample: Dict[str, Any] = {
            "x": pooled,
            "voxel": voxel,
            "label": label,
            "sequence_id": info.sequence_id,
            "sequence_path": str(info.path),
            "split": info.split,
            "global_index": int(global_index),
            "window_index": int(window_index),
            "t_start": t_start,
            "t_end": t_end,
            "t_label": t_label,
            "event_count": int(len(events["t"])),
            "h5_event_count": h5_event_count,
            "bins": bins,
            "height": height,
            "width": width,
            "polarity_mode": self.polarity_mode,
            "clip_max": self.clip_max,
            "spatial_pool": self.spatial_pool,
        }
        if return_events:
            sample["events"] = events
        return sample

    def __getitem__(self, global_index: int) -> Tuple[np.ndarray, int]:
        sample = self.get_sample(global_index, return_events=False)
        return sample["x"], sample["label"]


def sample_summary(sample: Dict[str, Any]) -> Dict[str, Any]:
    voxel = sample["voxel"]
    x = sample["x"]
    per_t = voxel.sum(axis=(1, 2, 3)).astype(float)
    per_c = voxel.sum(axis=(0, 2, 3)).astype(float)
    return {
        "sequence_id": sample["sequence_id"],
        "split": sample["split"],
        "global_index": sample["global_index"],
        "window_index": sample["window_index"],
        "label": sample["label"],
        "t_start": sample["t_start"],
        "t_end": sample["t_end"],
        "t_label": sample["t_label"],
        "event_count": sample["event_count"],
        "h5_event_count": sample["h5_event_count"],
        "voxel_shape": list(voxel.shape),
        "pooled_shape": list(x.shape),
        "voxel_sum": float(voxel.sum()),
        "pooled_sum": float(x.sum()),
        "nonzero_voxels": int(np.count_nonzero(voxel)),
        "nonzero_pooled_voxels": int(np.count_nonzero(x)),
        "per_polarity_sum": per_c.tolist(),
        "per_time_sum_min": float(per_t.min()) if len(per_t) else 0.0,
        "per_time_sum_max": float(per_t.max()) if len(per_t) else 0.0,
        "per_time_sum_mean": float(per_t.mean()) if len(per_t) else 0.0,
    }

