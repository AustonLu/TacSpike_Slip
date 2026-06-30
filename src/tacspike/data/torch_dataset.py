from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .h5_dataset import TacSpikeH5Dataset


class IndexedTacSpikeDataset(Dataset):
    """Torch dataset over a fixed list of TacSpike global window indices."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        indices: np.ndarray,
        polarity_mode: str = "both",
        clip_max: Optional[float] = 1.0,
        spatial_pool: int = 4,
        context_ms: Optional[float] = None,
        time_bins: Optional[int] = None,
    ) -> None:
        self.base = TacSpikeH5Dataset(
            data_root=data_root,
            split=split,
            polarity_mode=polarity_mode,
            clip_max=clip_max,
            spatial_pool=spatial_pool,
            context_ms=context_ms,
            time_bins=time_bins,
        )
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        global_index = int(self.indices[index])
        x, y = self.base[global_index]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def transition_distances(labels: np.ndarray) -> np.ndarray:
    """Distance in windows to the nearest binary label transition."""

    labels = np.asarray(labels)
    if labels.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    transitions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if transitions.size == 0:
        return np.full(labels.shape[0], np.inf, dtype=np.float32)
    positions = np.arange(labels.shape[0], dtype=np.int64)
    idx = np.searchsorted(transitions, positions)
    right = np.where(idx < transitions.size, transitions[np.minimum(idx, transitions.size - 1)], np.inf)
    left_idx = np.maximum(idx - 1, 0)
    left = np.where(idx > 0, transitions[left_idx], -np.inf)
    return np.minimum(np.abs(positions - left), np.abs(positions - right)).astype(np.float32)


def build_label_index_cache(
    data_root: Path,
    split: str,
    cache_dir: Path,
    force: bool = False,
) -> Path:
    """Build or reuse global index pools for slip/no-slip windows."""

    import h5py

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_label_indices.npz"
    if cache_path.exists() and not force:
        return cache_path

    base = TacSpikeH5Dataset(data_root=data_root, split=split)
    slip_parts = []
    no_slip_parts = []
    slip_distance_parts = []
    no_slip_distance_parts = []
    for seq_idx, info in enumerate(base.sequences):
        offset = base.offsets[seq_idx]
        with h5py.File(info.path, "r") as h5:
            labels = h5["label/slip"][:].astype(np.int8, copy=False)
        local = np.arange(labels.shape[0], dtype=np.int64) + int(offset)
        distances = transition_distances(labels)
        slip_mask = labels == 1
        no_slip_mask = labels == 0
        slip_parts.append(local[slip_mask])
        no_slip_parts.append(local[no_slip_mask])
        slip_distance_parts.append(distances[slip_mask])
        no_slip_distance_parts.append(distances[no_slip_mask])
    base.close()

    slip_indices = np.concatenate(slip_parts) if slip_parts else np.empty((0,), dtype=np.int64)
    no_slip_indices = np.concatenate(no_slip_parts) if no_slip_parts else np.empty((0,), dtype=np.int64)
    slip_distances = np.concatenate(slip_distance_parts) if slip_distance_parts else np.empty((0,), dtype=np.float32)
    no_slip_distances = (
        np.concatenate(no_slip_distance_parts) if no_slip_distance_parts else np.empty((0,), dtype=np.float32)
    )
    np.savez_compressed(
        cache_path,
        slip=slip_indices,
        no_slip=no_slip_indices,
        slip_transition_distance=slip_distances,
        no_slip_transition_distance=no_slip_distances,
    )
    return cache_path


def sample_epoch_indices(
    data_root: Path,
    split: str,
    cache_dir: Path,
    num_samples: int,
    seed: int,
    sampling: str = "balanced",
    ignore_transition_ms: float = 0.0,
) -> np.ndarray:
    """Sample global window indices for one epoch."""

    rng = np.random.default_rng(seed)
    if sampling == "random":
        if ignore_transition_ms > 0:
            cache_path = build_label_index_cache(data_root=data_root, split=split, cache_dir=cache_dir)
            with np.load(cache_path) as cache:
                min_distance = float(ignore_transition_ms)
                slip = cache["slip"][cache["slip_transition_distance"] >= min_distance]
                no_slip = cache["no_slip"][cache["no_slip_transition_distance"] >= min_distance]
                eligible = np.concatenate([slip, no_slip]).astype(np.int64, copy=False)
                if eligible.shape[0] == 0:
                    raise ValueError(f"ignore_transition_ms={ignore_transition_ms} removed all random samples")
            return rng.choice(eligible, size=num_samples, replace=num_samples > len(eligible)).astype(
                np.int64,
                copy=False,
            )
        base = TacSpikeH5Dataset(data_root=data_root, split=split)
        indices = rng.integers(0, len(base), size=num_samples, dtype=np.int64)
        base.close()
        return indices

    if sampling != "balanced":
        raise ValueError(f"Unsupported sampling mode: {sampling}")

    cache_path = build_label_index_cache(data_root=data_root, split=split, cache_dir=cache_dir)
    with np.load(cache_path) as cache:
        slip = cache["slip"]
        no_slip = cache["no_slip"]
        if ignore_transition_ms > 0 and "slip_transition_distance" in cache and "no_slip_transition_distance" in cache:
            min_distance = float(ignore_transition_ms)
            slip = slip[cache["slip_transition_distance"] >= min_distance]
            no_slip = no_slip[cache["no_slip_transition_distance"] >= min_distance]
            if slip.shape[0] == 0 or no_slip.shape[0] == 0:
                raise ValueError(
                    f"ignore_transition_ms={ignore_transition_ms} removed all samples "
                    f"(slip={slip.shape[0]}, no_slip={no_slip.shape[0]})"
                )

    n_slip = num_samples // 2
    n_no_slip = num_samples - n_slip
    sampled_slip = rng.choice(slip, size=n_slip, replace=n_slip > len(slip))
    sampled_no_slip = rng.choice(no_slip, size=n_no_slip, replace=n_no_slip > len(no_slip))
    indices = np.concatenate([sampled_slip, sampled_no_slip]).astype(np.int64, copy=False)
    rng.shuffle(indices)
    return indices
