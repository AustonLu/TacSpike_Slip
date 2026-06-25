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
    ) -> None:
        self.base = TacSpikeH5Dataset(
            data_root=data_root,
            split=split,
            polarity_mode=polarity_mode,
            clip_max=clip_max,
            spatial_pool=spatial_pool,
        )
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        global_index = int(self.indices[index])
        x, y = self.base[global_index]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


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
    for seq_idx, info in enumerate(base.sequences):
        offset = base.offsets[seq_idx]
        with h5py.File(info.path, "r") as h5:
            labels = h5["label/slip"][:].astype(np.int8, copy=False)
        local = np.arange(labels.shape[0], dtype=np.int64) + int(offset)
        slip_parts.append(local[labels == 1])
        no_slip_parts.append(local[labels == 0])
    base.close()

    slip_indices = np.concatenate(slip_parts) if slip_parts else np.empty((0,), dtype=np.int64)
    no_slip_indices = np.concatenate(no_slip_parts) if no_slip_parts else np.empty((0,), dtype=np.int64)
    np.savez_compressed(cache_path, slip=slip_indices, no_slip=no_slip_indices)
    return cache_path


def sample_epoch_indices(
    data_root: Path,
    split: str,
    cache_dir: Path,
    num_samples: int,
    seed: int,
    sampling: str = "balanced",
) -> np.ndarray:
    """Sample global window indices for one epoch."""

    rng = np.random.default_rng(seed)
    if sampling == "random":
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

    n_slip = num_samples // 2
    n_no_slip = num_samples - n_slip
    sampled_slip = rng.choice(slip, size=n_slip, replace=n_slip > len(slip))
    sampled_no_slip = rng.choice(no_slip, size=n_no_slip, replace=n_no_slip > len(no_slip))
    indices = np.concatenate([sampled_slip, sampled_no_slip]).astype(np.int64, copy=False)
    rng.shuffle(indices)
    return indices

