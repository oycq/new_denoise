"""Dataset for Neighbor2Neighbor Bayer denoising.

We use *only* the noisy raw frames (Neighbor2Neighbor is single-image self
supervised). The ISP frames are kept aside for visualisation/comparison.

Each item is a 4-channel packed RGGB patch in [0, 1] (float32), of shape
``(4, patch_size, patch_size)``. Random flips/transposes are applied for
augmentation, all *on the packed cells* so the Bayer pattern is preserved.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .raw_utils import RAW_MAX, pack_rggb, read_raw


def find_raw_files(data_root: str | Path) -> List[Path]:
    """Recursively gather all `*_raw.png` files under ``data_root``."""
    root = Path(data_root)
    return sorted(p for p in root.rglob("*_raw.png"))


class BayerN2NDataset(Dataset):
    """Random 4-channel packed RGGB patches from a list of raw frames.

    The dataset caches each raw frame in *packed float32* form on first read
    (one entry per file) so subsequent epochs are fast. Memory cost is roughly
    ``num_files * H/2 * W/2 * 4 * 4 bytes``; for our 140 frames at 1280x1088
    that is ~150 MB which is fine.
    """

    def __init__(
        self,
        files: Sequence[Path],
        patch_size: int = 128,
        samples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        if not files:
            raise ValueError("Empty file list passed to BayerN2NDataset")
        self.files = list(files)
        self.patch_size = patch_size  # in 4-channel (packed) pixels
        self.samples_per_epoch = samples_per_epoch
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_packed(self, idx: int) -> np.ndarray:
        if idx not in self._cache:
            raw = read_raw(self.files[idx]).astype(np.float32) / RAW_MAX
            packed = pack_rggb(raw)
            self._cache[idx] = packed.astype(np.float32, copy=False)
        return self._cache[idx]

    def __getitem__(self, _idx: int) -> torch.Tensor:
        f_idx = random.randrange(len(self.files))
        packed = self._load_packed(f_idx)
        h, w, _ = packed.shape
        ps = self.patch_size
        if h < ps or w < ps:
            raise RuntimeError(f"Image too small for patch: {packed.shape}")
        y = random.randrange(0, h - ps + 1)
        x = random.randrange(0, w - ps + 1)
        patch = packed[y : y + ps, x : x + ps]

        if random.random() < 0.5:
            patch = patch[:, ::-1]
        if random.random() < 0.5:
            patch = patch[::-1]
        if random.random() < 0.5:
            patch = patch.transpose(1, 0, 2)

        patch = np.ascontiguousarray(patch).transpose(2, 0, 1)  # (4, H, W)
        return torch.from_numpy(patch)
