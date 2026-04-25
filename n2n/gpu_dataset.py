"""GPU-resident dataset for Bayer N2N training."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .raw_utils import RAW_MAX, pack_rggb, read_raw


class GPUDataset:
    """All raw frames pre-packed and resident in GPU memory."""

    def __init__(
        self,
        files: Sequence[Path],
        device: torch.device,
        *,
        dtype: torch.dtype = torch.float32,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> None:
        if not files:
            raise ValueError("Empty file list")
        packed_list: list[torch.Tensor] = []
        for f in files:
            raw = read_raw(f).astype(np.float32) / RAW_MAX
            pkd = pack_rggb(raw).astype(np.float32, copy=False)
            packed_list.append(torch.from_numpy(pkd))
        # (N, H/2, W/2, 4) -> (N, 4, H/2, W/2)
        stacked = torch.stack(packed_list, dim=0).permute(0, 3, 1, 2).contiguous()
        self.frames = stacked.to(device=device, dtype=dtype, non_blocking=False)
        self.device = device
        self.n_files = self.frames.shape[0]
        self.h = self.frames.shape[2]
        self.w = self.frames.shape[3]
        self.dtype = dtype
        self.memory_format = memory_format

    def __len__(self) -> int:
        return self.n_files

    def memory_mb(self) -> float:
        return self.frames.numel() * self.frames.element_size() / (1024 * 1024)

    def sample_batch(self, batch_size: int, patch_size: int) -> torch.Tensor:
        """Return a freshly sampled (B, 4, ps, ps) batch on the GPU."""
        ps = patch_size
        if self.h < ps or self.w < ps:
            raise RuntimeError(
                f"GPUDataset frames ({self.h}, {self.w}) smaller than patch {ps}")
        crops: list[torch.Tensor] = []
        for _ in range(batch_size):
            i = random.randrange(self.n_files)
            y = random.randrange(0, self.h - ps + 1)
            x = random.randrange(0, self.w - ps + 1)
            patch = self.frames[i, :, y:y + ps, x:x + ps]
            if random.random() < 0.5:
                patch = patch.flip(-1)
            if random.random() < 0.5:
                patch = patch.flip(-2)
            if random.random() < 0.5:
                patch = patch.transpose(-1, -2)
            crops.append(patch)
        # Materialise the batch in the requested memory layout. This fuses
        # the stack + the channels_last conversion that the trainer used to
        # do as a separate `.to(memory_format=...)` call (saves one memcpy
        # of the (B, 4, ps, ps) batch per step).
        out = torch.stack(crops, dim=0)
        if self.memory_format != torch.contiguous_format:
            out = out.contiguous(memory_format=self.memory_format)
        else:
            out = out.contiguous()
        return out
