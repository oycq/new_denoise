"""Neighbor2Neighbor pair sub-sampler.

Re-implements the 2-pixel paired neighbour sampler from
"Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images"
(Huang et al., CVPR 2021). Each 2x2 cell of the input is split into a pair
of adjacent pixels, one going into sub-image ``g1`` and the other into
``g2`` - both halves are (H/2, W/2) and statistically independent in their
noise.

The implementation returns per-cell *indices* in [0, 4) directly (no
intermediate boolean mask + ``argmax`` round-trip), which is the only
performance trick worth caring about here. ``idx_pairs`` is cached per
device so we don''t recreate the same constant tensor every step.
"""
from __future__ import annotations

import torch

# Eight ordered (idx1, idx2) pairs inside a 2x2 cell flattened TL,TR,BL,BR.
# Each pair is two adjacent pixels (H-top, H-bot, V-left, V-right) in both
# orderings (so g1 and g2 are equally likely to get either neighbour).
_IDX_PAIRS_CPU = torch.tensor(
    [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [1, 0], [2, 0], [3, 1], [3, 2],
    ],
    dtype=torch.int64,
)
_IDX_PAIRS_CACHE: dict[torch.device, torch.Tensor] = {}


def _idx_pairs_for(device: torch.device) -> torch.Tensor:
    cached = _IDX_PAIRS_CACHE.get(device)
    if cached is None:
        cached = _IDX_PAIRS_CPU.to(device)
        _IDX_PAIRS_CACHE[device] = cached
    return cached


def generate_index_pair(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-cell indices ``(idx1, idx2)`` for the two sub-images.

    Each is shape ``(N, H/2, W/2)`` int64 with values in ``[0, 4)`` referring
    to the per-cell (TL, TR, BL, BR) order used by :func:`subimage_by_idx`.
    """
    n, c, h, w = img.shape
    if h % 2 or w % 2:
        raise ValueError(f"Need even spatial dims, got ({h}, {w})")
    h2, w2 = h // 2, w // 2
    rnd = torch.randint(0, 8, (n, h2, w2), device=img.device)
    pairs = _idx_pairs_for(img.device)            # (8, 2) int64
    chosen = pairs[rnd]                           # (N, H/2, W/2, 2)
    return chosen[..., 0].contiguous(), chosen[..., 1].contiguous()


def subimage_by_idx(img: torch.Tensor, idx_per_cell: torch.Tensor) -> torch.Tensor:
    """Gather one pixel per 2x2 cell using a per-cell index tensor.

    ``img``: (N, C, H, W); ``idx_per_cell``: (N, H/2, W/2) int64 in [0, 4).
    Returns (N, C, H/2, W/2).
    """
    n, c, h, w = img.shape
    # (N, C, H/2, 2, W/2, 2) -> (N, C, H/2, W/2, 4) with 4 = TL,TR,BL,BR.
    cells = (
        img.reshape(n, c, h // 2, 2, w // 2, 2)
           .permute(0, 1, 2, 4, 3, 5)
           .reshape(n, c, h // 2, w // 2, 4)
    )
    idx = idx_per_cell.unsqueeze(1).unsqueeze(-1).expand(n, c, h // 2, w // 2, 1)
    return cells.gather(-1, idx).squeeze(-1)
