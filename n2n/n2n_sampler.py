"""Neighbor2Neighbor sub-sampler & loss helpers.

Re-implements the 2-pixel paired neighbour sampler from
"Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images"
(Huang et al., CVPR 2021). Each 2x2 cell of the input is split along a random
axis into a pair of adjacent pixels, one going into sub-image ``g1`` and the
other into ``g2``. Both halves are therefore (H/2, W/2) and statistically
independent in their noise.

Performance-conscious implementation:

* Returns per-cell *indices* directly (shape ``(N, H/2, W/2)`` int64) instead
  of a flat boolean mask + ``argmax`` round trip - same math, fewer ops.
* The 8x2 ``idx_pairs`` constant is cached per device so every step doesn't
  re-create it.
"""
from __future__ import annotations

import torch

# 8 ordered pairs of (idx1, idx2) inside a 2x2 cell flattened TL,TR,BL,BR.
# These encode the 4 admissible neighbour pairs (H-top, H-bot, V-left, V-right)
# x 2 orderings (which one goes to g1 vs g2).
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

    Equivalent to ``generate_mask_pair`` followed by ``mask.argmax`` over each
    cell, but avoids materialising the boolean masks.
    """
    n, c, h, w = img.shape
    if h % 2 or w % 2:
        raise ValueError(f"Need even spatial dims, got ({h}, {w})")
    h2, w2 = h // 2, w // 2
    rnd = torch.randint(0, 8, (n, h2, w2), device=img.device)
    pairs = _idx_pairs_for(img.device)        # (8, 2) int64
    chosen = pairs[rnd]                       # (N, H/2, W/2, 2)
    return chosen[..., 0].contiguous(), chosen[..., 1].contiguous()


def _to_cells(img: torch.Tensor) -> torch.Tensor:
    """Reshape ``(N, C, H, W)`` into ``(N, C, H/2, W/2, 4)`` with the last
    axis enumerating the 4 pixels of each 2x2 cell in TL, TR, BL, BR order.

    The intermediate ``permute`` makes the data non-contiguous so the final
    ``reshape`` must materialise a fresh tensor; this is the single biggest
    memcpy in the per-step pipeline so we expose it as a helper to share
    across multiple gathers (g1 and g2 below).
    """
    n, c, h, w = img.shape
    return (
        img.reshape(n, c, h // 2, 2, w // 2, 2)
           .permute(0, 1, 2, 4, 3, 5)
           .reshape(n, c, h // 2, w // 2, 4)
    )


def subimage_by_idx(img: torch.Tensor, idx_per_cell: torch.Tensor) -> torch.Tensor:
    """Gather one pixel per 2x2 cell using a per-cell index tensor.

    ``img``: (N, C, H, W); ``idx_per_cell``: (N, H/2, W/2) int64 in [0, 4).
    Returns (N, C, H/2, W/2).
    """
    n, c, h, w = img.shape
    cells = _to_cells(img)
    idx = idx_per_cell.unsqueeze(1).unsqueeze(-1).expand(n, c, h // 2, w // 2, 1)
    return cells.gather(-1, idx).squeeze(-1)


def split_subimages(
    img: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute both g1 and g2 sub-images from a single ``cells`` reshape.

    ``subimage_by_idx`` rebuilds ``cells`` (an O(NCHW) memcpy because of the
    non-contiguous permute) on every call. When you need both halves of the
    same image, this helper amortises that cost across both gathers.
    """
    n, c, h, w = img.shape
    cells = _to_cells(img)
    h2, w2 = h // 2, w // 2
    e1 = idx1.unsqueeze(1).unsqueeze(-1).expand(n, c, h2, w2, 1)
    e2 = idx2.unsqueeze(1).unsqueeze(-1).expand(n, c, h2, w2, 1)
    return cells.gather(-1, e1).squeeze(-1), cells.gather(-1, e2).squeeze(-1)


# ---- backward-compatible thin wrappers -------------------------------------
# Older callers may still use the boolean-mask API. They produce the same
# subimages (bit-exact, modulo the indices being deterministic for the same
# RNG seed in `generate_index_pair`).

def generate_mask_pair(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean-mask form (legacy). Prefer :func:`generate_index_pair`."""
    n, c, h, w = img.shape
    h2, w2 = h // 2, w // 2
    idx1, idx2 = generate_index_pair(img)
    num_cells = n * h2 * w2
    base = torch.arange(num_cells, device=img.device) * 4
    mask1 = torch.zeros(num_cells * 4, dtype=torch.bool, device=img.device)
    mask2 = torch.zeros(num_cells * 4, dtype=torch.bool, device=img.device)
    mask1[base + idx1.reshape(-1)] = True
    mask2[base + idx2.reshape(-1)] = True
    return mask1, mask2


def generate_subimages(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Boolean-mask form (legacy). Prefer :func:`subimage_by_idx`."""
    n, c, h, w = img.shape
    mask_per_cell = mask.reshape(n, h // 2, w // 2, 4)
    idx = mask_per_cell.to(torch.int64).argmax(dim=-1)
    return subimage_by_idx(img, idx)
