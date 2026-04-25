"""Neighbor2Neighbor sub-sampler & loss helpers.

Re-implements the 2-pixel paired neighbour sampler from
"Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images"
(Huang et al., CVPR 2021). Each 2x2 cell of the input is split along a random
axis into a pair of adjacent pixels, one going into sub-image ``g1`` and the
other into ``g2``. Both halves are therefore (H/2, W/2) and statistically
independent in their noise.
"""
from __future__ import annotations

import torch


def generate_mask_pair(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two boolean masks selecting g1 / g2 pixels in 2x2 cells.

    Each cell randomly picks one of the four allowed neighbour pairs:
    horizontal-top, horizontal-bottom, vertical-left, vertical-right (i.e.
    each pair is two adjacent pixels). One of the two is assigned to mask1,
    the other to mask2; the remaining two pixels of the cell are unused.

    Returns flat masks of shape (N * H * W // 4) over the *same* flat order
    used by :func:`generate_subimages`.
    """
    n, c, h, w = img.shape
    if h % 2 or w % 2:
        raise ValueError(f"Need even spatial dims, got ({h}, {w})")
    num_cells = n * (h // 2) * (w // 2)
    total_pixels = num_cells * 4  # one cell = 4 pixels (TL, TR, BL, BR)
    mask1 = torch.zeros(total_pixels, dtype=torch.bool, device=img.device)
    mask2 = torch.zeros(total_pixels, dtype=torch.bool, device=img.device)

    # 8 ordered pairs of (idx1, idx2) inside a 2x2 cell flattened TL,TR,BL,BR
    idx_pairs = torch.tensor(
        [
            [0, 1], [0, 2], [1, 3], [2, 3],
            [1, 0], [2, 0], [3, 1], [3, 2],
        ],
        dtype=torch.int64,
        device=img.device,
    )
    rnd = torch.randint(0, 8, (num_cells,), device=img.device)
    chosen = idx_pairs[rnd]  # (num_cells, 2)

    base = torch.arange(num_cells, device=img.device) * 4
    mask1[base + chosen[:, 0]] = True
    mask2[base + chosen[:, 1]] = True
    return mask1, mask2


def generate_subimages(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a flat boolean mask (from :func:`generate_mask_pair`) to ``img``.

    Returns a tensor of shape (N, C, H/2, W/2). Vectorised across channels:
    each 2x2 cell is reshaped into a length-4 axis, then ``mask`` (which
    encodes one chosen pixel per cell) is converted to a per-cell index and
    used with :func:`torch.gather`.
    """
    n, c, h, w = img.shape
    # (N, C, H/2, 2, W/2, 2) -> (N, C, H/2, W/2, 4)  with 4 = TL,TR,BL,BR
    cells = img.reshape(n, c, h // 2, 2, w // 2, 2).permute(0, 1, 2, 4, 3, 5)
    cells = cells.reshape(n, c, h // 2, w // 2, 4)
    # mask is shape (N * H/2 * W/2 * 4,) with exactly one True per cell.
    # Recover the per-cell index in [0, 4).
    mask_per_cell = mask.reshape(n, h // 2, w // 2, 4)
    idx = mask_per_cell.to(torch.int64).argmax(dim=-1)  # (N, H/2, W/2)
    idx = idx.unsqueeze(1).unsqueeze(-1).expand(n, c, h // 2, w // 2, 1)
    sub = cells.gather(-1, idx).squeeze(-1)             # (N, C, H/2, W/2)
    return sub
