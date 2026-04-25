"""Loss functions used by the Neighbor2Neighbor trainer.

All loss functions take ``(den, target)`` tensors of shape ``(N, C, H, W)``
and return a scalar tensor. The ``den`` tensor is the model's denoised
output for the g1 sub-image; ``target`` is the g2 sub-image.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def l1(den: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain L1 (the user's stated baseline)."""
    return F.l1_loss(den, target)


def charbonnier(den: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Smooth L1 (a.k.a. Charbonnier).

    ``sqrt((x-y)^2 + eps^2)`` - quadratic near 0, linear far from 0. Avoids
    the L1 median collapse / posterisation in low-texture regions while
    keeping L1's robustness to outlier gradients.
    """
    diff = den - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def gradient_l1(den: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 of horizontal+vertical first-order differences.

    Encourages the denoiser to preserve edges instead of over-smoothing.
    """
    dx_d = den[..., :, 1:] - den[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_d = den[..., 1:, :] - den[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_d, dx_t) + F.l1_loss(dy_d, dy_t)


def l1_plus_grad(
    den: torch.Tensor, target: torch.Tensor, lam: float = 0.5
) -> torch.Tensor:
    """L1 + lam * gradient-L1: edge-preserving variant."""
    return F.l1_loss(den, target) + lam * gradient_l1(den, target)


def multiscale_l1(
    den: torch.Tensor, target: torch.Tensor, scales: int = 3
) -> torch.Tensor:
    """L1 averaged over multiple Gaussian-pooled scales.

    Pooling by ``avg_pool2d(2**s)`` for ``s in [0, scales)`` produces
    progressively coarser views; matching all scales discourages
    low-frequency blob residuals that single-scale L1 sometimes leaves.
    """
    total = F.l1_loss(den, target)
    for s in range(1, scales):
        f = 2 ** s
        d = F.avg_pool2d(den, f)
        t = F.avg_pool2d(target, f)
        total = total + F.l1_loss(d, t)
    return total / scales


def huber(
    den: torch.Tensor, target: torch.Tensor, delta: float = 0.01
) -> torch.Tensor:
    """Huber: L2 below ``delta``, L1 above.

    Quadratic regime gives smoother gradients in low-error regions, linear
    regime keeps robustness to large noise outliers.
    """
    return F.smooth_l1_loss(den, target, beta=delta)


def l1_plus_tv(
    den: torch.Tensor, target: torch.Tensor, lam: float = 0.05
) -> torch.Tensor:
    """L1 + lam * total variation on the denoised output.

    TV penalises ``|d/dx den| + |d/dy den|``, encouraging piecewise-smooth
    output (less low-frequency blob, but risks slight cartoonisation).
    """
    main = F.l1_loss(den, target)
    tv = (den[..., :, 1:] - den[..., :, :-1]).abs().mean() + (
        den[..., 1:, :] - den[..., :-1, :]
    ).abs().mean()
    return main + lam * tv


def ms_l1_charbonnier(
    den: torch.Tensor,
    target: torch.Tensor,
    scales: int = 4,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier at multiple Gaussian-pooled scales.

    Smooth-L1 robustness + multi-scale low-freq matching - the union of the
    two best single-axis ideas in the previous round.
    """
    def _ch(d, t):
        diff = d - t
        return torch.sqrt(diff * diff + eps * eps).mean()

    total = _ch(den, target)
    for s in range(1, scales):
        f = 2 ** s
        total = total + _ch(F.avg_pool2d(den, f), F.avg_pool2d(target, f))
    return total / scales


REGISTRY: dict[str, LossFn] = {
    "l1": l1,
    "charbonnier": charbonnier,
    "l1_plus_grad": l1_plus_grad,
    "multiscale_l1": multiscale_l1,
    "huber": huber,
    "l1_plus_tv": l1_plus_tv,
    "ms_l1_charbonnier": ms_l1_charbonnier,
}


def get(name: str) -> LossFn:
    if name not in REGISTRY:
        raise KeyError(f"unknown loss: {name}; available={list(REGISTRY)}")
    return REGISTRY[name]
