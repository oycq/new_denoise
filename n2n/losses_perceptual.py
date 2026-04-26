"""VGG perceptual loss for Neighbor2Neighbor training.

After a 6-candidate perceptual sweep (chroma split / CSF-weighted FFT /
MS-SSIM / VGG / combos — see ``docs/EXPERIMENTS_JOURNEY.md``) the user
picked **VGG-as-loss** as the only term that visibly changes residual
noise from the "unnatural blotch" look toward natural-looking noise.
The chroma split and edge-aware TV stay because they round out the
behaviour cheaply; the FFT/CSF and MS-SSIM terms were dropped from the
production code (kept only as text in the journey doc).

This module exposes:

* :func:`luma_chroma_l1` — YCoCg-R split L1 with HVS-motivated weights.
* :func:`vgg_perceptual_loss` — feature-space L1 over three VGG-16
  layers (relu1_2 / relu2_2 / relu3_3, the LPIPS layers). Runs on a
  cached singleton extractor in fp32 internally so it is autocast-safe.
* :func:`vgg_l1_eatv` — the production combo for the recommended
  ``P4_vgg`` candidate: luma_chroma_l1 + lam_eatv · edge-aware TV +
  lam_vgg · vgg_perceptual_loss. Drop-in replacement for
  ``l1_plus_tv``.
* :func:`luma_chroma_tv` — the chroma-only ablation (no VGG), kept so
  the journey doc's pure-chroma comparison is reproducible.

All losses operate on packed-RGGB ``(N, 4, H, W)`` tensors in [0, 1].

VGG only participates in the **training** loss. Inference uses the
unmodified UNet, so deploy targets (BPU / ONNX / TRT) see no change.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _grad_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def _edge_aware_tv(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """Total-variation regulariser whose weight at each pixel decays
    exponentially with the local gradient magnitude, so edges are
    preserved while flat regions still get smoothed."""
    dx, dy = _grad_xy(x)
    with torch.no_grad():
        wx = torch.exp(-dx.abs() / max(sigma, 1e-6))
        wy = torch.exp(-dy.abs() / max(sigma, 1e-6))
    return (wx * dx.abs()).mean() + (wy * dy.abs()).mean()


# ---------------------------------------------------------------------------
# Luma / chroma decomposition (YCoCg-R, reversible)
# ---------------------------------------------------------------------------
def ycocg_split(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed RGGB ``(N, 4, H, W)`` -> ``(Y, Co, Cg)`` each ``(N, 1, H, W)``.

    YCoCg-R formulas (lossless, integer-friendly)::

        G   = 0.5 * (Gr + Gb)
        Co  = R - B                (orange-blue chroma)
        Cg  = G - 0.5 * (R + B)    (green-magenta chroma)
        Y   = 0.25*R + 0.5*G + 0.25*B
    """
    R = packed[:, 0:1]
    Gr = packed[:, 1:2]
    Gb = packed[:, 2:3]
    B = packed[:, 3:4]
    G = 0.5 * (Gr + Gb)
    Co = R - B
    Cg = G - 0.5 * (R + B)
    Y = 0.25 * R + 0.5 * G + 0.25 * B
    return Y, Co, Cg


def luma_chroma_l1(
    den: torch.Tensor,
    target: torch.Tensor,
    *,
    w_luma: float = 4.0,
    w_chroma: float = 1.0,
) -> torch.Tensor:
    """L1 separately on Y vs (Co, Cg) with HVS-motivated weights.

    Default ``w_luma=4 · w_chroma=1`` reflects the rough luma:chroma
    sensitivity ratio used by JPEG quantisation (which itself is based
    on HVS measurements).
    """
    Yd, Cod, Cgd = ycocg_split(den)
    Yt, Cot, Cgt = ycocg_split(target)
    L_y = F.l1_loss(Yd, Yt)
    L_c = 0.5 * (F.l1_loss(Cod, Cot) + F.l1_loss(Cgd, Cgt))
    return w_luma * L_y + w_chroma * L_c


def luma_chroma_tv(
    den: torch.Tensor,
    target: torch.Tensor,
    *,
    w_luma: float = 4.0,
    w_chroma: float = 1.0,
    lam_tv: float = 0.03,
) -> torch.Tensor:
    """``luma_chroma_l1 + lam_tv · TV(den)`` — the chroma-only ablation."""
    base = luma_chroma_l1(den, target, w_luma=w_luma, w_chroma=w_chroma)
    dx, dy = _grad_xy(den)
    return base + lam_tv * (dx.abs().mean() + dy.abs().mean())


# ---------------------------------------------------------------------------
# VGG perceptual loss (NN-as-loss, true LPIPS-flavour)
# ---------------------------------------------------------------------------
class _VGGFeatExtractor(nn.Module):
    """Cached singleton VGG16 feature extractor used by
    :func:`vgg_perceptual_loss`. Returns feature maps from layers
    ``relu1_2``, ``relu2_2``, ``relu3_3`` (the LPIPS layers).
    """

    _SINGLETON: Optional["_VGGFeatExtractor"] = None
    _LAYERS = (3, 8, 15)

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: max(self._LAYERS) + 1]
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone.eval().to(device=device, dtype=dtype)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=dtype, device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=dtype, device=device).view(1, 3, 1, 1),
        )

    @classmethod
    def get(cls, device: torch.device, dtype: torch.dtype) -> "_VGGFeatExtractor":
        if cls._SINGLETON is None:
            cls._SINGLETON = cls(device, dtype)
        return cls._SINGLETON

    def forward(self, rgb01: torch.Tensor) -> list[torch.Tensor]:
        """Take RGB in [0, 1], emit a list of conv-feature tensors."""
        x = (rgb01 - self.mean) / self.std
        feats = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in self._LAYERS:
                feats.append(x)
            if idx == max(self._LAYERS):
                break
        return feats


def packed_to_rgb(packed: torch.Tensor) -> torch.Tensor:
    """Map ``(N, 4, H, W)`` RGGB -> ``(N, 3, H, W)`` RGB by collapsing
    Gr+Gb. The output has the same packed resolution; VGG receptive
    fields adapt without rescaling because we only care about
    feature-space similarity, not absolute scale.
    """
    R = packed[:, 0:1]
    G = 0.5 * (packed[:, 1:2] + packed[:, 2:3])
    B = packed[:, 3:4]
    return torch.cat([R, G, B], dim=1).clamp(0.0, 1.0)


def vgg_perceptual_loss(
    den: torch.Tensor,
    target: torch.Tensor,
    *,
    layer_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """L1 between VGG-16 features of ``den`` vs ``target``.

    Both are first converted to RGB (Gr+Gb collapsed to G). VGG features
    are computed in fp32 internally (autocast-safe).
    """
    rgb_d = packed_to_rgb(den).float()
    rgb_t = packed_to_rgb(target).float()
    extractor = _VGGFeatExtractor.get(rgb_d.device, rgb_d.dtype)
    feats_d = extractor(rgb_d)
    feats_t = extractor(rgb_t.detach())
    loss = sum(
        w * F.l1_loss(fd, ft) for w, fd, ft in zip(layer_weights, feats_d, feats_t)
    )
    return loss.to(den.dtype)


# ---------------------------------------------------------------------------
# Production combo (the loss `P4_vgg` actually uses)
# ---------------------------------------------------------------------------
def vgg_l1_eatv(
    den: torch.Tensor,
    target: torch.Tensor,
    *,
    w_luma: float = 4.0,
    w_chroma: float = 1.0,
    lam_eatv: float = 0.03,
    lam_vgg: float = 0.05,
    sigma: float = 0.03,
) -> torch.Tensor:
    """Luma/chroma L1 + edge-aware TV + VGG-16 feature L1.

    This is the recipe selected after the perceptual-loss sweep. Costs
    roughly 1.25× a bare L1+TV step due to the (cached, frozen) VGG
    forward pass; inference is unaffected because the loss only runs at
    training time.
    """
    base = luma_chroma_l1(den, target, w_luma=w_luma, w_chroma=w_chroma)
    eatv = _edge_aware_tv(den, sigma=sigma)
    vgg = vgg_perceptual_loss(den, target)
    return base + lam_eatv * eatv + lam_vgg * vgg


EXTRA_REGISTRY_PERCEPTUAL: dict[str, callable] = {
    "luma_chroma_l1": luma_chroma_l1,
    "luma_chroma_tv": luma_chroma_tv,
    "vgg_perceptual_loss": vgg_perceptual_loss,
    "vgg_l1_eatv": vgg_l1_eatv,
}
