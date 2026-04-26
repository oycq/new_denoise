"""VGG-16 perceptual loss for Neighbor2Neighbor training.

After many ablations (chroma split / edge-aware TV / EMA self-distill /
masking / R2R / sub-pixel jitter — all archived in
``docs/EXPERIMENTS_JOURNEY.md`` and the local ``_archive_local/`` tree)
the recipe collapsed to:

    L1(den, target)  +  lam_vgg * vgg_perceptual_loss(den, target)

The "extras" cost real wall-clock time and gave no measurable visual
improvement on top of plain L1 + VGG (the diff is invisible at minute 10
on the project's scenes; see ``tmp/`` vs ``tmp2/`` reports). They have
been removed from the live codebase. This module now exposes the single
non-trivial loss term.

VGG only participates in the **training** loss — inference uses the
unmodified UNet, so deploy targets (BPU / ONNX / TRT) see no change.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _VGGFeatExtractor(nn.Module):
    """Cached singleton VGG-16 feature extractor.

    Returns feature maps from ``relu1_2 / relu2_2 / relu3_3`` (the LPIPS
    layers). Weights are downloaded once from torchvision and frozen.
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
    Gr+Gb. Output keeps the packed resolution; VGG receptive fields adapt
    without rescaling because we only care about feature similarity, not
    absolute scale.
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
    are computed in fp32 internally so the call is autocast-safe.
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
