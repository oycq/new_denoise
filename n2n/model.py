"""Small U-Net for raw denoising.

Designed to fit comfortably in 8 GB GPU memory under FP16 with batch size
4-8 at 128x128 patch size. Predicts a *noise residual*: the final denoised
image is ``input - model(input)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Shallow encoder-decoder with skip connections, residual output.

    Depth 3 (3x max-pool, 8x total spatial reduction) keeps the receptive
    field local enough for small training patches, which avoids the smooth
    "blob"-shaped residuals that a deeper UNet tends to produce on
    textureless regions when trained with a single L1 loss.
    """

    def __init__(self, in_ch: int = 4, out_ch: int = 4, base: int = 48):
        super().__init__()
        self.enc1 = _DoubleConv(in_ch, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.enc3 = _DoubleConv(base * 2, base * 4)
        self.bottleneck = _DoubleConv(base * 4, base * 4)

        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.dec3 = _DoubleConv(base * 8, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_ch, 1)

        # Initialise the residual head near zero so early training is stable.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)  # residual noise estimate

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: subtract predicted noise from input."""
        return x - self.forward(x)
