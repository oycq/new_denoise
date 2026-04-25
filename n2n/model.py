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
    """Configurable-depth encoder-decoder with skip connections, residual head.

    ``depth=3`` (default) gives the local-feature small UNet that avoided
    "blob"-shaped low-freq residuals at small training patches; ``depth=4``
    is the standard N2N config which works fine when the patch is large
    enough (e.g. 256+).
    """

    def __init__(
        self,
        in_ch: int = 4,
        out_ch: int = 4,
        base: int = 48,
        depth: int = 3,
    ):
        super().__init__()
        if depth < 2 or depth > 5:
            raise ValueError(f"depth must be in [2, 5], got {depth}")
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth)]   # e.g. depth=3 -> [B, 2B, 4B]
        chs.append(chs[-1])  # bottleneck width = deepest enc width

        self.encs = nn.ModuleList(
            [_DoubleConv(in_ch if i == 0 else chs[i - 1], chs[i]) for i in range(depth)]
        )
        self.bottleneck = _DoubleConv(chs[depth - 1], chs[depth])

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for i in reversed(range(depth)):
            in_up = chs[i + 1] if i == depth - 1 else chs[i + 1]
            self.ups.append(nn.ConvTranspose2d(in_up, in_up, 2, stride=2))
            self.decs.append(_DoubleConv(in_up + chs[i], chs[i] if i > 0 else base))

        self.out_conv = nn.Conv2d(base, out_ch, 1)
        # Residual head near zero -> network starts as identity.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        cur = x
        for i, enc in enumerate(self.encs):
            cur = enc(cur if i == 0 else F.max_pool2d(cur, 2))
            skips.append(cur)
        cur = self.bottleneck(F.max_pool2d(cur, 2))
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            cur = dec(torch.cat([up(cur), skip], dim=1))
        return self.out_conv(cur)

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.forward(x)
