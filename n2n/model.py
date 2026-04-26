"""U-Net for raw Bayer denoising, following Neighbor2Neighbor (CVPR 2021).

The architecture matches the official N2N implementation
(github.com/TaoHuang2018/Neighbor2Neighbor) plus the two N2V2 (Hoeck et al.,
2022) anti-checkerboard fixes that the project's experimentation
confirmed (see ``docs/EXPERIMENTS_JOURNEY.md``):

* **BlurPool downsample** (instead of max-pool). Max-pool aliases — it
  picks the max of 4 random samples and locks to one of 4 phases per
  2×2 cell, which on real Bayer raw shows up as a visible 2×2 grid in
  flat dark regions. A 3-tap [1,2,1] separable blur followed by avg-pool
  is anti-aliased and removes the lock.
* **PixelShuffle upsample** (instead of ``ConvTranspose2d(c, c, 2, stride=2)``).
  Stride-2 transpose convolution is the textbook checkerboard-artefact
  source (https://distill.pub/2016/deconv-checkerboard/). PixelShuffle
  feeds an unstrided 1×1 conv into a deterministic channel→space
  rearrangement and avoids any phase the network could exploit.
* **Non-residual output head**. The N2V2 paper showed that residual
  heads (``denoised = input − net(input)``) accumulate a 2×2 grid in
  blind-spot / N2N self-supervised setups. Predicting the denoised
  image directly avoids that. The cost is sensitivity to short
  training runs (no zero-init identity prior) — the project's
  ``tmp4/`` experiments showed 5 minutes is unreliable, 10+ minutes
  is stable.
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


class _PixelShuffleUp(nn.Module):
    """2× upsample via 1×1 conv into 4·c channels then PixelShuffle.

    Mathematically equivalent to a learned 2×2 deconvolution but
    parameterised as a 1×1 conv on the input followed by a
    deterministic re-arrangement, which empirically avoids the per-2×2
    phase patterns that ConvTranspose2d's stride-2 layout invites.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 4, 1, bias=True)
        self.shuf = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuf(self.conv(x))


class _BlurDown(nn.Module):
    """Anti-aliased 2× downsample: 3-tap [1,2,1] separable blur + avg-pool 2."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([[1., 2., 1.],
                          [2., 4., 2.],
                          [1., 2., 1.]]) / 16.0
        self.register_buffer("kernel", k.view(1, 1, 3, 3), persistent=False)

    def forward(self, x):
        n, c, h, w = x.shape
        k = self.kernel.expand(c, 1, 3, 3).to(x.dtype)
        x = F.conv2d(x, k, padding=1, groups=c)
        return F.avg_pool2d(x, 2)


class UNet(nn.Module):
    """Configurable-depth encoder–decoder with skip connections.

    The output is the **denoised image directly** — there is no residual
    subtraction at the head (cf. N2V2 anti-checkerboard fix). At inference
    callers can use either ``model(x)`` or ``model.denoise(x)``; both return
    the denoised packed RGGB image.

    ``depth=3`` (default) gives the small UNet that converges in ~10 min on
    a 4090 and avoids the "blob"-shaped low-frequency residuals the deeper
    variant produces at small training patches.
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
            in_up = chs[i + 1]
            self.ups.append(_PixelShuffleUp(in_up))
            self.decs.append(_DoubleConv(in_up + chs[i], chs[i] if i > 0 else base))

        self.out_conv = nn.Conv2d(base, out_ch, 1)
        # Default PyTorch Kaiming init on the head — non-residual networks
        # need the head to learn a real mapping, not start at zero.
        self.blur = _BlurDown()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        cur = x
        for i, enc in enumerate(self.encs):
            cur = enc(cur if i == 0 else self.blur(cur))
            skips.append(cur)
        cur = self.bottleneck(self.blur(cur))
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            cur = dec(torch.cat([up(cur), skip], dim=1))
        return self.out_conv(cur)

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward` — kept for callers that previously
        relied on the residual-head ``denoise = x - forward(x)`` pattern."""
        return self.forward(x)
