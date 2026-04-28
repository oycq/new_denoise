from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ysstereo.ops.builder import OPERATORS
from ysstereo.utils.utils import x_bilinear_sampler


@OPERATORS.register_module()
class CorrLookup(nn.Module):
    """Correlation lookup operator.

    This operator is used in `RAFT<https://arxiv.org/pdf/2003.12039.pdf>`_

    Args:
        radius (int): the radius of the local neighborhood of the pixels.
            Default to 4.
        mode (str): interpolation mode to calculate output values 'bilinear'
            | 'nearest' | 'bicubic'. Default: 'bilinear' Note: mode='bicubic'
            supports only 4-D input.
        padding_mode (str): padding mode for outside grid values 'zeros' |
            'border' | 'reflection'. Default: 'zeros'
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s corner
            pixels. If set to False, they are instead considered as referring
            to the corner points of the input’s corner pixels, making the
            sampling more resolution agnostic. Default to True.
    """

    def __init__(self,
                 radius: int = 4,
                 mode: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 align_corners: bool = True) -> None:
        super().__init__()
        self.r = radius
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, corr_pyramid: Sequence[Tensor], flow: Tensor) -> Tensor:
        """Forward function of Correlation lookup.

        Args:
            corr_pyramid (Sequence[Tensor]): Correlation pyramid.
            flow (Tensor): Current estimated disparity.

        Returns:
            Tensor: Feature map by indexing from the correlation pyramid.
        """
        flow = flow[:, :1].permute(0, 2, 3, 1)
        B, H, W, _ = flow.shape
        out_corr_pyramid = []
        for i in range(len(corr_pyramid)):
            corr = corr_pyramid[i]
            dx = torch.linspace(-self.r, self.r, 2*self.r+1)
            dx = dx.view(1, 1, 2*self.r+1, 1).to(flow.device)
            x0 = dx + flow.reshape(B*H*W, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr = x_bilinear_sampler(corr, coords_lvl)

            corr = corr.view(B, H, W, -1)
            out_corr_pyramid.append(corr)

        out = torch.cat(out_corr_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()