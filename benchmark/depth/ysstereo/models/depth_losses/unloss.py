from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ysstereo.utils.utils import coords_grid
from ysstereo.models.depth_losses.ssim import weighted_ssim
from ysstereo.models.builder import LOSSES

def image_warping_loss(left_img:torch.Tensor, right_img:torch.Tensor, pred_disp:torch.Tensor,
                       ssim_weight:float=0.85):
    B, _, H, W = left_img.shape
    # generate coords
    coords:torch.Tensor = coords_grid(B, H, W, left_img.device) # b, 2, h, w
    x_coords = coords[:,0]
    y_coords = coords[:,1]
    x_coords = x_coords - pred_disp[:, 0] # flow is neg disp
    x_coords = x_coords / ((W-1.0)*0.5) - 1.0
    y_coords = y_coords / ((H-1.0)*0.5) - 1.0
    # grid sample
    sampled_img = F.grid_sample(
        right_img, mode='bilinear', grid=torch.stack([x_coords, y_coords], dim=-1),
        align_corners=True
    )
    ssim_loss = weighted_ssim(left_img, sampled_img).mean()
    pixel_loss = torch.abs(sampled_img - left_img).mean()

    return ssim_weight * ssim_loss + (1-ssim_weight) * pixel_loss

