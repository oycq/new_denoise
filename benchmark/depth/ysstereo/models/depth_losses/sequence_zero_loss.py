from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from ysstereo.models.builder import LOSSES


def sequence_addzero_loss(preds, disp_gt, w_zero, zero_thresh):
    """Compute sequence loss between prediction and ground truth.

    Args:
        preds (list(torch.Tensor)): List of disp prediction from
            disp_estimator.
        disp_gt (torch.Tensor): Ground truth disp map.
        gamma (float): Scale factor gamma in loss calculation.
        valid (torch.Tensor, optional): Tensor Used to exclude invalid pixels.
            Default: None.
        max_disp (int, optional): Used to exclude extremely large
            displacements. Default: 400.

    Returns:
        disp_loss (float): Total sequence loss.
    """
    zero_disp_loss = 0.
    # mag = torch.sum(disp_gt**2, dim=1).sqrt()
    disp_gt = disp_gt.squeeze(1)
    mag = (disp_gt ** 2).sqrt()
    invalid = (mag < zero_thresh).to(disp_gt)

    for i, pred in enumerate(preds):
        pred = pred.squeeze(1)
        i_loss = (pred - disp_gt).abs()
        zero_disp_loss += w_zero * (invalid * i_loss).sum() / invalid.sum()

    return zero_disp_loss


@LOSSES.register_module()
class SequenceAddzeroLoss(nn.Module):
    """Sequence Loss for RAFT.

    Args:
        gamma (float): The base of exponentially increasing weights. Default to
            0.8.
        max_disp (float): The maximum value of disparity, if some pixel's
            disparity of target is larger than it, this pixel is not valid.
                Default to 400.
    """

    def __init__(self, zero_thresh: float = 2, gamma: float = 0.8, max_disp: float = 400., weight: float = 1) -> None:
        super().__init__()
        self.gamma = gamma
        self.max_disp = max_disp
        self.weight = weight
        self.zero_thresh = zero_thresh

    def forward(self,
                disp_preds: Sequence[Tensor],
                disp_gt: Tensor,
                valid: Tensor = None) -> Tensor:
        """Forward function for MultiLevelEPE.

        Args:
            preds_dict Sequence[Tensor]: The list of predicted disparity.
            target (Tensor): Ground truth of disparity with shape
                (B, 2, H, W).
            valid (Tensor, optional): Valid mask for disparity.
                Defaults to None.

        Returns:
            Tensor: value of pixel-wise end point error loss.
        """
        return sequence_addzero_loss(disp_preds, disp_gt, self.w_zero, self.zero_thresh)
