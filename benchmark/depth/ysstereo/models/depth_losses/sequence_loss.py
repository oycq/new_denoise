from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ysstereo.models.builder import LOSSES


def sequence_loss(preds, disp_gt, gamma, valid=None, max_disp=400,
                  support_swing=False, factor=0, oa_param=0.5,
                  use_batch_weight=False):
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
    n_preds = len(preds)
    disp_loss = torch.zeros_like(disp_gt[0, 0, 0, 0])
    # mag = torch.sum(disp_gt**2, dim=1).sqrt()
    disp_gt = disp_gt.squeeze(1)
    valid = valid.squeeze(1)
    mag = (disp_gt**2).sqrt()
    if valid is None:
        valid = torch.ones(disp_gt[:, 0, :, :].shape).to(disp_gt)
    else:
        valid = ((valid >= 0.5) & (mag < max_disp)).to(disp_gt)

    if valid.sum() == 0:  # all samples in the batch are real samples without GT. 
        return disp_loss  

    # compute batch valid ratio
    if use_batch_weight:
        batch_valid_ratio = torch.sum(valid.view(valid.shape[0], -1), dim=1) / valid.view(valid.shape[0], -1).shape[1]
        batch_valid_weight= 1.0 / (batch_valid_ratio + 0.1)
        batch_valid_weight[batch_valid_weight < 1.0] = 1.0
        batch_valid_weight = batch_valid_weight.view(-1, 1, 1)

    for i, pred in enumerate(preds):
        pred = pred.squeeze(1)
        i_weight = gamma**(n_preds - i - 1)
        i_loss = torch.clamp(disp_gt-pred, min=0) + torch.clamp(pred-disp_gt, min=0)*oa_param
        if use_batch_weight:
            i_loss = i_loss * batch_valid_weight
        if support_swing:
            i_weight = i_weight + 2 ** (-disp_gt[valid.bool()]/3 + factor)
        disp_loss = disp_loss + (i_weight * i_loss[valid.bool()]).mean()

    return disp_loss


@LOSSES.register_module()
class SequenceLoss(nn.Module):
    """Sequence Loss for RAFT.

    Args:
        gamma (float): The base of exponentially increasing weights. Default to
            0.8.
        max_disp (float): The maximum value of disparity, if some pixel's
            disparity of target is larger than it, this pixel is not valid.
                Default to 400.
    """

    def __init__(self, gamma: float = 0.8, max_disp: float = 400., suppress_swing=False, factor=0, oa_param=0.5,
                 use_batch_weight=False, swing_weight_thres=(1.0, 1.0)) -> None:
        super().__init__()
        self.gamma = gamma
        self.max_disp = max_disp
        self.suppress_swing = suppress_swing
        self.factor = factor
        self.oa_param = oa_param
        self.swing_weight_thres = swing_weight_thres
        self.use_batch_weight = use_batch_weight

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
        return sequence_loss(disp_preds, disp_gt, self.gamma, valid,
                             self.max_disp, self.suppress_swing,
                             self.factor, self.oa_param, self.use_batch_weight)
