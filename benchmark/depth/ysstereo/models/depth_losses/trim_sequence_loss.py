from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ysstereo.models.builder import LOSSES


def sequence_trim_loss(preds, disp_gt, gamma, valid=None, max_disp=400,
                  support_swing=False, factor=0, oa_param=0.5, trim_rate=0.075):
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

    for i, pred in enumerate(preds):
        pred = pred.squeeze(1)
        i_weight = gamma**(n_preds - i - 1)
        i_loss = torch.clamp(disp_gt-pred, min=0) + torch.clamp(pred-disp_gt, min=0)*oa_param
        if support_swing:
            i_weight = i_weight + 2 ** (-disp_gt[valid.bool()]/3 + factor)
        valid_i_loss = i_loss[valid.bool()]
        # trim the top 5% of the loss
        ## sort the loss (small to large) and get index
        sorted_loss, indices = torch.sort(valid_i_loss)
        trim_num = int(len(sorted_loss) * trim_rate)
        if trim_num > 0:
            valid_i_loss = sorted_loss[:-trim_num]
        if support_swing:
            # sort i_weight according to the indices
            i_weight = i_weight[indices]
            if trim_num > 0:
                i_weight = i_weight[:-trim_num]
        disp_loss = disp_loss + (i_weight * valid_i_loss).mean()
    return disp_loss


@LOSSES.register_module()
class TrimSequenceLoss(nn.Module):
    """Sequence Loss for RAFT.

    Args:
        gamma (float): The base of exponentially increasing weights. Default to
            0.8.
        max_disp (float): The maximum value of disparity, if some pixel's
            disparity of target is larger than it, this pixel is not valid.
                Default to 400.
    """

    def __init__(self, gamma: float = 0.8, max_disp: float = 400., suppress_swing=False, factor=0, oa_param=0.5,
                 trim_rate=0.075) -> None:
        super().__init__()
        self.gamma = gamma
        self.max_disp = max_disp
        self.suppress_swing = suppress_swing
        self.factor = factor
        self.oa_param = oa_param
        self.trim_rate = trim_rate

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
        return sequence_trim_loss(disp_preds, disp_gt, self.gamma, valid,
                                self.max_disp, self.suppress_swing,
                                self.factor, self.oa_param, self.trim_rate)
