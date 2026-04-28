from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ysstereo.models.builder import LOSSES


def sequence_pseudo_loss(preds, pseudo_gt, gamma, valid, threshold=0.2):
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
    assert n_preds >= 1
    pseudo_loss = torch.zeros_like(pseudo_gt[0, 0, 0, 0])

    if valid.sum() == 0:
        return pseudo_loss
    bs = valid.shape[0]

    for i in range(n_preds):
        try:
            assert not torch.isnan(preds[i]).any() and not torch.isinf(preds[i]).any()
        except:
            return torch.zeros_like(pseudo_gt[0, 0, 0, 0])
        
        adjusted_loss_gamma = gamma**(15/(n_preds))
        i_weight = adjusted_loss_gamma**(n_preds - i - 1)

        for j in range(bs):
            if valid[j]:
                median = torch.median(preds[i][j])
                scale = torch.mean(torch.abs(preds[i][j]-median))

                loss_ij = (pseudo_gt[j] - (preds[i][j]-median)/(scale+10e-7)).abs()
                mask = loss_ij > min(threshold, loss_ij.max().item()-0.001)
                loss = i_weight*loss_ij[mask].mean()
                pseudo_loss += loss

    return pseudo_loss/valid.sum()


@LOSSES.register_module()
class SequencePseudoLoss(nn.Module):
    """Sequence Loss for RAFT.

    Args:
        gamma (float): The base of exponentially increasing weights. Default to
            0.8.
        max_disp (float): The maximum value of disparity, if some pixel's
            disparity of target is larger than it, this pixel is not valid.
                Default to 400.
    """

    def __init__(self, gamma: float = 0.8, threshold=0.2, weight=0.2) -> None:
        super().__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.weight = weight

    def forward(self,
                disp_preds: Sequence[Tensor],
                pseudo_gt: Tensor,
                valid: Tensor) -> Tensor:
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
        return self.weight*sequence_pseudo_loss(disp_preds, pseudo_gt, self.gamma, valid, self.threshold)
