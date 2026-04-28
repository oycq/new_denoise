from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ysstereo.models.builder import LOSSES


def sequence_fs_loss(preds: Sequence[Tensor],
                     keysets: Tensor,
                     lambda_sets: Tensor, gamma):
    """Compute sequence loss between prediction and ground truth.

    Args:
        preds (list(torch.Tensor)): List of disp prediction from
            disp_estimator.
        disp_gt (torch.Tensor): Ground truth disp map.
        keysets (torch.Tensor): Ground truth disp map.
                lambda_sets:
        gamma (float): Scale factor gamma in loss calculation.

    Returns:
        fs_loss (float): Total sequence loss.
    """
    n_preds = len(preds)
    fs_loss = 0.
    # mag = torch.sum(disp_gt**2, dim=1).sqrt()
    bs, _, h, w = preds[0].shape

    if lambda_sets.ndim == 4:
        lambda_sets = lambda_sets.squeeze(1)

    for i, pred in enumerate(preds):
        i_weight = gamma ** (n_preds - i - 1)

        flatten_pred_i = preds[i].view(bs, 1, h * w)
        disp_1 = torch.gather(flatten_pred_i, 2, keysets[:, :, 0])
        disp_2 = torch.gather(flatten_pred_i, 2, keysets[:, :, 1])
        disp_3 = torch.gather(flatten_pred_i, 2, keysets[:, :, 2])

        i_loss = (lambda_sets * (disp_2 - disp_1) - (disp_3 - disp_1)).abs()

        fs_loss = fs_loss + i_weight * i_loss.mean()

    return fs_loss


@LOSSES.register_module()
class SequenceFSLoss(nn.Module):
    """ Flatness and Strightness Loss function defined over sequence of disparity predictions .

    Args:
        gamma (float): The base of exponentially increasing weights. Default to
            0.8.
        max_disp (float): The maximum value of disparity, if some pixel's
            disparity of target is larger than it, this pixel is not valid.
                Default to 400.
    """

    def __init__(self, gamma: float = 0.8, max_disp: float = 400., weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.max_disp = max_disp
        self.weight = weight

    def forward(self,
                disp_preds: Sequence[Tensor],
                keysets: Sequence[Tensor] = None,
                lambda_sets: Sequence[Tensor] = None) -> tuple:
        """Forward function for MultiLevelEPE.

        Args:
            disp_preds: The list of predicted disparity.
            disp_gt (Tensor): Ground truth of disparity with shape
                (B, 2, H, W).
            lambda_sets:
            keysets:
            valid (Tensor, optional): Valid mask for disparity.
                Defaults to None.

        Returns:
            Tensor: value of pixel-wise end point error loss.
        """

        fs_loss = sequence_fs_loss(disp_preds, keysets, lambda_sets, self.gamma)
        fs_loss = fs_loss*self.weight

        return fs_loss
