import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class VideoIrtL1Loss(nn.Module):
    """L1 loss to improve the temporal consistency.

    Ref paper: Blind Video Temporal Consistency via Deep Video Prior (NeurIPS 2020)
    Ref code: https://github.com/ChenyangLEI/deep-video-prior
    """

    def __init__(self, loss_name='loss_irt_l1', coeff=1.0):
        super().__init__()
        self._loss_name = loss_name
        self.coeff = coeff

    def forward(self, data, it, **kwargs):
        losses = 0

        b, s, _, _, _ = data['gt'].shape

        for i in range(1, s):
            loss_i = 0
            for j in range(b):
                prediction = data['alphas_%d' % i][j:j + 1]
                prediction_main = prediction[:, :1]
                prediction_minor = prediction[:, 1:]
                net_gt = data['alphas'][j:j + 1, i]
                net_in = net_gt

                loss_l1 = F.l1_loss(prediction_main, net_gt)

                diff_map_main, _ = torch.max(
                    torch.abs(prediction_main - net_gt) / (net_in + 1e-1),
                    dim=1,
                    keepdim=True)
                diff_map_minor, _ = torch.max(
                    torch.abs(prediction_minor - net_gt) / (net_in + 1e-1),
                    dim=1,
                    keepdim=True)
                confidence_map = torch.lt(diff_map_main,
                                          diff_map_minor).float()
                confidence_map_l = torch.lt(diff_map_main,
                                            diff_map_minor).float()
                loss_major = F.l1_loss(prediction_main * confidence_map_l,
                                       net_gt * confidence_map_l)
                loss_minor = F.l1_loss(prediction_minor * (1 - confidence_map),
                                       net_gt * (1 - confidence_map))
                loss = loss_l1 + loss_major + loss_minor

                loss_i += loss / b

            losses += loss_i

        return self.coeff * losses

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name