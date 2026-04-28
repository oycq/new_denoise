import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class VideoTimeConsistentLoss(nn.Module):

    def __init__(self, loss_name='loss_time', coeff=1.0):
        super().__init__()
        self._loss_name = loss_name
        self.coeff = coeff

    def forward(self, data, it, **kwargs):
        losses = 0

        b, s, _, _, _ = data['gt'].shape
        selector = data.get('selector', None)

        loss_i = 0
        for j in range(b):
            if selector is not None and selector[j][1] > 0.5:
                loss = F.l1_loss(
                    data['alphas_2'][j:j + 1] - data['alphas_1'][j:j + 1],
                    data['alphas'][j:j + 1, 2] - data['alphas'][j:j + 1, 1])
            else:
                loss = F.l1_loss(
                    data['alphas_2'][j:j + 1, :2] -
                    data['alphas_1'][j:j + 1, :2],
                    data['alphas'][j:j + 1, 2] - data['alphas'][j:j + 1, 1])
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