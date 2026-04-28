import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class VideoL2Loss(nn.Module):

    def __init__(self,
                 loss_name='loss_l2',
                 gt_scale_factor=1,
                 align_corners=False,
                 coeff=1.0):
        super().__init__()
        self._loss_name = loss_name
        self.gt_scale_factor = gt_scale_factor
        self.align_corners = align_corners
        self.coeff = coeff

    def forward(self, data, it, **kwargs):
        losses = 0

        b, s, _, _, _ = data['gt'].shape
        selector = data.get('selector', None)

        for i in range(1, s):
            # Have to do it in a for-loop like this since not every entry has the second object
            # Well it's not a lot of iterations anyway
            loss_i = 0
            for j in range(b):
                if selector is not None and selector[j][1] > 0.5:
                    pred = data['alphas_%d' % i][j:j + 1]
                    gt = data['alphas'][j:j + 1, i]
                    gt = F.interpolate(
                        gt,
                        scale_factor=self.gt_scale_factor,
                        mode='bilinear',
                        align_corners=self.align_corners).float()
                    loss = F.mse_loss(pred, gt)
                else:
                    pred = data['alphas_%d' % i][j:j + 1, :2]
                    gt = data['alphas'][j:j + 1, i]
                    gt = F.interpolate(
                        gt,
                        scale_factor=self.gt_scale_factor,
                        mode='bilinear',
                        align_corners=self.align_corners).float()
                    loss = F.mse_loss(pred, gt)

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