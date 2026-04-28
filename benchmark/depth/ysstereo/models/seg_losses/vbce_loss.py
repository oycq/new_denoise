import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


class BootstrappedCE(nn.Module):
    """Bootstrapped cross entropy loss.
    Ref: https://stackoverflow.com/questions/63735255/how-do-i-compute-\
        bootstrapped-cross-entropy-loss-in-pytorch
    """

    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * (
                (self.end_warm - it) / (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


@LOSSES.register_module()
class VideoBootstrappedCELoss(nn.Module):

    def __init__(self, loss_name='loss_vbce'):
        super().__init__()
        self.bce = BootstrappedCE()
        self._loss_name = loss_name

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
                    loss, p = self.bce(data['logits_%d' % i][j:j + 1],
                                       data['cls_gt'][j:j + 1, i], it)
                else:
                    loss, p = self.bce(data['logits_%d' % i][j:j + 1, :2],
                                       data['cls_gt'][j:j + 1, i], it)

                loss_i += loss / b

            losses += loss_i

        return losses

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