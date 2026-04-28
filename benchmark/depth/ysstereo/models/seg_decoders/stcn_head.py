from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from ysstereo.models.builder import DECODERS, build_loss


@DECODERS.register_module()
class StcnHead(nn.Module):
    """Base class for StcnHead.

    Args:
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
    """

    def __init__(self, loss_decode=dict(type='VideoBootstrappedCELoss')):
        super(StcnHead, self).__init__()

        self.loss_decode = nn.ModuleList()

        if isinstance(loss_decode, dict):
            self.loss_decode.append(build_loss(loss_decode))
        elif isinstance(loss_decode, (list, tuple)):
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        self.fp16_enabled = False

    def forward(self, inputs):
        """Forward function."""
        return inputs

    def forward_train(self, inputs, iteration, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `ysvos/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        data = self.forward(inputs)
        data.update({'iteration': iteration})
        losses = self.losses(**data)
        return losses

    def forward_test(self, inputs, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `ysvos/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def losses(self,
               gt : torch.Tensor,
               cls_gt : torch.Tensor,
               iteration : torch.Tensor,
               **kwargs : torch.float32):
        """Compute segmentation loss."""
        data = kwargs
        data.update({'gt': gt, 'cls_gt': cls_gt})
        loss = dict()

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(data, iteration)
            else:
                loss[loss_decode.loss_name] += loss_decode(data, iteration)

        return loss
