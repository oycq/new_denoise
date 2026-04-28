from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from .mmseg.structures import build_pixel_sampler
from .mmseg.models.utils.wrappers import resize
from ..seg_losses import accuracy

from ysstereo.models.builder import DECODERS, build_loss


@DECODERS.register_module()
class STMHead(nn.Module, metaclass=ABCMeta):
    """Base class for STMHead.

    Args:
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(STMHead, self).__init__()
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def forward(self,
                inputs : torch.float16):
        """Forward function."""
        return inputs

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def losses(self,
               seg_logit: torch.float32,
               seg_label: torch.Tensor):
        """Compute segmentation loss."""
        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        def convert_onehot_to_seg(onehot):
            """Convert onehot shape [N, C, H, W] to [N, H, W]"""
            N, C, H, W = onehot.size()
            target_shape = (N, H, W)
            seg = onehot.new_zeros(target_shape)
            for k in range(C):
                seg[onehot[:, k, :, :] == 1] = k
            return seg

        N, T, C, H, W = seg_label.size()
        for t in range(1, T):
            pred = seg_logit[:, t - 1]
            label = seg_label[:, t].float()

            loss[f'loss_seg.t_{t}'] = self.loss_decode(
                pred, label, weight=seg_weight, ignore_index=self.ignore_index)

        return loss
