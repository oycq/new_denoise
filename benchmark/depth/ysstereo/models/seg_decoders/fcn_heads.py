import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.amp import autocast

from typing import Optional, Sequence, Dict
from ysstereo.models.builder import DECODERS
from .mmseg.models.decode_heads import FCNHead as _FCNHead
from .mmseg.models.losses import accuracy
from .mmseg.models.utils.wrappers import resize
from .mmseg.losses.cross_entropy_loss import cross_entropy
from .mmseg.losses.tversky_loss import tversky_loss
from .mmseg.utils.typing_utils import ConfigType

@DECODERS.register_module()
class FCNHead(_FCNHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    """
    def __init__(self, truncated_grad: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.truncated_grad = truncated_grad

    def _transform_inputs(self, inputs, upsample_mode='bilinear'):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(x, scale_factor=2**i, mode=upsample_mode)
                for i, x in enumerate(inputs)
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
    
    def forward_train(self,
                      inputs: torch.Tensor,
                      return_preds: bool,
                      seg_gt: torch.Tensor,
                      train_cfg: ConfigType,
                      valid: Optional[torch.Tensor] = None):
        if self.truncated_grad:
            inputs = [input.detach() for input in inputs]
        with autocast('cuda', enabled=False):
            inputs = [input.float() for input in inputs]
            seg_preds = self.forward(inputs)

        if return_preds:
            return self.losses(seg_logits=seg_preds, seg_label=seg_gt, valid=valid), seg_preds
        else:
            return self.losses(seg_logits=seg_preds, seg_label=seg_gt, valid=valid), None
        
    def forward_test(self,
                     inputs: torch.Tensor):
        with autocast('cuda', enabled=False):
            inputs = [input.float() for input in inputs]

            seg_logit = self.forward(inputs)
            seg_logit = F.interpolate(seg_logit, scale_factor=2**2, mode='bilinear')

        seg_logit_cls = seg_logit.argmax(dim=1, keepdim=True)
        seg_result = seg_logit_cls[:, 0].cpu().data.numpy()
        seg_result = list(seg_result)
        seg_result = [dict(seg=f) for f in seg_result]

        return seg_result

        
    def losses(self,
               seg_logits: Sequence[torch.Tensor],
               seg_label: torch.Tensor,
               valid: torch.Tensor = None, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute segmentation loss.

        Args:
            seg_pred (Sequence[Tensor]): The list of predicted segmentation.
            seg_gt (Tensor): The ground truth of segmentation.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        
        refs: .mmseg.models.decode_heads.loss
        """
        loss = dict()
        seg_label = seg_label.to(torch.long)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label[~valid] = self.ignore_index

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss


@DECODERS.register_module()
class MUTI_FCNHead(_FCNHead):
    def __init__(self,**kwargs):
        super(MUTI_FCNHead, self).__init__(**kwargs)

        self.conv_segs = nn.ModuleList([
            nn.Conv2d(self.channels, 1, kernel_size=3, padding=1) for _ in range(self.num_classes)
        ])
    
    def _transform_inputs(self, inputs, upsample_mode='bilinear'):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(x, scale_factor=2**i, mode=upsample_mode)
                for i, x in enumerate(inputs)
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        outputs = [self.conv_seg(feat)]
        for conv in self.conv_segs:
            outputs.append(conv(feat))
        return outputs


    def losses(self, seg_logit_list, seg_label):
        """Compute segmentation loss."""
        # loss = dict()
        loss = super(MUTI_FCNHead, self).losses(seg_logit_list[0], seg_label)
        seg_label = seg_label.squeeze(1)  # 形状变为 (N, H, W)

        total_loss = 0
        total_acc = 0

        for class_idx in range(1, self.num_classes):
            # 获取对应的 seg_logit，并调整尺寸
            seg_logit = resize(
                input=seg_logit_list[class_idx],
                size=seg_label.shape[1:],  # (H, W)
                mode='bilinear',
                align_corners=self.align_corners)

            # 创建二分类标签，当前类别为1，其他为0
            binary_label = (seg_label == class_idx).float()  # 形状为 (N, H, W)

            # 使用 BCEWithLogitsLoss，logit 输入形状应为 (N, 1, H, W)
            if seg_logit.shape[1] != 1:
                seg_logit = seg_logit[:, 0:1, :, :]  # 取第一个通道，确保形状为 (N, 1, H, W)

            # 计算损失
            bce_loss = F.binary_cross_entropy_with_logits(
                seg_logit,
                binary_label.unsqueeze(1),  # 调整形状为 (N, 1, H, W)
                reduction='mean')

            loss_name = 'loss_bce_class_{}'.format(class_idx)
            loss[loss_name] = bce_loss
            total_loss += bce_loss

            # 使用 mmseg 的 accuracy 函数计算准确率
            pred = (torch.sigmoid(seg_logit) > 0.5).float()  # 获取二分类预测
            acc = accuracy(pred, binary_label.long(), ignore_index=self.ignore_index)  # 使用 mmseg 计算准确率
            acc_name = 'acc_seg_class_{}'.format(class_idx)
            loss[acc_name] = acc
            total_acc += acc

        # 将总损失添加到损失字典中
        # loss['loss_seg'] = total_loss
        # 计算平均准确率
        # loss['acc_seg'] = total_acc / self.num_classes

        return loss


@DECODERS.register_module()
class UpFCNHead(FCNHead):

    def __init__(self, scale_factor=4, **kwargs):
        super(UpFCNHead, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        output = F.interpolate(
            output, scale_factor=self.scale_factor, mode='bilinear')
        return output


@DECODERS.register_module()
class FCNwithPixelShuffleHead(FCNHead):

    def __init__(self,
                 pixelshuffle_dim=4,
                 scale_factor=4,
                 upsample_mode='bilinear',
                 **kwargs):
        super(FCNwithPixelShuffleHead, self).__init__(**kwargs)
        reduce_dim = scale_factor * scale_factor * pixelshuffle_dim

        self.reduce_dim_conv = ConvModule(
            self.channels,
            reduce_dim,
            kernel_size=3,
            padding=1,
            dilation=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.pixelshuffle = nn.PixelShuffle(scale_factor)
        self.upsample_mode = upsample_mode

        self.conv_seg = nn.Conv2d(
            pixelshuffle_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs, upsample_mode=self.upsample_mode)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))

        output = self.reduce_dim_conv(output)
        output = self.pixelshuffle(output)

        output = self.cls_seg(output)
        return output
