import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ysstereo.ops.AGCL import AGCL
from ysstereo.models.builder import DECODERS, build_loss, build_encoder, build_components
from ysstereo.models.depth_decoders.base_decoder import BaseDecoder
from ysstereo.models.depth_decoders.decoder_submodules import MotionEncoder, ConvGRU, SeqConvGRU, XHead
from ysstereo.utils.utils import convex_upsample


class BasicUpdateBlock(nn.Module):
    def __init__(self, 
            radius: int,
            gru_type: str = 'SeqConv',
            h_channels: int = 32,
            cxt_channels: int = 16,
            feat_channels: int = 64,
            mask_channels: int = 64,
            conv_cfg: Optional[dict] = None,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()

        feat_channels = feat_channels if isinstance(tuple,
                                                    list) else [feat_channels]

        self.feat_channels = feat_channels
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels
        self.mask_channels = mask_channels * (2 * radius + 1)

        self.encoder = MotionEncoder(
            num_levels=4, # 4-group-wise correlation, thus 4 folds of channels
            radius=radius,
            net_type='Full',
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gru_type = gru_type

        self.gru = self.make_gru_block()

        self.flow_pred = XHead(self.h_channels, self.feat_channels, 2, x='flow')

        self.mask_pred = XHead(
            self.h_channels, self.feat_channels, self.mask_channels, x='mask')

    def make_gru_block(self):
        return SeqConvGRU(
            self.h_channels,
            self.encoder.out_channels[0] + 2 + self.cxt_channels, # fix a bug, as GRU input is motion + context
            net_type=self.gru_type)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(corr, flow)
        inp = torch.cat((inp, motion_features), dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_pred(net)

        # scale mask to balence gradients
        mask = .25 * self.mask_pred(net)
        return net, mask, delta_flow

@DECODERS.register_module()
class CreStereoDecoder(BaseDecoder):
    """The decoder of CreStereo Net.

    The decoder of CreStereo Net, which outputs list of upsampled stereo estimation.

    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        radius (int): Radius used when calculating correlation tensor.
        gru_type (str): Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
        feat_channels (Sequence(int)): features channels of prediction module.
        mask_channels (int): Output channels of mask prediction layer.
            Default: 64.
        conv_cfg (dict, optional): Config dict of convolution layers in motion
            encoder. Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in motion encoder.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in motion
            encoder. Default: None.
    """
    def __init__(
        self,
        radius: int,
        gru_type: str = 'SeqConv',
        cxt_channels: int = 16,
        h_channels: int = 32,
        d_model: int = 256,
        feat_channels: int = 64,
        mask_channels: int = 64,
        iters: int = 10,
        self_transformer_encoder: dict = None,
        cross_transformer_encoder: dict = None,
        position_encoding_sine: dict = None,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        disp_loss: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.self_att_fn = build_encoder(self_transformer_encoder)
        self.cross_att_fn = build_encoder(cross_transformer_encoder)
        self.AGCL = AGCL(att=self.cross_att_fn)
        self.pos_encoding_fn_small = build_components(position_encoding_sine)

        self.h_channels = h_channels
        self.iters = iters

        # adaptive search
        self.search_num = 9
        self.conv_offset_16 = nn.Conv2d(
            d_model, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_offset_8 = nn.Conv2d(
            d_model, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )

        self.update_block = BasicUpdateBlock(radius, gru_type,
                h_channels, cxt_channels, feat_channels, mask_channels, conv_cfg, norm_cfg, act_cfg)

        if disp_loss is not None:
            self.disp_loss = build_loss(disp_loss)

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_flow = torch.cat((_x, _y), dim=1).to(fmap.device)
        return zero_flow

    def forward(self, fmap1, fmap2, flow_init=None, test_mode=False):
        # 1/4 -> 1/8
        # feature
        fmap1_dw8 = F.avg_pool2d(fmap1, 2, stride=2)
        fmap2_dw8 = F.avg_pool2d(fmap2, 2, stride=2)

        # offset
        offset_dw8 = self.conv_offset_8(fmap1_dw8)
        offset_dw8 = (torch.sigmoid(offset_dw8) - 0.5) * 2.0

        # context
        net, inp = torch.split(fmap1, [self.h_channels,self.h_channels], dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)
        net_dw8 = F.avg_pool2d(net, 2, stride=2)
        inp_dw8 = F.avg_pool2d(inp, 2, stride=2)

        # 1/4 -> 1/16
        # feature
        fmap1_dw16 = F.avg_pool2d(fmap1, 4, stride=4)
        fmap2_dw16 = F.avg_pool2d(fmap2, 4, stride=4)
        offset_dw16 = self.conv_offset_16(fmap1_dw16)
        offset_dw16 = (torch.sigmoid(offset_dw16) - 0.5) * 2.0

        # context
        net_dw16 = F.avg_pool2d(net, 4, stride=4)
        inp_dw16 = F.avg_pool2d(inp, 4, stride=4)

        # positional encoding and self-attention
        # 'n c h w -> n (h w) c'
        x_tmp = self.pos_encoding_fn_small(fmap1_dw16)
        fmap1_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1])
        # 'n c h w -> n (h w) c'
        x_tmp = self.pos_encoding_fn_small(fmap2_dw16)
        fmap2_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1])

        fmap1_dw16, fmap2_dw16 = self.self_att_fn(fmap1_dw16, fmap2_dw16)
        fmap1_dw16, fmap2_dw16 = [
            x.reshape(x.shape[0], fmap1.shape[2] // 4, -1, x.shape[2]).permute(0, 3, 1, 2)
            for x in [fmap1_dw16, fmap2_dw16]
        ]

        # Cascaded refinement (1/16 + 1/8 + 1/4)
        predictions = []
        flow = None
        flow_up = None
        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = -scale * F.interpolate(
                flow_init,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
                )
        else:
            # zero initialization
            flow_dw16 = self.zero_init(fmap1_dw16)

            # Recurrent Update Module
            # RUM: 1/16
            for itr in range(self.iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw16 = flow_dw16.detach()
                out_corrs = self.AGCL(
                    fmap1_dw16, fmap2_dw16, flow_dw16, offset_dw16, small_patch=small_patch
                    )

                net_dw16, up_mask, delta_flow = self.update_block(
                    net_dw16, inp_dw16, out_corrs, flow_dw16
                )

                flow_dw16 = flow_dw16 + delta_flow
                flow = convex_upsample(flow_dw16, up_mask, rate=4)
                flow_up = -4 * F.interpolate(
                    flow,
                    size=(4 * flow.shape[2], 4 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_up = flow_up[:,:1]
                predictions.append(flow_up)

            scale = fmap1_dw8.shape[2] / flow.shape[2]
            flow_dw8 = -scale * F.interpolate(
                flow,
                size=(fmap1_dw8.shape[2], fmap1_dw8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

            # RUM: 1/8
            for itr in range(self.iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw8 = flow_dw8.detach()
                out_corrs = self.AGCL(fmap1_dw8, fmap2_dw8, flow_dw8, offset_dw8, small_patch=small_patch)

                net_dw8, up_mask, delta_flow = self.update_block(
                    net_dw8, inp_dw8, out_corrs, flow_dw8
                )

                flow_dw8 = flow_dw8 + delta_flow
                flow = convex_upsample(flow_dw8, up_mask, rate=4)
                flow_up = -2 * F.interpolate(
                    flow,
                    size=(2 * flow.shape[2], 2 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_up = flow_up[:,:1]
                predictions.append(flow_up)

            scale = fmap1.shape[2] / flow.shape[2]
            flow = -scale * F.interpolate(
                flow,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

        # RUM: 1/4
        for itr in range(self.iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = self.AGCL(fmap1, fmap2, flow, None, small_patch=small_patch, iter_mode=True)

            net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = -convex_upsample(flow, up_mask, rate=4)
            flow_up = flow_up[:,:1]
            predictions.append(flow_up)

        if test_mode:
            return flow_up

        return predictions

    def forward_train(self,
                    fmap1: torch.Tensor,
                    fmap2: torch.Tensor,
                    flow_init: torch.Tensor,
                    test_mode: bool,
                    disp_gt: torch.Tensor,
                    valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the left input image.
            feat2 (Tensor): The feature from the right input image.
            flow_init (Tensor): The init estimated flow from GRU cell.
            test_mode (bool): The flag of test mode.
            disp_gt (Tensor): The ground truth of disparity.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of model.
        """

        disp_pred = self.forward(fmap1, fmap2, flow_init, test_mode)

        return self.losses(disp_pred, disp_gt, valid=valid)

    def forward_test(self,
                    fmap1: torch.Tensor,
                    fmap2: torch.Tensor,
                    flow_init: torch.Tensor,
                    test_mode: bool = False,
                    img_metas=None) -> Sequence[Dict[str, np.ndarray]]:
        """Forward function when model training.

        Args:
            fmap1 (Tensor): The feature from the left input image.
            fmap2 (Tensor): The feature from the right input image.
            flow_init (Tensor): The init estimated flow from GRU cell.
            test_mode (bool): The flag of test mode.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the disparity to original ground truth size. Defaults to None.
        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted disparity
                with the same size of images before augmentation.
        """
        disp_pred = self.forward(fmap1, fmap2, flow_init, test_mode)

        # disp_result = disp_pred[-1]
        # print('disp result shape: ', disp_result.shape)
        # disparity maps with the shape [H, W, 2]
        disp_result = disp_pred.permute(0, 2, 3, 1).cpu().data.numpy()
        # unravel batch dim
        disp_result = list(disp_result)
        disp_result = [dict(disp=f) for f in disp_result]

        return self.get_disp(disp_result, img_metas=img_metas)

    def losses(self,
               disp_pred: Sequence[torch.Tensor],
               disp_gt: torch.Tensor,
               valid: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute disparity loss.

        Args:
            disp_pred (Sequence[Tensor]): The list of predicted disparity.
            disp_gt (Tensor): The ground truth of disparity.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """

        loss = dict()
        loss['loss_disp'] = self.disp_loss(disp_pred, disp_gt, valid)
        return loss