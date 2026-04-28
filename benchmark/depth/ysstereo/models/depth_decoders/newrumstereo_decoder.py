import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ysstereo.ops.AGCL import AGCL, get_correlation, corr_iter, onnx_corr_iter
from ysstereo.models.builder import DECODERS, build_loss, build_encoder, build_components
from ysstereo.models.depth_decoders.base_decoder import BaseDecoder
from ysstereo.models.depth_decoders.decoder_submodules import MotionEncoder, ConvGRU, SeqConvGRU, XHead
from ysstereo.utils.utils import convex_upsample, coords_grid


class BasicUpdateBlock(nn.Module):
    def __init__(self,
                 radius: int,
                 gru_type: str = 'SeqConv',
                 conv_type: str = 'Conv',
                 h_channels: int = 32,
                 cxt_channels: int = 32,
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
            radius=radius,
            net_type='Basic',
            conv_type=conv_type,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,)
        self.gru_type = gru_type

        self.gru = self.make_gru_block()

        self.flow_pred = XHead(self.h_channels, self.feat_channels, 2, x='flow')

        self.mask_pred = XHead(
            self.h_channels, self.feat_channels, self.mask_channels, x='mask')

    def make_gru_block(self):
        return SeqConvGRU(
            self.h_channels,
            self.encoder.out_channels[0] + 2 + self.cxt_channels,  # fix a bug, as GRU input is motion + context
            net_type=self.gru_type)

    def forward(self, net, inp=None, corr=None, flow=None, mask_flag=True, output_all=False):
        if output_all:
            motion_features = self.encoder(corr, flow)
            inp = torch.cat((inp, motion_features), dim=1)

            net = self.gru(net, inp)
            delta_flow = self.flow_pred(net)

            # scale mask to balence gradients
            mask = .25 * self.mask_pred(net)

            return net, mask, delta_flow
        if mask_flag:
            # scale mask to balence gradients
            mask = .25 * self.mask_pred(net)
            return mask
        motion_features = self.encoder(corr, flow)
        inp = torch.cat((inp, motion_features), dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_pred(net)

        return net, delta_flow


@DECODERS.register_module()
class NewRumStereoDecoder(BaseDecoder):
    """The decoder of RumStereo Net.

    The decoder of RumStereo Net, which outputs list of upsampled stereo estimation.

    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        radius (int): Radius used when calculating correlation tensor.
        gru_type (str): Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
        conv_type (str): Type of the Conv. Choices: ['Conv', 'InvertedResidual'].
            Default: 'Conv'.
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
            conv_type: str = 'Conv',
            cxt_channels: int = 32,
            h_channels: int = 32,
            feat_channels: int = 64,
            mask_channels: int = 64,
            iters: int = 10,
            frozen_features: bool = False,
            shared_components: Optional[dict] = None, # the shared components of two update blocks
            conv_cfg: Optional[dict] = None,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            disp_loss: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.h_channels = h_channels
        self.iters = iters
        self.frozen_features = frozen_features
        self.shared_components = shared_components

        # adaptive search
        self.search_num = 9
        self.update_block_8 = BasicUpdateBlock(radius, gru_type, conv_type,
                                               h_channels, cxt_channels, feat_channels, mask_channels, conv_cfg, norm_cfg,
                                               act_cfg)
        self.update_block_16 = BasicUpdateBlock(radius, gru_type, conv_type,
                                               h_channels, cxt_channels, feat_channels, mask_channels, conv_cfg, norm_cfg,
                                               act_cfg)
        if self.shared_components is not None:
            if 'encoder' in self.shared_components:
                del self.update_block_8.encoder
                self.update_block_8.encoder = self.update_block_16.encoder
            if 'gru' in self.shared_components:
                del self.update_block_8.gru
                self.update_block_8.gru = self.update_block_16.gru
            if 'head' in self.shared_components:
                del self.update_block_8.flow_pred
                del self.update_block_8.mask_pred
                self.update_block_8.flow_pred = self.update_block_16.flow_pred
                self.update_block_8.mask_pred = self.update_block_16.mask_pred

        if disp_loss is not None:
            self.loss_names = []
            self.loss_types = []
            for name, loss_setting in disp_loss.items():
                setattr(self, name, build_loss(loss_setting))
                self.loss_names.append(name)
                self.loss_types.append(loss_setting['type'])

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_flow = torch.cat((_x, _y), dim=1).to(fmap.device)
        return zero_flow

    def forward(self, fmaps1, fmaps2, flow_init=None, test_mode=False):
        # feature
        fmap1_16 = fmaps1[0]
        fmap2_16 = fmaps2[0]

        fmap1_8 = fmaps1[1]
        fmap2_8 = fmaps2[1]

        # context
        net_16, inp_16 = torch.split(fmap1_16, [self.h_channels, self.h_channels], dim=1)
        net_16 = torch.tanh(net_16)
        inp_16 = F.relu(inp_16)
        
        net_8, inp_8 = torch.split(fmap1_8, [self.h_channels, self.h_channels], dim=1)
        net_8 = torch.tanh(net_8)
        inp_8 = F.relu(inp_8)

        psize = (1, 9)
        dilate = (1, 1)
        # Cascaded refinement (1/16 + 1/8)
        predictions = []
        flow = None
        flow_up = None
        if flow_init is not None:
            scale = fmap1_8.shape[2] / flow_init.shape[2]
            flow = -scale * F.interpolate(
                flow_init,
                size=(fmap1_8.shape[2], fmap1_8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            # zero initialization
            flow_16 = self.zero_init(fmap1_16)

            # Recurrent Update Module
            # RUM: 1/16
            # for itr in range(self.iters // 2):
            for itr in range(5):
                flow_16 = flow_16.detach()
                out_corrs = corr_iter(fmap1_16, fmap2_16, flow_16, psize, dilate)

                net_16, up_mask, delta_flow = self.update_block_16(
                    net_16, inp_16, out_corrs, flow_16, output_all=True)

                flow_16 = flow_16 + delta_flow
                flow = convex_upsample(flow_16, up_mask, rate=8)
                flow_up = -2 * F.interpolate(
                    flow,
                    size=(2 * flow.shape[2], 2 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                disp_up = flow_up[:, :1]
                predictions.append(disp_up)

            scale = fmap1_8.shape[2] / flow.shape[2]
            flow = scale * F.interpolate(
                flow,
                size=(fmap1_8.shape[2], fmap1_8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

        # RUM: 1/8
        for itr in range(self.iters):
            flow = flow.detach()
            out_corrs = corr_iter(fmap1_8, fmap2_8, flow, psize, dilate)

            net_8, up_mask, delta_flow = self.update_block_8(net_8, inp_8, out_corrs, flow, output_all=True)

            flow = flow + delta_flow
            flow_up = -convex_upsample(flow, up_mask, rate=8)
            flow_up = flow_up[:, :1]
            predictions.append(flow_up)

        if test_mode:
            # return predictions
            # flow_up = -1 * flow[:, :1]
            return flow_up

        return predictions

    def forward_train(self,
                      fmaps1: torch.Tensor,
                      fmaps2: torch.Tensor,
                      flow_init: torch.Tensor,
                      test_mode: bool,
                      return_preds: bool,
                      disp_gt: torch.Tensor,
                      valid: Optional[torch.Tensor] = None,
                      *args, **kwargs) -> Dict[str, torch.Tensor]:
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
        if self.frozen_features:
            fmaps1 = [fmap.detach() for fmap in fmaps1]
            fmaps2 = [fmap.detach() for fmap in fmaps2]
        disp_pred = self.forward(fmaps1, fmaps2, flow_init, test_mode)

        if return_preds:
            self.losses(disp_pred, disp_gt, valid=valid, *args, **kwargs), disp_pred
        else:
            return self.losses(disp_pred, disp_gt, valid=valid, *args, **kwargs)

    def forward_test(self,
                     fmaps1: torch.Tensor,
                     fmaps2: torch.Tensor,
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
        disp_pred = self.forward(fmaps1, fmaps2, flow_init, test_mode)
        disp_result = disp_pred.permute(0, 2, 3, 1).cpu().data.numpy()
        #disp_result = [d.permute(0, 2, 3, 1).cpu().data.numpy() for d in disp_pred]
        # unravel batch dim
        disp_result = list(disp_result)
        disp_result = [dict(disp=f) for f in disp_result]
        #
        # disp_preds = self.forward(fmap1, fmap2, flow_init, test_mode)
        # disp_results = []
        # for disp_pred in disp_preds:
        #     disp = disp_pred.permute(0, 2, 3, 1).cpu().data.numpy()
        #     disp_results.append(disp)
        # disp_result = [dict(disp=f) for f in disp_results]

        return self.get_disp(disp_result, img_metas=img_metas)

    def forward_onnx_feat(self,
                          fmap1: torch.Tensor,
                          fmap2: torch.Tensor):
        """Forward function part 1.

        Args:
            fmap1 (Tensor): The feature from the left input image.
            fmap2 (Tensor): The feature from the right input image.
        """
        # context
        fmap1_dw16 = F.avg_pool2d(fmap1, 2, stride=2, padding=0, count_include_pad=False)
        fmap2_dw16 = F.avg_pool2d(fmap2, 2, stride=2, padding=0, count_include_pad=False)

        # context
        net, inp = torch.split(fmap1, [self.h_channels, self.h_channels], dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)
        net_dw16 = F.avg_pool2d(net, 2, stride=2, padding=0, count_include_pad=False)
        inp_dw16 = F.avg_pool2d(inp, 2, stride=2, padding=0, count_include_pad=False)


        return fmap1, fmap2, net, inp, fmap1_dw16, fmap2_dw16, net_dw16, inp_dw16

    def forward_onnx_dw16corr_update(self,
                                     fmap1_dw16: torch.Tensor,
                                     fmap2_dw16: torch.Tensor,
                                     net_dw16: torch.Tensor,
                                     inp_dw16: torch.Tensor,
                                     flow_dw16: torch.Tensor,
                                     coords0_dw16: torch.Tensor):
        """Forward function part 3.

        Args:
            fmap1_dw8 (Tensor): The small resolution feature from the left input image.
            fmap2_dw8 (Tensor): The small resolution feature from the right input image.
        """
        dw16_coords1 = coords0_dw16 + flow_dw16
        out_corrs = onnx_corr_iter(fmap1_dw16, fmap2_dw16, dw16_coords1)
        net_dw16, delta_flow = self.update_block(
            net_dw16, inp_dw16, out_corrs, flow_dw16, mask_flag=False)

        flow_dw16 = flow_dw16 + delta_flow

        return flow_dw16, net_dw16

    def forward_onnx_dw16flow_upsample(self, flow_dw16: torch.Tensor):
        scale = 2
        up_flow = -scale * F.interpolate(
            flow_dw16,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        return up_flow

    def forward_onnx_dw8corr_update(self,
                                    fmap1: torch.Tensor,
                                    fmap2: torch.Tensor,
                                    net: torch.Tensor,
                                    inp: torch.Tensor,
                                    flow: torch.Tensor,
                                    coords0: torch.Tensor):
        """Forward function part 4.

        Args:
            fmap1 (Tensor): The feature from the left input image.
            fmap2 (Tensor): The feature from the right input image.
        """
        for itr in range(self.iters):
            dw8_coords1 = coords0 + flow
            out_corrs = onnx_corr_iter(fmap1, fmap2, dw8_coords1)
            net, delta_flow = self.update_block(net, inp, out_corrs, flow, mask_flag=False)

            flow = flow + delta_flow

        # mask = self.update_block(net, mask_flag=True)
        return flow, net

    def forward_onnx_gridsample(self, fmap1: torch.Tensor):
        coords0 = coords_grid(fmap1_8.shape[0], fmap1_8.shape[2], fmap1_8.shape[3], fmap1_8.device)
        coords1 = coords0
        _, _, H, W = fmap1_8.shape
        paded_coords1 = torch.cat((coords1[:, :, :, 0:1].repeat(1, 1, 1, 4),
                                   coords1,
                                   coords1[:, :, :, W - 1:].repeat(1, 1, 1, 4)), 3)

        coords = paded_coords1.permute(0, 2, 3, 1)
        xgrid, ygrid = coords.split([1, 1], dim=3)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=3)
        warped_right_feature = F.grid_sample(fmap1, grid, align_corners=True)
        return warped_right_feature

    def forward_onnx_generatecorrs(self, fmap1: torch.Tensor, warped_fmap2: torch.Tensor):
        _, _, H, W = fmap1_8.shape
        corr_list = []
        for h in range(0, 0 * 2 + 1, 1):
            for w in range(0, 4 * 2 + 1, 1):
                right_crop = warped_fmap2[:, :, h: h + H, w: w + W]
                corr = torch.mean(fmap1 * right_crop, dim=1, keepdims=True)
                corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)
        return corr_final

    def forward_onnx_updateblock(self, net: torch.Tensor, inp: torch.Tensor,
                                 out_corrs: torch.Tensor, flow: torch.Tensor):
        net, delta_flow = self.update_block(net, inp, out_corrs, flow, mask_flag=False)
        return net, delta_flow

    def forward_onnx_mask(self, net: torch.Tensor):
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the left input image.
            feat2 (Tensor): The feature from the right input image.
        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted disparity
                with the same size of images before augmentation.
        """
        mask = self.update_block(net, mask_flag=True)
        return mask

    def forward_onnx_rumstereo(self,
                               fmap1: torch.Tensor,
                               fmap2: torch.Tensor):
        # context
        fmap1_dw8 = F.avg_pool2d(fmap1, 2, stride=2, padding=0, count_include_pad=False)
        fmap2_dw8 = F.avg_pool2d(fmap2, 2, stride=2, padding=0, count_include_pad=False)

        # context
        # net, inp = torch.split(fmap1, [self.h_channels, self.h_channels], dim=1)
        net = fmap1[:, 0:self.h_channels, :, :]
        inp = fmap1[:, self.h_channels:(self.h_channels + self.h_channels), :, :]
        net = torch.tanh(net)
        inp = F.relu(inp)
        net_dw8 = F.avg_pool2d(net, 2, stride=2, padding=0, count_include_pad=False)
        inp_dw8 = F.avg_pool2d(inp, 2, stride=2, padding=0, count_include_pad=False)

        flow_dw16 = self.zero_init(fmap1_dw8)
        dw16_coords0 = coords_grid(1, fmap1_dw8.shape[2], fmap1_dw8.shape[3], fmap1_dw8.device)

        for i in range(5):
            dw16_coords1 = dw16_coords0 + flow_dw16
            out_corrs = onnx_corr_iter(fmap1_dw8, fmap2_dw8, dw16_coords1)
            net_dw8, delta_flow = self.update_block(
                net_dw8, inp_dw8, out_corrs, flow_dw16, mask_flag=False)

            flow_dw16 = flow_dw16 + delta_flow

        # return flow_dw16, net_dw8
        scale = 2
        flow = scale * F.interpolate(
            flow_dw16,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

        dw8_coords0 = coords_grid(fmap1_8.shape[0], fmap1_8.shape[2], fmap1_8.shape[3], fmap1_8.device)
        for itr in range(self.iters):
            dw8_coords1 = dw8_coords0 + flow
            out_corrs = onnx_corr_iter(fmap1, fmap2, dw8_coords1)
            net, delta_flow = self.update_block(net, inp, out_corrs, flow, mask_flag=False)

            flow = flow + delta_flow

        return flow, net

    def losses(self,
               disp_pred: Sequence[torch.Tensor],
               disp_gt: torch.Tensor,
               valid: torch.Tensor = None, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute disparity loss.

        Args:
            disp_pred (Sequence[Tensor]): The list of predicted disparity.
            disp_gt (Tensor): The ground truth of disparity.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """

        loss = dict()
        for loss_name, loss_type in zip(self.loss_names, self.loss_types):
            loss_function = getattr(self, loss_name)
            if loss_type == 'SequenceFSLoss':
                loss[loss_name] = loss_function(disp_pred, *args, **kwargs)
            else:
                loss[loss_name] = loss_function(disp_pred, disp_gt, valid)
        return loss
