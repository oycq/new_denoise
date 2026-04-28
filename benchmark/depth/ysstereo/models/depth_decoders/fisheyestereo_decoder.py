import math
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ysstereo.ops import build_operators
from ysstereo.models.builder import DECODERS, build_loss
from ysstereo.models.depth_decoders.base_decoder import BaseDecoder
from ysstereo.models.depth_decoders.decoder_submodules import FisheyeCorrBlock1D, MotionEncoder, ConvGRU, XHead

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

@DECODERS.register_module()
class FisheyeStereoDecoder(BaseDecoder):
    """The decoder of FisheyeRAFTStereo Net.

    The decoder of Fisheye RAFT Net, which outputs list of upsampled stereo estimation.

    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        num_levels (int): Number of levels used when calculating
            correlation tensor.
        radius (int): Radius used when calculating correlation tensor.
        iters (int): Total iteration number of iterative update of RAFTDecoder.
        corr_op_cfg (dict): Config dict of correlation operator.
            Default: dict(type='CorrLookup').
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
        net_type: str,
        num_levels: int,
        num_downsamples: int,
        radius: int,
        iters: int,
        num_layers: int,
        corr_op_cfg: dict = dict(type='CorrLookup', align_corners=True),
        gru_type: str = 'SeqConv',
        hidden_channels: int = 16,
        update_channels: Optional[Sequence[int]] = None,
        feat_channels: Union[int, Sequence[int]] = 32,
        mask_channels: int = 64,
        valid_height: int = 161,
        valid_width: int = 83,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        disp_loss: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert net_type in ['Basic', 'Small']
        assert type(feat_channels) in (int, tuple, list)
        self.corr_block = FisheyeCorrBlock1D(num_levels=num_levels)
        self.num_downsamples = num_downsamples

        feat_channels = feat_channels if isinstance(tuple,
                                                    list) else [feat_channels]
        self.net_type = net_type
        self.num_levels = num_levels
        self.radius = radius
        self.num_layers = num_layers
        self.h_channels = hidden_channels
        self.cxt_channels = update_channels
        self.iters = iters
        self.mask_channels = mask_channels * (2 * radius + 1)
        self.valid_h = valid_height
        self.valid_w = valid_width
        corr_op_cfg['radius'] = radius
        self.corr_lookup = build_operators(corr_op_cfg)
        self.encoder = MotionEncoder(
            num_levels=num_levels,
            radius=radius,
            corr_channels=[self.h_channels, self.h_channels],
            flow_channels=[self.h_channels, self.h_channels],
            out_channels=self.cxt_channels[1]-2,
            net_type=net_type,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gru_type = gru_type
        # self.gru = self.make_gru_block()

        self.gru08 = self.make_gru08_block(num_layers)
        self.gru16 = self.make_gru16_block(num_layers)
        self.gru32 = self.make_gru32_block(num_layers)

        self.flow_pred = XHead(self.h_channels, feat_channels, 2, x='flow')

        if net_type == 'Basic':
            self.mask_pred = XHead(
                self.h_channels, feat_channels, self.mask_channels, x='mask')

        if disp_loss is not None:
            self.disp_loss = build_loss(disp_loss)

    def make_gru08_block(self, num_layers):
        return ConvGRU(
            self.h_channels,
            self.cxt_channels[1] + self.h_channels * (num_layers > 1),
            net_type=self.gru_type)

    def make_gru16_block(self, num_layers):
        return ConvGRU(
            self.h_channels,
            self.h_channels + self.h_channels * (num_layers == 3),
            net_type=self.gru_type)

    def make_gru32_block(self, num_layers):
        return ConvGRU(
            self.h_channels,
            self.h_channels,
            net_type=self.gru_type)

    def pool2x(self, x):
        return F.avg_pool2d(x, 3, stride=2, padding=1)

    def pool4x(self, x):
        return F.avg_pool2d(x, 5, stride=4, padding=1)

    def interp(self, x, dest):
        interp_args = {'mode': 'bilinear', 'align_corners': True}
        return F.interpolate(x, dest.shape[2:], **interp_args)

    def initialize_flow(self, flow):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = flow.shape

        # coords0 = torch.zeros(N, 2, self.valid_h, self.valid_w).to(flow.device)
        # coords1 = torch.zeros(N, 2, self.valid_h, self.valid_w).to(flow.device)

        coords0 = coords_grid(N, self.valid_h, self.valid_w).to(flow.device)
        coords1 = coords_grid(N, self.valid_h, self.valid_w).to(flow.device)

        return coords0, coords1

    def _upsample(self,
                  flow: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex
        combination.

        Args:
            flow (Tensor): The disparity with the shape [N, 2, H/8, W/8].
            mask (Tensor, optional): The leanable mask with shape
                [N, grid_size x scale x scale, H/8, H/8].

        Returns:
            Tensor: The output disparity with the shape [N, 2, H, W].
        """
        scale = 2**self.num_downsamples
        grid_size = self.radius * 2 + 1
        grid_side = int(math.sqrt(grid_size))
        N, D, H, W = flow.shape
        if mask is None:
            new_size = (scale * H, scale * W)
            return scale * F.interpolate(
                flow, size=new_size, mode='bilinear', align_corners=True)
        # predict a (Nx8×8×9xHxW) mask
        mask = mask.view(N, 1, grid_size, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        # extract local grid with 3x3 side  padding = grid_side//2
        upflow = F.unfold(scale * flow, [grid_side, grid_side], padding=1)
        # upflow with shape N, 2, 9, 1, 1, H, W
        upflow = upflow.view(N, D, grid_size, 1, 1, H, W)

        # take a weighted combination over the neighborhood grid 3x3
        # upflow with shape N, 2, 8, 8, H, W
        upflow = torch.sum(mask * upflow, dim=2)
        upflow = upflow.permute(0, 1, 4, 2, 5, 3)
        return upflow.reshape(N, D, scale * H, scale * W)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor,
                valid_grids: torch.Tensor, inv_valid_grids: torch.Tensor,
                corr_grids: torch.Tensor, flow_init: torch.Tensor,
                net_list: Sequence[torch.Tensor], inp_list: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """Forward function for RAFTDecoder.

        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            valid_grids (Tensor): The valid fisheye grids.
            inv_valid_grids (Tensor): The inverse valid fisheye grids.
            corr_grids (Tensor): The correlation of right fisheye grids.
            flow (Tensor): The initialized flow when warm start.
            net_list (Tensor): The hidden state for GRU cell.
            inp_list (Tensor): The contextual feature from the left image.

        Returns:
            Sequence[Tensor]: The list of predicted disparity.
        """

        corr_pyramid = self.corr_block(feat1, feat2, valid_grids, corr_grids)
        upflow_preds = []

        coords0, coords1 = self.initialize_flow(net_list[0])
        if flow_init is not None:
            coords1 = coords1 + flow_init

        for _ in range(self.iters):
            coords1 = coords1.detach()
            valid_corr = self.corr_lookup(corr_pyramid, coords1)
            valid_flow = coords1 - coords0

            N,C,_,_ = valid_corr.shape
            valid_corr = valid_corr.reshape(N, C, self.valid_h, self.valid_w)
            flow = F.grid_sample(valid_flow, inv_valid_grids, align_corners=True)
            corr = F.grid_sample(valid_corr, inv_valid_grids, align_corners=True)

            if self.num_layers == 3:
                net_list[2] = self.gru32(net_list[2], *(inp_list[2]), self.pool2x(net_list[1]))
            if self.num_layers > 2:
                net_list[1] = self.gru16(net_list[1], *(inp_list[1]), self.pool2x(net_list[0]), self.interp(net_list[2], net_list[1]))
            if self.num_layers == 2:
                net_list[1] = self.gru16(net_list[1], *(inp_list[1]), self.pool2x(net_list[0]))
            motion_feat = self.encoder(corr, flow)
            if self.num_layers > 1:
                net_list[0] = self.gru08(net_list[0], *(inp_list[0]), motion_feat, self.interp(net_list[1], net_list[0]))
            else:
                net_list[0] = self.gru08(net_list[0], *(inp_list[0]), motion_feat)

            delta_flow = self.flow_pred(net_list[0])
            delta_flow[:,1] = 0.0

            valid_delta_flow = F.grid_sample(delta_flow, valid_grids, align_corners=True)
            coords1 = coords1 + valid_delta_flow

            valid_delta_coords = coords1 - coords0
            delta_coords = F.grid_sample(valid_delta_coords, inv_valid_grids, align_corners=True)

            if hasattr(self, 'mask_pred'):
                mask = .25 * self.mask_pred(net_list[0])
            else:
                mask = None

            upflow = self._upsample(delta_coords, mask)
            upflow = upflow[:, :1]
            upflow_preds.append(upflow)

        return upflow_preds

    def forward_train(
            self,
            feat1: torch.Tensor,
            feat2: torch.Tensor,
            valid_grids: torch.Tensor,
            inv_valid_grids: torch.Tensor,
            corr_grids: torch.Tensor, 
            flow: torch.Tensor,
            net_list: Sequence[torch.Tensor],
            inp_list: Sequence[torch.Tensor],
            disp_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the left input image.
            feat2 (Tensor): The feature from the right input image.
            valid_grids (Tensor): The valid fisheye grids.
            inv_valid_grids (Tensor): The inverse valid fisheye grids.
            corr_grids (Tensor): The correlation of right fisheye grids.
            flow (Tensor): The last estimated flow from GRU cell.
            net_list (Tensor): The hidden state for GRU cell.
            inp_list (Tensor): The contextual feature from the left image.
            disp_gt (Tensor): The ground truth of disparity.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of model.
        """

        flow_pred = self.forward(feat1, feat2, valid_grids, inv_valid_grids, corr_grids,
            flow, net_list, inp_list)

        return self.losses(flow_pred, disp_gt, valid=valid)

    def forward_test(self,
                     feat1: torch.Tensor,
                     feat2: torch.Tensor,
                     valid_grids: torch.Tensor,
                     inv_valid_grids: torch.Tensor,
                     corr_grids: torch.Tensor, 
                     flow: torch.Tensor,
                     net_list: Sequence[torch.Tensor],
                     inp_list: Sequence[torch.Tensor],
                     img_metas=None) -> Sequence[Dict[str, np.ndarray]]:
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the left input image.
            feat2 (Tensor): The feature from the right input image.
            valid_grids (Tensor): The valid fisheye grids.
            inv_valid_grids (Tensor): The inverse valid fisheye grids.
            corr_grids (Tensor): The correlation of right fisheye grids.
            flow (Tensor): The last estimated flow from GRU cell.
            net_list (Tensor): The hidden state for GRU cell.
            inp_list (Tensor): The contextual feature from the left image.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted disparity
                with the same size of images before augmentation.
        """
        flow_pred = self.forward(feat1, feat2, valid_grids, inv_valid_grids, corr_grids,
            flow, net_list, inp_list)

        flow_result = flow_pred[-1]
        # flow maps with the shape [H, W, 2]
        flow_result = flow_result.permute(0, 2, 3, 1).cpu().data.numpy()
        # unravel batch dim
        flow_result = list(flow_result)
        flow_result = [dict(disp=f) for f in flow_result]

        return self.get_disp(flow_result, img_metas=img_metas)

    def forward_onnx_feat_corr(self,
                     feat1: torch.Tensor,
                     feat2: torch.Tensor,
                     valid_grids: torch.Tensor,
                     corr_grids: torch.Tensor):
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the left input image.
            feat2 (Tensor): The feature from the right input image.
            valid_grids (Tensor): The valid fisheye grids.
            inv_valid_grids (Tensor): The inverse valid fisheye grids.
            corr_grids (Tensor): The correlation of right fisheye grids.
        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted disparity
                with the same size of images before augmentation.
        """
        corr_pyramid = self.corr_block(feat1, feat2, valid_grids, corr_grids)
        return corr_pyramid

    def forward_onnx_update(self,
                     corr: torch.Tensor,
                     flow: torch.Tensor,
                     net_list: Sequence[torch.Tensor],
                     inp_array: Sequence[torch.Tensor]):
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the left input image.
            feat2 (Tensor): The feature from the right input image.
        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted disparity
                with the same size of images before augmentation.
        """
        inp_list = []
        for i in range(len(inp_array)):
            tmp_inp = list(torch.split(inp_array[i], 1, dim=0))
            inp_list.append(tmp_inp)
        if self.num_layers == 3:
            net_list[2] = self.gru32(net_list[2], *(inp_list[2]), self.pool2x(net_list[1]))
        if self.num_layers > 2:
            net_list[1] = self.gru16(net_list[1], *(inp_list[1]), self.pool2x(net_list[0]), self.interp(net_list[2], net_list[1]))
        if self.num_layers == 2:
            net_list[1] = self.gru16(net_list[1], *(inp_list[1]), self.pool2x(net_list[0]))
        motion_feat = self.encoder(corr, flow)
        if self.num_layers > 1:
            net_list[0] = self.gru08(net_list[0], *(inp_list[0]), motion_feat, self.interp(net_list[1], net_list[0]))
        else:
            net_list[0] = self.gru08(net_list[0], *(inp_list[0]), motion_feat)

        delta_flow = self.flow_pred(net_list[0])
        return net_list, delta_flow

    def forward_onnx_mask(self,
                     net_list0: torch.Tensor):
        """Forward function when model training.

        Args:
            feat1 (Tensor): The feature from the left input image.
            feat2 (Tensor): The feature from the right input image.
        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted disparity
                with the same size of images before augmentation.
        """
        mask = .25 * self.mask_pred(net_list0)
        return mask

    def losses(self,
               flow_pred: Sequence[torch.Tensor],
               disp_gt: torch.Tensor,
               valid: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute disparity loss.

        Args:
            flow_pred (Sequence[Tensor]): The list of predicted disparity.
            disp_gt (Tensor): The ground truth of disparity.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """

        loss = dict()
        loss['loss_disp'] = self.disp_loss(flow_pred, disp_gt, valid)
        return loss