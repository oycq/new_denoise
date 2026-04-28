from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from numpy import ndarray

from ysstereo.registry import MODELS
from ysstereo.models.builder import build_encoder, build_decoder, build_components
from ysstereo.models.stereo_estimators.base import StereoEstimator

@MODELS.register_module()
class RAFTStereo(StereoEstimator):
    """RAFT model.

    Args:
        num_levels (int): Number of levels in .
        radius (int): Number of radius in  .
        cxt_channels (int): Number of channels of context feature.
        h_channels (int): Number of channels of hidden feature in .
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
            Default: False.
    """

    def __init__(self,
                 encoder: dict,
                 decoder: dict,
                 feat_component: dict,
                 num_levels: int,
                 radius: int,
                 h_channels: int,
                 num_layers: int,
                 onnx_cfg: dict,
                 freeze_bn: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_levels = num_levels
        self.radius = radius
        self.h_channels = h_channels

        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)
        self.conv2 = build_components(feat_component)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(h_channels, h_channels*3, 3, padding=1) for i in range(num_layers)])

        assert self.num_levels == self.decoder.num_levels
        assert self.radius == self.decoder.radius
        assert self.h_channels == self.decoder.h_channels
        # assert self.h_channels + self.cxt_channels == self.context.out_channels

        self.onnx_cfg = onnx_cfg
        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_feat(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Tensor, Tensor, list[Tensor], list[Tensor]]: The feature from the first
                image, the feature from the second image, the hidden state
                feature for GRU cell and the contextual feature.
        """
        B, C , H , W = imgs.shape
        imgl = imgs[:,:3,:]
        imgr = imgs[:,3:,:]
        imgs = torch.cat([imgl, imgr], 0)

        *cnet_list, x = self.encoder(imgs)
        feat1, feat2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        return feat1, feat2, net_list, inp_list

    def forward_train(
            self,
            imgs: torch.Tensor,
            disp_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward function for RAFTStereo when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            disp_gt (Tensor): The ground truth of disparity.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Dict[str, Tensor]: The losses of output.
        """
        feat1, feat2, net_list, inp_list = self.extract_feat(imgs)

        return self.decoder.forward_train(
            feat1,
            feat2,
            flow=flow_init,
            net_list=net_list,
            inp_list=inp_list,
            disp_gt=disp_gt,
            valid=valid)

    def forward_test(
            self,
            imgs: torch.Tensor,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None) -> Sequence[ndarray]:
        """Forward function for RAFTStereo when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the disparity to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the batch of predicted disparity
                with the same size of images after augmentation.
        """
        train_iter = self.decoder.iters
        if self.test_cfg is not None and self.test_cfg.get(
                'iters') is not None:
            self.decoder.iters = self.test_cfg.get('iters')

        feat1, feat2, net_list, inp_list = self.extract_feat(imgs)
        B, _, H, W = feat1.shape

        if flow_init is None:
            flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

        results = self.decoder.forward_test(
            feat1=feat1,
            feat2=feat2,
            flow=flow_init,
            net_list=net_list,
            inp_list=inp_list,
            img_metas=img_metas)
        # recover iter in train
        self.decoder.iters = train_iter

        return results

    def forward_onnx(self, sub_type, imgs = None, *args, **kwargs):
        """Forward function for RAFTStereo when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
        """
        onnx_submodule = {
            'feat_corr_submodule': self.decoder.forward_onnx_feat_corr,
            'update_submodule': self.decoder.forward_onnx_update,
            'mask_submodule': self.decoder.forward_onnx_mask
        }

        if sub_type == 'feat_corr_submodule':
            imgs = torch.cat([imgs[0], imgs[1]], dim=1)
            feat1, feat2, net_list, inp_list = self.extract_feat(imgs)
            results = self.decoder.forward_onnx_feat_corr(feat1, feat2)
            results.append(net_list)
            inp_array = []
            inp_array.append(torch.cat([inp_list[0][0], inp_list[0][1], inp_list[0][2]], dim = 0))
            inp_array.append(torch.cat([inp_list[1][0], inp_list[1][1], inp_list[1][2]], dim = 0))
            results.append(inp_array)
        else:
            results = onnx_submodule[sub_type](*args, **kwargs)
        return results