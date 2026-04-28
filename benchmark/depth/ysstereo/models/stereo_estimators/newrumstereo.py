from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from numpy import ndarray

from ysstereo.registry import MODELS
from ysstereo.models.builder import build_encoder, build_decoder
from ysstereo.models.stereo_estimators.base import StereoEstimator
from ysstereo.utils.misc import colorMap


@MODELS.register_module()
class NewRumStereo(StereoEstimator):
    """CREStereo model.

    Args:
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
        Default: False.
    """

    def __init__(self,
                 encoder: dict,
                 decoder: dict,
                 onnx_cfg: dict = None,
                 freeze_bn: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)
        self.in_channels = encoder.in_channels

        self.onnx_cfg = onnx_cfg
        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward_train(
            self,
            imgs: torch.Tensor,
            flow_init: Optional[torch.Tensor] = None,
            return_preds: bool = False,
            **data_samples) -> Dict[str, torch.Tensor]:
        """Forward function for RAFTStereo when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            disp_gt (Tensor): The ground truth of disparity.
                Defaults to None.
            valid (Tensor, optional): The valid mask. Defaults to None.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.
        Returns:
            Dict[str, Tensor]: The losses of output.
        
        vis = torch.cat([imgs[:,:self.in_channels,:], imgs[:,self.in_channels:,:]], axis=3)
        vis_in = colorMap('inferno', vis[0][0].detach())
        vis_gt = colorMap('inferno', disp_gt[0][0].detach())
        vis_in = np.concatenate([vis_in, vis_gt], axis=1)
        cv2.imwrite('temp/vis_cre_inputs.png', vis_in)
        """
        imgl = imgs[:, :self.in_channels, :]
        imgr = imgs[:, self.in_channels:, :]
        imgs = torch.cat([imgl, imgr], 0)

        fmaps = self.encoder(imgs)
        fmaps1, fmaps2 = [], []
        for fmap in fmaps:
            fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            fmaps1.append(fmap1)
            fmaps2.append(fmap2)

        return self.decoder.forward_train(
                        fmaps1,
                        fmaps2,
                        flow_init=flow_init,
                        test_mode=False,
                        return_preds=return_preds,
                        **data_samples)

    def forward_test(
            self,
            imgs: torch.Tensor,
            flow_init: Optional[torch.Tensor] = None,
            **data_samples) -> Sequence[ndarray]:
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
        assert len(imgs.shape) > 2
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
        imgl = imgs[:,:self.in_channels,:]
        imgr = imgs[:,self.in_channels:,:]
        imgs = torch.cat([imgl, imgr], 0)

        fmaps = self.encoder(imgs)
        fmaps1, fmaps2 = [], []
        for fmap in fmaps:
            fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            fmaps1.append(fmap1)
            fmaps2.append(fmap2)

        results = self.decoder.forward_test(
            fmaps1=fmaps1,
            fmaps2=fmaps2,
            flow_init=flow_init,
            test_mode=True,
            **data_samples)
        # recover iter in train
        self.decoder.iters = train_iter

        return results

    def forward_onnx(self, sub_type, imgs = None, *args, **kwargs):
        """Forward function for RumStereo when model testing.

        Args:
            imgs (Tensor): The concatenated input images.
        """
        onnx_submodule = {
            'feat_submodule': self.decoder.forward_onnx_feat,
            # 'transform_submodule': self.decoder.forward_onnx_feat,
            'dw16corr_update_submodule': self.decoder.forward_onnx_dw16corr_update,
            'dw16flow_upsample_submodule': self.decoder.forward_onnx_dw16flow_upsample,
            'dw8corr_update_submodule': self.decoder.forward_onnx_dw8corr_update,
            'mask_submodule': self.decoder.forward_onnx_mask,
            # 'gridsample_submodule': self.decoder.forward_onnx_gridsample,
            # 'generatecorrs_submodule': self.decoder.forward_onnx_generatecorrs,
            # 'updateblock_submodule': self.decoder.forward_onnx_updateblock,
        }

        if sub_type == 'feat_submodule':
            # imgs = torch.cat([imgs[0], imgs[1]], dim=0)
            # fmap = self.encoder(imgs)
            # fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            left_input = 2 * (imgs[0] / 255.0) - 1.0
            right_input = 2 * (imgs[1] / 255.0) - 1.0

            fmap1 = self.encoder(left_input)
            fmap2 = self.encoder(right_input)
            results = self.decoder.forward_onnx_feat(fmap1, fmap2)
            # results = (fmap1, fmap2)
        elif sub_type == 'rumstereo':
            left_input = 2 * (imgs[0] / 255.0) - 1.0
            right_input = 2 * (imgs[1] / 255.0) - 1.0

            fmap1 = self.encoder(left_input)
            fmap2 = self.encoder(right_input)
            # images = torch.cat([imgs[0], imgs[1]], dim=0)
            # fmap = self.encoder(images)
            # fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            results = self.decoder.forward_onnx_rumstereo(fmap1, fmap2)
        else:
            results = onnx_submodule[sub_type](*args, **kwargs)
        return results
