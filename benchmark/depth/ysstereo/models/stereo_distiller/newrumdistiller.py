from typing import Dict, Optional, Sequence, Tuple
import cv2

import torch
import numpy as np

from ysstereo.registry import MODELS
from ysstereo.models.builder import build_loss, build_encoder, build_decoder
from ysstereo.models.stereo_estimators.newrumstereo import NewRumStereo
from ysstereo.utils.misc import colorMap

@MODELS.register_module()
class NewRumDistiller(NewRumStereo):
    """NewRumDistiller model.

    Args:
        distill (dict): teacher configs and pre-trained weights and distillation settings
        encoder (dict): student encoder configs
        decoder (dict): student decoder configs
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
        Default: False.
    """

    def __init__(self,
                 distill: dict,
                 encoder: dict,
                 decoder: dict,
                 onnx_cfg: dict = None,
                 freeze_bn: bool = False,
                 **kwargs) -> None:
        super().__init__(encoder, decoder, onnx_cfg, freeze_bn, **kwargs)

        # Initiaize teacher
        from ysstereo.apis.inference import init_model
        self.device = torch.cuda.current_device()
        self.teacher = init_model(distill.teacher_cfg,
                        distill.teacher_model_path, self.device)
        self.teacher._is_init = True

        self.teacher.eval()
        for m in self.teacher.modules():
            for param in m.parameters():
                param.requires_grad = False
        self.debug = distill.get("debug", False)

        # This can be done only when the decoders of teacher and student are the same. 
        if distill.get("copy_decoder_from_teacher_to_student", False):
            self.decoder.load_state_dict(self.teacher.decoder.state_dict())
            self.decoder._is_init = True

            if distill.get("fix_student_decoder", False):
                self.decoder.eval()
                for m in self.decoder.modules():
                    for param in m.parameters():
                        param.requires_grad = False

        if distill.loss is not None:
            self.loss_names = []
            self.loss_types = []
            for name, loss_setting in distill.loss.items():
                setattr(self, name, build_loss(loss_setting))
                self.loss_names.append(name)
                self.loss_types.append(loss_setting['type'])
        
    def forward_train(
            self,
            imgs: torch.Tensor,
            disp_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None,
            *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
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
        """
        vis = torch.cat([imgs[:,:self.in_channels,:], imgs[:,self.in_channels:,:]], axis=3)
        vis_in = colorMap('inferno', vis[0][0].detach())
        vis_gt = colorMap('inferno', disp_gt[0][0].detach())
        vis_in = np.concatenate([vis_in, vis_gt], axis=1)
        cv2.imwrite('temp/vis_cre_inputs.png', vis_in)

        imgl = imgs[:,:self.in_channels,:]
        imgr = imgs[:,self.in_channels:,:]
        imgs = torch.cat([imgl, imgr], 0)

        fmaps = self.encoder(imgs)
        fmaps1, fmaps2 = [], []
        for fmap in fmaps:
            fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)
            fmaps1.append(fmap1)
            fmaps2.append(fmap2)

        losses, preds =  self.decoder.forward_train(fmaps1, fmaps2, flow_init=flow_init, test_mode=False, 
                                                    return_preds=True, disp_gt=disp_gt, valid=valid, *args, **kwargs)
        
        t_fmaps = self.teacher.encoder(imgs)
        t_preds = None
        for loss_name, loss_type in zip(self.loss_names, self.loss_types):
            loss_function = getattr(self, loss_name)
            if 'loss_feat' in loss_name and loss_function.weight > 0:
                feat_loss = loss_function(fmaps, t_fmaps)
                losses.update({"distill_"+loss_name: feat_loss})
            if 'loss_pred' in loss_name and loss_function.weight > 0:
                if t_preds is None:
                    t_fmaps1, t_fmaps2 = [], []
                    for t_fmap in t_fmaps:
                        t_fmap1, t_fmap2 = torch.split(t_fmap, t_fmap.shape[0]//2, dim=0)
                        t_fmaps1.append(t_fmap1)
                        t_fmaps2.append(t_fmap2)
                    _, t_preds =  self.teacher.decoder.forward_train(t_fmaps1, t_fmaps2, flow_init=flow_init, test_mode=False,  
                                                                     return_preds=True, disp_gt=disp_gt, valid=valid)
                    num_s_preds = len(preds)
                    num_t_preds = len(t_preds)
                    if num_s_preds>num_t_preds:
                        t_preds = (num_s_preds-num_t_preds)*[t_preds[0]] + t_preds
                    if num_t_preds>num_s_preds:
                        t_preds = t_preds[-num_s_preds:]

                    t_preds = torch.cat(t_preds, dim=1)
                    preds = torch.cat(preds, dim=1)
                
                loss_function = getattr(self, loss_name)
                pred_loss = loss_function(preds, t_preds)
                losses.update({"distill_"+loss_name: pred_loss})

        return losses

