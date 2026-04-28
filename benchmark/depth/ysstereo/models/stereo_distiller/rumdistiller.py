from typing import Dict, Optional, Sequence, Tuple
import cv2

import torch
import numpy as np

from ysstereo.registry import MODELS
from ysstereo.models.stereo_estimators.rumstereo import RumStereo
from ysstereo.utils.misc import colorMap

@MODELS.register_module()
class RumDistiller(RumStereo):
    """RumDistiller model.

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

        self.distill_feat_weight = distill.get("distill_feat_weight", 0)
        self.distill_preds_weight = distill.get("distill_preds_weight", 0)

        self.teacher.eval()
        for m in self.teacher.modules():
            for param in m.parameters():
                param.requires_grad = False
        self.distill_warm_step = distill.distill_warm_step
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
        Returns:
            Dict[str, Tensor]: The losses of output.
        """
        vis = torch.cat([imgs[:,:3,:], imgs[:,3:,:]], axis=3)
        vis_in = colorMap('inferno', vis[0][0].detach())
        vis_gt = colorMap('inferno', disp_gt[0][0].detach())
        vis_in = np.concatenate([vis_in, vis_gt], axis=1)
        cv2.imwrite('temp/vis_cre_inputs.png', vis_in)

        imgl = imgs[:,:3,:]
        imgr = imgs[:,3:,:]
        imgs = torch.cat([imgl, imgr], 0)

        fmap = self.encoder(imgs)
        fmap1, fmap2 = torch.split(fmap, fmap.shape[0]//2, dim=0)

        if self.distill_preds_weight > 0:
            losses, preds =  self.decoder.forward_train(fmap1, fmap2, flow_init=flow_init, test_mode=False, 
                                                        return_preds=True, disp_gt=disp_gt, valid=valid)
        else:
            losses =  self.decoder.forward_train(fmap1, fmap2, flow_init=flow_init, test_mode=False, 
                                                 return_preds=False, disp_gt=disp_gt, valid=valid)
        distill_feat_loss = 0
        t_fmap = self.teacher.encoder(imgs) 

        if self.distill_feat_weight > 0 :
            distill_feat_loss = torch.pow((t_fmap - fmap), 2).mean()
            distill_feat_loss = distill_feat_loss * self.distill_feat_weight

            if self.debug:
                # if self._inner_iter == 10:
                #     breakpoint()
                print(self._inner_iter, distill_feat_loss)
            
            if self.distill_warm_step > self.iter:
                distill_feat_loss = (self.iter / self.distill_warm_step) * distill_feat_loss
            
            if self.distill_feat_weight:
                losses.update({"distill_feat_loss": distill_feat_loss})


        if self.distill_preds_weight > 0:
            t_fmap1, t_fmap2 = torch.split(t_fmap, t_fmap.shape[0]//2, dim=0)
            _, t_preds =  self.teacher.decoder.forward_train(t_fmap1, t_fmap2, flow_init=flow_init, test_mode=False,  
                                                             return_preds=True, disp_gt=disp_gt, valid=valid)
            
            num_s_preds = len(preds)
            num_t_preds = len(t_preds)
            if num_s_preds>num_t_preds:
                t_preds = (num_s_preds-num_t_preds)*[t_preds[0]] + t_preds
            if num_t_preds>num_s_preds:
                t_preds = t_preds[-num_s_preds:]

            distill_preds_loss = 0
            for s, t in zip(preds, t_preds):
                distill_preds_loss += (s-t).abs().mean()
            
            distill_preds_loss = distill_preds_loss*self.distill_preds_weight
            
            if self.debug:
                # if self._inner_iter == 10:
                #     breakpoint()
                print(self._inner_iter, distill_preds_loss)
        
            if self.distill_warm_step > self.iter:
                distill_preds_loss = (self.iter / self.distill_warm_step) * distill_preds_loss
            
            losses.update({"distill_preds_loss": distill_preds_loss})

        return losses

