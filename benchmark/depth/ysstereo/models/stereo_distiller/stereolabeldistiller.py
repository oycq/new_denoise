from typing import Dict, Optional, Sequence, Tuple
import cv2
from functools import partial
from collections import OrderedDict

import torch, copy
from torch.amp import autocast
import numpy as np
import torch.nn.functional as F
from ysstereo.registry import MODELS
from ysstereo.models.builder import build_loss, build_encoder, build_decoder
from ysstereo.models.stereo_estimators.newrumstereo import NewRumStereo
from ysstereo.models.stereo_estimators.base import StereoEstimator
from ysstereo.models.stereo_estimators.sepcxt_stereo import SepCxtNewRumStereo
from ysstereo.models.depth_decoders import IGEVStereoSlimDecoder
from ysstereo.models.seg_decoders import FCNHead
from ysstereo.models.depth_losses import MAELoss, CWDLoss
from ysstereo.utils.misc import colorMap
from ysstereo.models.depth_losses.unloss import image_warping_loss, coords_grid
from ysstereo.models.depth_losses.smooth_loss import smooth_1st_loss
from ysstereo.datasets import visualize_disp, visualize_depth, write_pfm

def error_rate_loss(pred_disp, disp_gt, valid):
    disp_gt = disp_gt.squeeze(1)
    pred_disp = pred_disp.squeeze(1)
    valid = valid.squeeze(1)
    disp_gt = disp_gt[valid.bool()]
    pred_disp = pred_disp[valid.bool()]
    # correct rate definition
    #  1. depth in 10m range: min(disp_error<0.4px, depth_error<0.1 * depth_gt)
    #  2. depth in 10m ~ 25m range: depth ratio error < 0.1~0.16
    #  3. depth > 25m range: make sure depth > 25m
    correct_num = 0
    disp_gt = torch.abs(disp_gt).clamp(min=0.01)
    pred_disp = torch.abs(pred_disp).clamp(min=0.01)
    pred_depth = 27.0 / pred_disp
    depth_gt = 27.0 / disp_gt
    mask_1 = depth_gt < 10.0
    valid_1 = mask_1 & (torch.abs(disp_gt - pred_disp) < 0.4) & (torch.abs(depth_gt - pred_depth) < 0.1 * depth_gt)
    correct_num += valid_1.sum()
    mask_2 = (depth_gt >= 10.0) & (depth_gt < 25.0)
    depth_ratio = torch.abs(depth_gt - pred_depth) / depth_gt
    thres = 0.1 + 0.06 * (depth_gt - 10.0) / 15.0
    valid_2 = mask_2 & (depth_ratio < thres)
    correct_num += valid_2.sum()
    mask_3 = depth_gt >= 25.0
    valid_3 = mask_3 & (pred_depth > 25.0)
    correct_num += valid_3.sum()
    return 1.0 - correct_num.float() / valid.sum() # error rate



def multiscale_warp_loss(left_image, right_image, disp_pred, ssim_weight:float=0.85, scale:int=4):
    if left_image.min() < 0:
        left_image = (left_image + 1.0) / 2.0
        right_image = (right_image + 1.0) / 2.0
    elif left_image.max() > 1.0:
        left_image = left_image / 255.0
        right_image = right_image / 255.0

    loss = 0.0
    H, W = left_image.shape[2:]
    for s in range(scale):
        if s > 0:
            times = (2 ** s)
            ratio = 1.0 / times
            li = F.interpolate(left_image, size=[H//times, W//times], scale_factor=None, mode='bilinear', align_corners=True)
            ri = F.interpolate(right_image, size=[H//times, W//times], scale_factor=None, mode='bilinear', align_corners=True)
            d = ratio * F.interpolate(disp_pred, size=[H//times, W//times], scale_factor=None, mode='bilinear', align_corners=True)
        else:
            ratio, li, ri, d = 1.0, left_image, right_image, disp_pred
        loss += ratio * image_warping_loss(li, ri, d, ssim_weight)

    return loss

def disp_gt_guide_smooth(disp_gt:torch.Tensor, pred_disp:torch.Tensor):
    assert disp_gt.ndim == 4 and pred_disp.ndim == 4
    # convert disp_gt into 3 channel rgb image
    b, c, h, w = disp_gt.shape
    disp_max, _ = torch.max(disp_gt.view(b, -1), dim=1)
    disp_max = disp_max.view(b, 1, 1, 1) + 1e-6
    disp_gt = disp_gt / (disp_max * 0.5)
    disp_gt_rgb = torch.cat([disp_gt]*3, dim=1)
    # use smooth loss
    smooth_loss = smooth_1st_loss(pred_disp, disp_gt_rgb, alpha=7.5)
    return smooth_loss

def rgb_guide_smooth(rgb:torch.Tensor, pred_disp:torch.Tensor):
    # rgb tensor is already normalized
    smooth_loss = smooth_1st_loss(pred_disp, rgb, alpha=7.0)
    return smooth_loss

@torch.no_grad()
def resize_batch(disps, imgs, tgt_h, tgt_w, psd_gts=None, max_ratio:float = 1.0):
    b, _, h, w = disps.shape
    min_ratio_h = tgt_h/h
    min_ratio_w = tgt_w/w
    min_ratio = max(min_ratio_h, min_ratio_w)
    new_disps, new_imgs, new_psd_gts = [],[],[]
    for i in range(b):
        # resize
        ratio = np.random.uniform(min_ratio, max_ratio)
        new_h, new_w = int(round(h*ratio)), int(round(w*ratio))
        new_h, new_w = max(new_h, tgt_h), max(new_w, tgt_w)
        cur_disp = (new_w/w) * F.interpolate(disps[i:(i+1)], size=(new_h, new_w), mode='nearest')
        cur_img = F.interpolate(imgs[i:(i+1)], size=(new_h, new_w), mode='bilinear', align_corners=False)
        if psd_gts is not None:
            cur_psd_gt = F.interpolate(psd_gts[i:(i+1)], size=(new_h, new_w), mode='nearest')
        # then crop
        margin_h = max(new_h - tgt_h, 0)
        margin_w = max(new_w - tgt_w, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + tgt_h
        crop_x1, crop_x2 = offset_w, offset_w + tgt_w
        cur_disp = cur_disp[:,:,crop_y1:crop_y2, crop_x1:crop_x2]
        cur_img = cur_img[:,:,crop_y1:crop_y2, crop_x1:crop_x2]
        new_disps.append(cur_disp)
        new_imgs.append(cur_img)
        if psd_gts is not None:
            cur_psd_gt = cur_psd_gt[:,:,crop_y1:crop_y2, crop_x1:crop_x2]
            new_psd_gts.append(cur_psd_gt)
    new_disps = torch.cat(new_disps, dim=0)
    new_imgs = torch.cat(new_imgs, dim=0)
    if len(new_psd_gts) > 0:
        new_psd_gts = torch.cat(new_psd_gts, dim=0)
    return new_disps, new_imgs, new_psd_gts


# ref to https://zhuanlan.zhihu.com/p/399214024 and https://github.com/LucasBoTang/GradNorm
class GradNormLoss(torch.nn.Module):
    def __init__(self, num_of_task, alpha=1.5):
        super(GradNormLoss, self).__init__()
        self.num_of_task = num_of_task
        self.alpha = alpha
        self.w = torch.nn.Parameter(torch.ones(num_of_task, dtype=torch.float))
        self.l1_loss = torch.nn.L1Loss()
        self.L_0 = None
        self.optimzer = torch.optim.Adam([self.w], lr=1e-4)

    def forward(self, grad_norm_weights: torch.nn.Module,
            total_loss: torch.Tensor):

        self.optimzer.zero_grad()
        if self.w.grad is not None:
            self.w.grad.data = self.w.grad.data * 0.0

        filtered_dict = {k: v for k, v in total_loss.items() if 'acc' not in k}
        self.L_t = torch.stack(list(filtered_dict.values()))

        # initialize the initial loss `Li_0`
        if self.L_0 is None:
            self.L_0 = self.L_t.detach() # detach

        self.GW_t = []
        for i in range(len(self.L_t)):
            # get the gradient of this task loss with respect to the shared parameters
            GiW_t = torch.autograd.grad(
                self.L_t[i], grad_norm_weights.parameters(), retain_graph=True)
            # compute the norm
            self.GW_t.append(torch.norm(GiW_t[0] * self.w[i]))
        self.GW_t = torch.stack(self.GW_t) # do not detatch

        self.bar_GW_t = self.GW_t.detach().mean()
        self.tilde_L_t = (self.L_t.detach() / self.L_0)
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))

        self.w.grad = torch.autograd.grad(grad_loss, self.w)[0]
        self.optimzer.step()

        weight_total_loss = dict()
        for i, (key, value) in enumerate(filtered_dict.items()):
            weight_total_loss[key] = total_loss[key] * self.w[i]
            weight_value = self.w[i].detach()
            weight_key = key.replace("loss", "weight")
            weight_total_loss[weight_key] = weight_value

        self.bar_GW_t, self.tilde_L_t, self.r_t = None, None, None
        # re-norm
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task

        return weight_total_loss

# ref to https://github.com/Mikoto10032/AutomaticWeightedLoss
class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, total_loss):

        weight_total_loss = {}
        filtered_dict = {k: v for k, v in total_loss.items() if 'acc' not in k}
        for i, (key, value) in enumerate(filtered_dict.items()):
            if i < 7:
                param = self.params[0]
            else:
                param = self.params[1]

            weight_loss = 0.5 / (param ** 2) * value + torch.log(1 + param ** 2)
            weight_total_loss[key] = weight_loss
            weight_value = param.detach()
            weight_key = key.replace("loss", "weight")
            weight_total_loss[weight_key] = weight_value

        # for i, (key, value) in enumerate(total_loss.items()):
        #     if 'acc' in key:
        #         weight_total_loss[key] = value
        #         continue

        #     weight_loss = 0.5 / (self.params[i] ** 2) * value + torch.log(1 + self.params[i] ** 2)
        #     weight_total_loss[key] = weight_loss

        #     weight_value = self.params[i].detach()
        #     weight_key = key.replace("loss", "weight")
        #     weight_total_loss[weight_key] = weight_value
        return weight_total_loss

#ref: https://github.com/uiuctml/ExcessMTL/blob/main/LibMTL/LibMTL/weighting/DWA.py
class DWALoss(torch.nn.Module):
    """Dynamic Weight Average (DWA).

    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 

    Args:
        T (float, default=2.0): The softmax temperature.

    """
    def __init__(self, task_num=2):
        super(DWALoss, self).__init__()
        self.train_loss_buffer = []
        self.task_num = task_num

    def forward(self, total_loss: torch.Tensor):

        filtered_dict = {k: v for k, v in total_loss.items() if 'acc' not in k}
        # self.L_t = torch.stack(list(filtered_dict.values()))

        # self.train_loss_buffer.append(self.L_t.detach())

        # if len(self.train_loss_buffer) > 1:
        #     w_i = torch.Tensor(self.train_loss_buffer[len(self.train_loss_buffer)-1]/(self.train_loss_buffer[len(self.train_loss_buffer)-2] + 1e-4))
        #     batch_weight = self.task_num * F.softmax(w_i / 2.0, dim=-1)
        # else:
        #     batch_weight = torch.ones_like(self.L_t)

        # if len(self.train_loss_buffer) > 10:
        #     self.train_loss_buffer.pop(0)

        # weight_total_loss = dict()
        # for i, (key, value) in enumerate(filtered_dict.items()):
        #     weight_total_loss[key] = total_loss[key] * batch_weight[i]
        #     weight_value = batch_weight[i].detach()
        #     weight_key = key.replace("loss", "weight")
        #     weight_total_loss[weight_key] = weight_value


        loss_disp_sum = 0
        loss_seg_sum  = 0
        for i, (key, value) in enumerate(filtered_dict.items()):
            if i < 7:
                loss_disp_sum += value
            else:
                loss_seg_sum += value
        loss_list = [loss_disp_sum, loss_seg_sum]
        self.L_t = torch.stack(loss_list)

        self.train_loss_buffer.append(self.L_t.detach())

        if len(self.train_loss_buffer) > 1:
            w_i = torch.Tensor(self.train_loss_buffer[len(self.train_loss_buffer)-1]/(self.train_loss_buffer[len(self.train_loss_buffer)-2] + 1e-4))
            batch_weight = self.task_num * F.softmax(w_i / 2.0, dim=-1)
        else:
            batch_weight = torch.ones_like(self.L_t)

        weight_total_loss = {}
        for i, (key, value) in enumerate(filtered_dict.items()):
            if i < 7:
                param = batch_weight[0]
            else:
                param = batch_weight[1]
            weight_total_loss[key] = total_loss[key] * param
            weight_value = param.detach()
            weight_key = key.replace("loss", "weight")
            weight_total_loss[weight_key] = weight_value

        if len(self.train_loss_buffer) > 10:
            self.train_loss_buffer.pop(0)

        return weight_total_loss

#ref: https://github.com/median-research-group/LibMTL
class ExcessMTL(torch.nn.Module):
    def __init__(self, task_num=2):
        super(ExcessMTL, self).__init__()
        self.task_num = task_num
        self.loss_weight = torch.tensor([1.0]*self.task_num, requires_grad=False)
        self.grad_sum = None
        self.first_epoch = True

    def forward(self, shared_weight: torch.nn.Module, total_loss: torch.Tensor):

        filtered_dict = {k: v for k, v in total_loss.items() if 'acc' not in k}

        loss_sum_list = []
        for i, (key, value) in enumerate(filtered_dict.items()):
            loss_sum_list.append(value)
        loss_sum_list = torch.stack(loss_sum_list)
        loss_list = [torch.sum(loss_sum_list[:7]), torch.sum(loss_sum_list[7:])]
        self.L_t = torch.stack(loss_list)

        self.grad_index = []
        for param in shared_weight.parameters():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

        grads = torch.zeros(self.task_num, self.grad_dim)
        for tn in range(self.task_num):
            grad = torch.autograd.grad(
                self.L_t[tn], shared_weight.parameters(), retain_graph=True)
            grads[tn] = torch.cat([g.view(-1) for g in grad])

        if self.grad_sum is None:
            self.grad_sum = torch.zeros_like(grads)

        w = torch.zeros(self.task_num)
        for i in range(self.task_num):
            self.grad_sum[i] += grads[i]**2
            grad_i = grads[i]
            h_i = torch.sqrt(self.grad_sum[i] + 1e-7)
            w[i] = grad_i * (1 / h_i) @ grad_i.t()

        if self.first_epoch:
            self.initial_w = w
            self.first_epoch = False
        else:
            w = w / self.initial_w
            self.loss_weight = self.loss_weight * torch.exp(w * 1e-2)
            self.loss_weight = self.loss_weight / self.loss_weight.sum() * self.task_num
            self.loss_weight = torch.clamp(self.loss_weight, min=0.1, max=5.0)
            self.loss_weight = self.loss_weight.detach().clone()

        weight_total_loss = {}
        for i, (key, value) in enumerate(filtered_dict.items()):
            if i < 7:
                param = self.loss_weight[0]
            else:
                param = self.loss_weight[1]
            weight_total_loss[key] = total_loss[key] * param

            weight_value = param.detach()
            weight_key = key.replace("loss", "weight")
            weight_total_loss[weight_key] = weight_value

        return weight_total_loss



@MODELS.register_module()
class StereoLabelDistiller(StereoEstimator):
    def __init__(self, student:dict, teacher: dict,
                 sr: float = 0.0,
                 sr_coe:float = 1.0,
                 resrep_mask:list = None,
                 freeze_bn: bool = False,
                 use_aw_loss: bool = False,
                 use_gradnorm_loss: bool = False,
                 use_daw_loss: bool = False,
                 use_excess_loss: bool = False,
                 mixed_training_weight: float = -1.0,
                 mixed_frozen: bool = False,
                 reinit_depth_decoder: bool = False,
                 loss_cfg = dict(),
                 train_depth: bool = True,
                 train_seg: bool = True,
                 data_preprocessor: Optional[dict] = None, **kwargs):
        super().__init__(data_preprocessor=data_preprocessor)
        student_ckpt = student.pop("ckpt", None)
        self.train_depth = train_depth
        self.train_seg = train_seg
        self.student:SepCxtNewRumStereo = MODELS.build(student)
        self.sr = sr
        self.sr_coe = sr_coe
        self.resrep_mask = resrep_mask
        self.small_res_regw = 1.0
        # assert isinstance(self.student, SepCxtNewRumStereo) and isinstance(self.student.decoder, IGEVStereoSlimDecoder)
        if student_ckpt is not None:
            sd = torch.load(student_ckpt, map_location='cpu')
            sd['state_dict'] = {k.replace('student.', ''): v for k, v in sd['state_dict'].items()}

            self.student.load_state_dict(sd['state_dict'], strict=True)
            print(f"Load Pretrained Student Weights from {student_ckpt}")

            if reinit_depth_decoder:
                self.student.reinit_depth_decoder()

        if teacher is not None:
            teacher_ckpt = teacher.pop("ckpt", None)
            self.teacher:SepCxtNewRumStereo = MODELS.build(teacher)
            if teacher_ckpt is not None:
                sd = torch.load(teacher_ckpt, map_location='cpu')
                self.teacher.load_state_dict(sd['state_dict'], strict=True)
                print(f"Load Pretrained Teacher Weights from {teacher_ckpt}")
            self.teacher.eval()
            for m in self.teacher.modules():
                for param in m.parameters():
                    param.requires_grad = False
        else:
            self.teacher = None

        self.use_freeze_bn = freeze_bn
        if self.use_freeze_bn:
            self.student.freeze_bn()

        self.use_aw_loss = use_aw_loss
        if self.use_aw_loss:
            print("use aw loss ...")
            self.aw_loss = AutomaticWeightedLoss(2)

        self.use_gradnorm_loss = use_gradnorm_loss
        if self.use_gradnorm_loss:
            print("use gradnorm loss ...")
            self.gradnorm_loss = GradNormLoss(9)

        self.use_daw_loss = use_daw_loss
        if self.use_daw_loss:
            print("use daw loss ...")
            self.daw_loss = DWALoss(2)

        self.use_excess_loss =use_excess_loss
        if self.use_excess_loss:
            print("use excess loss ...")
            self.excess_loss = ExcessMTL(2)
        
        self.loss_disp_dict = {}
        self.loss_disp_weights = {}
        self.loss_disp_types = {}
        for k in loss_cfg:
            loss_type = loss_cfg[k]['type']
            if loss_type == "warp":
                self.loss_disp_dict[k] = partial(
                    multiscale_warp_loss,
                    ssim_weight=loss_cfg[k].get("ssim_weight", 0.85),
                    scale=loss_cfg[k].get("scale", 2),
                )
            elif loss_type == "dav2_smooth":
                self.loss_disp_dict[k] = disp_gt_guide_smooth
            elif loss_type == "rgb_smooth":
                self.loss_disp_dict[k] = rgb_guide_smooth
            elif loss_type == "error_rate":
                self.loss_disp_dict[k] = error_rate_loss
            elif loss_type == "size_consistency":
                self.loss_disp_dict[k] = partial(
                    resize_batch,
                    tgt_h=loss_cfg[k].get("tgt_height"),
                    tgt_w=loss_cfg[k].get("tgt_width"),
                    max_ratio=loss_cfg[k].get("max_ratio", 1.0)
                )
            elif loss_type == "load_gt":
                # only use for teacher is not None
                self.loss_disp_dict[k] = True if self.teacher is not None else False
            self.loss_disp_weights[k] = loss_cfg[k].get("weight", 1.0)
            self.loss_disp_types[k] = loss_type
        self._is_init = True
        # self.idx = 0
        self.mixed_training_weight = mixed_training_weight
        self.use_mixed_training = mixed_training_weight > 0.0
        self.mixed_frozen = mixed_frozen
        
    @torch.no_grad()
    def teacher_infer(self, imgs):
        # only available for igev decoder
        with autocast('cuda', enabled=False): 
            return self.teacher.forward_train(
                imgs.float(), disp_gt=None, valid=None, flow_init=None, img_metas=None, return_preds=True,
            )
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        new_sd = OrderedDict()
        for k in sd:
            if k.startswith("teacher") or k.startswith("student_ema"):
                continue
            new_sd[k] = sd[k]
        return new_sd

    def forward_train(
            self,
            imgs: torch.Tensor,
            disp_gt: torch.Tensor,
            seg_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None,
            seg_valid: Optional[torch.Tensor] = None,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None,
            return_preds: bool = False,
            *args, **kwargs) -> Dict[str, torch.Tensor]:
        # use teacher to infer disp gt
        clear_imgs = kwargs.get('imgs_ori')
        if self.teacher is not None:
            _, teacher_disp_preds, teacher_seg_preds = self.teacher_infer(clear_imgs)
            teacher_disp_gt = teacher_disp_preds[-1]
            teacher_disp_valid = torch.ones_like(teacher_disp_gt)
            teacher_seg_gt = teacher_seg_preds[-1]
            teacher_seg_valid = torch.ones_like(teacher_seg_gt)
        else:
            # use loaded gt as teacher valid
            teacher_disp_preds = [disp_gt]
            teacher_disp_gt = teacher_disp_preds[-1]
            teacher_disp_valid = valid
            teacher_seg_preds = [seg_gt]
            teacher_seg_gt = teacher_seg_preds[-1]
            teacher_seg_valid = seg_valid

        # ---------- student forward ----------
        loss_disp_dict, loss_seg_dict, student_disp_preds, student_seg_preds = self.student.forward_train(
            imgs,
            disp_gt=teacher_disp_gt if self.train_depth else None,
            seg_gt=teacher_seg_gt if self.train_seg else None,
            depth_valid=teacher_disp_valid if self.train_depth else None,
            seg_valid=teacher_seg_valid if self.train_seg else None,
            flow_init=flow_init,
            img_metas=img_metas,
            return_preds=True,
            train_depth=self.train_depth,
            train_seg=self.train_seg,
            *args, **kwargs,
        )
        
        if self.train_depth:
            # additional weights
            for k in self.loss_disp_dict:
                loss_weight = self.loss_disp_weights[k]
                loss_type = self.loss_disp_types[k]
                if loss_type == 'warp':
                    loss = loss_weight * self.loss_disp_dict[k]((clear_imgs[:,:3]-128.0)/128.0, (clear_imgs[:,3:]-128.0)/128.0, student_disp_preds[-1])
                elif loss_type == 'load_gt_loss':
                    if self.loss_disp_dict[k]:
                        cur_loss_dict = self.student.decoder.losses(
                            student_disp_preds, disp_gt, valid,
                        )
                        cur_loss = 0.0
                        for k1 in cur_loss_dict:
                            if 'loss' in k1:
                                cur_loss += cur_loss_dict[k1]
                        loss = loss_weight * cur_loss
                    else:
                        continue
                elif loss_type == 'error_rate':
                    loss = loss_weight * self.loss_disp_dict[k](student_disp_preds[-1], teacher_disp_gt, valid)
                elif loss_type == 'dav2_smooth':
                    loss = loss_weight * self.loss_disp_dict[k](kwargs['pseudo_gt'], student_disp_preds[-1])
                elif loss_type == 'rgb_smooth':
                    loss = loss_weight * self.loss_disp_dict[k](clear_imgs[:,:3], student_disp_preds[-1])
                elif loss_type == "size_consistency":
                    origin_pred, origin_psd_gt = student_disp_preds[-1].detach(), kwargs['pseudo_gt']
                    resized_pred, resized_imgs, resized_psd_gt = self.loss_disp_dict[k](
                        disps=origin_pred.detach(), imgs=imgs, psd_gts=origin_psd_gt,
                    )
                    kwargs['pseudo_gt'] = resized_psd_gt
                    resized_loss_dict, _, resized_outputs, _ = self.student.forward_train(
                        resized_imgs, disp_gt=resized_pred, valid=torch.ones_like(resized_pred),
                        flow_init=flow_init, img_metas=img_metas, return_preds=True,
                        *args, **kwargs,
                    )
                    kwargs['pseudo_gt'] = origin_psd_gt
                    resized_loss = 0.0
                    for k1 in resized_loss_dict:
                        if 'loss' in k1:
                            resized_loss += resized_loss_dict[k1]
                    resized_loss = 0.3 * resized_loss # only as reg term
                    resized_loss = resized_loss + 1.5 * torch.abs(resized_outputs[-1] - resized_pred).mean()
                    loss = resized_loss * loss_weight
                else:
                    loss = 0.0
                loss_disp_dict[k] = loss

        if self.use_mixed_training:
            data_batch_B = kwargs["data_B"]
            imgs_B = data_batch_B.pop("inputs")
            data_samples_B = data_batch_B.pop("data_samples")
            loss_dict_B, _, _, _ = self.student.forward_train(
                imgs_B, return_preds=True, frozen_features=self.mixed_frozen, # detach features to avoid out-of-domain feature learning
                **data_samples_B,
            )
            mixed_loss = 0.0
            for k in loss_dict_B:
                if "loss" in k:
                    mixed_loss += loss_dict_B[k]
            mixed_loss = self.mixed_training_weight * mixed_loss
            loss_disp_dict['loss_data_batch_B'] = mixed_loss

        loss_whole_dict = dict()
        if self.train_depth:
            for disp_loss in loss_disp_dict:
                loss_whole_dict[disp_loss] = loss_disp_dict[disp_loss]
        if self.train_seg:
            for seg_loss in loss_seg_dict:
                loss_whole_dict[seg_loss] = loss_seg_dict[seg_loss]

        if self.use_aw_loss:
            loss_whole_dict = self.aw_loss.forward(loss_whole_dict)

        if self.use_gradnorm_loss:
            loss_whole_dict = self.gradnorm_loss.forward(list(self.student.encoder.modules())[-1], loss_whole_dict)

        if self.use_daw_loss:
            loss_whole_dict = self.daw_loss.forward(loss_whole_dict)

        if self.use_excess_loss:
            loss_whole_dict = self.excess_loss.forward(self.student.encoder, loss_whole_dict)

        if return_preds:
            return loss_whole_dict, student_disp_preds, student_seg_preds
        return loss_whole_dict

    def forward_test(
            self,
            imgs: torch.Tensor,
            flow_init: Optional[torch.Tensor] = None,
            img_metas: Optional[Sequence[dict]] = None):
        return self.student.forward_test(
            imgs, flow_init, img_metas,
        )

    def train(self, mode:bool=True):
        super().train(mode)
        # we need teacher model to be eval mode
        if self.teacher is not None:
            self.teacher.eval()
        if self.use_freeze_bn:
            self.student.freeze_bn()
    
    def forward_onnx(self, *args, **kwargs):
        """Placeholder for forward function of stereo estimator when converting onnx."""
        return None
