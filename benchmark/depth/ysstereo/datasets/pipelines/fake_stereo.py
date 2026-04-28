import os
import os.path as osp
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2, mmcv
import numpy as np
from skimage.filters import sobel
from functools import partial
from scipy.stats import truncnorm
from mmcv.transforms import BaseTransform, TRANSFORMS
from mmcv.image import adjust_brightness, adjust_color, adjust_contrast
from ..utils import adjust_gamma, adjust_hue
import mmengine.fileio as fileio

def softsplat_np(image:np.ndarray, disp:np.ndarray):
    # img channels
    img_channels= [image[:, :, i].astype(np.float64) for i in range(image.shape[2])]
    # normalize disparity to 25.0 for metric
    disp = disp.astype(np.float64)
    metric_factor = 2.0 * (25.0 / disp.max())
    metric = disp * metric_factor
    # clip to -20~40.0
    metric = np.clip(metric, -20.0, 40.0)
    # generate out_coords
    h, w = image.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    x = x.astype(np.float64)
    new_x = x - disp
    # compute valid mask
    x_floor = np.floor(new_x)
    factor_ceiling = new_x - x_floor
    factor_floor = 1.0 - factor_ceiling
    x_floor = x_floor.astype(int)
    x_ceiling = x_floor + 1
    mask = (x_floor >= 0) * (x_floor < w) * (x_ceiling >= 0) * (x_ceiling < w)
    mask = mask.astype(bool)
    # init out
    outs = [np.zeros_like(img_ch) for img_ch in img_channels]
    outs.append(np.zeros_like(metric))
    # compute out_coords
    y = y[mask]
    x_floor = x_floor[mask]
    x_ceiling = x_ceiling[mask]
    factor_floor = factor_floor[mask]
    factor_ceiling = factor_ceiling[mask]
    metric = np.exp(metric[mask])
    masked_img_chs = [img_ch[mask]*metric for img_ch in img_channels]
    # compute out
    inputs = masked_img_chs + [metric]
    for out, inp in zip(outs, inputs):
        np.add.at(out, (y, x_floor), inp * factor_floor)
        np.add.at(out, (y, x_ceiling), inp * factor_ceiling)
    out_imgs = outs[:-1]
    normalize_factor = outs[-1]
    # clip normalize_factor to avoid divide by zero
    normalize_factor = np.clip(normalize_factor, 1e-9, None)
    for out_img in out_imgs:
        out_img /= normalize_factor
    out_img = np.stack(out_imgs, axis=-1)
    out_img = out_img.astype(image.dtype)
    return out_img

@TRANSFORMS.register_module()
class LoadFakeStereoSample(BaseTransform):
    def __init__(self,
                 img_loading_config=dict(
                    color_type='color',
                    imdecode_backend='cv2',
                    to_float32=False,
                 ),
                 disp_loading_config=dict(
                    disp_gamma_ratio=0.1,
                    dav2_rate=0.85,
                    trunc_near_ratio=0.85,
                    # for simulate stereo images in small height
                    truncnorm_near_params=dict(
                        mu=20.0,
                        sigma=6.0,
                        lower=2.0,
                        upper=50.0,
                    ),
                    # for simulate stereo images in large height
                    truncnorm_far_params=dict(
                        mu=6.0,
                        sigma=3.0,
                        lower=0.7,
                        upper=12.0,
                    ),
                    disp_updown_rate=0.8,
                 ),
                 color_transform_config=dict(
                    # only apply to right image
                    color_jitter_params=dict(
                        ratio=0.7,
                        asymmetric=0.5,
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                    ),
                    gamma_params=dict(
                        ratio=0.7,
                        gamma_range=(0.7, 1.5),
                    ),
                 ),
                #  resize_config=dict(
                #     tgt_size=(640, 896),
                #     max_disp_bound:int=128,
                #  )
                 ):
        # img_loading conifg
        self.imload_to_float32 = img_loading_config.get('to_float32', False)
        self.imload_color_type = img_loading_config.get('color_type', 'color')
        self.imload_imdecode_backend = img_loading_config.get('imdecode_backend', 'cv2')
        # disp_loading config
        self.dispload_dav2_rate = disp_loading_config.get('dav2_rate', 0.85)
        self.dispload_gamma_ratio = disp_loading_config.get('disp_gamma_ratio', 0.1)
        self.dispload_trunc_near_ratio = disp_loading_config.get('trunc_near_ratio', 0.85)
        # for simulate stereo images in small height
        params = disp_loading_config.get('truncnorm_near_params')
        a, b = (params['lower'] - params['mu']) / params['sigma'], (params['upper'] - params['mu']) / params['sigma']
        self.dispload_truncnorm_near = partial(truncnorm.rvs, a=a, b=b, loc=params['mu'], scale=params['sigma'])
        # for simulate stereo images in large height
        params = disp_loading_config.get('truncnorm_far_params')
        a, b = (params['lower'] - params['mu']) / params['sigma'], (params['upper'] - params['mu']) / params['sigma']
        self.dispload_truncnorm_far = partial(truncnorm.rvs, a=a, b=b, loc=params['mu'], scale=params['sigma'])
        self.dispload_updown_rate = disp_loading_config.get('disp_updown_rate', 0.8)
        # color_transform config
        gamma_params = color_transform_config.get('gamma_params')
        self.gamma_ratio = gamma_params.get('ratio', 0.7)
        self.gamma_range = gamma_params.get('gamma_range', (0.7, 1.5))
        color_jitter_params = color_transform_config.get('color_jitter_params', dict(
            ratio=0.7, asymmetric=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
        ))
        self.cj_ratio = color_jitter_params.get('ratio', 0.7)
        self.cj_asymmetric = color_jitter_params.get('asymmetric', 0.5)
        self.cj_brightness = self._check_input(
            color_jitter_params.get('brightness', 0.2), 'brightness')
        self.cj_contrast = self._check_input(
            color_jitter_params.get('contrast', 0.2), 'contrast')
        self.cj_saturation = self._check_input(
            color_jitter_params.get('saturation', 0.2), 'saturation')
        self.cj_hue = self._check_input(
            color_jitter_params.get('hue', 0.1), 'hue', center=0, bound=(-0.5, 0.5),
            clip_first_on_zero=False)
        

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, (float, int)):

            if value < 0:
                raise ValueError(
                    f'If {name} is a single number, it must be non negative.')
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f'{name} values should be between {bound}')

        elif isinstance(value, (tuple, list)) and len(value) == 2:

            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f'{name} values should be between {bound}')

        else:
            raise TypeError(
                f'{name} should be a single number or a list/tuple with '
                f'length 2, but got {value}.')

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value
 
    def _load_img(self, results:dict, rotate=False,
                  tgt_h:int=None, tgt_w:int=None) -> dict:
        filename1 = results['img_info']['filename1']
        if (not osp.isfile(filename1)):
            raise RuntimeError(
                f'Cannot load file from {filename1}')
        img1_bytes = fileio.get(filename1)
        img1 = mmcv.imfrombytes(
            img1_bytes, flag=self.imload_color_type, backend=self.imload_imdecode_backend)
        if rotate:
            # updown stereo
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if tgt_h is not None and tgt_w is not None:
            ori_h, ori_w = img1.shape[:2]
            ori_area = ori_h * ori_w
            tgt_area = tgt_h * tgt_w
            # for resize, use INTER_LANCZOS4 for upsample, INTER_AREA for downsample
            if tgt_area > ori_area:
                img1 = cv2.resize(img1, (tgt_w, tgt_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                img1 = cv2.resize(img1, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        if self.imload_to_float32:
            img1 = img1.astype(np.float32)
        results['filename1'] = filename1
        results['ori_filename1'] = osp.split(filename1)[-1]
        results['filename2'] = filename1
        results['ori_filename2'] = osp.split(filename1)[-1]

        results['imgl'] = img1
        results['imgr'] = img1.copy()
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        results['pad_shape'] = img1.shape
        results['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        num_channels = 1 if len(img1.shape) < 3 else img1.shape[2]
        results['img_norm_cfg'] = dict(
            mean = np.zeros(num_channels, dtype=np.float32),
            std = np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results
    
    def _load_disp(self, results:dict, rotate=False) -> dict:
        filename_disp = results['ann_info']['filename_disp']
        # decide dav2 or dpro to load disparity
        use_dpro = np.random.rand() > self.dispload_dav2_rate
        if use_dpro:
            # default use dav2, we need to change to dpro
            filename_disp = filename_disp.replace('dav2', 'dpro')
        if not osp.isfile(filename_disp):
            raise RuntimeError(f'Cannot load file from {filename_disp}')
        # load exr disp via cv2
        disp = cv2.imread(filename_disp, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
        if len(disp.shape) > 2:
            disp = disp[..., 0]
        if rotate:
            # updown stereo
            disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # gamma correction
        if self.dispload_gamma_ratio > np.random.rand():
            # do gamma augmentation for disparity
            gamma = np.random.uniform(0.9, 1.15)
            # normalize to 0~1
            disp_max = disp.max()
            disp = disp / disp_max # make sure disp is in 0~1
            disp = disp ** gamma # gamma correction
            disp = disp * disp_max
        # decide max disp value
        truncnorm_fun = self.dispload_truncnorm_near
        if np.random.rand() > self.dispload_trunc_near_ratio:
            truncnorm_fun = self.dispload_truncnorm_far
        max_disp = truncnorm_fun()
        ori_disp_max = disp.max()
        if ori_disp_max < 1.0:
            # this is a really small disparity, we needn't to scale it
            new_disp = disp
        else:
            new_disp = disp * max_disp / ori_disp_max
        # disp sharpen via sobel
        new_disp[new_disp < 0] = 0 # set negative value to 0
        edge_mask = sobel(new_disp)>3.2
        new_disp[edge_mask] = 0 # mask out the edge
        results['filename_disp'] = filename_disp
        results['ori_filename_disp'] = osp.split(filename_disp)[-1]
        results['disp_gt'] = new_disp
        results['valid'] = ~edge_mask # inverse edge mask is valid
        results['ann_fields'].append('disp_gt')
        return results

    def _color_jitter(self, imgs):
        fn_idx = np.random.permutation(4)
        b = None if self.cj_brightness is None else np.random.uniform(
            self.cj_brightness[0], self.cj_brightness[1])
        c = None if self.cj_contrast is None else np.random.uniform(
            self.cj_contrast[0], self.cj_contrast[1])
        s = None if self.cj_saturation is None else np.random.uniform(
            self.cj_saturation[0], self.cj_saturation[1])
        h = None if self.cj_hue is None else np.random.uniform(
            self.cj_hue[0], self.cj_hue[1])
        for i in fn_idx:
            if i == 0 and b:
                imgs = [adjust_brightness(img, b) for img in imgs]
            if i == 1 and c:
                imgs = [adjust_contrast(img, c) for img in imgs]
            if i == 2 and s:
                imgs = [adjust_color(img, s) for img in imgs]
            if i == 3 and h:
                imgs = [adjust_hue(img, h) for img in imgs]
        return imgs

    def _color_transform(self, results:dict) -> dict:
        imgs = [results['imgl'], results['imgr']]
        use_gamma = np.random.rand() < self.gamma_ratio
        if use_gamma:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            results['gamma'] = gamma
            imgs = [adjust_gamma(img, gamma) for img in imgs]
        else:
            results['gamma'] = 1.0
        use_color_jitter = np.random.rand() < self.cj_ratio
        if use_color_jitter:
            asymmetric = np.random.rand() < self.cj_asymmetric
            if asymmetric:
                imgs = [self._color_jitter([v])[0] for v in imgs]
            else:
                imgs = self._color_jitter(imgs)
        results['imgl'] = imgs[0]
        results['imgr'] = imgs[1]
        return results
    
    def _warp_to_right(self, results:dict) -> dict:
        img2 = results['imgr']
        disp_gt = results['disp_gt']
        img2 = softsplat_np(img2, disp_gt)
        results['imgr'] = img2
        return results
    
    def transform(self, results:dict) -> dict:
        rotate = np.random.rand() < self.dispload_updown_rate
        results = self._load_disp(results, rotate)
        tgt_h, tgt_w = results['disp_gt'].shape[:2]
        results = self._load_img(results, rotate, tgt_h, tgt_w)
        results = self._color_transform(results)
        results = self._warp_to_right(results)
        return results
