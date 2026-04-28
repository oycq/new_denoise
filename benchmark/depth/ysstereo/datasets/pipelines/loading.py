import os.path as osp
import json
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import pyexr

import mmcv
import numpy as np
import random
from mmcv import sparse_flow_from_bytes
import yaml
import torch

import mmengine.fileio as fileio
from mmcv.transforms import BaseTransform, TRANSFORMS
from mmcv.transforms import LoadImageFromFile, LoadAnnotations
from ysstereo.datasets.utils import disp_from_bytes
from ysstereo.datasets.utils.fisheyecam import FisheyeCamModel, pixelToGrid, toNumpy
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R
from skimage.util import img_as_float
from skimage.segmentation import felzenszwalb
from ysstereo.datasets.utils import read_pfm

@TRANSFORMS.register_module()
class ComputeDispDist(BaseTransform):
    def __init__(self, max_disp:int=512, dist_key:str='disp_gt_dist'):
        self.max_disp = max_disp
        self.dist_key = dist_key
    
    def transform(self, results:dict):
        disp_gt = results['disp_gt']
        valid = results.get('valid', (disp_gt > 1e-3) & (disp_gt < float(self.max_disp)))
        disp_gt:np.ndarray = disp_gt[valid]
        dist = np.zeros((1, self.max_disp+10)).reshape(-1)
        medians = np.zeros((1, self.max_disp+10)).reshape(-1)
        if disp_gt.size > 0:
            disp_gt = disp_gt.astype(np.uint32)
            large_mask = disp_gt >= self.max_disp
            dist[self.max_disp] = np.sum(large_mask)
            disp_gt = disp_gt[~large_mask]
            values, counts = np.unique(disp_gt, return_counts=True)
            median = np.median(disp_gt)
            dist[values] = counts
            medians[np.uint32(median)]=1
        results['median']=medians
        results[self.dist_key] = dist
        return results

@TRANSFORMS.register_module()
class Dupano2Pinhole(BaseTransform):
    """convert simdupano image1, image2 and depth to pinhole ones.
    """

    def __init__(self,
                 max_disp: float = 512.0,
                 output_width: int = 768,
                 output_height: int = 768,
                 xfov: int = 100,
                 return_depth: bool = False,
                 test_mode: bool = False,
                 max_dist: float = 200) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.w = output_width
        self.h = output_height
        self.xfov = xfov / 180 * np.pi

        self.u, self.v = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))

        self.pinhole_focal = self.w / 2 / np.tan(self.xfov/2)
        self.X = (self.u - self.w / 2+0.5)
        self.Y = (self.v - self.h / 2+0.5)
        self.Z = np.ones_like(self.X) * self.pinhole_focal

        self.P0 = np.stack([self.X, self.Y, self.Z], axis=0).astype(np.float32)
        self.P = self.P0 / LA.norm(self.P0, axis=0)

        self.theta = np.arccos(self.P[0])
        self.phi = np.arctan2(self.P[1], self.P[2])
        
        self.u1 = 1-2*self.theta/np.pi
        self.v1 = self.phi/np.pi

        self.grid = torch.tensor(np.stack([self.u1, self.v1], axis=2)).unsqueeze(0)

        self.return_depth = return_depth
        self.test_mode = test_mode
        self.max_dist = max_dist

    def transform(self, results: dict) -> dict:

        cam_1_idx = int(results['filename1'].split('/')[-3].split('_')[1])
        cam_2_idx = int(results['filename2'].split('/')[-3].split('_')[1])

        if (cam_1_idx==0 and cam_2_idx==2) or (cam_1_idx==2 and cam_2_idx==0):
            baseline = 0.4
        else:
            baseline = 0.2
        
        up_down = True if cam_2_idx > cam_1_idx else False
        if up_down:
            pano_img1 = cv2.rotate(results['imgl'], cv2.ROTATE_90_COUNTERCLOCKWISE)
            pano_img2 = cv2.rotate(results['imgr'], cv2.ROTATE_90_COUNTERCLOCKWISE)
            pano_distance = cv2.rotate(results['disp_gt'], cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            pano_img1 = cv2.rotate(results['imgl'], cv2.ROTATE_90_CLOCKWISE)
            pano_img2 = cv2.rotate(results['imgr'], cv2.ROTATE_90_CLOCKWISE)
            pano_distance = cv2.rotate(results['disp_gt'], cv2.ROTATE_90_CLOCKWISE)

        del results['imgl'], results['imgr'], results['disp_gt'], results['valid']

        if not self.test_mode:
            if random.random() > 0.5:
                pano_img1 = cv2.flip(pano_img1, 0)
                pano_img2 = cv2.flip(pano_img2, 0)
                pano_distance = cv2.flip(pano_distance, 0)
            h, w = pano_distance.shape
            roll_idx = random.randint(0, h)
            pano_img1 = np.roll(pano_img1, roll_idx, 0)
            pano_img2 = np.roll(pano_img2, roll_idx, 0)
            pano_distance = np.roll(pano_distance, roll_idx, 0)

        pano_img1 = torch.tensor(pano_img1.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        pinhole_img1 = torch.nn.functional.grid_sample(pano_img1, self.grid, align_corners=False)[0] \
            .permute(1, 2, 0).numpy().astype(np.uint8)
        
        pano_img2 = torch.tensor(pano_img2.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        pinhole_img2 = torch.nn.functional.grid_sample(pano_img2, self.grid, align_corners=False)[0] \
            .permute(1, 2, 0).numpy().astype(np.uint8)

        results['imgl'] = pinhole_img1
        results['imgr'] = pinhole_img2
        results['img_shape'] = pinhole_img1.shape
        results['ori_shape'] = pinhole_img1.shape
        results['pad_shape'] = pinhole_img1.shape

        if self.test_mode:
            return results

        pano_distance = torch.tensor(pano_distance).unsqueeze(0).unsqueeze(0)
        pinhole_distance = torch.nn.functional.grid_sample(pano_distance, self.grid,
                                                           align_corners=False, mode='nearest')[0, 0].numpy()
        pinhole_depth = pinhole_distance*self.P[2]
        
        disp = baseline * self.pinhole_focal / pinhole_depth

        valid = ((np.abs(disp) <= self.max_disp) & (np.abs(disp) > 1e-3) & (np.abs(pinhole_distance) > 1e-2))

        # # make as filter
        # min_d = pinhole_depth.min()
        # if min_d < 0.6:
        #     # set valid as all False
        #     valid = np.zeros_like(valid).astype(bool)

        results['disp_gt'] = disp
        results['valid'] = valid

        if self.return_depth:
            pinhole_depth[pinhole_depth < 1e-2] = 1e-2
            pinhole_depth[pinhole_depth > self.max_dist] = self.max_dist
            results['depth'] = pinhole_depth

        return results


@TRANSFORMS.register_module()
class GetFsTriple(BaseTransform):
    def __init__(self, num_fs_triple: int = 32, disable: bool = False) -> None:
        self.num_fs_triple = num_fs_triple
        self.disable = disable

    def get_fs_triple(self, img):
        keysets = []
        lambda_sets = []
        acc_num = 0

        # extract lines
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        length_threshold = int(np.sqrt(h ** 2 + w ** 2) / 10)
        fld = cv2.ximgproc.createFastLineDetector(length_threshold)
        lines = fld.detect(image)
        if lines is not None:
            lines = np.squeeze(lines, 1)
            lengths = np.sqrt((lines[:, 0] - lines[:, 2]) ** 2 + (lines[:, 1] - lines[:, 3]) ** 2)
            sum_lengths = np.sum(lengths)

            # sample triple from lines
            for k in range(lines.shape[0]):
                x1, y1, x2, y2 = lines[k]
                int_x1 = min(max(0, int(x1)), w - 1)
                int_y1 = min(max(0, int(y1)), h - 1)
                int_x2 = min(max(0, int(x2)), w - 1)
                int_y2 = min(max(0, int(y2)), h - 1)
                num_samp = max(1, int(self.num_fs_triple * lengths[k] / sum_lengths))
                lambdas = np.random.rand(num_samp)
                acc_num += num_samp
                for i in range(num_samp):
                    x3 = lambdas[i] * (x2 - x1) + x1
                    y3 = lambdas[i] * (y2 - y1) + y1
                    int_x3 = min(max(0, int(x3)), w - 1)
                    int_y3 = min(max(0, int(y3)), h - 1)
                    keysets.append(np.array([int_y1 * w + int_x1,
                                             int_y2 * w + int_x2,
                                             int_y3 * w + int_x3]))  # save index of flatten disp pixels
                    lambda_sets.append(lambdas[i])

        # extract superpixels
        image = img_as_float(img)
        segment = felzenszwalb(image, scale=100, sigma=0.8, min_size=20)
        n = 1
        for i in range(segment.max()):
            npixel = np.sum(segment == (i + 1))
            if npixel > w * h / 400:
                segment[segment == (i + 1)] = n
                n += 1
            else:
                segment[segment == (i + 1)] = 0

        # sample triple from superpixels
        segment = segment.flatten()
        num_struct_pixels = np.sum(segment > 0)
        num_struct = segment.max()
        for j in range(num_struct):
            pixels_j = np.sum(segment == j + 1)
            num_j = int(np.ceil(2 * self.num_fs_triple * pixels_j / num_struct_pixels))  # each group have two lines
            inx_j = np.argwhere(segment == j + 1)

            keyset_j = inx_j[
                np.random.randint(inx_j.shape[0], size=num_j * 2)]  # select four points to devide two lines
            keyset_j = np.reshape(keyset_j, (2, num_j))

            int_x1_j = keyset_j[0] % w
            int_y1_j = keyset_j[0] // w
            int_x2_j = keyset_j[1] % w
            int_y2_j = keyset_j[1] // w

            lambdas = np.random.rand(num_j)

            int_x3_j = ((int_x2_j - int_x1_j) * lambdas).round().astype(int) + int_x1_j
            int_y3_j = ((int_y2_j - int_y1_j) * lambdas).round().astype(int) + int_y1_j
            keyset_3_j = int_y3_j * w + int_x3_j
            for i in range(num_j):
                if segment[keyset_3_j[i]] == j + 1:
                    keysets.append(np.array([keyset_j[0][i], keyset_j[1][i], keyset_3_j[i]]))
                    lambda_sets.append(lambdas[i])
                    acc_num += 1

        if len(keysets):
            keysets = np.stack(keysets, axis=1)
            lambda_sets = np.array(lambda_sets)
            idxs = np.random.randint(keysets.shape[1], size=self.num_fs_triple)
            keysets = keysets[:, idxs]
            lambda_sets = lambda_sets[idxs]
        else:
            # no a keyset found (no detected planes or lines)
            keysets = np.zeros((3, self.num_fs_triple), dtype=np.int64)
            lambda_sets = np.zeros((self.num_fs_triple))

        return keysets, lambda_sets

    def transform(self, results: dict) -> dict:
        img1 = results['imgl'].copy()
        if self.disable:
            keysets = np.zeros((3, self.num_fs_triple), dtype=np.int64)
            lambda_sets = np.zeros((self.num_fs_triple))
        else:
            keysets, lambda_sets = self.get_fs_triple(img1)

        keysets = torch.from_numpy(keysets[None]).long()
        lambda_sets = torch.from_numpy(lambda_sets[None]).float()

        results['keysets'] = keysets
        results['lambda_sets'] = lambda_sets

        return results


@TRANSFORMS.register_module()
class GetFsTripleFast(BaseTransform):
    def __init__(self, num_fs_triple: int = 32, disable: bool = False) -> None:
        self.num_fs_triple = num_fs_triple
        self.disable = disable

    def get_fs_triple(self, img):
        keysets = []
        lambda_sets = []
        acc_num = 0

        # extract lines
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        length_threshold = int(np.sqrt(h ** 2 + w ** 2) / 10)
        fld = cv2.ximgproc.createFastLineDetector(length_threshold)
        lines = fld.detect(image)
        if lines is not None:
            lines = np.squeeze(lines, 1)
            lengths = np.sqrt((lines[:, 0] - lines[:, 2]) ** 2 + (lines[:, 1] - lines[:, 3]) ** 2)
            sum_lengths = np.sum(lengths)

            # get sample number & lambdas
            num_samp = np.clip((self.num_fs_triple * lengths/sum_lengths + 0.5).astype(np.int64), a_min=1, a_max=None)
            num_lines = np.sum(num_samp)
            lambdas = np.random.rand(num_lines) # 0-1 ratios
            # generate line idx
            line_idx = np.linspace(start=0, stop=len(lines)-1, num=len(lines), dtype=np.int64)
            samp_idx = np.repeat(line_idx, num_samp, axis=0)
            # get samples
            samp_lines = lines[samp_idx] # N,4
            x3 = lambdas * (samp_lines[:,2] - samp_lines[:,0]) + samp_lines[:,0]
            y3 = lambdas * (samp_lines[:,3] - samp_lines[:,1]) + samp_lines[:,1]
            samp_lines = np.clip((samp_lines+0.5).astype(np.int64), a_min=0, a_max=[w-1, h-1, w-1, h-1])
            int_y3 = np.clip(y3.astype(np.int64), a_min=0, a_max=h-1)
            int_x3 = np.clip(x3.astype(np.int64), a_min=0, a_max=w-1)
            samples = np.stack(
                [samp_lines[:,1] * w + samp_lines[:,0], samp_lines[:,3] * w + samp_lines[:,2], int_y3 * w + int_x3], axis=-1,
            )
            acc_num += num_lines
            keysets.append(samples)
            lambda_sets.append(lambdas)

        # extract superpixels
        image = img_as_float(img)
        segment = felzenszwalb(image, scale=100, sigma=0.8, min_size=20) # 0->n
        # re-index
        _, indice, counts = np.unique(segment.reshape(-1), return_counts=True, return_inverse=True)
        valid_seg_mask = counts>w*h/400
        new_seg_idx = np.cumsum(valid_seg_mask) * valid_seg_mask
        segment = new_seg_idx[indice] # h*w, 1
        counts:np.ndarray = counts[valid_seg_mask] # 1-N
        # sample
        num_struct_pixels = np.sum(counts)
        pixels_j = counts
        invalid_px_count = len(segment) - np.sum(counts)
        num_j = (np.ceil(2 * self.num_fs_triple * pixels_j)/num_struct_pixels).astype(np.int64) # N sp
        inx_j = np.argsort(segment)
        inx_j = inx_j[invalid_px_count:]
        high = np.cumsum(counts)
        low = high - counts
        low = np.repeat(low, num_j*2, axis=0)
        high = np.repeat(high, num_j*2, axis=0)
        sample_idx = np.random.randint(low=low, high=high, size=high.shape)
        keyset_j = inx_j[sample_idx].reshape(-1, 2)
        int_x1_j = keyset_j[:, 0] % w
        int_y1_j = keyset_j[:, 0] // w
        int_x2_j = keyset_j[:, 1] % w
        int_y2_j = keyset_j[:, 1] // w
        lambdas = np.random.rand(len(keyset_j))
        int_x3_j = ((int_x2_j - int_x1_j) * lambdas).round().astype(int) + int_x1_j
        int_y3_j = ((int_y2_j - int_y1_j) * lambdas).round().astype(int) + int_y1_j
        keyset_3_j = int_y3_j * w + int_x3_j
        k3_j_seg_id = segment[keyset_3_j]
        k1_j_seg_id = segment[keyset_j[:, 0]]
        valid_sample_mask = (k3_j_seg_id==k1_j_seg_id)
        valid_sample_num = np.sum(valid_sample_mask)
        if valid_sample_num > 0:
            keysets.append(
                np.stack((keyset_j[:, 0], keyset_j[:, 1], keyset_3_j), axis=-1)[valid_sample_mask]
            )
            lambda_sets.append(lambdas[valid_sample_mask])
        acc_num += valid_sample_num

        if len(keysets):
            keysets = np.concatenate(keysets, axis=0).transpose()
            lambda_sets = np.concatenate(lambda_sets, axis=0)
            idxs = np.random.randint(keysets.shape[1], size=self.num_fs_triple)
            keysets = keysets[:, idxs]
            lambda_sets = lambda_sets[idxs]
        else:
            # no a keyset found (no detected planes or lines)
            keysets = np.zeros((3, self.num_fs_triple), dtype=np.int64)
            lambda_sets = np.zeros((self.num_fs_triple))

        return keysets, lambda_sets
    
    def transform(self, results: dict) -> dict:
        img1 = results['imgl'].copy()
        if self.disable:
            keysets = np.zeros((3, self.num_fs_triple), dtype=np.int64)
            lambda_sets = np.zeros((self.num_fs_triple))
        else:
            keysets, lambda_sets = self.get_fs_triple(img1)

        keysets = torch.from_numpy(keysets[None]).long()
        lambda_sets = torch.from_numpy(lambda_sets[None]).float()

        results['keysets'] = keysets
        results['lambda_sets'] = lambda_sets

        return results


@TRANSFORMS.register_module()
class Fisheye2TransEqui(BaseTransform):
    """convert fisheye image1, image2 and depth to trans. euqi. forms.
    """

    def __init__(self,
                 max_disp: float = 512.0,
                 f1: float = 315.712,
                 c1: list = [512.0, 512.0],
                 k1: list = [1., -1.7700345707950937e-01, 2.8355590110360829e-01, -1.1359963826630388e-01],
                 R1: list = [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                 T1: list = [0., 0., 0.],
                 f2: float = 315.712,
                 c2: list = [512.0, 512.0],
                 k2: list = [1., -1.7700345707950937e-01, 2.8355590110360829e-01, -1.1359963826630388e-01],
                 R2: list = [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                 T2: list = [0.6, 0., 0.],
                 output_width: int = 768,
                 output_height: int = 768,
                 hfov: int = 180,
                 vfov: int = 180,
                 return_distance: bool = False,
                 test_mode: bool = False,
                 max_dist: float = 200,
                 old_data: bool = True,
                 samllT_boundary: int = 19301) -> None:
        super().__init__()
        self.max_disp = max_disp

        self.f1 = f1
        self.c1 = c1
        self.k1 = np.array(k1).astype(np.float32)
        self.R1w = np.array(R1).reshape(3, 3).astype(np.float32)  # 相机1 在世界坐标系中的pose
        self.T1w = np.array(T1).reshape(3, 1).astype(np.float32)  # 相机1 在世界坐标系中的位置
        self.f2 = f2
        self.c2 = c2
        self.k2 = np.array(k2).astype(np.float32)
        self.R2w = np.array(R2).reshape(3, 3).astype(np.float32)  # 相机2 在世界坐标系中的pose
        self.T2w = np.array(T2).reshape(3, 1).astype(np.float32)  # 相机2 在世界坐标系中的位置
        # self.R21 = self.R2@self.R1.transpose()

        self.old_data = old_data
        self.samllT_boundary = samllT_boundary

        self.w = output_width
        self.h = output_height
        self.hfov = hfov / 180 * np.pi
        self.vfov = vfov / 180 * np.pi
        self.u, self.v = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))
        self.angle_x = (1 - (self.u + 0.5) / self.w) * self.hfov + (
                    np.pi - self.hfov) / 2  # the angle of the 3D vector to X-axis
        self.angle_yz = ((self.v + 0.5) / self.h - 0.5) * self.vfov  # the angle of the mapped 2D vector on the Y-Z plane to Z-axis

        self.X = np.cos(self.angle_x.flatten())
        self.Y = np.sin(self.angle_x.flatten()) * np.sin(self.angle_yz.flatten())
        self.Z = np.sin(self.angle_x.flatten()) * np.cos(self.angle_yz.flatten())

        T12 = self.T2w - self.T1w # 在世界坐标系中，相机1原点到相机2原点的向量
        cx = T12[:, 0] / LA.norm(T12)   # T12的方向作为rectification后的新双目系统的x轴方向, 仍是在世界坐标系下的表达
        z_sum = self.R1w[:, 2] + self.R2w[:, 2] # 相机1和相机2的z轴方向在世界坐标系下的和向量
        cz_ = z_sum / LA.norm(z_sum) # z_sum作为rectification后的新双目系统的z轴方向, 可以保证两图像有最大的overlap, 但由于不能保证和cx完全垂直只能作为一个粗略结果
        cy = np.cross(cz_, cx) # 根据右手坐标系原理求出rectification后的新双目系统的y轴方向
        cy = cy / LA.norm(cy)
        cz = np.cross(cx, cy) # 根据右手坐标系原理求出最终的rectification后的新双目系统的z轴方向
        cz = cz / LA.norm(cz)
        self.Rcw = np.stack([cx, cy, cz], axis=1)  # rectification后的新双目系统 在世界坐标系中的pose  

        # 对原双目相机进行 rectification
        self.R1c = self.R1w @ self.Rcw.T  # 相机1 在rectification后的新双目系统中的pose
        self.R2c = self.R2w @ self.Rcw.T  # 相机2 在rectification后的新双目系统中的pose
        self.T12_c = self.Rcw.T @ T12  # baseline vertor 在rectification后的新双目系统中的表达，实际会退化为只有x轴上有值

        self.P = np.stack([self.X, self.Y, self.Z], axis=0).astype(np.float32)  # rectification后的新双目系统中上的三维采样点
        self.P1 = np.matmul(self.R1c.T, self.P)  # 旋转至原相机1系统中

        self.theta1 = np.arccos(self.P1[2])
        self.phi1 = np.arctan2(self.P1[1], self.P1[0])
        self.r1 = self.f1 * self.k1 @ np.stack([self.theta1, self.theta1 ** 2, self.theta1 ** 3, self.theta1 ** 4], axis=0)
        self.u1 = (self.r1 * np.cos(self.phi1) + self.c1[0]).reshape(self.h, self.w)
        self.v1 = (self.r1 * np.sin(self.phi1) + self.c1[1]).reshape(self.h, self.w)

        self.P2 = np.matmul(self.R2c.T, self.P)  # 旋转至原相机2系统中

        self.theta2 = np.arccos(self.P2[2])
        self.phi2 = np.arctan2(self.P2[1], self.P2[0])
        self.r2 = self.f2 * self.k2 @ np.stack([self.theta2, self.theta2 ** 2, self.theta2 ** 3, self.theta2 ** 4], axis=0)
        self.u2 = (self.r2 * np.cos(self.phi2) + self.c2[0]).reshape(self.h, self.w)
        self.v2 = (self.r2 * np.sin(self.phi2) + self.c2[1]).reshape(self.h, self.w)

        self.return_distance = return_distance
        self.test_mode = test_mode
        self.max_dist = max_dist
        self.fs = self.w / self.hfov
        if return_distance:
            u, v = np.meshgrid(range(self.w), range(self.h))
            self.angle_x = (1 - (u + 0.5) / self.w) * self.hfov + (np.pi - self.hfov) / 2

    def rodrigues(self, r: np.ndarray) -> np.ndarray:
        if r.size == 3:
            return R.from_euler('xyz', r.squeeze(), degrees=False).as_matrix()
        else:
            return R.from_matrix(r).as_euler('xyz', degrees=False).reshape((3, 1))

    def transform(self, results: dict) -> dict:

        idx = int(results['filename1'].split('/')[-1].split('.')[0])

        if self.old_data and idx <= self.samllT_boundary:
            baseline_v = self.T12_c / 3
        else:
            baseline_v = self.T12_c

        fisheye_img1 = results['imgl'].copy()
        fisheye_img2 = results['imgr'].copy()
        del results['imgl'], results['imgr']

        h1, w1, _ = fisheye_img1.shape
        fisheye_img1 = torch.tensor(fisheye_img1.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        grid1 = torch.tensor(np.stack([2*self.u1/w1-1, 2*self.v1/h1-1], axis=2)).unsqueeze(0)
        transequi_img1 = torch.nn.functional.grid_sample(
            fisheye_img1, grid1, align_corners=False)[0].permute(1, 2, 0).numpy().astype(np.uint8)

        h2, w2, _ = fisheye_img2.shape
        fisheye_img2 = torch.tensor(fisheye_img2.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        grid2 = torch.tensor(np.stack([2 * self.u2 / w2 - 1, 2 * self.v2 / h2 - 1], axis=2)).unsqueeze(0)
        transequi_img2 = torch.nn.functional.grid_sample(
            fisheye_img2, grid2, align_corners=False)[0].permute(1, 2, 0).numpy().astype(np.uint8)

        results['imgl'] = transequi_img1
        results['imgr'] = transequi_img2
        results['img_shape'] = transequi_img1.shape
        results['ori_shape'] = transequi_img1.shape
        results['pad_shape'] = transequi_img1.shape

        if self.test_mode:
            return results

        fisheye_distance = results['disp_gt'].copy()
        del results['disp_gt'], results['valid']
        fisheye_distance = torch.tensor(fisheye_distance).unsqueeze(0).unsqueeze(0)
        transequi_distance = \
            torch.nn.functional.grid_sample(fisheye_distance, grid1, align_corners=False, mode='nearest')[0, 0].numpy()

        P3 = np.stack([self.X * transequi_distance.flatten(), self.Y * transequi_distance.flatten(),
                       self.Z * transequi_distance.flatten()], axis=0) - baseline_v
        norm_P3 = P3 / np.linalg.norm(P3, axis=0, keepdims=True)
        angle_x_3 = np.arccos(norm_P3[0].reshape(self.h, self.w))
        u3 = self.w * (0.5 + (np.pi / 2 - angle_x_3) / self.hfov) - 0.5
        disp = (self.u - u3)
        valid = ((np.abs(disp) <= self.max_disp) & (np.abs(disp) > 1e-3) & (np.abs(transequi_distance) > 1e-2))

        results['disp_gt'] = disp
        results['valid'] = valid

        if self.return_distance:
            transequi_distance[transequi_distance < 1e-2] = 1e-2
            transequi_distance[transequi_distance > self.max_dist] = self.max_dist
            results['distance'] = transequi_distance

        return results

    def disp2depth(self, disp_pred: np.array, idx: int = -1):
        disp_pred[disp_pred < 0] = 0
        disp_pred_n = disp_pred / self.fs
        if len(disp_pred_n.shape) == 3:
            disp_pred_n = disp_pred_n[:, :, 0]

        if self.old_data and idx <= self.samllT_boundary:
            baseline_v = self.T12_c / 3
        else:
            baseline_v = self.T12_c

        dep_pred = baseline_v[0, 0] * np.sin(np.pi - (self.angle_x + disp_pred_n)) / (np.sin(disp_pred_n) + 10e-5)
        dep_pred[dep_pred < 1e-2] = 1e-2
        dep_pred[dep_pred > self.max_dist] = self.max_dist

        return dep_pred

    def disp2PointCloud(self, depth_pred: np.array):
        fovX = 150 / 180 * np.pi
        X = depth_pred.flatten() * self.P1[0]
        Y = depth_pred.flatten() * self.P1[1]
        Z = depth_pred.flatten() * self.P1[2]
        valid_mask = (self.angle_x > ((np.pi - fovX) / 2)) & (self.angle_x < ((np.pi + fovX) / 2))
        mask = (depth_pred.flatten() < 500) & (valid_mask.flatten())
        P = np.stack([X, Y, Z], axis=1)[mask, :]

        return P, mask

    def transfisheye2Equi(self, fisheye_img1: np.array):
        h1, w1, _ = fisheye_img1.shape
        fisheye_img1 = torch.tensor(fisheye_img1.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        grid1 = torch.tensor(np.stack([2*self.u1/w1-1, 2*self.v1/h1-1], axis=2)).unsqueeze(0)
        transequi_img1 = torch.nn.functional.grid_sample(fisheye_img1, grid1, align_corners=False)[0].permute(1, 2, 0).numpy().astype(np.uint8)

        return transequi_img1


@TRANSFORMS.register_module()
class Fisheye2Pinhole(BaseTransform):
    """convert fisheye image1, image2 and depth to pinhole ones.
    """

    def __init__(self,
                 max_disp: float = 512.0,
                 f1: float = 315.712,
                 c1: list = [512.0, 512.0],
                 k1: list = [1., -1.7700345707950937e-01, 2.8355590110360829e-01, -1.1359963826630388e-01],
                 R1: list = [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                 T1: list = [0., 0., 0.],
                 f2: float = 315.712,
                 c2: list = [512.0, 512.0],
                 k2: list = [1., -1.7700345707950937e-01, 2.8355590110360829e-01, -1.1359963826630388e-01],
                 R2: list = [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                 T2: list = [0.6, 0., 0.],
                 output_width: int = 768,
                 output_height: int = 768,
                 xfov: int = 100,
                 return_depth: bool = False,
                 test_mode: bool = False,
                 max_dist: float = 200,
                 old_data: bool = True,
                 samllT_boundary: int = 19301) -> None:
        super().__init__()
        self.max_disp = max_disp

        self.f1 = f1
        self.c1 = c1
        self.k1 = np.array(k1).astype(np.float32)
        self.R1w = np.array(R1).reshape(3, 3).astype(np.float32)  # 相机1 在世界坐标系中的pose
        self.T1w = np.array(T1).reshape(3, 1).astype(np.float32)  # 相机1 在世界坐标系中的位置
        self.f2 = f2
        self.c2 = c2
        self.k2 = np.array(k2).astype(np.float32)
        self.R2w = np.array(R2).reshape(3, 3).astype(np.float32)  # 相机2 在世界坐标系中的pose
        self.T2w = np.array(T2).reshape(3, 1).astype(np.float32)  # 相机2 在世界坐标系中的位置
        # self.R21 = self.R2@self.R1.transpose()

        self.old_data = old_data
        self.samllT_boundary = samllT_boundary

        self.w = output_width
        self.h = output_height
        self.xfov = xfov / 180 * np.pi

        T12 = self.T2w - self.T1w # 在世界坐标系中，相机1原点到相机2原点的向量
        cx = T12[:, 0] / LA.norm(T12)   # T12的方向作为rectification后的新双目系统的x轴方向, 仍是在世界坐标系下的表达
        z_sum = self.R1w[:, 2] + self.R2w[:, 2] # 相机1和相机2的z轴方向在世界坐标系下的和向量
        cz_ = z_sum / LA.norm(z_sum) # z_sum作为rectification后的新双目系统的z轴方向, 可以保证两图像有最大的overlap, 但由于不能保证和cx完全垂直只能作为一个粗略结果
        cy = np.cross(cz_, cx) # 根据右手坐标系原理求出rectification后的新双目系统的y轴方向
        cy = cy / LA.norm(cy)
        cz = np.cross(cx, cy) # 根据右手坐标系原理求出最终的rectification后的新双目系统的z轴方向
        cz = cz / LA.norm(cz)
        self.Rcw = np.stack([cx, cy, cz], axis=1)  # rectification后的新双目系统 在世界坐标系中的pose  

        # 对原双目相机进行 rectification
        self.R1c = self.R1w @ self.Rcw.T  # 相机1 在rectification后的新双目系统中的pose
        self.R2c = self.R2w @ self.Rcw.T  # 相机2 在rectification后的新双目系统中的pose
        self.T12_c = self.Rcw.T @ T12  # baseline vertor 在rectification后的新双目系统中的表达，实际会退化为只有x轴上有值

        self.u, self.v = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))

        self.pinhole_focal = self.w / 2 / np.tan(self.xfov/2)
        self.X = (self.u - self.w / 2+0.5).flatten()
        self.Y = (self.v - self.h / 2+0.5).flatten()
        self.Z = np.ones_like(self.X) * self.pinhole_focal

        self.P0 = np.stack([self.X, self.Y, self.Z], axis=0).astype(np.float32)  # rectification后的新双目系统中上的三维采样点
        self.P = self.P0 / LA.norm(self.P0, axis=0)

        # left
        self.P1 = np.matmul(self.R1c.T, self.P)  # 旋转至原相机1系统中
        self.angle_z1 = np.arccos(self.P1[2])

        self.theta1 = np.arccos(self.P1[2])
        self.phi1 = np.arctan2(self.P1[1], self.P1[0])
        self.r1 = self.f1 * self.k1 @ np.stack([self.theta1, self.theta1 ** 2, self.theta1 ** 3, self.theta1 ** 4],
                                               axis=0)
        self.u1 = (self.r1 * np.cos(self.phi1) + self.c1[0]).reshape(self.h, self.w)
        self.v1 = (self.r1 * np.sin(self.phi1) + self.c1[1]).reshape(self.h, self.w)

        # right
        self.P2 = np.matmul(self.R2c.T, self.P)  # 旋转至原相机2系统中
        self.angle_z2 = np.arccos(self.P2[2])

        self.theta2 = np.arccos(self.P2[2])
        self.phi2 = np.arctan2(self.P2[1], self.P2[0])
        self.r2 = self.f2 * self.k2 @ np.stack([self.theta2, self.theta2 ** 2, self.theta2 ** 3, self.theta2 ** 4],
                                               axis=0)
        self.u2 = (self.r2 * np.cos(self.phi2) + self.c2[0]).reshape(self.h, self.w)
        self.v2 = (self.r2 * np.sin(self.phi2) + self.c2[1]).reshape(self.h, self.w)

        self.return_depth = return_depth
        self.test_mode = test_mode
        self.max_dist = max_dist

    # def rodrigues(self, r: np.ndarray) -> np.ndarray:
    #     if r.size == 3:
    #         return R.from_euler('xyz', r.squeeze(), degrees=False).as_matrix()
    #     else:
    #         return R.from_matrix(r).as_euler('xyz', degrees=False).reshape((3, 1))

    def transform(self, results: dict) -> dict:

        idx = int(results['filename1'].split('/')[-1].split('.')[0])

        if self.old_data and idx <= self.samllT_boundary:
            baseline_v = self.T12_c / 3
        else:
            baseline_v = self.T12_c

        fisheye_img1 = results['imgl'].copy()
        fisheye_img2 = results['imgr'].copy()
        del results['imgl'], results['imgr']

        h1, w1, _ = fisheye_img1.shape
        fisheye_img1 = torch.tensor(fisheye_img1.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        grid1 = torch.tensor(np.stack([2 * self.u1 / w1 - 1, 2 * self.v1 / h1 - 1], axis=2)).unsqueeze(0)
        pinhole_img1 = torch.nn.functional.grid_sample(fisheye_img1, grid1, align_corners=False)[0] \
            .permute(1, 2, 0).numpy().astype(np.uint8)

        h2, w2, _ = fisheye_img2.shape
        fisheye_img2 = torch.tensor(fisheye_img2.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        grid2 = torch.tensor(np.stack([2 * self.u2 / w2 - 1, 2 * self.v2 / h2 - 1], axis=2)).unsqueeze(0)
        pinhole_img2 = torch.nn.functional.grid_sample(fisheye_img2, grid2, align_corners=False)[0] \
            .permute(1, 2, 0).numpy().astype(np.uint8)

        results['imgl'] = pinhole_img1
        results['imgr'] = pinhole_img2
        results['img_shape'] = pinhole_img1.shape
        results['ori_shape'] = pinhole_img1.shape
        results['pad_shape'] = pinhole_img1.shape

        if self.test_mode:
            return results

        fisheye_distance = results['disp_gt'].copy()
        del results['disp_gt'], results['valid']
        fisheye_distance = torch.tensor(fisheye_distance).unsqueeze(0).unsqueeze(0)
        pinhole_distance = torch.nn.functional.grid_sample(fisheye_distance, grid1,
                                                           align_corners=False, mode='nearest')[0, 0].numpy()
        """ bug
        depth_res = LA.norm(baseline_v) * self.cube_focal / (pinhole_distance + 1e-9)
        f = LA.norm(self.P0, axis=0).reshape(self.h, self.w)
        disp = f * depth_res / self.cube_focal
        """
        
        pinhole_depth = pinhole_distance * self.P[2].reshape(self.h, self.w)
        disp = LA.norm(baseline_v) * self.pinhole_focal / pinhole_depth

        valid = ((np.abs(disp) <= self.max_disp) & (np.abs(disp) > 1e-3) & (np.abs(pinhole_distance) > 1e-2))

        results['disp_gt'] = disp
        results['valid'] = valid

        if self.return_depth:
            pinhole_depth[pinhole_depth < 1e-2] = 1e-2
            pinhole_depth[pinhole_depth > self.max_dist] = self.max_dist
            results['depth'] = pinhole_depth

        return results


@TRANSFORMS.register_module()
class LoadStereoImageFromFile(LoadImageFromFile):
    """Load image1 and image2 from file.

    Required keys are "img1_info" (dict that must contain the key "filename"
    and "filename2"). Added or updated keys are "img1", "img2", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0, 1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 rotate: bool = False,
                 imdecode_backend: str = 'cv2') -> None:
        super().__init__()
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.rotate = rotate

    def transform(self, results: dict) -> dict:
        """Call function to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename1 = results['img_info']['filename1']
        filename2 = results['img_info']['filename2']
        if (not osp.isfile(filename1)) or (not osp.isfile(filename2)):
            raise RuntimeError(
                f'Cannot load file from {filename1} or {filename2}')

        img1_bytes = fileio.get(filename1)
        img2_bytes = fileio.get(filename2)

        img1 = mmcv.imfrombytes(
            img1_bytes, flag=self.color_type, backend=self.imdecode_backend)
        img2 = mmcv.imfrombytes(
            img2_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.rotate:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # if img1 is None:
        #     print(filename1)
        # if img2 is None:
        #     print(filename2)
        assert img1 is not None

        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['filename1'] = filename1
        results['filename2'] = filename2
        results['ori_filename1'] = osp.split(filename1)[-1]
        results['ori_filename2'] = osp.split(filename2)[-1]

        results['imgl'] = img1
        results['imgr'] = img2

        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img1.shape
        results['scale_factor'] = np.array([1.0, 1.0])
        num_channels = 1 if len(img1.shape) < 3 else img1.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@TRANSFORMS.register_module()
class LoadDispAnnotations(LoadAnnotations):
    """Load disparity from file.

    Args:
        with_occ (bool): whether to parse and load occlusion mask.
            Default to False.
        sparse (bool): whether the disparity is sparse. Default to False.
    """

    def __init__(
            self,
            eth3d_training: bool = False,
            with_occ: bool = False,
            sparse: bool = False,
            max_disp: float = 512.0,
            disp_divison: float = 32.0,
            sintel_mode: bool = False,
            fat_mode: bool = False,
            tartanair_mode: bool = False,
            kenburns_mode: bool = False,
            davanet_mode: bool = False,
            realscene_mode: bool = False,
            dev3_mode: bool = False,
            dev3_lidar_mode: bool = False,
            hand_lidar_mode: bool = False,
            snow_syn_mode: bool = False,
            dilate_disp: bool = False,
            dilate_kernel: int = 3,
            zero_disp_thres: float = 0.0, # recommend 0.75 or 0.85 or 0.95
    ) -> None:
        self.eth3d_training = eth3d_training
        self.with_occ = with_occ
        self.sparse = sparse
        self.max_disp = max_disp
        self.disp_divison = disp_divison
        self.sintel_mode = sintel_mode
        self.fat_mode = fat_mode
        self.tartanair_mode = tartanair_mode
        self.kenburns_mode = kenburns_mode
        self.davanet_mode = davanet_mode
        self.realscene_mode = realscene_mode
        self.dev3_mode = dev3_mode
        self.dev3_lidar_mode = dev3_lidar_mode
        self.hand_lidar_mode = hand_lidar_mode
        self.snow_syn_mode = snow_syn_mode
        self.dilate_disp = dilate_disp
        self.dilate_kernel = (dilate_kernel, dilate_kernel) if isinstance(dilate_kernel, int) else dilate_kernel
        self.zero_disp_thres = zero_disp_thres

    def transform(self, results: dict) -> dict:
        """Call function to load disparity and occlusion mask (optional).

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """


        if self.sparse:
            results = self._load_sparse_disp(results)
        else:
            results = self._load_disp(results)
        if self.with_occ:
            results = self._load_occ(results)
        return results

    def _load_handlidar_cam_params(self, folder, split):
        yml_file = osp.join(folder, split, 'lidar_stereo_camera_param.yml')
        fs = cv2.FileStorage(yml_file, cv2.FILE_STORAGE_READ)
        K_l = fs.getNode('K_l').mat().reshape(-1).tolist()
        T_cl_cr = fs.getNode('T_cl_cr').mat().reshape(-1).tolist()
        focal = K_l[0]
        baseline = T_cl_cr[0] if split == "horizon" else T_cl_cr[1]
        return focal, baseline
    
    def _check_lidar_path(self, filename_disp):
        return not "pgt" in filename_disp

    def _load_disp(self, results: dict) -> dict:
        """load dense disparity function.

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        filenames = list(results['ann_info'].keys())
        skip_len = len('filename_')

        for filename in filenames:
            if filename.find('disp') > -1:

                filename_disp = results['ann_info'][filename]
                if not os.path.exists(filename_disp):
                    img_l = results['imgl']
                    disp = np.zeros((img_l.shape[0], img_l.shape[1]), dtype=np.float32)
                elif self.sintel_mode:
                    f_in = cv2.imread(filename_disp, cv2.IMREAD_UNCHANGED)
                    d_r = f_in[:, :, 2].astype(np.float32)
                    d_g = f_in[:, :, 1].astype(np.float32)
                    d_b = f_in[:, :, 0].astype(np.float32)
                    disp = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
                elif self.fat_mode:
                    f_in = cv2.imread(filename_disp, cv2.IMREAD_UNCHANGED)
                    depth = f_in.astype(np.float32) / 10000
                    baseline = 0.06
                    camera_file = osp.dirname(filename_disp) + "/_camera_settings.json"
                    camera = json.load(open(camera_file))
                    fx = camera["camera_settings"][0]["intrinsic_settings"]["fx"]
                    disp = fx * baseline / depth
                elif self.tartanair_mode:
                    depth = np.load(filename_disp)
                    disp = 80.0 / depth
                elif self.kenburns_mode:
                    depth = cv2.imread(filename_disp, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    meta_file_name = filename_disp.replace('-depth', '')[:-7]+'-meta.json'
                    fltFov = json.loads(open(meta_file_name, 'r').read())['fltFov']
                    fltFocal = 0.5 * 512 * np.tan(np.radians(90.0) - (0.5 * np.radians(fltFov)))
                    fltBaseline = 40.0
                    disp = (fltFocal * fltBaseline) / depth
                elif self.davanet_mode:
                    disp = pyexr.open(filename_disp).get()[:, :, 0]
                    disp[np.isnan(disp)] = -1
                    disp[np.isinf(disp)] = -1
                elif self.realscene_mode:
                    if "defom_pgt" in filename_disp:
                        disp = read_pfm(filename_disp)
                        disp[disp<0.0011] = 0.0011 # to avoid being mased as invalid
                    else:
                        h, w = results['img_shape'][0], results['img_shape'][1]
                        disp = np.zeros([h, w], dtype=np.float32)
                elif self.dev3_mode:
                    if "defom_pgt" in filename_disp:
                        disp = read_pfm(filename_disp)
                        disp[disp<0.0011] = 0.0011 # to avoid being mased as invalid
                    elif "combined_uint16" in filename_disp:
                        disp = cv2.imread(filename_disp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                        disp = disp.astype(np.float32)/1000
                        disp[disp<0.0011] = 0.0011
                    else:
                        h, w = results['img_shape'][0], results['img_shape'][1]
                        disp = np.zeros([h, w], dtype=np.float32)
                elif self.dev3_lidar_mode:
                    #disp = read_pfm(filename_disp)
                    if filename_disp[-3:] == 'pfm': #ckk
                        disp = read_pfm(filename_disp)
                    elif filename_disp[-3:] == 'exr':
                        disp = cv2.imread(filename_disp, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if self._check_lidar_path(filename_disp):
                        lidar_depth = disp
                        invalid_mask = lidar_depth < 0.001
                        lidar_depth[invalid_mask] = 0.001
                        bf = 0.05 * 570.96
                        disp = bf / lidar_depth
                        disp[invalid_mask] = 0
                    else:
                        disp[disp<0.0011] = 0.0011
                    disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE) # lidar data's gt is saved as vertical
                elif self.hand_lidar_mode:
                    if self._check_lidar_path(filename_disp):
                        # lidar points
                        if filename_disp[-3:] == 'pfm': #ckk
                            lidar_depth = read_pfm(filename_disp)
                        elif filename_disp[-3:] == 'exr':
                            lidar_depth = cv2.imread(filename_disp, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                            if len(lidar_depth.shape) > 2:
                                lidar_depth = lidar_depth[..., 0]
                        invalid_mask = lidar_depth < 0.001
                        lidar_depth[invalid_mask] = 0.001
                        split_folder = osp.dirname(osp.dirname(filename_disp))
                        split = osp.basename(split_folder)
                        seq_folder = osp.dirname(split_folder)
                        f, b = self._load_handlidar_cam_params(seq_folder, split)
                        bf = b * f
                        disp = bf / lidar_depth
                        disp[invalid_mask] = 0
                        disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE) # lidar data's gt is saved as vertical
                    else:
                        # defom disparity
                        disp = read_pfm(filename_disp)
                        disp[disp<0.0011] = 0.0011
                        disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE) # lidar data's gt is saved as vertical
                elif self.snow_syn_mode:
                    depth = cv2.imread(filename_disp,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    invalid_mask = depth < 0.001
                    depth[invalid_mask] = 0.001
                    fx = 544
                    baseline = 0.05
                    if ('bottom' in filename_disp):
                        fx = 240
                        baseline = 0.047
                    disp = fx * baseline / depth
                    invalid = np.isnan(disp) | np.isinf(disp)
                    disp[disp < 0.0011] = 0.0011
                    disp[invalid] = 0
                    disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE) # lidar data's gt is saved as vertical
                elif filename_disp[-3:] == 'png':
                    disp = cv2.imread(filename_disp, cv2.IMREAD_UNCHANGED)
                    disp = disp.astype(np.float32) / self.disp_divison
                elif filename_disp[-3:] == 'exr':
                    disp = cv2.imread(filename_disp, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if len(disp.shape) > 2:
                        disp = disp[..., 0]
                elif filename_disp[-3:] == 'npy':
                    disp = np.load(filename_disp)
                else:
                    disp_bytes = fileio.get(filename_disp)
                    disp = disp_from_bytes(disp_bytes, filename_disp[-3:])
                
                if self.zero_disp_thres > 0.0:
                    zero_mask = (disp > 1e-3) & (disp < self.zero_disp_thres)
                    disp[zero_mask] = 0.0011
                
                if self.dilate_disp:
                    # use 3x3 kernel to dilate the disparity map
                    disp = cv2.dilate(disp, np.ones(self.dilate_kernel, np.uint8), iterations=1)

                if self.eth3d_training:
                    disp = disp.copy()
                    disp[np.isinf(disp)] = 0.0

                valid = (disp <= self.max_disp) & (disp > 1e-3)
                results[filename] = filename_disp
                results['ori_' + filename] = osp.split(filename_disp)[-1]
                ann_key = filename[skip_len:] + '_gt'
                results[ann_key] = disp
                results['valid'] = valid
                results['ann_fields'].append(ann_key)

        return results

    def _load_sparse_disp(self, results: dict) -> dict:
        """load sparse disparity function.

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        filenames = list(results['ann_info'].keys())
        skip_len = len('filename_')

        for filename in filenames:

            if filename.find('disp') > -1:
                filename_disp = results['ann_info'][filename]
                disp_bytes = fileio.get(filename_disp)
                disp, valid = sparse_flow_from_bytes(disp_bytes)

                results[filename] = filename_disp
                results['ori_' + filename] = osp.split(filename_disp)[-1]
                ann_key = filename[skip_len:] + '_gt'
                # sparse disparity dataset don't include backward disparity
                results['valid'] = valid
                results[ann_key] = disp
                results['ann_fields'].append(ann_key)

        return results

    def _load_occ(self, results: dict) -> dict:
        """load annotation function.

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        filenames = list(results['ann_info'].keys())
        skip_len = len('filename_')

        for filename in filenames:
            if filename.find('occ') > -1:
                filename_occ = results['ann_info'][filename]
                occ_bytes = fileio.get(filename_occ)
                occ = (mmcv.imfrombytes(occ_bytes, flag='grayscale') /
                       255).astype(np.float32)

                results[filename] = filename_occ
                results['ori_' + filename] = osp.split(filename_occ)[-1]
                ann_key = filename[skip_len:] + '_gt'
                results[ann_key] = occ
                results['ann_fields'].append(ann_key)

        return results


@TRANSFORMS.register_module()
class LoadPseudoAnnotations(LoadAnnotations):
    """Load disparity from file.

    Args:
        davanet_mode: for DAVANet dataset
        realscene_mode: for realscene dataset
    """

    REALSCENE_MAP = {
        "v1": "left_depth_anything_v1",
        "v2": "left_depth_anything_v2",
    }
    DEV3_MAP = {
        "v2": "left_depth_anything_v2",
    }
    HANDLIDAR_MAP = {
        "v2": "dav2_uint16",
    }
    DEV3LIDAR_MAP = {
        "v2": "DAdepth_0",
    }
    PSD_MAPS = {
        "realscene":REALSCENE_MAP,
        "dev3":DEV3_MAP,
        "hand_lidar":HANDLIDAR_MAP,
        "dev3_lidar":DEV3LIDAR_MAP,
    }

    def __init__(
            self,
            gt_suffix:str=None, # should be aligned with dataset
            davanet_mode: bool = False,
            realscene_mode: bool = False,
            dev3_mode: bool = False,
            dev3_lidar_mode: bool = False,
            hand_lidar_mode: bool = False,
            use_true_dav2 = False,
            dan_v: str = "v2",
    ) -> None:
        self.gt_suffix = gt_suffix
        self.davanet_mode = davanet_mode
        self.realscene_mode = realscene_mode
        self.dev3_mode = dev3_mode
        self.dev3_lidar_mode = dev3_lidar_mode
        self.hand_lidar_mode = hand_lidar_mode
        self.use_true_dav2 = use_true_dav2
        self.dan_v = dan_v
        self.epsilon = 1e-10

    def transform(self, results: dict) -> dict:
        """Call function to load disparity and occlusion mask (optional).

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """

        results = self._load_disp(results)
        return results

    def _gen_psd_path(self, filename_disp, key:str):
        if not self.use_true_dav2:
            return filename_disp
        ss = filename_disp.split('/')[-2]
        ret = filename_disp.replace(ss, self.PSD_MAPS[key][self.dan_v])
        if self.gt_suffix is None:
            raise ValueError("gt_suffix is not set, psd file can not be found correctly without it.")
        ret = ret.replace(self.gt_suffix, ".pfm")
        if not osp.exists(ret):
            ret = ret.replace(".pfm", ".png")
        if not osp.exists(ret):
            ret = ret.replace(".png", ".exr")
        return ret

    def _load_disp(self, results: dict) -> dict:
        """load dense disparity function.

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        filenames = list(results['ann_info'].keys())
        skip_len = len('filename_')

        for filename in filenames:

            if filename.find('disp') > -1:

                filename_disp = results['ann_info'][filename]
                right_psd_disp = None
                if not os.path.exists(filename_disp):
                    pseudo_disp = results['disp_gt']
                    valid = np.zeros_like(pseudo_disp[:1, :1])
                if self.davanet_mode:
                    # depth_anything_file = filename_disp.replace("disparity", "depth_anything")
                    depth_anything_file = filename_disp.replace("disparity_left", "depth_anything_left_"+self.dan_v).replace("disparity_right", "depth_anything_right_"+self.dan_v)
                    pseudo_disp = cv2.imread(depth_anything_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if len(pseudo_disp.shape) > 2:
                        pseudo_disp = pseudo_disp[..., 0]
                    valid = np.ones_like(pseudo_disp[:1, :1])
                elif self.realscene_mode:
                    depth_anything_file = self._gen_psd_path(filename_disp, "realscene")
                    pseudo_disp = cv2.imread(depth_anything_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if len(pseudo_disp.shape) > 2:
                        pseudo_disp = pseudo_disp[..., 0]
                    valid = np.ones_like(pseudo_disp[:1, :1])
                    
                elif self.dev3_mode:
                    depth_anything_file = self._gen_psd_path(filename_disp, "dev3")
                    #print(depth_anything_file)
                    if np.max(results['disp_gt'])<0.1:
                        pseudo_disp = np.random.rand(1088,1280).astype(np.float32)
                        valid = np.zeros_like(pseudo_disp[:1, :1])
                    else:
                        pseudo_disp = cv2.imread(depth_anything_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        if len(pseudo_disp.shape) > 2:
                            pseudo_disp = pseudo_disp[..., 0]
                        valid = np.ones_like(pseudo_disp[:1, :1])
                    # pseudo_disp = cv2.imread(depth_anything_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    # print(depth_anything_file)
                    # if len(pseudo_disp.shape) > 2:
                    #     pseudo_disp = pseudo_disp[..., 0]
                    # if np.max(results['disp_gt'])<0.1:
                    #     valid = np.zeros_like(pseudo_disp[:1, :1])
                    # else:
                    #     valid = np.ones_like(pseudo_disp[:1, :1])
                elif self.dev3_lidar_mode:
                    depth_anything_file = self._gen_psd_path(filename_disp, "dev3_lidar")
                    pseudo_disp = cv2.imread(depth_anything_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if len(pseudo_disp.shape) > 2:
                        pseudo_disp = pseudo_disp[..., 0]
                    # rotate 
                    pseudo_disp = cv2.rotate(pseudo_disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    valid = np.ones_like(pseudo_disp[:1, :1])
                elif self.hand_lidar_mode:
                    # currently only support defom_pgt as pseudogt
                    depth_anything_file = self._gen_psd_path(filename_disp, "hand_lidar")
                    pseudo_disp = cv2.imread(depth_anything_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    if len(pseudo_disp.shape) > 2:
                        pseudo_disp = pseudo_disp[..., 0]
                    pseudo_disp = cv2.rotate(pseudo_disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    valid = np.ones_like(pseudo_disp[:1, :1])
                else:
                    depth_anything_file = filename_disp
                    pseudo_disp = results['disp_gt']
                    valid = np.zeros_like(pseudo_disp[:1, :1])

                median = np.median(pseudo_disp)
                scele = np.mean(np.abs(pseudo_disp-median))

                pseudo_disp = (pseudo_disp-median)/(scele + self.epsilon)

                results["filename_pseudo"] = depth_anything_file
                results['ori_' + "filename_pseudo"] = osp.split(depth_anything_file)[-1]
                ann_key = 'pseudo_gt'
                results[ann_key] = pseudo_disp
                results['pseudo_valid'] = valid
                results['ann_fields'].append(ann_key)

        return results


@TRANSFORMS.register_module()
class LoadFisheyeGrid(BaseTransform):
    """Load fisheye grid table from file.

    Args:
        with_occ (bool): whether to parse and load occlusion mask.
            Default to False.
        sparse (bool): whether the disparity is sparse. Default to False.
    """

    def __init__(
            self,
            root: str = None,
            config_file: str = None,
            load_lut: bool = False,
            remove_75_more=False,
    ) -> None:

        self.config_file = config_file
        self.load_lut = load_lut
        self.root = root
        self.remove_75_more = remove_75_more

    def transform(self, results: dict) -> dict:
        """Call function to load disparity and occlusion mask (optional).

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        config = yaml.safe_load(open(self.config_file))
        cameras_cfg = config['cameras']
        self.fcams = FisheyeCamModel()
        self.fcams.setConfig(cameras_cfg)

        self.out_size = config['config']['out_size']
        self.pano_size = config['config']['pano_size']
        self.valid_out_size = config['config']['valid_out_size']
        self.valid_lut_fmt = 'valid_lt_(%d,%d).hwd'
        self.inv_valid_lut_fmt = 'inv_valid_lt_(%d,%d).hwd'
        self.corr_lut_fmt = 'corr_lt_(%d,%d,%d).hwd'
        self.f2vpano_lut_fmt = 'f2vpano_lt_(%d,%d).hwd'
        self.vpano2f_lut_fmt = 'vpano2f_lt_(%d,%d).hwd'
        self.num_disp = 192
        self.min_disp = config['config']['min_disp']
        self.max_disp = 192
        self.sample_step_disp = 1
        self.fov = config['config']['fov']
        self.__initSweep(self.load_lut)
        results['valid_grids'] = self.valid_grids
        results['inv_valid_grids'] = self.inv_valid_grids
        results['corr_grids'] = self.corr_grids
        results['f2vpano_grids'] = self.f2vpano_grids
        results['vpano2f_grids'] = self.vpano2f_grids

        self.__generateValidRegions(results)

        return results

    def __generateValidRegions(self, results):

        h, w = results['valid'].shape
        if not osp.exists(osp.join(self.root, 'valid_0_15.hwd')):
            factor = h / self.fcams.height
            focus = self.fcams.f1 * factor
            cx, cy = self.fcams.cx1 * factor, self.fcams.cy1 * factor

            max_fov_rad = self.fcams.max_theta * 0.5 / 180.0 * np.pi
            max_r = focus * self.fcams.UndisortTheta(max_fov_rad, self.fcams.k1)

            x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            y = y.reshape(1, -1) - cy
            x = x.reshape(1, -1) - cx
            rho = np.sqrt(x * x + y * y)

            theta = []
            for i in range(0, rho.shape[1]):
                if rho[0, i] > max_r:
                    theta.append(self.fcams.max_theta / 180.0 * np.pi)
                    continue
                theta.append(self.fcams.getThetaInDistort(rho[0, i], focus, self.fcams.k1))
            theta = (np.array(theta)).reshape(1, -1).astype(np.float32)

            fi = np.arctan2(y, x)
            x1 = np.sin(theta) * np.cos(fi)
            lat = np.abs(np.pi / 2 - np.arccos(x1))
            lat = lat.reshape(h, w)

            valid_0_15 = (lat <= np.pi / 12)
            valid_15_30 = (lat > np.pi / 12) & (lat <= np.pi / 6)
            valid_30_45 = (lat > np.pi / 6) & (lat <= np.pi / 4)
            valid_45_60 = (lat > np.pi / 4) & (lat <= np.pi / 3)
            valid_60_75 = (lat > np.pi / 3) & (lat <= 5 * np.pi / 12)
            valid_75_more = (lat > 5 * np.pi / 12)

            valid_0_15.tofile(osp.join(self.root, 'valid_0_15.hwd'))
            valid_15_30.tofile(osp.join(self.root, 'valid_15_30.hwd'))
            valid_30_45.tofile(osp.join(self.root, 'valid_30_45.hwd'))
            valid_45_60.tofile(osp.join(self.root, 'valid_45_60.hwd'))
            valid_60_75.tofile(osp.join(self.root, 'valid_60_75.hwd'))
            valid_75_more.tofile(osp.join(self.root, 'valid_75_more.hwd'))
        else:
            valid_0_15 = np.fromfile(osp.join(self.root, 'valid_0_15.hwd'), dtype=np.bool_).reshape(h, w)
            valid_15_30 = np.fromfile(osp.join(self.root, 'valid_15_30.hwd'), dtype=np.bool_).reshape(h, w)
            valid_30_45 = np.fromfile(osp.join(self.root, 'valid_30_45.hwd'), dtype=np.bool_).reshape(h, w)
            valid_45_60 = np.fromfile(osp.join(self.root, 'valid_45_60.hwd'), dtype=np.bool_).reshape(h, w)
            valid_60_75 = np.fromfile(osp.join(self.root, 'valid_60_75.hwd'), dtype=np.bool_).reshape(h, w)
            valid_75_more = np.fromfile(osp.join(self.root, 'valid_75_more.hwd'), dtype=np.bool_).reshape(h, w)

        results["valid_0_15"] = valid_0_15 & results['valid']
        results["valid_15_30"] = valid_15_30 & results['valid']
        results["valid_30_45"] = valid_30_45 & results['valid']
        results["valid_45_60"] = valid_45_60 & results['valid']
        results["valid_60_75"] = valid_60_75 & results['valid']
        results["valid_75_more"] = valid_75_more & results['valid']

        if self.remove_75_more:
            results['valid'] = results['valid'] & (~valid_75_more)

    def __initSweep(self, load_lut=True):
        self.left_f2s = self.fcams.calc_left_fisheye2sphere()
        self.right_s2f = self.fcams.calc_right_pano2fisheye()

        delta_t = np.abs(self.fcams.R1 * self.fcams.T2 -
                         self.fcams.R2 * self.fcams.T1)
        # = self.fcams.T1-self.fcams.R1@self.fcams.R2.transpose()@self.fcams.T2

        f = max(self.fcams.f1, self.fcams.f2)
        bf = np.linalg.norm(delta_t, ord=2) * f

        self.disp = np.arange(self.min_disp, self.max_disp, self.sample_step_disp, dtype=np.float64)
        self.depth = bf / self.disp

        if load_lut: self.__loadOrBuildLookupTable()

    def __loadOrBuildLookupTable(self) -> None:
        out_h, out_w = self.out_size
        valid_out_h, valid_out_w = self.valid_out_size
        valid_path = osp.join(self.root, self.valid_lut_fmt % (valid_out_h, valid_out_w))
        inv_valid_path = osp.join(self.root, self.inv_valid_lut_fmt % (out_h, out_w))
        corr_path = osp.join(self.root, self.corr_lut_fmt % (out_h, out_w, self.num_disp))
        f2vpano_path = osp.join(self.root, self.f2vpano_lut_fmt % (self.pano_size[0], self.pano_size[1]))
        vpano2f_path = osp.join(self.root, self.vpano2f_lut_fmt % (self.fcams.height, self.fcams.width))

        if not osp.exists(valid_path) or not osp.exists(inv_valid_path) or not osp.exists(corr_path) \
                or not osp.exists(f2vpano_path) or not osp.exists(vpano2f_path):
            self.f2vpano_grids = self.buildFisheye2VerticalPanoTable()
            self.vpano2f_grids = self.buildVerticalPano2FisheyeTable()
            self.corr_grids = self.buildRightCorrTable()
            self.valid_grids = self.buildFisheyeValidTable()
            self.inv_valid_grids = self.buildInvFisheyeValidTable()
            np.concatenate(toNumpy(self.f2vpano_grids)[np.newaxis, ...], axis=0).tofile(f2vpano_path)
            np.concatenate(toNumpy(self.vpano2f_grids)[np.newaxis, ...], axis=0).tofile(vpano2f_path)
            np.concatenate(toNumpy(self.valid_grids)[np.newaxis, ...], axis=0).tofile(valid_path)
            np.concatenate(toNumpy(self.inv_valid_grids)[np.newaxis, ...], axis=0).tofile(inv_valid_path)
            np.concatenate(toNumpy(self.corr_grids)[np.newaxis, ...], axis=0).tofile(corr_path)
        else:
            self.corr_grids = np.fromfile(corr_path, dtype=np.float32).reshape(
                [int(self.num_disp / 2), out_h, out_w, 2])
            self.valid_grids = np.fromfile(valid_path, dtype=np.float32).reshape(
                [valid_out_h, valid_out_w, 2])
            self.inv_valid_grids = np.fromfile(inv_valid_path, dtype=np.float32).reshape(
                [out_h, out_w, 2])
            self.f2vpano_grids = np.fromfile(f2vpano_path, dtype=np.float32).reshape(
                [self.pano_size[0], self.pano_size[1], 2])
            self.vpano2f_grids = np.fromfile(vpano2f_path, dtype=np.float32).reshape(
                [self.fcams.height, self.fcams.width, 2])

    def buildRightCorrTable(self, num_disp=None):
        if num_disp is None: num_disp = self.num_disp
        num_disp_2 = int(num_disp / 2)
        # all cameras have the same distortion model and parameters
        w = self.fcams.width
        h = self.fcams.height

        grids = np.zeros((num_disp_2, h, w, 2), dtype=np.float32)
        for d in range(num_disp_2):
            depth = self.depth[2 * d]
            pts = self.fcams.calc_l2r_viewmap(depth, self.left_f2s, self.right_s2f)
            grid = pixelToGrid(pts, (self.fcams.height, self.fcams.width),
                               (h, w))
            grids[d, ...] = grid.astype(np.float32)
        return grids

    def buildFisheyeValidTable(self):
        # all cameras have the same distortion model and parameters
        pts = self.fcams.calc_left_valid_fisheye()
        grid = pixelToGrid(pts, (self.fcams.valid_height, self.fcams.valid_width),
                           (self.fcams.height, self.fcams.width))
        grid = grid.astype(np.float32)
        return grid

    def buildInvFisheyeValidTable(self):
        # all cameras have the same distortion model and parameters
        pts = self.fcams.calc_left_inv_valid_fisheye()
        grid = pixelToGrid(pts, (self.fcams.height, self.fcams.width),
                           (self.fcams.valid_height, self.fcams.valid_width))
        grid = grid.astype(np.float32)
        return grid

    def buildVerticalPano2FisheyeTable(self):
        # all cameras have the same distortion model and parameters
        pts = self.fcams.calc_left_verticalpano2fisheye(self.pano_size[0], self.pano_size[1], self.fov)
        grid = pixelToGrid(pts, (self.fcams.height, self.fcams.width), self.pano_size)
        grid = grid.astype(np.float32)
        return grid

    def buildFisheye2VerticalPanoTable(self):
        # all cameras have the same distortion model and parameters
        pts = self.fcams.calc_left_fisheye2verticalpano(self.pano_size[0], self.pano_size[1],
                                                        self.fov)
        grid = pixelToGrid(pts, self.pano_size, (self.fcams.height, self.fcams.width))
        grid = grid.astype(np.float32)
        return grid


@TRANSFORMS.register_module()
class LoadImageFromWebcam(LoadStereoImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def transform(self, results: dict) -> dict:
        """Call function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        imgl = results['imgl']
        imgr = results['imgr']
        if self.to_float32:
            imgl = imgl.astype(np.float32)
            imgr = imgr.astype(np.float32)

        results['filename1'] = None
        results['ori_filename1'] = None
        results['filename2'] = None
        results['ori_filename2'] = None
        results['imgl'] = imgl
        results['imgr'] = imgr
        results['img_shape'] = imgl.shape
        results['ori_shape'] = imgl.shape
        results['img_fields'] = ['imgl', 'imgr']
        # Set initial values for default meta_keys
        results['pad_shape'] = imgl.shape
        results['scale_factor'] = np.array([1.0, 1.0])

        print("LoadStereoImageFromFile results key", results["ann_info"].keys())

        return results
    
@TRANSFORMS.register_module()
class LoadSegAnnotations(LoadAnnotations):
    """Load segmentation from file.

    Args:
        with_occ (bool): whether to parse and load invalid mask.
            Default to False.
        sparse (bool): whether the segmentation is sparse. Default to False.
    """

    _bad_seg_cache = {}

    def __init__(self,
                 rotate: bool = False,
                 bad_seg_list_txt: str = None):
        self.rotate = rotate
        # 仅通过 txt 文件读取需屏蔽的分割样本
        self._bad_seg_set = set()
        if bad_seg_list_txt is not None and isinstance(bad_seg_list_txt, str) and os.path.exists(bad_seg_list_txt):
            cached = LoadSegAnnotations._bad_seg_cache.get(bad_seg_list_txt)
            if cached is None:
                loaded = set()
                try:
                    with open(bad_seg_list_txt, 'r', encoding='utf-8') as f:
                        for line in f:
                            s = line.strip()
                            if s:
                                loaded.add(s)
                except Exception:
                    loaded = set()
                try:
                    print(f"LoadSegAnnotations: 已加载坏分割列表 {bad_seg_list_txt}，共 {len(loaded)} 条")
                except Exception:
                    pass
                LoadSegAnnotations._bad_seg_cache[bad_seg_list_txt] = loaded
                cached = loaded
            self._bad_seg_set = cached

    def transform(self, results: dict) -> dict:
        """Call function to load disparity and occlusion mask (optional).

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        results = self._load_seg(results)
        return results

    def _load_seg(self, results: dict) -> dict:
        """load dense segmentation function.

        Args:
            results (dict): Result dict from :obj:`ysstereo.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        filenames = list(results['ann_info'].keys())
        skip_len = len('filename_')

        for filename in filenames:
            if filename.find('seg') > -1:
                filename_seg = results['ann_info'][filename]
                is_bad = filename_seg in self._bad_seg_set
                if not os.path.exists(filename_seg) or is_bad:
                    img_l = results['imgl']
                    seg_in = np.ones((img_l.shape[0], img_l.shape[1]), dtype=np.uint8) * 255
                    valid  = np.zeros((img_l.shape[0], img_l.shape[1]), dtype=bool)
                else:
                    seg_in = cv2.imread(filename_seg, cv2.IMREAD_UNCHANGED)
                    if self.rotate:
                        seg_in = cv2.rotate(seg_in, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    valid  = np.ones((seg_in.shape[0], seg_in.shape[1]), dtype=bool)

                results[filename] = filename_seg
                ann_key = filename[skip_len:] + '_gt'
                results[ann_key] = seg_in
                results['seg_valid'] = valid
                results['ann_fields'].append(ann_key)

        return results



    