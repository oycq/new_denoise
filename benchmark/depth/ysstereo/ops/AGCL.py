import torch
import torch.nn as nn
import torch.nn.functional as F

from ysstereo.ops.builder import OPERATORS
from ysstereo.utils.utils import bilinear_sampler, coords_grid, manual_pad

@OPERATORS.register_module()
class AGCL(nn.Module):
    """
    Implementation of Adaptive Group Correlation Layer (AGCL).
    """

    def __init__(self, att=None):
        super().__init__()
        self.att = att

    def forward(self, fmap1, fmap2, flow, extra_offset=None, small_patch=False, iter_mode=False):
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)
        if iter_mode:
            corr = self.corr_iter(fmap1, fmap2, flow, small_patch)
        else:
            corr = self.corr_att_offset(
                fmap1, fmap2, flow, extra_offset, small_patch
            )
        return corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):

        N, C, H, W = left_feature.shape

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        right_pad = manual_pad(right_feature, pady, padx)

        corr_list = []
        for h in range(0, pady * 2 + 1, di_y):
            for w in range(0, padx * 2 + 1, di_x):
                right_crop = right_pad[:, :, h: h + H, w: w + W]
                assert right_crop.shape == left_feature.shape
                corr = torch.mean(left_feature * right_crop, dim=1, keepdims=True)
                corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)

        return corr_final

    def corr_iter(self, left_feature, right_feature, flow, small_patch):

        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        right_feature = bilinear_sampler(right_feature, coords)

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        N, C, H, W = left_feature.shape
        lefts = torch.split(left_feature, left_feature.shape[1] // 4, dim=1)
        rights = torch.split(right_feature, right_feature.shape[1] // 4, dim=1)

        corrs = []
        for i in range(len(psize_list)):
            corr = self.get_correlation(
                lefts[i], rights[i], psize_list[i], dilate_list[i]
            )
            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)

        return final_corr

    def corr_att_offset(
            self, left_feature, right_feature, flow, extra_offset, small_patch
    ):

        N, C, H, W = left_feature.shape

        if self.att is not None:
            left_feature = left_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)  # 'n c h w -> n (h w) c'
            right_feature = right_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)  # 'n c h w -> n (h w) c'
            # 'n (h w) c -> n c h w'
            left_feature, right_feature = self.att(left_feature, right_feature)
            # 'n (h w) c -> n c h w'
            left_feature, right_feature = [
                x.reshape(N, H, W, C).permute(0, 3, 1, 2)
                for x in [left_feature, right_feature]
            ]

        lefts = torch.split(left_feature, left_feature.shape[1] // 4, dim=1)
        rights = torch.split(right_feature, right_feature.shape[1] // 4, dim=1)

        C = C // 4

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        search_num = 9
        extra_offset = extra_offset.reshape(N, search_num, 2, H, W).permute(0, 1, 3, 4, 2)  # [N, search_num, 1, 1, 2]

        corrs = []
        for i in range(len(psize_list)):
            left_feature, right_feature = lefts[i], rights[i]
            psize, dilate = psize_list[i], dilate_list[i]

            psizey, psizex = psize[0], psize[1]
            dilatey, dilatex = dilate[0], dilate[1]

            ry = psizey // 2 * dilatey
            rx = psizex // 2 * dilatex
            x_grid, y_grid = torch.meshgrid(torch.arange(-rx, rx + 1, dilatex, device=flow.device),
                                            torch.arange(-ry, ry + 1, dilatey, device=flow.device),
                                            indexing='ij')

            offsets = torch.stack((x_grid, y_grid))
            offsets = offsets.reshape(2, -1).permute(1, 0)
            for d in sorted((0, 2, 3)):
                offsets = offsets.unsqueeze(d)
            offsets = offsets.repeat_interleave(N, dim=0)
            offsets = offsets + extra_offset

            coords = self.coords + flow  # [N, 2, H, W]
            coords = coords.permute(0, 2, 3, 1)  # [N, H, W, 2]
            coords = torch.unsqueeze(coords, 1) + offsets
            coords = coords.reshape(N, -1, W, 2)  # [N, search_num*H, W, 2]

            right_feature = bilinear_sampler(
                right_feature, coords
            )  # [N, C, search_num*H, W]
            right_feature = right_feature.reshape(N, C, -1, H, W)  # [N, C, search_num, H, W]
            left_feature = left_feature.unsqueeze(2).repeat_interleave(right_feature.shape[2], dim=2)

            corr = torch.mean(left_feature * right_feature, dim=1)

            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)

        return final_corr


def corr_iter_raft(left_feature, right_feature, flow, psize=(3,3), dilate=(1,1)):
    N, C, H, W = left_feature.shape

    coords0 = coords_grid(left_feature.shape[0], left_feature.shape[2], left_feature.shape[3], left_feature.device)
    coords = coords0
    coords[:, 0:1] = coords[:, 0:1] + flow[:, 0:1] # base flow
    coords = coords.permute(0, 2, 3, 1) # N, H, W, 2

    corr_list = []
    di_y, di_x = dilate[0], dilate[1]
    pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x
    for w in range(0, padx * 2 + 1, di_x):
        offset = w - padx
        current_coords = coords.clone()
        current_coords[...,:0:1] = current_coords[...,:0:1] + offset
        sampled_right_feature = bilinear_sampler(right_feature, current_coords)
        corr = torch.mean(left_feature * sampled_right_feature, dim=1, keepdims=True)
        corr_list.append(corr)
    corr_final = torch.cat(corr_list, dim=1)

    return corr_final

def corr_iter(left_feature, right_feature, flow, psize=(3, 3), dilate=(1, 1)):
    N, C, H, W = left_feature.shape

    coords0 = coords_grid(left_feature.shape[0], left_feature.shape[2], left_feature.shape[3], left_feature.device)
    coords = coords0.clone()
    coords[:, 0:1] = coords0[:, 0:1] + flow
    coords = coords.permute(0, 2, 3, 1)
    right_feature = bilinear_sampler(right_feature, coords)

    di_y, di_x = dilate[0], dilate[1]
    pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

    right_pad = manual_pad(right_feature, pady, padx)

    corr_list = []
    for h in range(0, pady * 2 + 1, di_y):
        for w in range(0, padx * 2 + 1, di_x):
            right_crop = right_pad[:, :, h: h + H, w: w + W]
            assert right_crop.shape == left_feature.shape
            corr = torch.mean(left_feature * right_crop, dim=1, keepdims=True)
            corr_list.append(corr)

    corr_final = torch.cat(corr_list, dim=1)

    return corr_final


def get_correlation(left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):
    N, C, H, W = left_feature.shape

    di_y, di_x = dilate[0], dilate[1]
    pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

    # right_pad = manual_pad(right_feature, pady, padx)
    right_pad = right_feature

    corr_list = []
    for h in range(0, pady * 2 + 1, di_y):
        for w in range(0, padx * 2 + 1, di_x):
            right_crop = right_pad[:, :, h: h + H, w: w + W]
            assert right_crop.shape == left_feature.shape
            corr = torch.mean(left_feature * right_crop, dim=1, keepdims=True)
            corr_list.append(corr)

    corr_final = torch.cat(corr_list, dim=1)

    return corr_final


def onnx_corr_iter(left_feature, right_feature, coords1):
    _, _, H, W = left_feature.shape
    paded_coords1 = torch.cat((coords1[:, :, :, 0:1].repeat(1, 1, 1, 4),
                               coords1,
                               coords1[:, :, :, W-1:].repeat(1, 1, 1, 4)), 3)

    # p1d = (4, 4, 0, 0)
    # paded_coords1 = F.pad(coords1, p1d, "constant", value=0)
    # paded_coords1[:, :, :, W+4] = paded_coords1[:, :, :, W+3]
    # paded_coords1[:, :, :, W+5] = paded_coords1[:, :, :, W+3]
    # paded_coords1[:, :, :, W+6] = paded_coords1[:, :, :, W+3]
    # paded_coords1[:, :, :, W+7] = paded_coords1[:, :, :, W+3]
    # p2d = (0, 4, 0, 0)
    # paded_coords1 = F.pad(paded_coords1, p2d, "constant", value=W-1)
    # paded_coords1

    coords = paded_coords1.permute(0, 2, 3, 1)

    xgrid, ygrid = coords.split([1, 1], dim=3)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=3)
    warped_right_feature = F.grid_sample(right_feature, grid, align_corners=True)

    corr_list = []
    for h in range(0, 0 * 2 + 1, 1):
        for w in range(0, 4 * 2 + 1, 1):
            right_crop = warped_right_feature[:, :, h: h + H, w: w + W]
            corr = torch.mean(left_feature * right_crop, dim=1, keepdims=True)
            corr_list.append(corr)

    corr_final = torch.cat(corr_list, dim=1)
    return corr_final
