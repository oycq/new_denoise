# Copyright (c) OpenMMLab. All rights reserved.
# refs: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/structures/sampler/builder.py
import warnings

from ysstereo.registry import TASK_UTILS

PIXEL_SAMPLERS = TASK_UTILS


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    warnings.warn(
        '``build_pixel_sampler`` would be deprecated soon, please use '
        '``mmseg.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)
