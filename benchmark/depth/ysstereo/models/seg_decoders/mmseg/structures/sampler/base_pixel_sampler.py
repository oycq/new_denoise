# Copyright (c) OpenMMLab. All rights reserved.
# refs: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/structures/sampler/base_pixel_sampler.py
from abc import ABCMeta, abstractmethod


class BasePixelSampler(metaclass=ABCMeta):
    """Base class of pixel sampler."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def sample(self, seg_logit, seg_label):
        """Placeholder for sample function."""
