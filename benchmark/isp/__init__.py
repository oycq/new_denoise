"""Minimal ISP for the ysstereo benchmark pipeline.

Pipeline (single-channel bayer_01 in [0, 1] -> uint8 BGR for stereo):
    BL subtract (9 in 8-bit) -> demosaic (BGGR) -> LSC -> AWB+CCM -> sRGB gamma.

The whole chain is driven by :func:`isp_process`. All calibration assets
are loaded once at module import from ``isp/assets/``.
"""
from .isp import isp_process

__all__ = ["isp_process"]
