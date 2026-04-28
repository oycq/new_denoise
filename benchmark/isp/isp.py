"""Bayer raw -> sRGB uint8 BGR.

    bayer_01 (H, W) float32 in [0, 1]
        -> subtract 8-bit black-level 9
        -> demosaic (BGGR -> linear BGR)
        -> LSC (per cam_id)
        -> AWB gains + CCM
        -> sRGB gamma -> uint8 BGR (H, W, 3)

The dead `tmo()` tone-map step from the original ISP has been dropped — it
was authored but never wired into ``isp_process``.
"""
from __future__ import annotations

import cv2
import numpy as np

from .awb import awb_analysis
from .lsc import apply_lsc

BLACK_LEVEL_8BIT = 9


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """sRGB OETF on a [0, 1] float linear-light image. Returns uint8."""
    img = linear.copy()
    mask = img > 0.0031308
    img[mask] = 1.055 * (img[mask] ** (1 / 2.4)) - 0.055
    img[~mask] *= 12.92
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def isp_process(bayer_01: np.ndarray, cam_id: int) -> np.ndarray:
    if bayer_01.ndim != 2:
        raise ValueError(f"expected (H, W), got {bayer_01.shape}")

    raw_u16 = (bayer_01 * 65535.0).astype(np.uint16)
    bl16 = BLACK_LEVEL_8BIT * 256
    raw_u16 = np.clip(raw_u16.astype(np.int32) - bl16, 0, 65535).astype(np.uint16)

    bgr_u16 = cv2.cvtColor(raw_u16, cv2.COLOR_BayerBGGR2BGR_EA)
    bgr_f = bgr_u16.astype(np.float32) / 65535.0

    bgr_f = apply_lsc(bgr_f, cam_id)
    bgr_f = np.clip(bgr_f, 0.0, 1.0)

    k_b, k_r, ccm = awb_analysis(bgr_f)
    bgr_f[..., 0] *= k_b
    bgr_f[..., 2] *= k_r
    bgr_f = np.clip(bgr_f, 0.0, 1.0)

    rgb_f = bgr_f[..., ::-1]
    h, w = rgb_f.shape[:2]
    rgb_f = (rgb_f.reshape(-1, 3) @ ccm.T).reshape(h, w, 3)
    rgb_f = np.clip(rgb_f, 0.0, 1.0)
    bgr_f = rgb_f[..., ::-1]

    return linear_to_srgb(bgr_f)
