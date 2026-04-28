"""Auto white-balance + colour-correction matrix interpolation.

Two outputs per call:
    - White-balance gains (``k_b, k_r``) from a frame-wise grey-world estimate
      restricted to the planckian-locus white-point envelope.
    - A colour-correction matrix (CCM) interpolated by the inferred colour
      temperature between the calibrated reference points (HZ / A / D65 / D75).

All calibration data (locus envelope, temperature fit, reference CCMs) lives
in ``isp/assets/awb_calib.json`` and is loaded once at import.

The original implementation drew an OpenCV scatter-plot debugging window every
call; that's been removed — the function is pure numpy now.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"

LUM_MIN = 10 / 255.0
LUM_MAX = 220 / 255.0

with (ASSETS / "awb_calib.json").open("r") as _f:
    _data = json.load(_f)

_calib = _data["calibration_results"]
_upper_a = _data["planck_y_max"]["a"]
_upper_b = _data["planck_y_max"]["b"]
_lower_a = _data["planck_y_min"]["a"]
_lower_b = _data["planck_y_min"]["b"]
_temp_a = _data["temp_fit"]["a"]
_temp_b = _data["temp_fit"]["b"]
_x_min = _data["rg_limits"]["min"]
_x_max = _data["rg_limits"]["max"]

_REF_LABELS = ["HZ", "A", "D65", "D75"]
_REF_TEMPS = np.array([2300, 2856, 6500, 7500], dtype=np.float64)
_REF_CCMS = [np.array(_calib[lbl]["ccm"], dtype=np.float64) for lbl in _REF_LABELS]

_DEFAULT_RG, _DEFAULT_BG = 0.54, 0.62


def _interp_ccm(temp: float) -> np.ndarray:
    """Piecewise-linear interpolation between the 4 reference CCMs by colour
    temperature, then row-normalised so each row sums to 1."""
    t = float(np.clip(temp, _REF_TEMPS[0], _REF_TEMPS[-1]))
    for i in range(len(_REF_TEMPS) - 1):
        lo, hi = _REF_TEMPS[i], _REF_TEMPS[i + 1]
        if lo <= t <= hi:
            w = (t - lo) / (hi - lo)
            ccm = _REF_CCMS[i] * (1 - w) + _REF_CCMS[i + 1] * w
            break
    else:
        ccm = _REF_CCMS[0] if t <= _REF_TEMPS[0] else _REF_CCMS[-1]

    row_sums = ccm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return ccm / row_sums


def _estimate_white_point(small_bgr: np.ndarray) -> tuple[float, float]:
    """Mean (rg, bg) of pixels that fall inside the planckian-locus envelope
    after dropping over/under-exposed ones."""
    b, g, r = small_bgr[..., 0], small_bgr[..., 1], small_bgr[..., 2]
    max_v = np.maximum(np.maximum(b, g), r)
    avg_v = (b + g + r) / 3.0
    valid = (max_v < LUM_MAX) & (avg_v > LUM_MIN) & (g > 0)

    rg = np.where(valid, r / np.where(g > 0, g, 1.0), 0.0)
    bg = np.where(valid, b / np.where(g > 0, g, 1.0), 0.0)
    in_x = (rg >= _x_min) & (rg <= _x_max)

    upper = _upper_a / np.where(rg > 0, rg, 1.0) + _upper_b
    lower = _lower_a / np.where(rg > 0, rg, 1.0) + _lower_b
    in_y = (bg >= lower) & (bg <= upper)
    keep = valid & in_x & in_y

    if not keep.any():
        return _DEFAULT_RG, _DEFAULT_BG
    return float(rg[keep].mean()), float(bg[keep].mean())


def awb_analysis(img: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Estimate (k_b, k_r, CCM) from a linear-BGR float32 image in [0, 1].

    Mirrors the original API:
        k_b = 1 / avg_bg   (multiplied into the B channel)
        k_r = 1 / avg_rg   (multiplied into the R channel)
        CCM is row-normalised, applied as RGB' = RGB · CCM.T.
    """
    small = cv2.resize(img, (32, 32)).astype(np.float32)
    avg_rg, avg_bg = _estimate_white_point(small)
    temp = _temp_a / max(avg_rg, 1e-6) + _temp_b
    ccm = _interp_ccm(temp)
    return 1.0 / avg_bg, 1.0 / avg_rg, ccm
