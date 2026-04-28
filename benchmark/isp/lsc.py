"""Lens shading correction.

Per-camera gain map is built once (at import time) from a flat-field PNG and
applied as a multiply in linear-BGR float32. ``apply_lsc`` is a hot path —
keeping the gain maps preloaded avoids re-doing the 21x21 mean blur per call.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
BLACK_LEVEL_8BIT = 9.0
_KERNEL = (21, 21)


def _build_gain_map(cam_id: int) -> np.ndarray:
    path = ASSETS / f"cam{cam_id}_lsc.png"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"missing LSC asset: {path}")
    if img.dtype != np.uint16:
        raise ValueError(f"LSC PNG must be uint16, got {img.dtype}")

    bl16 = BLACK_LEVEL_8BIT * 256.0
    corrected = np.clip(img.astype(np.int32) - bl16, 0, 65535).astype(np.uint16)
    bgr = cv2.cvtColor(corrected, cv2.COLOR_BayerBGGR2BGR).astype(np.float32)
    blurred = cv2.blur(bgr, _KERNEL)

    h, w = blurred.shape[:2]
    ref = blurred[h // 2, w // 2]
    eps = 1e-6
    return (ref / (blurred + eps)).astype(np.float32)


_GAIN_MAPS: dict[int, np.ndarray] = {cam: _build_gain_map(cam) for cam in (2, 3)}


def apply_lsc(img: np.ndarray, cam_id: int) -> np.ndarray:
    gm = _GAIN_MAPS.get(cam_id)
    if gm is None:
        raise ValueError(f"no LSC gain map for cam_id={cam_id}")
    if img.shape != gm.shape:
        raise ValueError(f"shape mismatch: img={img.shape} vs gain={gm.shape}")
    return img * gm
