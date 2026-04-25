"""Raw I/O, 4-channel Bayer packing, and the display pipeline.

Conventions
-----------
- Sensor raw is stored as 16-bit single-channel PNG, **RGGB** Bayer (verified
  against ISP reference: brown bench color matches only with RGGB).
- Network input is `raw / 65535.0` -> float32 / float16 in [0, 1].
- Display pipeline (per user spec):
    1. Convert to 0-255 domain: ``img8 = raw_uint16 / 256``.
    2. Subtract black level **9** (in 0-255 domain).
    3. Demosaic.
    4. Apply gamma 2.2 for display.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

BAYER_PATTERN: str = "RGGB"
BLACK_LEVEL_8BIT: float = 9.0
DISPLAY_GAMMA: float = 2.2
RAW_MAX: float = 65535.0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_raw(path: str | Path) -> np.ndarray:
    """Read a 16-bit Bayer raw PNG. Returns ``uint16`` array of shape (H, W)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read raw image: {path}")
    if img.dtype != np.uint16:
        raise ValueError(f"Expected uint16 raw, got {img.dtype} at {path}")
    if img.ndim != 2:
        raise ValueError(f"Expected single-channel Bayer raw, got shape {img.shape}")
    return img


def read_isp(path: str | Path) -> np.ndarray:
    """Read the matching ISP reference (uint8 BGR)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read isp image: {path}")
    return img


# ---------------------------------------------------------------------------
# 4-channel packing for RGGB
# ---------------------------------------------------------------------------
def pack_rggb(bayer: np.ndarray) -> np.ndarray:
    """Pack a single-channel RGGB Bayer image (H, W) into 4-channel (H/2, W/2, 4).

    Channel order: ``R, Gr, Gb, B``.

    Accepts float32 / float16 / uint16 input. Output dtype follows input
    (with uint16 promoted to float32 by the caller if desired).
    """
    if bayer.ndim != 2:
        raise ValueError(f"Expected (H, W), got {bayer.shape}")
    h, w = bayer.shape
    if h % 2 or w % 2:
        raise ValueError(f"Bayer dims must be even, got ({h}, {w})")
    r = bayer[0::2, 0::2]
    gr = bayer[0::2, 1::2]
    gb = bayer[1::2, 0::2]
    b = bayer[1::2, 1::2]
    return np.stack([r, gr, gb, b], axis=-1)


def unpack_rggb(packed: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_rggb`. Input shape (H/2, W/2, 4) -> (H, W)."""
    if packed.ndim != 3 or packed.shape[-1] != 4:
        raise ValueError(f"Expected (h, w, 4), got {packed.shape}")
    h, w, _ = packed.shape
    out = np.empty((h * 2, w * 2), dtype=packed.dtype)
    out[0::2, 0::2] = packed[..., 0]
    out[0::2, 1::2] = packed[..., 1]
    out[1::2, 0::2] = packed[..., 2]
    out[1::2, 1::2] = packed[..., 3]
    return out


# ---------------------------------------------------------------------------
# Display pipeline
# ---------------------------------------------------------------------------
def gray_world_gains(bgr_linear: np.ndarray) -> tuple[float, float, float]:
    """Compute per-channel multiplicative gains so that all channel means equal G.

    Operates in *linear* (pre-gamma) BGR domain.
    """
    means = bgr_linear.reshape(-1, 3).mean(axis=0).astype(np.float64) + 1e-6
    g = means[1]
    return float(g / means[0]), 1.0, float(g / means[2])  # gains for B, G, R


def raw_to_display(
    raw: np.ndarray,
    *,
    black_level_8bit: float = BLACK_LEVEL_8BIT,
    gamma: float = DISPLAY_GAMMA,
    apply_white_balance: bool = True,
    wb_gains: tuple[float, float, float] | None = None,
    return_rgb: bool = False,
) -> np.ndarray:
    """Run the full display pipeline on an RGGB Bayer raw.

    Pipeline: ``raw_uint16 / 256`` -> subtract 8-bit black level -> demosaic ->
    white balance (gray-world unless ``wb_gains`` is given, optional) ->
    gamma 2.2.

    Parameters
    ----------
    raw : np.ndarray
        Either ``uint16`` (raw 0-65535) or ``float32`` already in 0-65535 range,
        or a normalized float in [0, 1] (auto-detected: max <= 1.5 -> normalized).
    apply_white_balance : bool, default True
        If True, multiplies B/G/R by gray-world gains in linear domain.
    wb_gains : (gB, gG, gR) | None
        If supplied, used instead of computing per-image gray-world gains.
        Useful to apply identical WB to a noisy/denoised pair so the
        comparison is colour-consistent.
    return_rgb : bool, default False
        If True returns RGB (matplotlib-style); else BGR (OpenCV style).

    Returns
    -------
    img : ``uint8`` ``(H, W, 3)``
    """
    if raw.ndim != 2:
        raise ValueError(f"Expected (H, W) Bayer, got {raw.shape}")

    arr = raw.astype(np.float32, copy=False)
    if arr.max() <= 1.5:  # normalized [0,1] input
        arr = arr * RAW_MAX

    img8 = arr / 256.0  # 0-255 domain
    img8 = img8 - black_level_8bit
    img8 = np.clip(img8, 0.0, 255.0).astype(np.uint8)

    bgr = cv2.cvtColor(img8, cv2.COLOR_BayerRG2BGR)  # RGGB pattern
    bgr_f = bgr.astype(np.float32)

    if apply_white_balance:
        gb, gg, gr = wb_gains if wb_gains is not None else gray_world_gains(bgr_f)
        bgr_f[..., 0] *= gb
        bgr_f[..., 1] *= gg
        bgr_f[..., 2] *= gr

    bgr_f = np.clip(bgr_f / 255.0, 0.0, 1.0)
    bgr_f = np.power(bgr_f, 1.0 / gamma)
    out = (bgr_f * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    if return_rgb:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


def packed_to_display(packed: np.ndarray, **kwargs) -> np.ndarray:
    """Display helper: 4-channel packed RGGB -> demosaiced uint8 (BGR/RGB)."""
    if packed.ndim != 3 or packed.shape[-1] != 4:
        raise ValueError(f"Expected packed (h, w, 4), got {packed.shape}")
    bayer = unpack_rggb(packed)
    return raw_to_display(bayer, **kwargs)
