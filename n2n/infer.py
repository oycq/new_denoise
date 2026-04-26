"""Inference helpers + final result generation.

Given a trained checkpoint, run the model on every raw frame in the dataset
and write a side-by-side comparison into ``result/``::

    raw -> ISP-style display    |    denoised raw -> ISP-style display

Each output PNG is concatenated horizontally for easy A/B inspection. We
also keep the original ISP reference for context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from torch.amp import autocast

from .dataset import find_raw_files
from .model import UNet
from .raw_utils import (
    RAW_MAX,
    gray_world_gains,
    pack_rggb,
    raw_to_display,
    raw_to_linear_bgr,
    read_isp,
    read_raw,
    unpack_rggb,
)


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[UNet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    base = cfg.get("base_channels", 48)
    depth = cfg.get("unet_depth", 3)
    model = UNet(in_ch=4, out_ch=4, base=base, depth=depth).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, cfg


@torch.no_grad()
def denoise_full_raw(
    model: UNet,
    raw_uint16: np.ndarray,
    *,
    use_fp16: bool = True,
    tile: int = 512,
    overlap: int = 32,
) -> np.ndarray:
    """Run the model on a full raw frame using overlapping tiles in 4-ch space.

    Returns a uint16 Bayer image of the same shape as the input.
    """
    device = next(model.parameters()).device
    norm = raw_uint16.astype(np.float32) / RAW_MAX
    packed = pack_rggb(norm).transpose(2, 0, 1)  # (4, H/2, W/2)
    _, h, w = packed.shape

    out = np.zeros_like(packed, dtype=np.float32)
    weight = np.zeros((1, h, w), dtype=np.float32)

    # Hann-style triangular blend window so seams are invisible
    win1d = np.linspace(0.0, 1.0, overlap, endpoint=False) if overlap > 0 else None

    step = max(1, tile - overlap)
    for y in range(0, h, step):
        y0 = min(y, max(0, h - tile))
        y1 = min(h, y0 + tile)
        for x in range(0, w, step):
            x0 = min(x, max(0, w - tile))
            x1 = min(w, x0 + tile)
            patch = packed[:, y0:y1, x0:x1]
            t = torch.from_numpy(patch).unsqueeze(0).to(device)
            with autocast("cuda", enabled=use_fp16):
                pred = model.denoise(t)
            pred = pred.float().clamp(0, 1).squeeze(0).cpu().numpy()

            ph, pw = pred.shape[1], pred.shape[2]
            wmask = np.ones((1, ph, pw), dtype=np.float32)
            if win1d is not None and overlap > 0:
                yramp = np.ones(ph, dtype=np.float32)
                yramp[:overlap] = np.minimum(yramp[:overlap], win1d)
                yramp[-overlap:] = np.minimum(yramp[-overlap:], win1d[::-1])
                xramp = np.ones(pw, dtype=np.float32)
                xramp[:overlap] = np.minimum(xramp[:overlap], win1d)
                xramp[-overlap:] = np.minimum(xramp[-overlap:], win1d[::-1])
                wmask = (yramp[:, None] * xramp[None, :])[None]

            out[:, y0:y1, x0:x1] += pred * wmask
            weight[:, y0:y1, x0:x1] += wmask
            if x1 == w:
                break
        if y1 == h:
            break

    out = out / np.maximum(weight, 1e-6)
    out = np.clip(out, 0.0, 1.0).transpose(1, 2, 0)  # (H/2, W/2, 4)
    bayer = unpack_rggb(out)
    bayer_u16 = (bayer * RAW_MAX + 0.5).clip(0, RAW_MAX).astype(np.uint16)
    return bayer_u16


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 32), (0, 0, 0), thickness=-1)
    cv2.putText(
        img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
    )
    return img


def make_comparison_panel(
    raw_uint16: np.ndarray,
    denoised_uint16: np.ndarray,
    isp_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a single BGR uint8 image with three labelled panels.

    The noisy and denoised panels share the white-balance gains (computed
    on the 16-bit demosaiced *denoised* image, whose stats are less noisy)
    so the comparison is colour-consistent.
    """
    shared_gains = gray_world_gains(raw_to_linear_bgr(denoised_uint16))

    noisy_disp = raw_to_display(raw_uint16, wb_gains=shared_gains)
    denoised_disp = raw_to_display(denoised_uint16, wb_gains=shared_gains)

    panels = [_label(noisy_disp, "noisy raw -> ISP"),
              _label(denoised_disp, "denoised raw -> ISP")]
    if isp_bgr is not None:
        panels.append(_label(isp_bgr, "vendor ISP reference"))
    return np.concatenate(panels, axis=1)


def run_inference_set(
    ckpt_path: str | Path,
    data_root: str | Path = "train_data/Data",
    out_root: str | Path = "result",
    *,
    use_fp16: bool = True,
    progress_cb=None,
    is_cancelled=None,
) -> list[Path]:
    """Run inference on every ``*_raw.png`` under ``data_root`` and write results.

    For each input ``<scene>/<iso>/sensorN_raw.png`` produces:

    - ``sensorN_compare.png``           : 3-panel comparison (noisy / denoised / vendor ISP)
    - ``sensorN_isp_without_denoise.png`` : full-size ISP-style render of the noisy raw
    - ``sensorN_isp_with_denoise.png``   : full-size ISP-style render of the denoised raw
    - ``sensorN_denoised_raw.png``       : the 16-bit denoised Bayer raw

    The two ``isp_*`` files share the same WB gains and dimensions, so they
    can be loaded directly into ImageJ as a stack for an A/B flicker test.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(ckpt_path, device)

    files = find_raw_files(data_root)
    written: list[Path] = []
    for i, f in enumerate(files, 1):
        if is_cancelled and is_cancelled():
            break
        raw = read_raw(f)
        denoised = denoise_full_raw(model, raw, use_fp16=use_fp16)

        isp_path = Path(str(f).replace("_raw.png", "_isp.png"))
        isp = read_isp(isp_path) if isp_path.exists() else None

        # Shared WB gains computed on the 16-bit demosaiced denoised image
        # (no uint8 quantisation in the gain pipeline) so noisy & denoised
        # ISP renders are colour-aligned and free of staircase artifacts.
        shared_gains = gray_world_gains(raw_to_linear_bgr(denoised))

        isp_no_dn = raw_to_display(raw, wb_gains=shared_gains)
        isp_dn = raw_to_display(denoised, wb_gains=shared_gains)

        rel = Path(f).relative_to(Path(data_root))
        out_dir = (out_root / rel).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = rel.stem.replace("_raw", "")  # e.g. "sensor2"

        # Three-panel comparison (uses shared gains internally too).
        panels = [_label(isp_no_dn, "noisy raw -> ISP"),
                  _label(isp_dn, "denoised raw -> ISP")]
        if isp is not None:
            panels.append(_label(isp, "vendor ISP reference"))
        panel = np.concatenate(panels, axis=1)
        compare_path = out_dir / f"{stem}_compare.png"
        cv2.imwrite(str(compare_path), panel)
        written.append(compare_path)

        # ImageJ-friendly stackable pair (identical dimensions, no labels).
        cv2.imwrite(str(out_dir / f"{stem}_isp_without_denoise.png"), isp_no_dn)
        cv2.imwrite(str(out_dir / f"{stem}_isp_with_denoise.png"), isp_dn)
        cv2.imwrite(str(out_dir / f"{stem}_denoised_raw.png"), denoised)

        if progress_cb:
            progress_cb(i, len(files), compare_path)

    return written
