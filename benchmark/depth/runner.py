"""Run the stereo-depth network on rectified ``cam2`` / ``cam3`` PNG pairs.

The original ``ysstereo/depth/main.py`` was an argparse-driven script with
the model config and checkpoint hard-coded into a sister ``run.sh``. Here
that's all collapsed into one function whose defaults point at the bundled
``model_config.py`` / ``weights.pth``.

For every pair ``imgs1[i] / imgs2[i]`` the runner writes:

    out_dir/depth/{stem}.png    16-bit disparity * 100 (used by EPE eval)
    out_dir/visual/{stem}.png   side-by-side colour visualisation
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Quiet the noisy mmengine / TorchScript warnings without nuking real ones.
_NOISY = ("TorchScript", "avg_non_ignore", "weights_only")
_orig_show = warnings.showwarning


def _filtered(message, category, filename, lineno, file=None, line=None):
    if any(s in str(message) for s in _NOISY):
        return
    return _orig_show(message, category, filename, lineno, file, line)


warnings.showwarning = _filtered

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mmengine import Config  # noqa: E402
from mmengine.dataset import Compose  # noqa: E402

from ysstereo.apis import inference_model, init_model  # noqa: E402,F401
from ysstereo.datasets import visualize_depth  # noqa: E402

DEFAULT_CONFIG = _HERE / "model_config.py"
DEFAULT_WEIGHTS = _HERE / "weights.pth"

_IS_ROTATED = True  # input images are landscape; the net expects portrait
_DEFAULT_TEST_PIPELINE = [
    dict(type="LoadStereoImageFromFile", rotate=_IS_ROTATED),
    dict(max_disp=512, type="LoadDispAnnotations"),
    dict(exponent=6, type="InputPad"),
    dict(type="TestFormatBundle"),
    dict(
        keys=["imgs"],
        meta_keys=[
            "disp_gt", "valid", "filename1", "filename2",
            "ori_filename1", "ori_filename2", "ori_shape", "img_shape",
            "img_norm_cfg", "scale_factor", "pad_shape", "pad",
        ],
        type="Collect",
    ),
]


def _build_test_pipeline(cfg):
    """Replicate the in-place mutation that ``inference_model`` does once,
    instead of doing it per image."""
    cfg = cfg.copy() if hasattr(cfg, "copy") else cfg
    for i in range(len(cfg.test_pipeline)):
        if cfg.test_pipeline[i].type == "LoadDispAnnotations":
            del cfg.test_pipeline[i]
            break
    for i in range(len(cfg.test_pipeline)):
        if cfg.test_pipeline[i].type == "Fisheye2TransEqui":
            cfg.test_pipeline[i].test_mode = True
            break
    cfg.test_pipeline = list(filter(
        lambda i: i["type"] != "RandomCrop", cfg.test_pipeline))
    last_meta = cfg.test_pipeline[-1]["meta_keys"]
    for k in ("disp_gt", "disp_fw_gt", "disp_bw_gt", "valid", "distance"):
        if k in last_meta:
            last_meta.remove(k)
    return Compose(cfg.test_pipeline)


class _StereoPairDataset(Dataset):
    """Run the test pipeline (PNG decode, rotate, transpose, concat) inside
    DataLoader workers so CPU work overlaps with GPU inference."""

    def __init__(self, imgs1, imgs2, cfg):
        self.imgs1 = imgs1
        self.imgs2 = imgs2
        self.cfg = cfg
        self._pipeline = None

    def __len__(self):
        return len(self.imgs1)

    def __getitem__(self, idx):
        if self._pipeline is None:
            self._pipeline = _build_test_pipeline(self.cfg)
        data = self._pipeline(dict(
            img_info=dict(filename1=self.imgs1[idx], filename2=self.imgs2[idx]),
            img1_prefix=None, img2_prefix=None,
            img_fields=["imgl", "imgr"],
        ))
        return {
            "inputs": data["inputs"],
            "data_samples": data["data_samples"],
            "imfile1": self.imgs1[idx],
            "idx": Path(self.imgs1[idx]).stem,
        }


def _collate_single(batch):
    assert len(batch) == 1, "batch_size must be 1"
    return batch[0]


def _save_outputs(out_dir: Path, idx, imfile1, disp, depth_map):
    img = cv2.imread(imfile1, cv2.IMREAD_COLOR)
    disp_u16 = (disp.clip(0, 500) * 100).astype(np.uint16)
    if _IS_ROTATED:
        disp_u16 = cv2.rotate(disp_u16, cv2.ROTATE_90_CLOCKWISE)
        depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(out_dir / "visual" / f"{idx}.png"),
                np.hstack((img, depth_map)))
    cv2.imwrite(str(out_dir / "depth"  / f"{idx}.png"), disp_u16)


def run_depth(
    img1_dir: str | Path,
    img2_dir: str | Path,
    out_dir: str | Path,
    *,
    config: str | Path = DEFAULT_CONFIG,
    checkpoint: str | Path = DEFAULT_WEIGHTS,
    device: str = "cuda:0",
    num_workers: int = 8,
    prefetch_factor: int = 2,
    num_io_threads: int = 4,
    fp16: bool = False,
) -> int:
    """Run stereo depth on every paired PNG in (img1_dir, img2_dir).

    Returns the number of frames processed.
    """
    img1_dir = Path(img1_dir)
    img2_dir = Path(img2_dir)
    out_dir = Path(out_dir)

    cfg = Config.fromfile(str(config))
    cfg["test_pipeline"] = cfg.get("test_pipeline", _DEFAULT_TEST_PIPELINE)
    model = init_model(cfg, str(checkpoint), device=device)
    model.eval()
    torch.backends.cudnn.benchmark = True

    imgs1, imgs2 = [], []
    for ext in ("png", "jpg"):
        imgs1 += sorted(glob.glob(str(img1_dir / f"*.{ext}")))
        imgs2 += sorted(glob.glob(str(img2_dir / f"*.{ext}")))
    if not imgs1 or len(imgs1) != len(imgs2):
        raise RuntimeError(
            f"img1_dir={img1_dir} ({len(imgs1)} files) and "
            f"img2_dir={img2_dir} ({len(imgs2)} files) are not balanced")

    # Wipe just our two subtrees, not the whole ``out_dir`` — callers may
    # legitimately point ``out_dir`` at a shared output root that already
    # holds other artefacts (the ``isp/`` tree from stage 1, the summary
    # plot from stage 3) which we have no business deleting.
    for sub in ("depth", "visual"):
        if (out_dir / sub).exists():
            shutil.rmtree(out_dir / sub)
    (out_dir / "depth").mkdir(parents=True)
    (out_dir / "visual").mkdir(parents=True)

    dataset = _StereoPairDataset(imgs1, imgs2, model.cfg)
    nw = min(num_workers, max(1, len(dataset)))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=nw,
        prefetch_factor=prefetch_factor if nw > 0 else None,
        persistent_workers=nw > 0, pin_memory=True,
        collate_fn=_collate_single,
    )

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if fp16 and device.startswith("cuda")
        else torch.amp.autocast("cuda", enabled=False)
    )

    io_pool = ThreadPoolExecutor(max_workers=max(1, num_io_threads))
    futures = []
    n_done = 0
    t0 = time.time()
    try:
        with torch.inference_mode():
            for sample in tqdm(loader, total=len(dataset), desc="[depth]"):
                with autocast_ctx:
                    results = model.predict({
                        "inputs": sample["inputs"],
                        "data_samples": sample["data_samples"],
                    })
                disp = np.abs(results[0]["disp"].squeeze())
                bf = 25.0
                disp_for_depth = np.where(disp < 0.01, 0.01, disp)
                depth = bf / disp_for_depth
                depth_map = visualize_depth(depth, max_depth=6)
                futures.append(io_pool.submit(
                    _save_outputs, out_dir, sample["idx"],
                    sample["imfile1"], disp, depth_map))
                n_done += 1
    finally:
        for fut in futures:
            fut.result()
        io_pool.shutdown(wait=True)

    if n_done > 0:
        dt = time.time() - t0
        print(f"[depth] processed {n_done} pairs in {dt:.1f}s "
              f"({dt / n_done * 1000:.1f} ms/img, {n_done / dt:.2f} img/s)")
    return n_done
