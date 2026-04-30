"""End-to-end benchmark: PyTorch denoiser -> ISP -> rectify -> stereo depth -> EPE.

Three stages, all reachable from a single function call:

    1. prepare:  for every <scene>/<gain>/sensor[23]_raw.png,
                 torch-denoise -> ISP -> rectify -> output/isp/{2,3}/.
                 The matching ``sensor[23]_isp.png`` files are rectified-only
                 (the "no-denoise" branch the EPE eval compares against).
    2. depth:    stereo network on the (cam2, cam3) pairs ->
                 output/depth/  (16-bit disparity * 100, used by EPE)
                 output/visual/ (side-by-side colour visualisations)
    3. eval:     scenario / ISO aggregation -> EvalResult (score, per-scene).
                 Score = mean denoised EPE across ISO 100..1600 vs each
                 scene's ``(iso=100, denoise=0)`` baseline. Single number,
                 no QC threshold.

Stage 1 used to drive the model through onnxruntime on CPU, which scales
linearly with model size and was the dominant cost as the network grew.
It now loads the training checkpoint (``.pt``) directly and runs the
UNet on CUDA — same numerical contract (fp32, single-channel bayer_01
in / out), an order of magnitude faster.

Usage:
    # As a function
    from benchmark.run import run_benchmark
    res = run_benchmark(ckpt_path=".../model.pt",
                        data_root="train_data/Data")
    print(res.score)

    # As a CLI
    python benchmark/run.py --ckpt checkpoints/n2n_model.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isp import isp_process            # noqa: E402
from rectify import rectify            # noqa: E402
from eval import evaluate, EvalResult  # noqa: E402

from n2n.model import UNet              # noqa: E402


# ---------------------------------------------------------------------------
# Stage 1: PyTorch denoise + ISP + rectify
# ---------------------------------------------------------------------------
class _TorchDenoiser:
    """Single-channel bayer_01 in / out — runs the residual UNet on CUDA.

    The denoiser is the (PixelUnshuffle -> UNet -> subtract residual ->
    PixelShuffle) wrapper that ``export_l1tv_onnx.py`` used to bake into
    the ONNX. It is rebuilt directly from a training checkpoint, so the
    benchmark no longer depends on an ONNX export step.

    The thread-pool in :func:`stage_prepare` calls this from up to
    ``num_workers`` threads at once; the CUDA kernel calls are protected
    by a lock so the GPU sees one inference at a time while the per-frame
    PNG decode / ISP / rectify / encode work parallelises across threads.
    """

    def __init__(self, ckpt_path: Path, device: str | torch.device | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {}) or {}
        base = int(cfg.get("base_channels", 48))
        depth = int(cfg.get("unet_depth", 3))

        unet = UNet(in_ch=4, out_ch=4, base=base, depth=depth)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        unet.load_state_dict(sd)

        self.pack = nn.PixelUnshuffle(2).to(self.device)
        self.unpack = nn.PixelShuffle(2).to(self.device)
        self.unet = unet.to(self.device).eval()
        self._lock = threading.Lock()

    @torch.inference_mode()
    def __call__(self, bayer_01: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(bayer_01)).to(
            self.device, non_blocking=True
        )[None, None]
        with self._lock:
            packed = self.pack(x)
            residual = self.unet(packed)
            den_packed = packed - residual
            y = self.unpack(den_packed)
        return y[0, 0].clamp_(0, 1).float().cpu().numpy()


def _parse_raw_path(p: Path) -> tuple[str, int, int, bool]:
    """``train_data/Data/<scene>/<gain>/sensor<cam>_raw.png`` ->
    (scene, cam_id, gain, is_raw). ``sensor<cam>_isp.png`` is also accepted."""
    parts = p.parts
    scene = parts[-3]
    gain = int(parts[-2])
    fname = parts[-1]
    cam_id = int(fname.split("sensor")[-1].split("_")[0])
    is_raw = "raw" in fname
    return scene, cam_id, gain, is_raw


def _prepare_one(src: Path, denoiser, out_root: Path):
    """Both the no-denoise (denoise=0) and denoise=1 outputs are generated
    from the SAME raw input, going through the SAME offline ISP. Only
    difference: denoise=1 applies ``denoiser`` to the bayer first.

    This makes ``denoiser=identity`` a real null-op — denoise=0 and
    denoise=1 outputs become byte-equal — instead of comparing two
    different ISP pipelines (vendor pre-rendered ``sensor*_isp.png`` vs
    our ``isp_process``). vendor ``*_isp.png`` files in the data tree
    are ignored.
    """
    scene, cam_id, gain, is_raw = _parse_raw_path(src)
    if not is_raw:
        return  # vendor ISPs no longer participate in the eval

    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    bayer_orig = raw.astype(np.float32) / 65535.0
    bayer_dn   = denoiser(bayer_orig)

    rgb_orig = rectify(isp_process(bayer_orig, cam_id), cam_id)
    rgb_dn   = rectify(isp_process(bayer_dn,   cam_id), cam_id)

    scene_flat = scene.replace("_", "")
    cv2.imwrite(
        str(out_root / str(cam_id) / f"{scene_flat}_{gain}_0.png"), rgb_orig)
    cv2.imwrite(
        str(out_root / str(cam_id) / f"{scene_flat}_{gain}_1.png"), rgb_dn)


def stage_prepare(
    ckpt_path: Path, data_root: Path, out_root: Path, *,
    num_workers: int = 8, device: str | torch.device | None = None,
) -> int:
    """Stage 1: drive the PyTorch denoiser over the full Data tree, render to
    rectified BGR, and write to ``out_root/{2,3}/<scene>_<gain>_<denoised>.png``.
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "2").mkdir(parents=True)
    (out_root / "3").mkdir(parents=True)

    raws = sorted(Path(data_root).glob("*/*/*.png"))
    if not raws:
        raise RuntimeError(f"no input PNGs under {data_root}")

    denoiser = _TorchDenoiser(ckpt_path, device=device)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(lambda p: _prepare_one(p, denoiser, out_root), raws))
    dt = time.time() - t0
    print(f"[prepare] {len(raws)} files in {dt:.1f}s "
          f"({dt / len(raws) * 1000:.1f} ms/img) on {denoiser.device}")
    return len(raws)


# ---------------------------------------------------------------------------
# Stage 2 + 3 are thin wrappers over depth.run_depth and eval.evaluate
# ---------------------------------------------------------------------------
def stage_depth(prepared_root: Path, depth_root: Path, **kw) -> int:
    from depth import run_depth
    return run_depth(prepared_root / "2", prepared_root / "3", depth_root, **kw)


def stage_eval(disp_dir: Path, plot_dir: Path, *, quiet: bool = False) -> EvalResult:
    return evaluate(disp_dir, plot_dir, quiet=quiet)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = ROOT / "train_data" / "Data"
DEFAULT_OUTPUT    = HERE / "output"
DEFAULT_CKPT      = HERE / "checkpoint.pt"     # local-only, gitignored


def run_benchmark(
    ckpt_path: str | Path,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    skip_prepare: bool = False,
    skip_depth: bool = False,
    num_workers: int = 8,
    device: str = "cuda:0",
    quiet: bool = False,
) -> EvalResult:
    """Run the full benchmark and return an :class:`EvalResult`.

    Parameters
    ----------
    ckpt_path
        Path to a training checkpoint produced by :mod:`n2n.trainer`
        (``.pt`` containing ``{"model": state_dict, "config": {...}}``).
        The denoiser is rebuilt as ``PixelUnshuffle -> UNet -> residual
        subtract -> PixelShuffle`` and run on ``device`` (CUDA by default).
    data_root
        Root with ``<scene>/<gain>/sensor[23]_{raw,isp}.png`` (defaults to
        ``train_data/Data`` next to the repo root).
    output_dir
        Where intermediate stages and the final summary plot land. Defaults
        to ``benchmark/output``.
    skip_prepare / skip_depth
        Reuse cached intermediates from a previous run (handy when only the
        eval changed).
    device
        CUDA / CPU device for *both* the denoiser (stage 1) and the stereo
        depth network (stage 2).
    """
    ckpt_path  = Path(ckpt_path)
    data_root  = Path(data_root)
    output_dir = Path(output_dir)
    prepared   = output_dir / "isp"

    if not skip_prepare:
        stage_prepare(ckpt_path, data_root, prepared,
                      num_workers=num_workers, device=device)
    if not skip_depth:
        # ``stage_depth`` writes ``output_dir/depth/`` and
        # ``output_dir/visual/`` — flat under the run's output root,
        # alongside ``isp/`` and the final summary plot.
        stage_depth(prepared, output_dir, device=device)
    return stage_eval(output_dir / "depth", output_dir, quiet=quiet)


def _cli():
    p = argparse.ArgumentParser(description="End-to-end ysstereo benchmark.")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                   help=f"denoiser training checkpoint (.pt). Default: "
                        f"{DEFAULT_CKPT} (gitignored — drop your trained "
                        "model here for no-args invocation).")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-depth",   action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    res = run_benchmark(
        ckpt_path=args.ckpt, data_root=args.data_root,
        output_dir=args.output_dir, device=args.device,
        num_workers=args.num_workers, quiet=args.quiet,
        skip_prepare=args.skip_prepare, skip_depth=args.skip_depth,
    )
    return 0 if res.score is not None else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
