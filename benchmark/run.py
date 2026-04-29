"""End-to-end benchmark: ONNX denoiser -> ISP -> rectify -> stereo depth -> EPE.

Three stages, all reachable from a single function call:

    1. prepare:  for every <scene>/<gain>/sensor[23]_raw.png,
                 onnx-denoise -> ISP -> rectify -> output/isp/{2,3}/.
                 The matching ``sensor[23]_isp.png`` files are rectified-only
                 (the "no-denoise" branch the EPE eval compares against).
    2. depth:    stereo network on the (cam2, cam3) pairs ->
                 output/depth/  (16-bit disparity * 100, used by EPE)
                 output/visual/ (side-by-side colour visualisations)
    3. eval:     scenario / ISO aggregation -> EvalResult (score, per-scene).
                 Score = mean denoised EPE across ISO 100..1600 vs each
                 scene's ``(iso=100, denoise=0)`` baseline. Single number,
                 no QC threshold.

Usage:
    # As a function
    from benchmark.run import run_benchmark
    res = run_benchmark(onnx_path=".../model.onnx",
                        data_root="train_data/Data")
    print(res.score)

    # As a CLI
    python benchmark/run.py --onnx checkpoints/_benchmark.onnx
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from isp import isp_process            # noqa: E402
from rectify import rectify            # noqa: E402
from eval import evaluate, EvalResult  # noqa: E402


# ---------------------------------------------------------------------------
# Stage 1: ONNX denoise + ISP + rectify
# ---------------------------------------------------------------------------
class _OnnxDenoiser:
    """Single-channel bayer_01 in / out — matches the contract of every
    ONNX exported by ``export_l1tv_onnx.py`` (mode 2, no FPN, no offset)."""

    def __init__(self, onnx_path: Path, providers=None):
        import onnxruntime as ort
        prov = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), providers=prov)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def __call__(self, bayer_01: np.ndarray) -> np.ndarray:
        x = bayer_01[None, None].astype(np.float32, copy=False)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        return np.clip(np.squeeze(y, axis=(0, 1)).astype(np.float32), 0, 1)


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
    onnx_path: Path, data_root: Path, out_root: Path, *,
    num_workers: int = 8, providers=None,
) -> int:
    """Stage 1: drive the ONNX denoiser over the full Data tree, render to
    rectified BGR, and write to ``out_root/{2,3}/<scene>_<gain>_<denoised>.png``.
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "2").mkdir(parents=True)
    (out_root / "3").mkdir(parents=True)

    raws = sorted(Path(data_root).glob("*/*/*.png"))
    if not raws:
        raise RuntimeError(f"no input PNGs under {data_root}")

    denoiser = _OnnxDenoiser(onnx_path, providers=providers)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(lambda p: _prepare_one(p, denoiser, out_root), raws))
    dt = time.time() - t0
    print(f"[prepare] {len(raws)} files in {dt:.1f}s "
          f"({dt / len(raws) * 1000:.1f} ms/img)")
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
DEFAULT_ONNX      = HERE / "checkpoint.onnx"   # local-only, gitignored


def run_benchmark(
    onnx_path: str | Path,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    skip_prepare: bool = False,
    skip_depth: bool = False,
    num_workers: int = 8,
    providers=None,
    device: str = "cuda:0",
    quiet: bool = False,
) -> EvalResult:
    """Run the full benchmark and return an :class:`EvalResult`.

    Parameters
    ----------
    onnx_path
        Path to the (1,1,H,W)->(1,1,H,W) bayer_01 denoiser ONNX (mode 2).
    data_root
        Root with ``<scene>/<gain>/sensor[23]_{raw,isp}.png`` (defaults to
        ``train_data/Data`` next to the repo root).
    output_dir
        Where intermediate stages and the final summary plot land. Defaults
        to ``benchmark/output``.
    skip_prepare / skip_depth
        Reuse cached intermediates from a previous run (handy when only the
        eval changed).
    """
    onnx_path  = Path(onnx_path)
    data_root  = Path(data_root)
    output_dir = Path(output_dir)
    prepared   = output_dir / "isp"

    if not skip_prepare:
        stage_prepare(onnx_path, data_root, prepared,
                      num_workers=num_workers, providers=providers)
    if not skip_depth:
        # ``stage_depth`` writes ``output_dir/depth/`` and
        # ``output_dir/visual/`` — flat under the run's output root,
        # alongside ``isp/`` and the final summary plot.
        stage_depth(prepared, output_dir, device=device)
    return stage_eval(output_dir / "depth", output_dir, quiet=quiet)


def _cli():
    p = argparse.ArgumentParser(description="End-to-end ysstereo benchmark.")
    p.add_argument("--onnx", default=str(DEFAULT_ONNX),
                   help=f"denoiser ONNX (mode 2). Default: {DEFAULT_ONNX} "
                        "(gitignored — drop your trained model here for "
                        "no-args invocation).")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-depth",   action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    res = run_benchmark(
        onnx_path=args.onnx, data_root=args.data_root,
        output_dir=args.output_dir, device=args.device,
        num_workers=args.num_workers, quiet=args.quiet,
        skip_prepare=args.skip_prepare, skip_depth=args.skip_depth,
    )
    return 0 if res.score is not None else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
