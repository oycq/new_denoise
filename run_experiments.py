"""Sequential N2N denoising experiments harness.

For every entry in ``STRATEGIES`` we

  1. train for ``SECONDS`` wall-clock seconds with the given config
  2. run inference on a fixed noisy reference frame (the noisiest ISO=16
     ``changqiao_qiaodi/sensor2_raw.png``)
  3. write the resulting denoised ISP-style render to
     ``experiments/<name>/sensor2_isp_with_denoise.png`` (plus the matching
     ``sensor2_isp_without_denoise.png`` for an ImageJ stack)

Logs are appended to ``experiments/log.txt``. Results are easy to A/B
inspect by loading ``experiments/*/sensor2_isp_with_denoise.png`` as a
stack in ImageJ.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from n2n.infer import denoise_full_raw, load_model
from n2n.raw_utils import gray_world_gains, raw_to_display, raw_to_linear_bgr, read_isp, read_raw
from n2n.trainer import TrainConfig, Trainer

REF_RAW = Path("train_data/Data/changqiao_qiaodi/16/sensor2_raw.png")
REF_ISP = Path("train_data/Data/changqiao_qiaodi/16/sensor2_isp.png")
EXP_ROOT = Path("experiments")


@dataclass
class Strategy:
    """One experiment configuration."""

    name: str
    description: str
    overrides: dict = field(default_factory=dict)


def default_strategies() -> list[Strategy]:
    return [
        # ---- round 1 (loss family + simple knobs) ----
        Strategy(
            name="00_baseline_l1",
            description="3-level UNet, patch 192, L1 main loss only (current default).",
            overrides={"loss_name": "l1"},
        ),
        Strategy(
            name="01_charbonnier",
            description="Charbonnier (smoothed L1) - quadratic near zero, no posterisation.",
            overrides={"loss_name": "charbonnier"},
        ),
        Strategy(
            name="02_l1_plus_grad",
            description="L1 + 0.5 * L1 of spatial gradients (edge-preserving).",
            overrides={"loss_name": "l1_plus_grad"},
        ),
        Strategy(
            name="03_multiscale_l1",
            description="Multi-scale L1 (3 dyadic pooled scales) - kills low-freq blobs.",
            overrides={"loss_name": "multiscale_l1"},
        ),
        Strategy(
            name="04_huber",
            description="Huber (smooth_l1, beta=0.01) - quadratic core, L1 tails.",
            overrides={"loss_name": "huber"},
        ),
        Strategy(
            name="05_l1_patch256",
            description="L1 main, patch 256 (more spatial context per training crop).",
            overrides={"loss_name": "l1", "patch_size": 256, "batch_size": 6},
        ),
        Strategy(
            name="06_multiscale_patch256",
            description="Multi-scale L1 + patch 256 - top 2 strategies combined.",
            overrides={"loss_name": "multiscale_l1", "patch_size": 256, "batch_size": 6},
        ),
        # ---- round 2 (architecture / scales) ----
        Strategy(
            name="07_unet4_l1_p256",
            description="Standard 4-level UNet (5.5M) + L1 + patch 256.",
            overrides={
                "loss_name": "l1", "patch_size": 256, "batch_size": 4,
                "unet_depth": 4,
            },
        ),
        Strategy(
            name="08_ms4_p192",
            description="Multi-scale L1 with 4 scales (down to 16x), patch 192.",
            overrides={"loss_name": "multiscale_l1", "loss_kwargs": {"scales": 4}},
        ),
        Strategy(
            name="09_ms4_p256",
            description="Multi-scale L1 (4 scales) + patch 256.",
            overrides={
                "loss_name": "multiscale_l1", "loss_kwargs": {"scales": 4},
                "patch_size": 256, "batch_size": 6,
            },
        ),
        Strategy(
            name="10_ms_charbonnier_p256",
            description="Multi-scale Charbonnier (4 scales) + patch 256 - smoothness combined.",
            overrides={
                "loss_name": "ms_l1_charbonnier", "loss_kwargs": {"scales": 4},
                "patch_size": 256, "batch_size": 6,
            },
        ),
        Strategy(
            name="11_charb_p256",
            description="Charbonnier + patch 256 - smoother loss with bigger context.",
            overrides={"loss_name": "charbonnier", "patch_size": 256, "batch_size": 6},
        ),
        # ---- round 3 (preprocessing / regularisation) ----
        Strategy(
            name="12_bl_subtract_l1",
            description="L1 main, but subtract black level from input (zero-mean) for fp16 stability.",
            overrides={"loss_name": "l1", "subtract_black_level": True},
        ),
        Strategy(
            name="13_l1_plus_tv",
            description="L1 + 0.05 * total variation - encourage piecewise smooth.",
            overrides={"loss_name": "l1_plus_tv"},
        ),
        Strategy(
            name="14_unet4_ms_p256",
            description="4-level UNet + multi-scale L1 (4 scales) + patch 256 - max quality.",
            overrides={
                "loss_name": "multiscale_l1", "loss_kwargs": {"scales": 4},
                "patch_size": 256, "batch_size": 4, "unet_depth": 4,
            },
        ),
        Strategy(
            name="15_unet4_ms_charb_p256",
            description="4-level UNet + multi-scale Charbonnier - heavyweight option.",
            overrides={
                "loss_name": "ms_l1_charbonnier", "loss_kwargs": {"scales": 4},
                "patch_size": 256, "batch_size": 4, "unet_depth": 4,
            },
        ),
        # ---- TV lambda sweep (refines strategy 13) ----
        Strategy(
            name="13b_l1_plus_tv_lam02",
            description="L1 + 0.02 * TV - moderate piecewise smoothing.",
            overrides={"loss_name": "l1_plus_tv", "loss_kwargs": {"lam": 0.02}},
        ),
        Strategy(
            name="13c_l1_plus_tv_lam01",
            description="L1 + 0.01 * TV - mild piecewise smoothing.",
            overrides={"loss_name": "l1_plus_tv", "loss_kwargs": {"lam": 0.01}},
        ),
        Strategy(
            name="13d_l1_plus_tv_lam005",
            description="L1 + 0.005 * TV - very light TV touch.",
            overrides={"loss_name": "l1_plus_tv", "loss_kwargs": {"lam": 0.005}},
        ),
        Strategy(
            name="13c_l1_plus_tv_lam01_10min",
            description="L1 + 0.01 * TV - same as 13c but trained twice as long.",
            overrides={"loss_name": "l1_plus_tv", "loss_kwargs": {"lam": 0.01}},
        ),
        Strategy(
            name="13b_l1_plus_tv_lam02_10min",
            description="L1 + 0.02 * TV - same as 13b but 10-min training.",
            overrides={"loss_name": "l1_plus_tv", "loss_kwargs": {"lam": 0.02}},
        ),
        Strategy(
            name="13_l1_plus_tv_lam05_10min",
            description="L1 + 0.05 * TV - same as 13 but 10-min training.",
            overrides={"loss_name": "l1_plus_tv", "loss_kwargs": {"lam": 0.05}},
        ),
        Strategy(
            name="00_baseline_l1_v2",
            description="Fresh L1 baseline with current arch (depth=3 patch=256 batch=6).",
            overrides={"loss_name": "l1", "loss_kwargs": {}},
        ),
    ]


def render_pair(
    model, raw_u16: np.ndarray, out_dir: Path,
    *, subtract_black_level: bool = False,
) -> tuple[Path, Path]:
    denoised = denoise_full_raw(
        model, raw_u16, subtract_black_level=subtract_black_level
    )
    # Shared WB gains computed in 16-bit linear domain on the denoised image.
    shared = gray_world_gains(raw_to_linear_bgr(denoised))
    no_dn = raw_to_display(raw_u16, wb_gains=shared)
    with_dn = raw_to_display(denoised, wb_gains=shared)
    out_dir.mkdir(parents=True, exist_ok=True)
    no_path = out_dir / "sensor2_isp_without_denoise.png"
    yes_path = out_dir / "sensor2_isp_with_denoise.png"
    cv2.imwrite(str(no_path), no_dn)
    cv2.imwrite(str(yes_path), with_dn)
    return no_path, yes_path


def run_strategy(strategy: Strategy, seconds: float, log_path: Path) -> dict:
    out_dir = EXP_ROOT / strategy.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_kwargs = dict(
        train_seconds=float(seconds),
        ckpt_path=str(out_dir / "model.pt"),
        # Sensible defaults for the harness; individual strategies override.
        patch_size=192,
        batch_size=8,
        base_channels=48,
        lr=2e-4,
        samples_per_epoch=1024,
        seed=1234,
    )
    cfg_kwargs.update(strategy.overrides)
    cfg = TrainConfig(**cfg_kwargs)

    print(f"\n=== {strategy.name} ===", flush=True)
    print(f"    {strategy.description}", flush=True)
    print(f"    cfg: loss={cfg.loss_name} patch={cfg.patch_size} batch={cfg.batch_size}", flush=True)

    losses_per_epoch: list[tuple[int, float]] = []
    t0 = time.time()

    def on_step(s):
        losses_per_epoch.append((s.epoch, s.main_loss))
        if s.epoch % 5 == 0 or s.epoch == 1:
            print(
                f"    ep={s.epoch:3d} step={s.step:5d} main={s.main_loss:.5f} "
                f"sps={s.steps_per_sec:5.1f} t={s.elapsed:5.1f}s",
                flush=True,
            )

    trainer = Trainer(cfg, on_step=on_step)
    trainer.run()
    train_dt = time.time() - t0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_cfg = load_model(cfg.ckpt_path, device)
    raw = read_raw(REF_RAW)
    no_path, yes_path = render_pair(
        model, raw, out_dir,
        subtract_black_level=ckpt_cfg.get("subtract_black_level", False),
    )
    if REF_ISP.exists():
        shutil.copy(REF_ISP, out_dir / "vendor_isp_reference.png")

    final_main = float(np.mean([l for _, l in losses_per_epoch[-5:]])) if losses_per_epoch else float("nan")

    summary = {
        "name": strategy.name,
        "description": strategy.description,
        "config": asdict(cfg),
        "train_seconds": train_dt,
        "epochs_completed": losses_per_epoch[-1][0] if losses_per_epoch else 0,
        "final_main_loss_avg5": final_main,
        "isp_with_denoise_path": str(yes_path),
        "isp_without_denoise_path": str(no_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description="Run a sweep of N2N denoising strategies")
    p.add_argument("--seconds", type=float, default=300.0, help="train budget per strategy")
    p.add_argument("--only", nargs="*", default=None, help="run only these strategy names")
    args = p.parse_args()

    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = EXP_ROOT / "log.txt"
    log_path.write_text("")  # truncate

    strategies = default_strategies()
    if args.only:
        strategies = [s for s in strategies if s.name in set(args.only)]

    results = []
    t0 = time.time()
    for s in strategies:
        try:
            summary = run_strategy(s, args.seconds, log_path)
            results.append(summary)
        except Exception as exc:  # noqa: BLE001
            print(f"!!! {s.name} failed: {exc}", flush=True)

    print("\n=== sweep complete ===")
    print(f"total wall: {time.time() - t0:.0f}s")
    for r in results:
        print(f"  {r['name']:24s}  main_loss={r['final_main_loss_avg5']:.5f}  -> {r['isp_with_denoise_path']}")


if __name__ == "__main__":
    main()
