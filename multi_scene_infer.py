"""Run 4 trained models on the ISO=16 sensor2 frame of every scene.

Saves the resulting ISP renders into ``experiments/_eval/multi_scene/<scene>/<model>.png``.

Used to power the 20-ROI cross-scene visual comparison report.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from n2n.infer import denoise_full_raw, load_model
from n2n.model import UNet
from n2n.raw_utils import gray_world_gains, raw_to_display, raw_to_linear_bgr, read_raw


def _remap_old_keys(state_dict: dict, depth: int = 3) -> dict:
    """Translate the older named-submodule UNet (enc1/up1/dec1) into the
    new ModuleList layout (encs.0/ups.0/decs.0). Older checkpoints from
    `00_baseline_l1` etc. use the legacy naming."""
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        # enc1 -> encs.0, enc2 -> encs.1, ...
        for i in range(1, depth + 1):
            nk = nk.replace(f"enc{i}.", f"encs.{i-1}.")
        # up1/dec1 is the *shallowest* (last) up; up{depth} is the deepest (first).
        # In new code: ups[0] = deepest, decs[0] = deepest.
        for i in range(1, depth + 1):
            nk = nk.replace(f"up{i}.",  f"ups.{depth-i}.")
            nk = nk.replace(f"dec{i}.", f"decs.{depth-i}.")
        new_sd[nk] = v
    return new_sd


def load_model_compat(ckpt_path: str | Path, device: torch.device):
    """Like ``load_model`` but transparently handles legacy enc{i}/dec{i} naming."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    base = cfg.get("base_channels", 48)
    depth = cfg.get("unet_depth", 3)
    model = UNet(in_ch=4, out_ch=4, base=base, depth=depth).to(device).eval()
    sd = ckpt["model"]
    if any(k.startswith("enc1.") for k in sd):
        sd = _remap_old_keys(sd, depth=depth)
    model.load_state_dict(sd)
    return model, cfg

MODELS = {
    "baseline_5min":  "experiments/00_baseline_l1_v2/model.pt",
    "lam01_10min":    "experiments/13c_l1_plus_tv_lam01_10min/model.pt",
    "lam02_10min":    "experiments/13b_l1_plus_tv_lam02_10min/model.pt",
    "lam05_10min":    "experiments/13_l1_plus_tv_lam05_10min/model.pt",
}

SCENES = [
    "changqiao_qiaodi", "langan_louti", "louti_shumu_caoguan",
    "qiang_qiaodi", "shumu_guanmu_che", "shumu_langan", "zixingche_shumu_jiedao",
]

OUT_ROOT = Path("experiments/_eval/multi_scene")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default="sensor2", help="sensor2 / sensor3")
    parser.add_argument("--data-root", default="train_data/Data")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all raw images first (cheap), so we can stream through models.
    raws = {}
    for s in SCENES:
        f = Path(args.data_root) / s / "16" / f"{args.sensor}_raw.png"
        if not f.exists():
            print(f"  skip {s}: missing {f}")
            continue
        raws[s] = read_raw(f)
        print(f"  loaded {s}  shape={raws[s].shape}")

    # Per scene: write the noisy ISP once (will share WB gains computed
    # from the cleanest model later, but for now we use gray-world on raw
    # directly so the noisy panel is consistent).
    for model_name, ckpt in MODELS.items():
        if not Path(ckpt).exists():
            print(f"  skip model {model_name}: missing ckpt")
            continue
        print(f"\n=== Running {model_name} ===")
        model, cfg = load_model_compat(ckpt, device)
        for scene, raw in raws.items():
            scene_dir = OUT_ROOT / scene
            scene_dir.mkdir(parents=True, exist_ok=True)
            denoised = denoise_full_raw(model, raw, use_fp16=True)
            # WB shared per-scene-per-model (computed on this model's denoised
            # output, since it has the cleanest stats). This keeps within-model
            # before/after comparison consistent.
            gains = gray_world_gains(raw_to_linear_bgr(denoised))
            isp_dn = raw_to_display(denoised, wb_gains=gains)
            cv2.imwrite(str(scene_dir / f"{model_name}_with_denoise.png"), isp_dn)
            # Also write the noisy ISP for this scene with the cleanest model's
            # WB so colours align - only save once (skip if exists).
            noisy_path = scene_dir / "noisy_isp.png"
            if not noisy_path.exists():
                isp_no = raw_to_display(raw, wb_gains=gains)
                cv2.imwrite(str(noisy_path), isp_no)
            print(f"    {scene} -> {model_name}_with_denoise.png")
        del model
        torch.cuda.empty_cache()

    print(f"\nAll outputs in {OUT_ROOT}/")


if __name__ == "__main__":
    main()
