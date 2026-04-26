"""Train the recommended VGG perceptual recipe (★).

Trains two ckpts back-to-back (resumable, skips ones that already exist):

* ``ckpts/P0_baseline.pt`` — robust training (EMA + R2R + mask + jitter)
  with the locked-in **L1 + 0.03·TV** loss. Apples-to-apples reference
  for VGG (same training schedule, different loss only).

* ``ckpts/P4_vgg.pt`` ★ — robust training with **VGG-16 feature L1**
  (``vgg_l1_eatv``: ``luma_chroma_l1 + 0.03·EATV + 0.05·VGG``).
  This is the configuration the perceptual sweep selected as the
  visual winner; see ``docs/EXPERIMENTS_JOURNEY.md`` for why VGG
  specifically (not chroma / CSF / MS-SSIM) was kept.

Both ckpts are inference-identical to the baseline UNet — VGG and the
robust mechanisms only run at training time, so BPU / ONNX / TRT
deployment is unchanged.

Usage::

    python train_vgg.py                      # train both, default ckpt dir
    python train_vgg.py --ckpt-dir my_ckpts/ # custom output
    python train_vgg.py --only P4_vgg        # train just one of them

Inference::

    from n2n.infer import run_inference_set
    run_inference_set(ckpt_path="ckpts/P4_vgg.pt",
                      data_root="train_data/Data",
                      out_root="result_vgg")
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from n2n.robust_trainer import RobustTrainer, RobustTrainConfig


# ---------------------------------------------------------------------------
# Candidate definitions — the two recipes worth reproducing
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    desc: str
    loss_name: str
    loss_kwargs: dict = field(default_factory=dict)
    use_ema_distill: bool = True
    ema_decay: float = 0.999
    ema_lam: float = 0.10
    use_r2r: bool = True
    r2r_sigma: float = 0.005
    use_mask: bool = True
    mask_ratio: float = 0.05
    use_jitter: bool = True
    jitter_amount: float = 0.5
    train_seconds: float = 360.0

    def to_train_cfg_kwargs(self) -> dict:
        keys = {
            "loss_name", "loss_kwargs",
            "use_ema_distill", "ema_decay", "ema_lam",
            "use_r2r", "r2r_sigma",
            "use_mask", "mask_ratio",
            "use_jitter", "jitter_amount",
            "train_seconds",
        }
        d = asdict(self)
        return {k: v for k, v in d.items() if k in keys}


CANDIDATES: List[Candidate] = [
    Candidate(
        name="P0_baseline",
        desc="对照组 · L1 + 0.03·TV (= 同款 robust 训练，无感知损失)",
        loss_name="l1_plus_tv",
        loss_kwargs={"lam": 0.03},
        train_seconds=300.0,
    ),
    Candidate(
        name="P4_vgg",
        desc="★ VGG-16 特征 L1 (NN-as-loss / LPIPS 风格) + 亮度/色度 L1 + EATV",
        loss_name="vgg_l1_eatv",
        loss_kwargs={
            "w_luma": 4.0, "w_chroma": 1.0,
            "lam_eatv": 0.03,
            "lam_vgg": 0.05,
        },
        train_seconds=420.0,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--ckpt-dir", default="ckpts",
        help="Output directory for ckpts + per-candidate logs (default: ckpts/)"
    )
    parser.add_argument(
        "--data-root", default=str(ROOT / "train_data" / "Data"),
        help="Training data root (default: ./train_data/Data)"
    )
    parser.add_argument(
        "--only", choices=[c.name for c in CANDIDATES], default=None,
        help="Train only this one candidate (default: train all)"
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Training seed (default: 1234)"
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    log_dir = ckpt_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    t_total = time.time()
    for c in CANDIDATES:
        if args.only and c.name != args.only:
            continue

        ckpt = ckpt_dir / f"{c.name}.pt"
        log_path = log_dir / f"{c.name}.log"
        if ckpt.exists():
            print(f"[train_vgg] {c.name}: ckpt exists, skip.")
            summary.append((c.name, "skip", float(c.train_seconds)))
            continue

        print(f"\n[train_vgg] === {c.name} === {c.desc}")
        sys.stdout.flush()

        kwargs = c.to_train_cfg_kwargs()
        cfg = RobustTrainConfig(
            data_root=args.data_root,
            seed=args.seed,
            ckpt_path=str(ckpt),
            log_every=100,
            **kwargs,
        )
        t0 = time.time()
        with open(log_path, "w") as f:
            stdout_save = sys.stdout
            sys.stdout = _Tee(stdout_save, f)
            try:
                RobustTrainer(cfg).run()
            finally:
                sys.stdout = stdout_save

        dt = time.time() - t0
        summary.append((c.name, "trained", dt))
        print(f"[train_vgg] {c.name}: done in {dt:.1f}s")

    print("\n[train_vgg] === ALL DONE ===")
    for n, st, dt in summary:
        print(f"  {n:20s} {st:8s} {dt:7.1f}s")
    print(f"  TOTAL: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
