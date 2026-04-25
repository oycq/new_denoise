"""3-minute throughput benchmark.

Just run it - no arguments. Reports the achievable step/sec for the
default `TrainConfig` and how many epochs you'd hit at 5 / 10 / 20 min,
plus a recommended ``train_seconds`` to match the reference dev box
(258 epochs at 20 min on RTX 2070).

    python benchmark.py
"""
from __future__ import annotations
import time
from dataclasses import replace

from n2n.trainer import StepInfo, Trainer, TrainConfig

BENCHMARK_SECONDS = 180.0  # 3 minutes
DEV_TARGET_EPOCHS = 258    # epochs at the 20-min mark on the dev box (RTX 2070)


def main() -> None:
    cfg = replace(
        TrainConfig(),
        train_seconds=BENCHMARK_SECONDS,
        intermediate_save_seconds=(),
        ckpt_path="checkpoints/_benchmark.pt",
    )
    print(f"=== benchmark · {cfg.loss_name}({cfg.loss_kwargs}) · "
          f"patch={cfg.patch_size} batch={cfg.batch_size} depth={cfg.unet_depth} ===")
    print(f"  data_root : {cfg.data_root}")
    print(f"  duration  : {cfg.train_seconds:.0f}s ({cfg.train_seconds/60:.1f} min)\n")

    last = {"step": 0, "epoch": 0, "elapsed": 0.0}

    def _on_step(s: StepInfo) -> None:
        if s.epoch % 5 == 0:
            print(f"   ep={s.epoch:3d}  step={s.step:5d}  t={s.elapsed:5.0f}s  "
                  f"sps={s.steps_per_sec:.1f}")
        last["step"] = s.step
        last["epoch"] = s.epoch
        last["elapsed"] = s.elapsed

    t0 = time.time()
    Trainer(cfg, on_step=_on_step).run()
    elapsed = time.time() - t0

    steps  = last["step"]
    epochs = last["epoch"]
    sps    = steps / max(1e-3, elapsed)
    steps_per_epoch = round(cfg.samples_per_epoch / cfg.batch_size)

    print()
    print("=== summary ===")
    print(f"  wall-clock        : {elapsed:6.1f} s  ({elapsed/60:.2f} min)")
    print(f"  steps             : {steps}")
    print(f"  epochs            : {epochs}")
    print(f"  steps / sec       : {sps:.1f}")
    print(f"  steps / epoch     : {steps_per_epoch}  ({cfg.samples_per_epoch} / {cfg.batch_size})")
    print()
    print(f"  projected epochs : 5 min -> {int(round(sps*300/steps_per_epoch))}   "
          f"10 min -> {int(round(sps*600/steps_per_epoch))}   "
          f"20 min -> {int(round(sps*1200/steps_per_epoch))}")
    print()

    target_full = DEV_TARGET_EPOCHS * steps_per_epoch / max(1e-3, sps)
    print(f"  reference: dev box (RTX 2070) reaches {DEV_TARGET_EPOCHS} epochs at 20 min.")
    print(f"  to match the same epoch count on THIS machine:")
    print(f"     train_seconds = {target_full:6.0f} s  ({target_full/60:.1f} min)")
    print()
    print("Set n2n/trainer.py TrainConfig.train_seconds to this value if your")
    print("machine is slower / faster than the dev box.")


if __name__ == "__main__":
    main()
