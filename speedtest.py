"""1-minute throughput speedtest.

无参数运行。报告的是 ``torch.compile`` 首编译期之后的 **稳态** sps
（取所有 warmup 后 epoch 的中位数），而不是含 warmup 的整体平均——
后者会被 1 min 测试里 ~30 s 的编译期严重拉低，参考意义不大。

注意：本脚本只测训练吞吐量，与 ``benchmark/`` 目录下的双目深度降噪评估无关。

    python speedtest.py
"""
from __future__ import annotations

import time
from dataclasses import replace
from statistics import median

from n2n.trainer import StepInfo, Trainer, TrainConfig

BENCHMARK_SECONDS = 60.0    # 1 min total budget
WARMUP_SECONDS    = 30.0    # torch.compile reduce-overhead 首编译期约 25–30 s
DEV_TARGET_EPOCHS = 977     # epochs at the 10-min mark on the dev box (RTX 4090)


def main() -> None:
    cfg = replace(
        TrainConfig(),
        train_seconds=BENCHMARK_SECONDS,
        intermediate_save_seconds=(),
        ckpt_path="checkpoints/_speedtest.pt",
    )
    print(f"=== speedtest · L1 + {cfg.tv_lambda}*TV · "
          f"patch={cfg.patch_size} batch={cfg.batch_size} depth={cfg.unet_depth} ===")
    print(f"  data_root      : {cfg.data_root}")
    print(f"  duration       : {cfg.train_seconds:.0f}s")
    print(f"  warmup ignored : 0-{WARMUP_SECONDS:.0f}s "
          f"(torch.compile first-compile, marked '*' below)\n")

    # (elapsed, sps_this_epoch, epoch, step)
    epoch_data: list[tuple[float, float, int, int]] = []

    def _on_step(s: StepInfo) -> None:
        in_warmup = s.elapsed < WARMUP_SECONDS
        marker = "*" if in_warmup else " "
        # Print every warmup epoch (so the slow start is visible) and
        # every 5th steady-state epoch (otherwise the output is too long).
        if in_warmup or s.epoch % 5 == 0:
            print(f"   {marker} ep={s.epoch:4d}  step={s.step:6d}  t={s.elapsed:5.0f}s  "
                  f"sps={s.steps_per_sec:5.1f}  loss={s.loss:.4f}")
        epoch_data.append((s.elapsed, s.steps_per_sec, s.epoch, s.step))

    Trainer(cfg, on_step=_on_step).run()

    # Steady-state sps = epochs that *completed* after the warmup window.
    stable = [(e, sps) for e, sps, _, _ in epoch_data if e >= WARMUP_SECONDS]
    if len(stable) < 3:
        # Very slow GPU / very short benchmark — fall back to the last
        # half so we always have something to report.
        tail = epoch_data[len(epoch_data) // 2:]
        stable = [(e, sps) for e, sps, _, _ in tail]
        print(f"\n  ⚠ only {len(stable)} epochs after {WARMUP_SECONDS:.0f}s; "
              f"using last 50% as fallback.")

    sps_values = [s for _, s in stable]
    sps_med = median(sps_values)
    sps_min = min(sps_values)
    sps_max = max(sps_values)

    steps_per_epoch = round(cfg.samples_per_epoch / cfg.batch_size)
    last_step = epoch_data[-1][3] if epoch_data else 0
    last_epoch = epoch_data[-1][2] if epoch_data else 0

    print()
    print("=== summary ===")
    print(f"  warmup-excluded epochs : {len(stable)} / {len(epoch_data)}")
    print(f"  steady-state sps       : {sps_med:5.1f}  "
          f"(min {sps_min:.1f} · max {sps_max:.1f})")
    print(f"  total steps  / epochs  : {last_step} / {last_epoch}")
    print(f"  steps / epoch          : {steps_per_epoch}  "
          f"({cfg.samples_per_epoch} / {cfg.batch_size})")
    print()
    print("  projected epochs (extrapolated from steady-state):")
    print(f"      5 min  -> {int(round(sps_med * 300 / steps_per_epoch))}")
    print(f"      10 min -> {int(round(sps_med * 600 / steps_per_epoch))}")
    print(f"      20 min -> {int(round(sps_med * 1200 / steps_per_epoch))}")
    print()

    target_full = DEV_TARGET_EPOCHS * steps_per_epoch / max(1e-3, sps_med)
    print(f"  reference: dev box (RTX 4090) reaches {DEV_TARGET_EPOCHS} "
          f"epochs at 10 min.")
    print(f"  to match the same epoch count on this machine:")
    print(f"      train_seconds = {target_full:6.0f} s  ({target_full/60:.1f} min)")


if __name__ == "__main__":
    main()
