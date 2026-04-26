"""Robust Neighbor2Neighbor trainer for the 3h push.

Goal
----
Make the denoiser produce *visually clean* output even when the sensor
has strong fixed-pattern noise (FPN: DSNU / PRNU / hot pixels). Vanilla
N2N's "two sub-images of one frame" assumption breaks down under FPN
because FPN is *perfectly correlated* across the two sub-images, so the
network learns to keep it.

This trainer adds five orthogonal mechanisms, all of which are
*training-time only* and do not change the inference network at all
(BPU deploy stays identical):

1.  **EMA self-distillation (NN-as-loss)** — maintain an exponential
    moving average copy of the model. On each step, compute
    ``L1(student_denoised, teacher_denoised)`` on a fresh full-resolution
    batch, weighted by ``lam_ema``. EMA acts as a soft "smoothness"
    teacher: per-batch FPN bias gets averaged out across thousands of
    steps, and the student is then pushed back toward that smoother target.

2.  **R2R noise injection** — at probability ``p_r2r`` per step, add a
    small i.i.d. Gaussian to the input. The injected noise is independent
    of FPN, breaking the perfect input/target FPN correlation.

3.  **Random pixel masking** — replace ``mask_ratio`` of input pixels
    with their 4-neighbour average before the N2N sampler runs. Forces
    the network to predict *from context*, so it cannot memorise the
    exact value of any single (potentially-FPN-biased) pixel
    (Noise2Self / J-invariance trick).

4.  **Sub-pixel spatial jitter** — apply a per-batch fractional shift
    via bilinear sampling (default ±0.5 packed-pixel). FPN is hard-binned
    to integer pixel positions, so a sub-pixel resampled FPN looks
    different at every training step — the network can't lock to any
    spatial pattern.

5.  **Dark-region loss weighting (optional, via ``loss_name``)** — see
    ``n2n.losses_v3``. Routes the dark-corner gradient signal up so the
    network spends capacity *there*.

Mechanisms 1-4 are stacked on top of *any* loss; mechanism 5 is a loss
function choice. The candidate sweep in ``run_all.py`` enables them
individually and in combinations.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Reuse trainer building blocks from the prod codebase.
from .dataset import find_raw_files
from .gpu_dataset import GPUDataset
from .losses import get as get_loss
from .model import UNet
from .n2n_sampler import generate_index_pair, subimage_by_idx


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RobustTrainConfig:
    data_root: str = "train_data/Data"
    seed: int = 1234
    train_seconds: float = 300.0           # default 5 min
    patch_size: int = 256                  # packed-pixel
    batch_size: int = 6
    lr: float = 2e-4
    samples_per_epoch: int = 1024
    base_channels: int = 48
    unet_depth: int = 3
    use_fp16: bool = True

    # Loss
    loss_name: str = "l1_plus_tv"
    loss_kwargs: dict = field(default_factory=lambda: {"lam": 0.03})

    # --- Mechanism 1: EMA self-distillation -------------------------------
    use_ema_distill: bool = False
    ema_decay: float = 0.999
    ema_lam: float = 0.10
    # Steps before we start using the teacher (let EMA settle from zero-init).
    ema_warmup_steps: int = 200
    # Re-use the same g1 batch as student for teacher input (saves a forward).
    # Teacher's denoised g1 is detached and used as soft target for student.

    # --- Mechanism 2: R2R noise injection ---------------------------------
    use_r2r: bool = False
    r2r_sigma: float = 0.005               # injected Gaussian std (in [0,1] domain)
    r2r_prob: float = 1.0                  # apply on every step (R2R is stable)

    # --- Mechanism 3: Random pixel masking (J-invariance) ----------------
    use_mask: bool = False
    mask_ratio: float = 0.05               # fraction of pixels to replace

    # --- Mechanism 4: Sub-pixel spatial jitter ----------------------------
    use_jitter: bool = False
    jitter_amount: float = 0.5             # fractional packed-pixel shift, ±

    # --- Resume / continuation
    # If provided and exists, load model + EMA + step counter + loss_log
    # from this ckpt before training. The new ``train_seconds`` budget is
    # then *additional* wall-clock time on top of whatever the ckpt
    # already accumulated. Use this to extend training of an existing
    # candidate without losing the prior loss-curve history.
    resume_from: str = ""

    # --- Output
    ckpt_path: str = "checkpoints/robust.pt"
    log_every: int = 100


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------
class EMAModel:
    """Maintains an EMA copy of a model's parameters and buffers."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for name, ema_param in self.module.state_dict().items():
            cur = msd[name]
            if ema_param.dtype.is_floating_point:
                ema_param.mul_(self.decay).add_(cur.detach(), alpha=1.0 - self.decay)
            else:
                ema_param.copy_(cur)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


# ---------------------------------------------------------------------------
# Augmentations (training-time only)
# ---------------------------------------------------------------------------
def random_pixel_mask_(x: torch.Tensor, ratio: float = 0.05) -> torch.Tensor:
    """In-place: replace a random fraction of pixels with a 4-neighbour avg.

    Operates on packed (N, C, H, W) tensor. The chosen pixels are
    replaced with the mean of their up/down/left/right neighbours. This
    is the J-invariant masking trick used in Noise2Self / B-AS pipelines.
    """
    n, c, h, w = x.shape
    # 4-neighbour average via padding+slicing (avoids a conv).
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    nbr = (xp[..., 1:-1, :-2] + xp[..., 1:-1, 2:] + xp[..., :-2, 1:-1] + xp[..., 2:, 1:-1]) * 0.25
    # Per-batch mask
    mask = torch.empty(n, 1, h, w, device=x.device).uniform_() < ratio
    x = torch.where(mask.expand_as(x), nbr, x)
    return x


def subpixel_jitter(x: torch.Tensor, max_shift: float = 0.5) -> torch.Tensor:
    """Apply a per-batch random sub-pixel translation via grid_sample.

    The shift breaks the network's ability to memorise FPN at exact pixel
    positions across batches. Reflect padding keeps borders sane.
    """
    n, c, h, w = x.shape
    # Per-batch random shift in [-max_shift, max_shift]
    sh_x = (torch.rand(n, device=x.device) * 2.0 - 1.0) * max_shift
    sh_y = (torch.rand(n, device=x.device) * 2.0 - 1.0) * max_shift
    # grid_sample expects normalised coords in [-1, 1]; pixel shift d -> 2d/(W-1)
    # We build identity grid and add the shift.
    yy = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, h, 1, 1).expand(n, h, w, 1)
    xx = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, w, 1).expand(n, h, w, 1)
    sx = (sh_x.view(n, 1, 1, 1) * 2.0 / max(w - 1, 1)).expand(n, h, w, 1)
    sy = (sh_y.view(n, 1, 1, 1) * 2.0 / max(h - 1, 1)).expand(n, h, w, 1)
    grid = torch.cat([xx + sx, yy + sy], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def r2r_corrupt(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add i.i.d. Gaussian noise of std ``sigma`` to a batch."""
    return x + torch.randn_like(x) * sigma


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class RobustTrainer:
    def __init__(
        self,
        cfg: RobustTrainConfig,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ):
        self.cfg = cfg
        self.is_cancelled = is_cancelled or (lambda: False)

    def _build_model(self, device: torch.device) -> UNet:
        return UNet(
            in_ch=4, out_ch=4,
            base=self.cfg.base_channels, depth=self.cfg.unet_depth,
        ).to(device)

    def _autocast_dtype(self, device: torch.device) -> Optional[torch.dtype]:
        if device.type != "cuda" or not self.cfg.use_fp16:
            return None
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16

    def run(self) -> Path:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        files = find_raw_files(cfg.data_root)
        if not files:
            raise RuntimeError(f"No raw files under {cfg.data_root}")

        ds = GPUDataset(files, device=device, dtype=torch.float32)
        print(f"[robust] dataset: {len(ds)} frames, {ds.memory_mb():.0f} MB on {device}")

        model = self._build_model(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        ema = EMAModel(model, decay=cfg.ema_decay) if cfg.use_ema_distill else None

        # Optional: resume model + EMA + step counter + loss_log from an
        # existing ckpt. The saved "model" key is the EMA copy when EMA
        # was on (per the trainer's save() convention), so we use it for
        # both student and EMA init. This way "more training" is just an
        # extension of the old run, not a fresh start.
        prev_steps = 0
        prev_loss_log: list[tuple[int, float]] = []
        prev_elapsed = 0.0
        if cfg.resume_from and Path(cfg.resume_from).exists():
            try:
                ck = torch.load(cfg.resume_from, map_location=device, weights_only=False)
                model.load_state_dict(ck["model"])
                if ema is not None:
                    ema.module.load_state_dict(ck["model"])
                prev_steps = int(ck.get("steps", 0) or 0)
                prev_elapsed = float(ck.get("elapsed", 0.0) or 0.0)
                prev_loss_log = list(ck.get("loss_log", []) or [])
                print(
                    f"[robust] resumed from {cfg.resume_from} "
                    f"(prev_steps={prev_steps}, prev_elapsed={prev_elapsed:.0f}s, "
                    f"prev_log_pts={len(prev_loss_log)})"
                )
            except Exception as exc:  # pragma: no cover
                print(f"[robust] WARNING: resume from {cfg.resume_from} failed: {exc}")

        loss_base = get_loss(cfg.loss_name)
        loss_kwargs = dict(cfg.loss_kwargs) if cfg.loss_kwargs else {}

        def main_loss_fn(d, t):
            if loss_kwargs:
                return loss_base(d, t, **loss_kwargs)
            return loss_base(d, t)

        amp_dtype = self._autocast_dtype(device)
        amp_enabled = amp_dtype is not None

        Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        step = prev_steps
        epoch = 0
        budget = max(1.0, cfg.train_seconds)
        steps_per_epoch = max(1, cfg.samples_per_epoch // cfg.batch_size)

        loss_log: list[tuple[int, float]] = list(prev_loss_log)  # (step, loss)
        recent: list[float] = []
        recent_main: list[float] = []
        recent_ema: list[float] = []

        model.train()
        stop = False
        last_log_t = t0

        while not stop:
            epoch += 1
            for _ in range(steps_per_epoch):
                if self.is_cancelled():
                    stop = True
                    break
                elapsed = time.time() - t0
                if elapsed >= budget:
                    stop = True
                    break

                batch = ds.sample_batch(cfg.batch_size, cfg.patch_size)

                # Mechanism 4: sub-pixel jitter on the full input
                if cfg.use_jitter and cfg.jitter_amount > 0:
                    batch = subpixel_jitter(batch, max_shift=cfg.jitter_amount)

                # Mechanism 3: random pixel mask
                masked_batch = batch
                if cfg.use_mask and cfg.mask_ratio > 0:
                    masked_batch = random_pixel_mask_(batch.clone(), ratio=cfg.mask_ratio)

                # Mechanism 2: R2R input corruption
                input_batch = masked_batch
                if cfg.use_r2r and cfg.r2r_sigma > 0:
                    if random.random() < cfg.r2r_prob:
                        input_batch = r2r_corrupt(masked_batch, cfg.r2r_sigma)

                # N2N sub-image sampling
                idx1, idx2 = generate_index_pair(input_batch)
                g1 = subimage_by_idx(input_batch, idx1)
                # IMPORTANT: target g2 is from the *original* batch (no R2R / no mask)
                # so the network's optimum is still the clean signal, not the
                # masked/r2r-corrupted version.
                g2 = subimage_by_idx(batch, idx2)

                with torch.amp.autocast(
                    "cuda",
                    enabled=amp_enabled,
                    dtype=amp_dtype if amp_enabled else torch.float16,
                ):
                    pred_noise = model(g1)
                    den_g1 = g1 - pred_noise
                    main = main_loss_fn(den_g1, g2)

                    ema_term = torch.tensor(0.0, device=device, dtype=main.dtype)
                    if ema is not None and step >= cfg.ema_warmup_steps:
                        with torch.no_grad():
                            teacher_noise = ema.module(g1)
                            teacher_den = g1 - teacher_noise
                        ema_term = F.l1_loss(den_g1, teacher_den.detach())
                        loss = main + cfg.ema_lam * ema_term
                    else:
                        loss = main

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if ema is not None:
                    ema.update(model)

                step += 1
                recent.append(float(loss.detach().item()))
                recent_main.append(float(main.detach().item()))
                if ema is not None:
                    recent_ema.append(float(ema_term.detach().item()))

                if step % cfg.log_every == 0:
                    sps = cfg.log_every / max(1e-6, time.time() - last_log_t)
                    last_log_t = time.time()
                    main_avg = sum(recent_main[-cfg.log_every:]) / cfg.log_every
                    ema_avg = sum(recent_ema[-cfg.log_every:]) / max(1, len(recent_ema[-cfg.log_every:])) if recent_ema else 0.0
                    print(
                        f"[robust] step={step:6d} ep={epoch:3d} "
                        f"loss={sum(recent[-cfg.log_every:])/cfg.log_every:.5f} "
                        f"main={main_avg:.5f} ema={ema_avg:.5f} "
                        f"elapsed={elapsed:.1f}s sps={sps:.1f}"
                    )
                    loss_log.append((step, sum(recent[-cfg.log_every:]) / cfg.log_every))

        # Save the *EMA* model weights when EMA is active (smoother), else
        # save the trained student. EMA also pulled toward the same optimum
        # but with much-reduced batch noise.
        save_model = ema.module if ema is not None else model
        ckpt = Path(cfg.ckpt_path)
        torch.save(
            {
                "model": save_model.state_dict(),
                "config": cfg.__dict__,
                "steps": step,
                "elapsed": prev_elapsed + (time.time() - t0),
                "loss_log": loss_log,
                "trained_with_ema": ema is not None,
            },
            ckpt,
        )
        print(f"[robust] done. saved {ckpt} after {step} steps / {time.time()-t0:.1f}s")
        return ckpt
