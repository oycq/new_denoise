"""Training loop for Neighbor2Neighbor Bayer denoising.

Loss is fixed to the **Neighbor2Neighbor (CVPR 2021) recipe**:

::

    L = || f(g₁) − g₂ ||²  +  γ(t) · || (f(g₁) − g₂) − (g₁(f(y)) − g₂(f(y))) ||²

where :math:`f` is the network, :math:`g_1, g_2` are random sub-samples of
the noisy input :math:`y`, the second term is the **consistency regulariser**
proposed in the paper, and :math:`\\gamma(t)` ramps linearly from 0 to 2
over the budget so the model has a chance to converge before the regulariser
takes over.

The project's experimentation (see ``docs/EXPERIMENTS_JOURNEY.md``)
explored many alternatives — VGG perceptual loss, mode penalties, Gr-Gb
consistency, channel std equalisation — and found they all *hurt* either
through over-smoothing, colour shifts, or new artefacts. The paper's
two-term loss alone, combined with a non-residual UNet (see ``model.py``),
is the simplest configuration that produces a clean output without the
2×2 grid that the residual + L1 + VGG variants suffered from.

Recommended budget: ≥ 600 s (10 min) on a 4090. Below ~5 min the
non-residual UNet has not yet escaped the random-init basin and may
output a catastrophic 2×2 grid.
"""
from __future__ import annotations

import contextlib as _contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.4
except ImportError:
    from torch.cuda.amp import GradScaler, autocast as _autocast_old  # PyTorch < 2.4

    @_contextlib.contextmanager
    def autocast(device_type="cuda", enabled=True, dtype=None, cache_enabled=True):
        kw = {"enabled": enabled, "cache_enabled": cache_enabled}
        if dtype is not None:
            kw["dtype"] = dtype
        with _autocast_old(**kw):
            yield
from torch.utils.data import DataLoader

from .dataset import BayerN2NDataset, find_raw_files
from .gpu_dataset import GPUDataset
from .model import UNet
from .n2n_sampler import generate_index_pair, subimage_by_idx
from .raw_utils import RAW_MAX, pack_rggb, read_raw


# ---------------------------------------------------------------------------
# Device-aware optimisation toggles
# ---------------------------------------------------------------------------
def _autodetect_amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """Pick the best autocast dtype for the current GPU."""
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def _try_torch_compile(model: torch.nn.Module) -> torch.nn.Module:
    """``torch.compile(mode='reduce-overhead')`` if available, else eager."""
    try:
        import triton  # noqa: F401
    except Exception:
        return model
    try:
        return torch.compile(model, mode="reduce-overhead", dynamic=False)
    except Exception as exc:
        print(f"[trainer] torch.compile unavailable, using eager: {exc}")
        return model


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    data_root: str = "train_data/Data"
    patch_size: int = 256         # in packed-pixel units (=> 512 in Bayer)
    batch_size: int = 6
    base_channels: int = 48
    unet_depth: int = 3
    lr: float = 3e-4
    samples_per_epoch: int = 1024
    num_workers: int = 4          # only used by the legacy DataLoader fallback
    use_gpu_dataset: bool = True
    use_fp16: bool = True
    preview_size: int = 64        # patch size in packed-pixel units
    preview_iso_dir: str = "16"
    seed: int = 1234
    ckpt_path: str = "checkpoints/n2n_model.pt"
    # Default 10 min — empirically the smallest budget where the non-residual
    # UNet reliably converges out of random init. 5 min is a danger zone
    # (sometimes works, sometimes outputs a catastrophic 2×2 grid).
    train_seconds: float = 600.0
    # Final value of γ in the consistency regulariser. The paper uses 1.0
    # for raw-RGB experiments and 2.0 for synthetic-noise sRGB; 2.0 also
    # works fine on this dataset and gives a slightly cleaner output.
    gamma_final: float = 2.0
    # Optional intermediate checkpoint times (seconds).
    intermediate_save_seconds: tuple = (300.0, 600.0)
    # GradScaler is only meaningful for fp16; bf16 has fp32 dynamic range
    # so it never under/overflows and we always skip it.
    use_grad_scaler: bool = False
    use_torch_compile: bool = True
    save_final: bool = True
    use_fused_adam: bool = True
    use_channels_last: bool = True


@dataclass
class StepInfo:
    """One emission *per epoch* (mean loss components, plus progress info)."""
    step: int
    epoch: int
    loss: float       # mean total loss (L_rec + γ·L_reg) across the epoch
    rec_loss: float   # mean reconstruction term ‖f(g₁) − g₂‖²
    reg_loss: float   # mean consistency regulariser ‖(f(g₁)−g₂) − (g₁(f(y))−g₂(f(y)))‖²
    gamma: float      # current γ value (ramps linearly 0 → cfg.gamma_final)
    elapsed: float
    progress: float   # in [0, 1]
    steps_per_sec: float


@dataclass
class PreviewInfo:
    step: int
    epoch: int
    noisy_packed: np.ndarray     # (H, W, 4) float32 in [0, 1]
    denoised_packed: np.ndarray  # (H, W, 4) float32 in [0, 1]


@dataclass
class TrainState:
    losses: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        on_step: Optional[Callable[[StepInfo], None]] = None,
        on_preview: Optional[Callable[[PreviewInfo], None]] = None,
        on_finish: Optional[Callable[[Path], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.cfg = cfg
        self.on_step = on_step
        self.on_preview = on_preview
        self.on_finish = on_finish
        self.is_cancelled = is_cancelled or (lambda: False)
        self.state = TrainState()

    # -- helpers -----------------------------------------------------------
    def _build(self) -> tuple[UNet, torch.optim.Optimizer, "object", torch.Tensor]:
        torch.manual_seed(self.cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        files = find_raw_files(self.cfg.data_root)
        if not files:
            raise RuntimeError(
                f"No *_raw.png found under {self.cfg.data_root}. "
                "Check the data path."
            )

        if self.cfg.use_gpu_dataset and device.type == "cuda":
            mem_fmt = (torch.channels_last
                       if self.cfg.use_channels_last
                       else torch.contiguous_format)
            data_src = GPUDataset(files, device=device, memory_format=mem_fmt)
            print(f"[trainer] GPUDataset: {len(data_src)} frames, "
                  f"{data_src.memory_mb():.0f} MB on {device}")
        else:
            ds_cpu = BayerN2NDataset(
                files,
                patch_size=self.cfg.patch_size,
                samples_per_epoch=self.cfg.samples_per_epoch,
            )
            loader_kwargs: dict = dict(
                batch_size=self.cfg.batch_size, shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=(device.type == "cuda"), drop_last=True,
            )
            if self.cfg.num_workers > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = 4
            data_src = DataLoader(ds_cpu, **loader_kwargs)

        model = UNet(
            in_ch=4, out_ch=4,
            base=self.cfg.base_channels, depth=self.cfg.unet_depth,
        ).to(device)
        if device.type == "cuda" and self.cfg.use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        adam_kwargs = dict(lr=self.cfg.lr)
        if device.type == "cuda" and self.cfg.use_fused_adam:
            adam_kwargs["fused"] = True
        opt = torch.optim.Adam(model.parameters(), **adam_kwargs)

        preview_path = self._find_preview_file(files)
        np_raw = read_raw(preview_path).astype(np.float32) / RAW_MAX
        preview_full = pack_rggb(np_raw)
        ps = self.cfg.preview_size
        h0 = (preview_full.shape[0] - ps) // 2
        w0 = (preview_full.shape[1] - ps) // 2
        preview_patch = preview_full[h0:h0 + ps, w0:w0 + ps]
        preview_t = (torch.from_numpy(preview_patch.transpose(2, 0, 1).copy())
                     .unsqueeze(0).to(device))
        return model, opt, data_src, preview_t

    def _find_preview_file(self, files: list) -> Path:
        target = self.cfg.preview_iso_dir
        for f in files:
            if Path(f).parent.name == target:
                return Path(f)
        return Path(files[0])

    def _run_preview(
        self,
        model: UNet,
        preview_t: torch.Tensor,
        step: int,
        epoch: int,
    ) -> None:
        if self.on_preview is None:
            return
        model.eval()
        with torch.no_grad():
            with autocast("cuda", enabled=self.cfg.use_fp16):
                den = model.denoise(preview_t)
        den = den.float().clamp(0, 1)
        noisy_np = preview_t[0].float().cpu().numpy().transpose(1, 2, 0)
        den_np = den[0].cpu().numpy().transpose(1, 2, 0)
        self.on_preview(PreviewInfo(
            step=step, epoch=epoch,
            noisy_packed=noisy_np, denoised_packed=den_np,
        ))
        model.train()

    # -- main loop ---------------------------------------------------------
    def run(self) -> Path:
        cfg = self.cfg
        model, opt, loader, preview_t = self._build()
        device = preview_t.device

        amp_dtype = _autodetect_amp_dtype(device) if cfg.use_fp16 else None
        amp_enabled = amp_dtype is not None
        need_scaler = (
            cfg.use_grad_scaler
            and amp_dtype is torch.float16
            and device.type == "cuda"
        )
        scaler = GradScaler("cuda", enabled=need_scaler)
        print(f"[trainer] amp_dtype={amp_dtype}  scaler={need_scaler}  "
              f"gamma_final={cfg.gamma_final}  budget={cfg.train_seconds:.0f}s")

        if cfg.use_torch_compile:
            model_compiled = _try_torch_compile(model)
            if model_compiled is not model:
                print("[trainer] torch.compile active (reduce-overhead)")
        else:
            model_compiled = model

        Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        step = 0
        epoch = 0
        budget = max(1.0, cfg.train_seconds)
        model.train()

        pending_saves = sorted(set(
            float(s) for s in (cfg.intermediate_save_seconds or ())
            if 0 < float(s) < budget
        ))

        loss_sum = torch.zeros((), device=device)
        rec_sum = torch.zeros((), device=device)
        reg_sum = torch.zeros((), device=device)
        loss_count = 0
        epoch_t0 = time.time()

        self._run_preview(model, preview_t, step=0, epoch=0)

        is_gpu_ds = isinstance(loader, GPUDataset)
        steps_per_epoch = max(1, cfg.samples_per_epoch // cfg.batch_size)

        stop = False
        while not stop:
            epoch += 1
            batch_iter = (
                (loader.sample_batch(cfg.batch_size, cfg.patch_size)
                 for _ in range(steps_per_epoch))
                if is_gpu_ds else loader
            )
            for batch in batch_iter:
                if self.is_cancelled():
                    stop = True
                    break
                elapsed = time.time() - t0
                if elapsed >= budget:
                    stop = True
                    break

                if not is_gpu_ds:
                    batch = batch.to(device, non_blocking=True)
                    if device.type == "cuda" and self.cfg.use_channels_last:
                        batch = batch.to(memory_format=torch.channels_last)

                # γ ramps linearly from 0 to gamma_final across the budget so the
                # model has a chance to converge before the regulariser takes over.
                gamma_t = (elapsed / budget) * cfg.gamma_final

                # N2N pair sampling stays in fp32 (no autocast benefit).
                idx1, idx2 = generate_index_pair(batch)
                g1 = subimage_by_idx(batch, idx1)
                g2 = subimage_by_idx(batch, idx2)

                with autocast(
                    "cuda",
                    enabled=amp_enabled,
                    dtype=amp_dtype if amp_enabled else torch.float16,
                ):
                    # Reconstruction term: f(g₁) ↦ g₂
                    den_g1 = model_compiled(g1)
                    diff = den_g1 - g2

                    # Consistency regulariser: full-image inference (no grad)
                    # sub-sampled to compare to (f(g₁) − g₂).
                    with torch.no_grad():
                        full_denoised = model_compiled(batch)
                        full_g1 = subimage_by_idx(full_denoised, idx1)
                        full_g2 = subimage_by_idx(full_denoised, idx2)
                    exp_diff = full_g1 - full_g2

                    rec_term = (diff ** 2).mean()
                    reg_term = ((diff - exp_diff) ** 2).mean()
                    loss = rec_term + gamma_t * reg_term

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                step += 1
                loss_sum = loss_sum + loss.detach()
                rec_sum = rec_sum + rec_term.detach()
                reg_sum = reg_sum + reg_term.detach()
                loss_count += 1

            # End of an epoch: emit step info on a single sync.
            if loss_count > 0:
                mean_loss = float((loss_sum / loss_count).item())
                mean_rec = float((rec_sum / loss_count).item())
                mean_reg = float((reg_sum / loss_count).item())
                loss_sum = torch.zeros((), device=device)
                rec_sum = torch.zeros((), device=device)
                reg_sum = torch.zeros((), device=device)
                ep_dt = time.time() - epoch_t0
                sps = loss_count / max(1e-6, ep_dt)
                self.state.losses.append(mean_loss)
                self.state.epochs.append(epoch)

                if self.on_step:
                    self.on_step(StepInfo(
                        step=step, epoch=epoch,
                        loss=mean_loss, rec_loss=mean_rec, reg_loss=mean_reg,
                        gamma=gamma_t,
                        elapsed=time.time() - t0,
                        progress=min(1.0, (time.time() - t0) / budget),
                        steps_per_sec=sps,
                    ))
                self._run_preview(model, preview_t, step=step, epoch=epoch)
                loss_count = 0
                epoch_t0 = time.time()

            # Intermediate snapshots (epoch-aligned).
            now = time.time() - t0
            while pending_saves and pending_saves[0] <= now:
                t = pending_saves.pop(0)
                ipath = Path(cfg.ckpt_path).with_name(
                    Path(cfg.ckpt_path).stem + f"_{int(round(t))}s"
                    + Path(cfg.ckpt_path).suffix
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.__dict__,
                        "steps": step,
                        "elapsed": now,
                        "snapshot_seconds": t,
                    },
                    ipath,
                )

        ckpt = Path(cfg.ckpt_path)
        if cfg.save_final:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "steps": step,
                    "elapsed": time.time() - t0,
                },
                ckpt,
            )
        if self.on_finish:
            self.on_finish(ckpt)
        return ckpt
