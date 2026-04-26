"""Training loop for Neighbor2Neighbor Bayer denoising.

Loss is fixed to ``L1(den, g2) + lam_vgg * VGG_perceptual(den, g2)``.
That is the only configuration the project ships after the experiments
documented in ``docs/EXPERIMENTS_JOURNEY.md``: every other variant
(L1+TV, EATV, luma/chroma split, EMA self-distill, R2R, masking,
sub-pixel jitter, …) was either no improvement or just made the step
slower without measurable visual gain.

The :class:`Trainer` is designed to be driven by a GUI: it exposes
callbacks that fire after every epoch (mean L1 / VGG / total) and after
every preview interval (a 64x64 RGGB preview of noisy vs denoised). It
also stops automatically once a wall-clock budget is reached.
"""
from __future__ import annotations

import contextlib as _contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

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
from .losses_perceptual import vgg_perceptual_loss
from .model import UNet
from .n2n_sampler import generate_index_pair, subimage_by_idx
from .raw_utils import RAW_MAX, pack_rggb, read_raw


# ---------------------------------------------------------------------------
# Device-aware optimisation toggles. These are pure detection helpers so the
# same TrainConfig can run optimally on a Turing 2070 (Windows) and on an
# Ampere/Ada 4090 (Linux) without manual flags.
# ---------------------------------------------------------------------------


def _autodetect_amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """Pick the best autocast dtype for the current GPU.

    * Ampere or newer (sm_80+, includes 4090): bfloat16. Same dynamic range
      as fp32, never under/overflows on small losses, and Ampere+ has
      native bf16 tensor cores. Crucially this also lets us drop GradScaler
      entirely (no inf/nan host-side check, no per-step sync).
    * Turing (sm_75, e.g. RTX 2070): float16. fp16 tensor cores exist;
      bf16 would fall back to a slow software path.
    * CPU / older GPU: ``None`` (autocast disabled).
    """
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def _try_torch_compile(model: torch.nn.Module) -> torch.nn.Module:
    """``torch.compile(model, mode='reduce-overhead')`` if available.

    The inductor backend needs Triton, which isn't shipped on Windows; on
    Linux + a recent CUDA toolkit it gives 1.3-1.5x on Ampere+. We probe
    by trying to import the backend; on failure return the eager model.
    """
    try:
        import triton  # noqa: F401  (presence test only)
    except Exception:
        return model
    try:
        return torch.compile(model, mode="reduce-overhead", dynamic=False)
    except Exception as exc:
        print(f"[trainer] torch.compile unavailable, using eager: {exc}")
        return model


@dataclass
class TrainConfig:
    data_root: str = "train_data/Data"
    # Loss is fixed to L1 + lam_vgg * VGG (see module docstring). The
    # only knob is the VGG weight; 0.05 was the value that consistently
    # cleaned up "blotchy" residual noise without over-smoothing.
    lam_vgg: float = 0.05
    patch_size: int = 256         # in packed-pixel units (=> 512 in Bayer)
    batch_size: int = 6
    base_channels: int = 48
    unet_depth: int = 3
    lr: float = 2e-4
    samples_per_epoch: int = 1024
    # ``num_workers`` is the legacy CPU-DataLoader path. The default is now
    # the GPU-resident dataset (see ``use_gpu_dataset``); ``num_workers``
    # is only consulted when that fallback is selected.
    num_workers: int = 4
    # GPU-resident dataset: load every packed RGGB frame onto the GPU at
    # init and do random crop + per-item flip/transpose entirely on the
    # GPU. Removes the CPU H2D copy + DataLoader bottleneck.
    use_gpu_dataset: bool = True
    use_fp16: bool = True
    preview_size: int = 64        # patch size in packed-pixel units
    preview_iso_dir: str = "16"   # subdirectory name for the noisiest ISO
    seed: int = 1234
    ckpt_path: str = "checkpoints/n2n_model.pt"
    train_seconds: float = 1200.0  # wall-clock budget (default 20 min)
    # Optional intermediate checkpoint times (seconds). When any of these
    # is crossed, a snapshot is written next to ``ckpt_path`` with a
    # ``_<seconds>s`` suffix, so a single 20-min run yields multiple
    # comparable checkpoints (e.g. 2/5/10/20 min).
    intermediate_save_seconds: tuple = (120.0, 300.0, 600.0)
    # GradScaler does an inf/nan check + host-side skip decision on every
    # step, which silently inserts a CUDA sync. With L1 + VGG in [0, 1]
    # gradients stay well within fp16 dynamic range; disabling the scaler
    # keeps fp16 forward/backward via autocast but removes the per-step
    # sync. (When autocast picks bf16 we never need it anyway.)
    use_grad_scaler: bool = False
    # On Ampere+ GPUs, ``torch.compile(mode='reduce-overhead')`` captures
    # the model into a CUDA graph and gives ~1.3-1.5x. Requires Triton,
    # which isn't shipped on Windows; the trainer auto-detects and falls
    # back to eager. Set False to force-disable.
    use_torch_compile: bool = True
    # Save the model checkpoint to the configured path even if a separate
    # 2-min snapshot was scheduled. Some experiments use only the final
    # ckpt and want the intermediate saves disabled.
    save_final: bool = True
    # Fused Adam fails on heterogeneous-layout models (notably depthwise
    # separable convs once the model is .to(memory_format=channels_last)).
    # Set False to fall back to the eager Adam path. Keep True by default.
    use_fused_adam: bool = True
    # The trainer applies channels_last to the model on CUDA, which gives
    # ~+20% on baseline convs but fights with depthwise / certain BN
    # layouts. Set False to keep contiguous NCHW for the model only.
    use_channels_last: bool = True


@dataclass
class StepInfo:
    """One emission *per epoch* (mean loss components, plus progress info)."""
    step: int        # global step count at end of this epoch
    epoch: int
    loss: float      # mean total loss (L1 + lam_vgg * VGG) across the epoch
    l1_loss: float   # mean plain-L1 component (the denoising error - noise floor)
    vgg_loss: float  # mean VGG perceptual L1 (raw, before lam_vgg scaling)
    elapsed: float
    progress: float  # in [0, 1]
    steps_per_sec: float


@dataclass
class PreviewInfo:
    step: int
    epoch: int
    noisy_packed: np.ndarray     # (H, W, 4) float32 in [0, 1]
    denoised_packed: np.ndarray  # (H, W, 4) float32 in [0, 1]


@dataclass
class TrainState:
    losses: List[float] = field(default_factory=list)  # per-epoch mean total losses
    epochs: List[int] = field(default_factory=list)


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
    def _build(
        self,
    ) -> tuple[UNet, torch.optim.Optimizer, "object", torch.Tensor]:
        """Build model + optimizer + data source + preview tensor.

        ``data source`` is either a ``GPUDataset`` (default) or a CPU
        ``DataLoader`` (legacy fallback). The training loop disambiguates.
        """
        torch.manual_seed(self.cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            # cuDNN selects the fastest conv algorithm per input shape on the
            # first step then reuses it. Big win for our fixed-shape pipeline.
            torch.backends.cudnn.benchmark = True
            # Allow tf32 fast paths where the hardware supports them
            # (Ampere+; no-op on Turing but cheap to enable).
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        files = find_raw_files(self.cfg.data_root)
        if not files:
            raise RuntimeError(
                f"No *_raw.png found under {self.cfg.data_root}. "
                "Check the data path."
            )

        if self.cfg.use_gpu_dataset and device.type == "cuda":
            mem_fmt = (
                torch.channels_last
                if self.cfg.use_channels_last
                else torch.contiguous_format
            )
            data_src = GPUDataset(
                files, device=device, memory_format=mem_fmt)
            print(f"[trainer] GPUDataset: {len(data_src)} frames, "
                  f"{data_src.memory_mb():.0f} MB on {device}")
        else:
            ds_cpu = BayerN2NDataset(
                files,
                patch_size=self.cfg.patch_size,
                samples_per_epoch=self.cfg.samples_per_epoch,
            )
            loader_kwargs: dict = dict(
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=True,
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

        # Preview is taken from the noisiest ISO, so visual quality is
        # tracked on the hardest case: ``<data_root>/<scene>/<preview_iso_dir>/sensorN_raw.png``.
        preview_path = self._find_preview_file(files)
        np_raw = read_raw(preview_path).astype(np.float32) / RAW_MAX
        preview_full = pack_rggb(np_raw)
        ps = self.cfg.preview_size
        h0 = (preview_full.shape[0] - ps) // 2
        w0 = (preview_full.shape[1] - ps) // 2
        preview_patch = preview_full[h0 : h0 + ps, w0 : w0 + ps]
        preview_t = (
            torch.from_numpy(preview_patch.transpose(2, 0, 1).copy())
            .unsqueeze(0)
            .to(device)
        )
        return model, opt, data_src, preview_t

    def _find_preview_file(self, files: list) -> Path:
        """Pick a stable preview file from the noisiest ISO subdirectory.

        Walks ``files`` looking for one whose parent folder name equals
        ``cfg.preview_iso_dir`` (default "16"); falls back to the first file.
        """
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
        self.on_preview(
            PreviewInfo(
                step=step, epoch=epoch, noisy_packed=noisy_np, denoised_packed=den_np
            )
        )
        model.train()

    # -- main loop ---------------------------------------------------------
    def run(self) -> Path:
        cfg = self.cfg
        model, opt, loader, preview_t = self._build()
        device = preview_t.device

        amp_dtype = _autodetect_amp_dtype(device) if cfg.use_fp16 else None
        amp_enabled = amp_dtype is not None
        # GradScaler is only meaningful for fp16; bf16 has fp32 dynamic range
        # so it never under/overflows and we always skip it.
        need_scaler = (
            cfg.use_grad_scaler
            and amp_dtype is torch.float16
            and device.type == "cuda"
        )
        scaler = GradScaler("cuda", enabled=need_scaler)
        print(f"[trainer] amp_dtype={amp_dtype}  scaler={need_scaler}  "
              f"lam_vgg={cfg.lam_vgg}")

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

        # GPU-resident accumulators avoid per-step .cpu() syncs.
        loss_sum = torch.zeros((), device=device)
        l1_sum = torch.zeros((), device=device)
        vgg_sum = torch.zeros((), device=device)
        loss_count = 0
        epoch_t0 = time.time()

        self._run_preview(model, preview_t, step=0, epoch=0)

        # Two source modes:
        #   - GPUDataset: drive the loop ourselves with
        #     ``steps_per_epoch = samples_per_epoch / batch_size``.
        #   - DataLoader: iterate over its batches per epoch.
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

                # N2N pair sampling stays in fp32 (no autocast benefit and
                # avoids int<->half overhead). ``idx1`` / ``idx2`` are int64
                # per-cell indices.
                idx1, idx2 = generate_index_pair(batch)
                g1 = subimage_by_idx(batch, idx1)
                g2 = subimage_by_idx(batch, idx2)

                with autocast(
                    "cuda",
                    enabled=amp_enabled,
                    dtype=amp_dtype if amp_enabled else torch.float16,
                ):
                    pred_noise_g1 = model_compiled(g1)
                    den_g1 = g1 - pred_noise_g1
                    l1_term = F.l1_loss(den_g1, g2)
                    vgg_term = vgg_perceptual_loss(den_g1, g2)
                    loss = l1_term + cfg.lam_vgg * vgg_term

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                step += 1
                loss_sum = loss_sum + loss.detach()
                l1_sum = l1_sum + l1_term.detach()
                vgg_sum = vgg_sum + vgg_term.detach()
                loss_count += 1

            # End of an epoch (or stopped mid-epoch): summarise on GPU then
            # move the scalars to CPU all at once.
            if loss_count > 0:
                mean_loss = float((loss_sum / loss_count).item())
                mean_l1 = float((l1_sum / loss_count).item())
                mean_vgg = float((vgg_sum / loss_count).item())
                loss_sum = torch.zeros((), device=device)
                l1_sum = torch.zeros((), device=device)
                vgg_sum = torch.zeros((), device=device)
                ep_dt = time.time() - epoch_t0
                sps = loss_count / max(1e-6, ep_dt)
                self.state.losses.append(mean_loss)
                self.state.epochs.append(epoch)

                if self.on_step:
                    self.on_step(
                        StepInfo(
                            step=step,
                            epoch=epoch,
                            loss=mean_loss,
                            l1_loss=mean_l1,
                            vgg_loss=mean_vgg,
                            elapsed=time.time() - t0,
                            progress=min(1.0, (time.time() - t0) / budget),
                            steps_per_sec=sps,
                        )
                    )
                self._run_preview(model, preview_t, step=step, epoch=epoch)
                loss_count = 0
                epoch_t0 = time.time()

            # After each epoch, dump intermediate checkpoints whose
            # time-budget has elapsed. Done at epoch boundaries so the
            # weights match the loss curve point we just emitted.
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
