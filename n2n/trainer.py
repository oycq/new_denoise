"""Training loop for Neighbor2Neighbor Bayer denoising.

The :class:`Trainer` is designed to be driven by a GUI: it exposes
callbacks that fire after every step (loss, progress) and after every
``preview_every`` steps (a 64x64 RGGB preview of noisy vs denoised). It
also stops automatically once a wall-clock budget is reached.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .dataset import BayerN2NDataset, find_raw_files
from .losses import get as get_loss
from .model import UNet
from .n2n_sampler import generate_mask_pair, generate_subimages
from .raw_utils import RAW_MAX, pack_rggb, read_raw


@dataclass
class TrainConfig:
    data_root: str = "train_data/Data"
    # Defaults reflect the winning strategy from `experiments/REPORT.md`:
    # Multi-scale Charbonnier (4 scales) + patch 256 - best blob suppression
    # AND best edge preservation among 16 tested strategies.
    patch_size: int = 256         # in packed-pixel units (=> 512 in Bayer)
    batch_size: int = 6
    base_channels: int = 48
    unet_depth: int = 3           # 3 = local-feature (default), 4 = standard N2N
    lr: float = 2e-4
    samples_per_epoch: int = 1024
    num_workers: int = 0          # Windows + small data: 0 is faster
    use_fp16: bool = True
    # Weight of the N2N consistency regulariser. The user requested the
    # simplest setup: a single L1 main loss only, so the default is 0 -
    # i.e. ``loss == main == L1(g1 - f(g1), g2)``. Set to e.g. 1.0 to
    # re-enable the regulariser (Huang et al. 2021 Eq.4).
    n2n_lambda: float = 0.0
    preview_size: int = 64        # patch size in packed-pixel units
    preview_iso_dir: str = "16"   # subdirectory name for the noisiest ISO
    seed: int = 1234
    ckpt_path: str = "checkpoints/n2n_model.pt"
    train_seconds: float = 600.0  # wall-clock budget
    # Name of the main loss function used for the N2N main term, looked up
    # in ``n2n.losses.REGISTRY``. The reg term (when n2n_lambda > 0) always
    # uses plain L1 for stability.
    loss_name: str = "ms_l1_charbonnier"
    # Optional kwargs for the loss (e.g. multiscale_l1 scales=4).
    loss_kwargs: dict = field(default_factory=lambda: {"scales": 4})
    # If True, subtract ~9 (in 0-255 domain, equivalently 2304/65535 normalised)
    # from input before feeding to the model. Centres the input near zero,
    # may improve fp16 numerics.
    subtract_black_level: bool = False


@dataclass
class StepInfo:
    """One emission *per epoch* (mean loss, plus progress info)."""
    step: int        # global step count at end of this epoch
    epoch: int
    loss: float      # mean total loss across the epoch (main + lambda * reg)
    main_loss: float # mean L1 main term (the actual denoising error - noise floor)
    reg_loss: float  # mean L1 regulariser (consistency); ramps from 0 over training
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
    losses: List[float] = field(default_factory=list)  # per-epoch mean losses
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
    def _build(self) -> tuple[UNet, torch.optim.Optimizer, DataLoader, torch.Tensor]:
        torch.manual_seed(self.cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        files = find_raw_files(self.cfg.data_root)
        if not files:
            raise RuntimeError(
                f"No *_raw.png found under {self.cfg.data_root}. "
                "Check the data path."
            )
        ds = BayerN2NDataset(
            files,
            patch_size=self.cfg.patch_size,
            samples_per_epoch=self.cfg.samples_per_epoch,
        )
        loader = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        model = UNet(
            in_ch=4, out_ch=4,
            base=self.cfg.base_channels, depth=self.cfg.unet_depth,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

        # The preview is the most informative when taken from the noisiest ISO:
        # we look for `<data_root>/<scene>/<preview_iso_dir>/sensorN_raw.png`.
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
        return model, opt, loader, preview_t

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
            inp = preview_t
            if self.cfg.subtract_black_level:
                inp = inp - (9.0 * 256.0 / RAW_MAX)
            with autocast("cuda", enabled=self.cfg.use_fp16):
                den = model.denoise(inp)
            if self.cfg.subtract_black_level:
                den = den + (9.0 * 256.0 / RAW_MAX)
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
        scaler = GradScaler("cuda", enabled=cfg.use_fp16 and device.type == "cuda")
        loss_base = get_loss(cfg.loss_name)
        if cfg.loss_kwargs:
            loss_kw = dict(cfg.loss_kwargs)
            def main_loss_fn(d, t, _f=loss_base, _kw=loss_kw):
                return _f(d, t, **_kw)
        else:
            main_loss_fn = loss_base

        # Optional zero-mean preprocessing: subtract a constant equal to the
        # 8-bit black-level scaled into the [0, 1] input domain.
        bl_norm = 0.0
        if cfg.subtract_black_level:
            bl_norm = 9.0 * 256.0 / RAW_MAX  # ~0.0352

        Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        step = 0
        epoch = 0
        budget = max(1.0, cfg.train_seconds)
        model.train()

        # Loss accumulators on the GPU - avoids per-step .cpu() sync
        loss_sum = torch.zeros((), device=device)
        main_sum = torch.zeros((), device=device)
        reg_sum = torch.zeros((), device=device)
        loss_count = 0
        epoch_t0 = time.time()

        # Emit a starting preview (epoch 0, untrained)
        self._run_preview(model, preview_t, step=0, epoch=0)

        stop = False
        while not stop:
            epoch += 1
            for batch in loader:
                if self.is_cancelled():
                    stop = True
                    break
                elapsed = time.time() - t0
                if elapsed >= budget:
                    stop = True
                    break

                batch = batch.to(device, non_blocking=True)
                if bl_norm > 0:
                    batch = batch - bl_norm

                # Mask sampling does no benefit from autocast and only adds
                # int<->half overhead; keep it in fp32 outside autocast.
                mask1, mask2 = generate_mask_pair(batch)
                g1 = generate_subimages(batch, mask1)
                g2 = generate_subimages(batch, mask2)

                with autocast("cuda", enabled=cfg.use_fp16):
                    pred_noise_g1 = model(g1)
                    den_g1 = g1 - pred_noise_g1
                    main = main_loss_fn(den_g1, g2)

                    if cfg.n2n_lambda > 0:
                        # N2N consistency regulariser (paper Eq.4). Skipped
                        # when lambda == 0 to halve the forward cost.
                        with torch.no_grad():
                            full_noise = model(batch)
                            full_den = batch - full_noise
                            full_g1 = generate_subimages(full_den, mask1)
                            full_g2 = generate_subimages(full_den, mask2)
                        reg = F.l1_loss(den_g1 - g2, full_g1 - full_g2)
                        loss = main + cfg.n2n_lambda * reg
                    else:
                        reg = torch.zeros((), device=device)
                        loss = main

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                step += 1
                loss_sum = loss_sum + loss.detach()
                main_sum = main_sum + main.detach()
                reg_sum = reg_sum + reg.detach()
                loss_count += 1

            # End of an epoch (or stopped mid-epoch): summarise on GPU then
            # move the scalars to CPU all at once.
            if loss_count > 0:
                mean_loss = float((loss_sum / loss_count).item())
                mean_main = float((main_sum / loss_count).item())
                mean_reg = float((reg_sum / loss_count).item())
                loss_sum = torch.zeros((), device=device)
                main_sum = torch.zeros((), device=device)
                reg_sum = torch.zeros((), device=device)
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
                            main_loss=mean_main,
                            reg_loss=mean_reg,
                            elapsed=time.time() - t0,
                            progress=min(1.0, (time.time() - t0) / budget),
                            steps_per_sec=sps,
                        )
                    )
                self._run_preview(model, preview_t, step=step, epoch=epoch)
                loss_count = 0
                epoch_t0 = time.time()

        ckpt = Path(cfg.ckpt_path)
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
