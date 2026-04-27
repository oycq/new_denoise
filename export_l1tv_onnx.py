"""Export the L1+0.01·TV residual UNet as a (1,1,H,W) -> (1,1,H,W) ONNX.

The wrapper bakes packing/unpacking and the residual subtraction into the
graph so the ONNX has the same I/O contract as `/home/bobiou/nikon/network.onnx`:
input/output are single-channel bayer_01 of shape (1, 1, 1280, 1088).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from n2n.model import UNet


class WrappedDenoiser(nn.Module):
    """Single-channel bayer_01 in -> single-channel bayer_01 out."""

    def __init__(self, unet: UNet) -> None:
        super().__init__()
        self.pack = nn.PixelUnshuffle(2)
        self.unpack = nn.PixelShuffle(2)
        self.unet = unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.pack(x)
        residual = self.unet(packed)
        denoised_packed = packed - residual
        return self.unpack(denoised_packed)


def _strip_compile_prefix(sd: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/l1tv_lam01_10min.pt")
    ap.add_argument("--out", default="checkpoints/l1tv_lam01_10min.onnx")
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--width", type=int, default=1088)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ckpt not found: {ckpt_path}", file=sys.stderr)
        return 1
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    base = int(cfg.get("base_channels", 48))
    depth = int(cfg.get("unet_depth", 3))

    unet = UNet(in_ch=4, out_ch=4, base=base, depth=depth)
    sd = _strip_compile_prefix(ckpt["model"])
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"missing={missing}\nunexpected={unexpected}", file=sys.stderr)
        return 2
    unet.eval()

    wrapped = WrappedDenoiser(unet).eval()
    h, w = args.height, args.width
    if h % 2 or w % 2:
        print(f"H/W must be even, got ({h}, {w})", file=sys.stderr)
        return 3
    dummy = torch.randn(1, 1, h, w, dtype=torch.float32)

    with torch.no_grad():
        y = wrapped(dummy)
    assert y.shape == dummy.shape, (y.shape, dummy.shape)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"exported -> {out_path}  ({h}x{w})")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        in_meta = sess.get_inputs()[0]
        out_meta = sess.get_outputs()[0]
        print(f"onnx input : {in_meta.name} {in_meta.shape} {in_meta.type}")
        print(f"onnx output: {out_meta.name} {out_meta.shape} {out_meta.type}")
        x = np.random.rand(1, 1, h, w).astype(np.float32)
        ort_y = sess.run([out_meta.name], {in_meta.name: x})[0]
        with torch.no_grad():
            torch_y = wrapped(torch.from_numpy(x)).numpy()
        diff = float(np.abs(ort_y - torch_y).max())
        print(f"onnx vs torch max|diff| = {diff:.2e}")
    except Exception as exc:
        print(f"onnxruntime check skipped: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
