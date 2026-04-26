"""Convenience entrypoint: launch the PyQt5 training GUI for N seconds.

Usage
-----
    python run_train.py [SECONDS]

This is a thin wrapper around ``train_gui.py`` that pre-fills the training
budget, auto-starts training, and auto-runs the full-set inference into
``result/`` once training finishes - so you watch the live loss curve and
the 64x64 noisy/denoised preview in the UI without clicking anything.

Examples
--------
    python run_train.py 600        # 10 minutes (recommended minimum)
    python run_train.py 1200       # 20 minutes
"""
from __future__ import annotations

import sys

import train_gui


def main():
    # 600 s is the smallest budget where the non-residual UNet reliably
    # converges out of random init. Going lower risks the catastrophic
    # 2×2-grid output the experiments in tmp4/ documented.
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    sys.argv = [sys.argv[0], str(seconds), "--auto-start", "--auto-result"]
    train_gui.main()


if __name__ == "__main__":
    main()
