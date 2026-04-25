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
    python run_train.py 300        # 5 minutes
    python run_train.py 60         # quick 1-minute smoke run
    python run_train.py 1800       # 30 minutes
"""
from __future__ import annotations

import sys

import train_gui


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    sys.argv = [sys.argv[0], str(seconds), "--auto-start", "--auto-result"]
    train_gui.main()


if __name__ == "__main__":
    main()
