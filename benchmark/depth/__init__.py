"""Stereo-depth inference module.

The vendored ``ysstereo`` package (mmengine-based) lives next to this file.
``run_depth(...)`` is the single public entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Put the directory holding the `ysstereo` package on sys.path before any
# downstream import touches it, so `from ysstereo.apis import ...` works no
# matter where the user invoked the script from.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from .runner import run_depth  # noqa: E402

__all__ = ["run_depth"]
