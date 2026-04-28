"""Per-scene EPE aggregation and QC pass/fail check."""
from .epe import EvalResult, evaluate

__all__ = ["evaluate", "EvalResult"]
