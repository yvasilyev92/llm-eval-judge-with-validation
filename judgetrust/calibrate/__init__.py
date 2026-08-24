"""Calibration mode: run the judge against a human-labeled set. No generation."""

from judgetrust.calibrate.dataset import load_calibration_set
from judgetrust.calibrate.metrics import cohen_kappa, kappa_band, raw_agreement
from judgetrust.calibrate.runner import format_report, persist_report, run_calibration

__all__ = [
    "cohen_kappa",
    "format_report",
    "kappa_band",
    "load_calibration_set",
    "persist_report",
    "raw_agreement",
    "run_calibration",
]
