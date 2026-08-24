"""UI modules import without launching Streamlit widgets."""

from __future__ import annotations

from judgetrust.report import assemble_trust_report
from judgetrust.ui.styles import DISCLAIMER


def test_disclaimer_mentions_eval_not_medical() -> None:
    text = DISCLAIMER.lower()
    assert "test the judge" in text
    assert "not medical advice" in text or "not to give medical advice" in text
    assert "reviewed" in text


def test_assemble_from_disk_without_results() -> None:
    report = assemble_trust_report(load_missing_from_disk=True)
    assert "calibration" in report.missing or report.kappa is not None
    assert report.verdict
