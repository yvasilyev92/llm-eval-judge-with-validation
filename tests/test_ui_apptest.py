"""Streamlit AppTest: shell, empty live input, missing-key error. No live API runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_app_shell_and_tabs() -> None:
    at = _app()
    assert not at.exception
    assert at.title[0].value == "Judge Trust"
    assert [tab.label for tab in at.tabs] == [
        "Calibrate judge",
        "Compare prompts (live)",
        "Run bias probe",
    ]
    assert {button.label for button in at.button} == {
        "Run calibration",
        "Compare prompts",
        "Run bias probe",
    }
    markdown = " ".join(item.value for item in at.markdown)
    captions = " ".join(item.value for item in at.caption)
    blob = f"{markdown} {captions}".lower()
    assert "test the judge" in markdown.lower()
    assert "not to give medical advice" in markdown.lower()
    assert "Self-preference caveat" in markdown or "self-preference" in markdown.lower()
    assert "judge trust report" in markdown.lower()
    assert "labeled pairs" in blob
    assert "rigged pairs" in blob
    assert "No live comparison in this session yet." in captions


def test_empty_live_question_warns() -> None:
    at = _app()
    button = next(item for item in at.button if item.label == "Compare prompts")
    button.click().run()
    assert at.warning
    assert "Enter a question" in at.warning[0].value
    assert not at.exception


def test_calibrate_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = _app()
    button = next(item for item in at.button if item.label == "Run calibration")
    button.click().run()
    assert at.error
    assert "OPENAI_API_KEY" in at.error[0].value
    assert not at.exception
