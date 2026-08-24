"""Calibration runner with an injected judge — no live API."""

from __future__ import annotations

from pathlib import Path

import pytest

from judgetrust.calibrate.dataset import load_calibration_set
from judgetrust.calibrate.runner import (
    format_report,
    load_report,
    missing_api_key,
    persist_report,
    run_calibration,
)
from judgetrust.models import PresentedJudgment


def _judgment(winner: str | None, *, error: str | None = None) -> PresentedJudgment:
    return PresentedJudgment(
        winner=winner,  # type: ignore[arg-type]
        reason="test",
        confidence=0.8,
        rubric=None,
        raw_output="",
        error=error,
    )


def _prefer_human():
    rows = load_calibration_set()
    by_question = {row.question: row for row in rows}

    def evaluate_fn(question: str, answer_1: str, answer_2: str) -> PresentedJudgment:
        row = by_question[question]
        if row.human_winner == "tie":
            return _judgment("tie")
        preferred = row.answer_a if row.human_winner == "A" else row.answer_b
        if answer_1 == preferred:
            return _judgment("1")
        return _judgment("2")

    return evaluate_fn


def test_perfect_agreement_is_almost_perfect() -> None:
    report = run_calibration(evaluate_fn=_prefer_human())
    assert report.mode == "calibration"
    assert report.n == report.n_scored
    assert report.n_errors == 0
    assert report.raw_agreement == 1.0
    assert report.kappa == 1.0
    assert report.kappa_band == "almost-perfect"
    assert report.position_consistency == 1.0
    assert report.disagreements == ()
    assert report.panel_dissent_rate == 0.0
    assert len(report.judge_kappas) == 3
    assert all(item.kappa == 1.0 for item in report.judge_kappas)


def test_inverted_labels_disagree() -> None:
    rows = load_calibration_set()
    by_question = {row.question: row for row in rows}

    def evaluate_fn(question: str, answer_1: str, answer_2: str) -> PresentedJudgment:
        row = by_question[question]
        if row.human_winner == "tie":
            return _judgment("1")
        preferred = row.answer_b if row.human_winner == "A" else row.answer_a
        if answer_1 == preferred:
            return _judgment("1")
        if answer_2 == preferred:
            return _judgment("2")
        return _judgment("1")

    report = run_calibration(evaluate_fn=evaluate_fn)
    assert report.raw_agreement is not None
    assert report.raw_agreement < 0.2
    assert report.disagreements
    assert report.kappa is not None
    assert report.kappa < 0.2


def test_always_first_is_position_bias() -> None:
    def evaluate_fn(_q: str, _a1: str, _a2: str) -> PresentedJudgment:
        return _judgment("1")

    report = run_calibration(evaluate_fn=evaluate_fn)
    assert report.position_consistency == 0.0
    assert all(row.position_bias for row in report.rows)
    assert all(row.judge_winner == "tie" for row in report.rows)


def test_both_orderings_failed_excluded_from_kappa() -> None:
    def evaluate_fn(_q: str, _a1: str, _a2: str) -> PresentedJudgment:
        return _judgment(None, error="malformed_or_missing_json")

    report = run_calibration(evaluate_fn=evaluate_fn)
    assert report.n_scored == 0
    assert report.n_errors == report.n
    assert report.kappa is None
    assert report.raw_agreement is None
    assert report.kappa_band is None
    assert report.position_consistency == 0.0


def test_persist_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    report = run_calibration(evaluate_fn=_prefer_human())
    written = persist_report(report, path)
    loaded = load_report(written)
    assert loaded["kappa"] == 1.0
    assert loaded["kappa_band"] == "almost-perfect"
    assert loaded["mode"] == "calibration"
    assert loaded["n"] == report.n
    assert "Kappa corrects for chance" in loaded["raw_agreement_note"]
    assert loaded["judge_models"] == list(report.judge_models)
    assert loaded["panel_dissent_rate"] == 0.0


def test_format_report_headlines_kappa() -> None:
    report = run_calibration(evaluate_fn=_prefer_human())
    text = format_report(report)
    assert "Cohen's kappa" in text
    assert "almost-perfect" in text
    assert "Raw agreement" in text
    assert "Position consistency" in text


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert missing_api_key() is True
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert missing_api_key() is False


def test_cli_exits_when_key_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from judgetrust.calibrate.__main__ import main

    assert main() == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
