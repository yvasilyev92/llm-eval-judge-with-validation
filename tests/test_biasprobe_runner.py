"""Bias probe runner with an injected judge — no live API."""

from __future__ import annotations

from pathlib import Path

import pytest

from judgetrust.biasprobe.__main__ import main as probe_main
from judgetrust.biasprobe.dataset import load_bias_probe_set
from judgetrust.biasprobe.metrics import length_bias_rate, position_bias_rate
from judgetrust.biasprobe.runner import (
    format_report,
    load_report,
    persist_report,
    run_bias_probe,
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


def _prefer_text(target_side: str):
    rows = load_bias_probe_set()
    by_question = {row.question: row for row in rows}

    def evaluate_fn(question: str, answer_1: str, answer_2: str) -> PresentedJudgment:
        row = by_question[question]
        if target_side == "longer":
            preferred = row.answer_a if row.longer_worse == "A" else row.answer_b
        else:
            preferred = row.answer_b if row.longer_worse == "A" else row.answer_a
        if answer_1 == preferred:
            return _judgment("1")
        return _judgment("2")

    return evaluate_fn


def test_length_bias_rate_helper() -> None:
    assert length_bias_rate([True, True, False]) == pytest.approx(2 / 3)
    assert length_bias_rate([]) is None
    assert position_bias_rate([True, False, False]) == pytest.approx(1 / 3)


def test_always_pick_longer_is_full_length_bias() -> None:
    report = run_bias_probe(evaluate_fn=_prefer_text("longer"))
    assert report.mode == "bias_probe"
    assert report.n_errors == 0
    assert report.length_bias_rate == 1.0
    assert report.position_bias_rate == 0.0
    assert all(row.length_bias_hit for row in report.rows)


def test_always_pick_shorter_is_zero_length_bias() -> None:
    report = run_bias_probe(evaluate_fn=_prefer_text("shorter"))
    assert report.length_bias_rate == 0.0
    assert not any(row.length_bias_hit for row in report.rows)
    assert report.position_bias_rate == 0.0


def test_always_first_is_position_bias_not_length_hit() -> None:
    def evaluate_fn(_q: str, _a1: str, _a2: str) -> PresentedJudgment:
        return _judgment("1")

    report = run_bias_probe(evaluate_fn=evaluate_fn)
    assert report.position_bias_rate == 1.0
    assert report.length_bias_rate == 0.0
    assert all(row.judge_winner == "tie" for row in report.rows)


def test_both_failed_excluded() -> None:
    def evaluate_fn(_q: str, _a1: str, _a2: str) -> PresentedJudgment:
        return _judgment(None, error="malformed_or_missing_json")

    report = run_bias_probe(evaluate_fn=evaluate_fn)
    assert report.n_scored == 0
    assert report.length_bias_rate is None
    assert report.n_errors == report.n


def test_persist_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "bias_probe.json"
    report = run_bias_probe(evaluate_fn=_prefer_text("shorter"))
    persist_report(report, path)
    loaded = load_report(path)
    assert loaded["mode"] == "bias_probe"
    assert loaded["length_bias_rate"] == 0.0


def test_format_report() -> None:
    text = format_report(run_bias_probe(evaluate_fn=_prefer_text("longer")))
    assert "Length-bias rate" in text
    assert "Position-bias rate" in text


def test_cli_missing_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert probe_main() == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
