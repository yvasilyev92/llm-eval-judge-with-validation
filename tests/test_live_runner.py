"""Live runner with injected generate/evaluate — no live API."""

from __future__ import annotations

from pathlib import Path

import pytest

from judgetrust.config import Settings
from judgetrust.live.__main__ import main as live_main
from judgetrust.live.runner import format_report, load_report, persist_report, run_live
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


def _generate(question: str, task_prompt: str, model: str) -> str:
    kind = "B" if "doctor" in task_prompt.lower() or "risk" in task_prompt.lower() else "A"
    return f"{model}|{kind}|{question[:20]}"


def _prefer_b(_question: str, answer_1: str, answer_2: str) -> PresentedJudgment:
    if "|B|" in answer_1 and "|B|" not in answer_2:
        return _judgment("1")
    if "|B|" in answer_2:
        return _judgment("2")
    return _judgment("tie")


def test_b_wins_all_models() -> None:
    report = run_live(
        "Is sunscreen useful on a cloudy day?",
        generate_fn=_generate,
        evaluate_fn=_prefer_b,
    )
    assert report.mode == "live"
    assert report.n == 3
    assert report.n_scored == 3
    assert report.prompt_b_win_rate == 1.0
    assert report.cross_model_agreement == 1.0
    assert report.position_consistency == 1.0
    assert all(duel.winner == "B" and duel.stable for duel in report.duels)
    assert report.panel_dissent_rate == 0.0
    assert all(len(duel.votes) == 3 for duel in report.duels)


def test_mixed_winners_agreement() -> None:
    def evaluate_fn(_q: str, answer_1: str, answer_2: str) -> PresentedJudgment:
        blob = answer_1 + answer_2
        preferred = "|A|" if "gpt-4.1-mini" in blob else "|B|"
        if preferred in answer_1:
            return _judgment("1")
        return _judgment("2")

    report = run_live("q", generate_fn=_generate, evaluate_fn=evaluate_fn)
    assert report.prompt_b_win_rate == pytest.approx(2 / 3)
    assert report.cross_model_agreement == pytest.approx(2 / 3)


def test_always_first_is_inconsistent() -> None:
    def evaluate_fn(_q: str, _a1: str, _a2: str) -> PresentedJudgment:
        return _judgment("1")

    report = run_live("q", generate_fn=_generate, evaluate_fn=evaluate_fn)
    assert report.position_consistency == 0.0
    assert all(duel.position_bias and duel.winner == "tie" for duel in report.duels)


def test_generate_failure_is_recorded() -> None:
    def generate_fn(_q: str, _p: str, _m: str) -> str:
        raise RuntimeError("boom")

    report = run_live("q", generate_fn=generate_fn, evaluate_fn=_prefer_b)
    assert report.n_scored == 0
    assert report.n_errors == 3
    assert report.prompt_b_win_rate is None
    assert all(duel.error and duel.error.startswith("generate_failed") for duel in report.duels)


def test_empty_question_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_live("  ", generate_fn=_generate, evaluate_fn=_prefer_b)


def test_custom_models_from_settings() -> None:
    settings = Settings(generator_models=("m1", "m2"))
    report = run_live(
        "q",
        generate_fn=_generate,
        evaluate_fn=_prefer_b,
        settings=settings,
    )
    assert report.n == 2
    assert report.generator_models == ("m1", "m2")


def test_persist_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "live.json"
    report = run_live(
        "cloudy sunscreen?",
        question_id="lq-05",
        generate_fn=_generate,
        evaluate_fn=_prefer_b,
    )
    persist_report(report, path)
    loaded = load_report(path)
    assert loaded["mode"] == "live"
    assert loaded["prompt_b_win_rate"] == 1.0
    assert loaded["question_id"] == "lq-05"
    assert len(loaded["duels"]) == 3


def test_format_report() -> None:
    text = format_report(
        run_live("q", generate_fn=_generate, evaluate_fn=_prefer_b)
    )
    assert "Prompt B win rate" in text
    assert "Cross-model agreement" in text


def test_cli_requires_question_or_sample() -> None:
    with pytest.raises(SystemExit):
        live_main([])


def test_cli_missing_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert live_main(["--question", "hello"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
