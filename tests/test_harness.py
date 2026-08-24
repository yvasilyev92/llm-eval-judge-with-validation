"""Flip-back mapping, stability rule, and both-orderings harness."""

from __future__ import annotations

from judgetrust.judge.harness import (
    compare_both_orderings,
    map_presented_winner,
    resolve_pairwise,
)
from judgetrust.models import Ordering, PresentedJudgment


def _judgment(winner: str | None, *, error: str | None = None) -> PresentedJudgment:
    return PresentedJudgment(
        winner=winner,  # type: ignore[arg-type]
        reason="test",
        confidence=0.7,
        rubric=None,
        raw_output='{"winner": "%s"}' % (winner or ""),
        error=error,
    )


def test_map_a_first_passthrough() -> None:
    assert map_presented_winner("1", Ordering.A_FIRST) == "A"
    assert map_presented_winner("2", Ordering.A_FIRST) == "B"
    assert map_presented_winner("tie", Ordering.A_FIRST) == "tie"


def test_map_b_first_flips() -> None:
    assert map_presented_winner("1", Ordering.B_FIRST) == "B"
    assert map_presented_winner("2", Ordering.B_FIRST) == "A"
    assert map_presented_winner("tie", Ordering.B_FIRST) == "tie"


def test_resolve_stable_agreement() -> None:
    winner, stable, bias = resolve_pairwise("A", "A")
    assert (winner, stable, bias) == ("A", True, False)
    winner, stable, bias = resolve_pairwise("tie", "tie")
    assert (winner, stable, bias) == ("tie", True, False)


def test_resolve_position_bias() -> None:
    winner, stable, bias = resolve_pairwise("A", "B")
    assert winner == "tie"
    assert stable is False
    assert bias is True


def test_resolve_missing_verdict_is_not_position_bias() -> None:
    winner, stable, bias = resolve_pairwise("A", None)
    assert winner == "tie"
    assert stable is False
    assert bias is False
    winner, stable, bias = resolve_pairwise(None, None)
    assert (winner, stable, bias) == ("tie", False, False)


def test_compare_prefers_better_answer_both_orderings() -> None:
    def evaluate_fn(_question: str, answer_1: str, answer_2: str) -> PresentedJudgment:
        if answer_1 == "better":
            return _judgment("1")
        if answer_2 == "better":
            return _judgment("2")
        return _judgment("tie")

    result = compare_both_orderings(
        "Is acetaminophen safe with alcohol?",
        "better",
        "worse",
        evaluate_fn=evaluate_fn,
    )
    assert result.final_winner == "A"
    assert result.stable is True
    assert result.position_bias is False
    assert result.run_a_first.verdict is not None
    assert result.run_a_first.verdict.winner == "A"
    assert result.run_b_first.verdict is not None
    assert result.run_b_first.verdict.winner == "A"


def test_compare_detects_first_position_bias() -> None:
    def evaluate_fn(_question: str, _answer_1: str, _answer_2: str) -> PresentedJudgment:
        return _judgment("1")

    result = compare_both_orderings("q", "answer A", "answer B", evaluate_fn=evaluate_fn)
    assert result.position_bias is True
    assert result.stable is False
    assert result.final_winner == "tie"
    assert result.run_a_first.verdict is not None
    assert result.run_a_first.verdict.winner == "A"
    assert result.run_b_first.verdict is not None
    assert result.run_b_first.verdict.winner == "B"


def test_compare_malformed_output_is_non_decisive() -> None:
    def evaluate_fn(_question: str, _a1: str, _a2: str) -> PresentedJudgment:
        return _judgment(None, error="malformed_or_missing_json")

    result = compare_both_orderings("q", "a", "b", evaluate_fn=evaluate_fn)
    assert result.final_winner == "tie"
    assert result.stable is False
    assert result.position_bias is False
    assert result.run_a_first.verdict is None
    assert result.run_a_first.error == "malformed_or_missing_json"


def test_compare_b_wins_when_stable() -> None:
    def evaluate_fn(_question: str, answer_1: str, answer_2: str) -> PresentedJudgment:
        preferred = "safety-tuned"
        if answer_1 == preferred:
            return _judgment("1")
        if answer_2 == preferred:
            return _judgment("2")
        return _judgment("tie")

    result = compare_both_orderings("q", "plain", "safety-tuned", evaluate_fn=evaluate_fn)
    assert result.final_winner == "B"
    assert result.stable is True
    assert result.position_bias is False
