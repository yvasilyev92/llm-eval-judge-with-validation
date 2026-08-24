"""Panel majority vote, abstain, dissent, and both-orderings resolve — no API."""

from __future__ import annotations

from judgetrust.judge.panel import (
    compare_panel,
    dissent_rate,
    majority_winner,
    resolve_panel,
    vote_from_pairwise,
)
from judgetrust.models import PresentedJudgment


def _judgment(winner: str | None, *, error: str | None = None) -> PresentedJudgment:
    return PresentedJudgment(
        winner=winner,  # type: ignore[arg-type]
        reason="test",
        confidence=0.7,
        rubric=None,
        raw_output="",
        error=error,
    )


def _prefer(text: str):
    def evaluate_fn(_question: str, answer_1: str, answer_2: str) -> PresentedJudgment:
        if answer_1 == text:
            return _judgment("1")
        if answer_2 == text:
            return _judgment("2")
        return _judgment("tie")

    return evaluate_fn


def test_majority_winner_two_of_three() -> None:
    assert majority_winner(["A", "A", "B"]) == "A"
    assert majority_winner(["B", "B", "tie"]) == "B"
    assert majority_winner(["tie", "tie", "A"]) == "tie"


def test_majority_winner_split_and_short() -> None:
    assert majority_winner(["A", "B", "tie"]) is None
    assert majority_winner(["A"]) is None
    assert majority_winner([]) is None
    assert majority_winner(["A", "A"]) == "A"
    assert majority_winner(["A", "B"]) is None


def test_panel_two_one_for_b() -> None:
    result = compare_panel(
        "q",
        "plain",
        "safety",
        models=("j1", "j2", "j3"),
        evaluate_fns={
            "j1": _prefer("safety"),
            "j2": _prefer("safety"),
            "j3": _prefer("plain"),
        },
    )
    assert result.final_winner == "B"
    assert result.dissent is True
    assert result.n_votes == 3
    assert result.error is None
    assert result.stable is True
    assert result.position_bias is False
    by_model = {vote.model: vote.winner for vote in result.votes}
    assert by_model == {"j1": "B", "j2": "B", "j3": "A"}


def test_panel_three_way_split_is_tie_with_dissent() -> None:
    result = compare_panel(
        "q",
        "plain",
        "safety",
        models=("j1", "j2", "j3"),
        evaluate_fns={
            "j1": _prefer("plain"),
            "j2": _prefer("safety"),
            "j3": lambda _q, _a1, _a2: _judgment("tie"),
        },
    )
    assert result.final_winner == "tie"
    assert result.dissent is True
    assert result.error is None


def test_panel_unanimous_no_dissent() -> None:
    result = compare_panel(
        "q",
        "plain",
        "safety",
        models=("j1", "j2", "j3"),
        evaluate_fn=_prefer("safety"),
    )
    assert result.final_winner == "B"
    assert result.dissent is False
    assert result.n_votes == 3


def test_one_abstain_two_agree() -> None:
    result = compare_panel(
        "q",
        "plain",
        "safety",
        models=("j1", "j2", "j3"),
        evaluate_fns={
            "j1": _prefer("safety"),
            "j2": _prefer("safety"),
            "j3": lambda _q, _a1, _a2: _judgment(None, error="malformed_or_missing_json"),
        },
    )
    assert result.final_winner == "B"
    assert result.n_votes == 2
    assert result.error is None
    assert result.votes[2].error == "malformed_or_missing_json"


def test_two_abstain_is_insufficient() -> None:
    def fail(_q: str, _a1: str, _a2: str) -> PresentedJudgment:
        return _judgment(None, error="malformed_or_missing_json")

    result = compare_panel(
        "q",
        "a",
        "b",
        models=("j1", "j2", "j3"),
        evaluate_fns={
            "j1": _prefer("a"),
            "j2": fail,
            "j3": fail,
        },
    )
    assert result.error == "insufficient_votes"
    assert result.final_winner == "tie"
    assert result.dissent is False
    assert result.n_votes == 1


def test_always_first_is_panel_position_bias() -> None:
    def evaluate_fn(_q: str, _a1: str, _a2: str) -> PresentedJudgment:
        return _judgment("1")

    result = compare_panel(
        "q",
        "a",
        "b",
        models=("j1", "j2", "j3"),
        evaluate_fn=evaluate_fn,
    )
    assert result.final_winner == "tie"
    assert result.position_bias is True
    assert result.stable is False
    assert result.dissent is False


def test_dissent_rate() -> None:
    assert dissent_rate([True, False, False, True]) == 0.5
    assert dissent_rate([]) is None


def test_vote_from_pairwise_abstains_on_malformed() -> None:
    from judgetrust.judge.harness import compare_both_orderings

    pairwise = compare_both_orderings(
        "q",
        "a",
        "b",
        evaluate_fn=lambda _q, _a1, _a2: _judgment(None, error="malformed_or_missing_json"),
    )
    vote = vote_from_pairwise("j1", pairwise)
    assert vote.winner is None
    assert vote.error == "malformed_or_missing_json"


def test_resolve_panel_dissent_flag() -> None:
    from judgetrust.judge.harness import compare_both_orderings

    a_win = compare_both_orderings("q", "plain", "safety", evaluate_fn=_prefer("plain"))
    b_win = compare_both_orderings("q", "plain", "safety", evaluate_fn=_prefer("safety"))
    votes = [
        vote_from_pairwise("j1", b_win),
        vote_from_pairwise("j2", b_win),
        vote_from_pairwise("j3", a_win),
    ]
    winner, stable, bias, dissent, error, n_votes = resolve_panel(votes, [b_win, b_win, a_win])
    assert (winner, stable, bias, dissent, error, n_votes) == ("B", True, False, True, None, 3)
