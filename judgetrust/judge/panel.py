"""Majority-vote panel over both-orderings pairwise judges."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from judgetrust.config import Settings, get_settings
from judgetrust.judge.chain import Judge
from judgetrust.judge.harness import EvaluateFn, compare_both_orderings, resolve_pairwise
from judgetrust.logging_setup import get_logger
from judgetrust.models import (
    JudgeVote,
    Ordering,
    PairwiseResult,
    PanelResult,
    Winner,
)

logger = get_logger("judge.panel")


def majority_winner(winners: Sequence[Winner]) -> Winner | None:
    """Strict majority. None if fewer than 2 ballots or no label has more than n/2."""

    if len(winners) < 2:
        return None
    label, count = Counter(winners).most_common(1)[0]
    if count > len(winners) / 2:
        return label
    return None


def pairwise_error(pairwise: PairwiseResult) -> str | None:
    """Error if either ordering failed. Both must succeed for a vote."""

    if pairwise.run_a_first.verdict is not None and pairwise.run_b_first.verdict is not None:
        return None
    return (
        pairwise.run_a_first.error
        or pairwise.run_b_first.error
        or "ordering_failed"
    )


def vote_from_pairwise(model: str, pairwise: PairwiseResult) -> JudgeVote:
    """Map a both-orderings result onto a panel ballot. Abstain on any ordering error."""

    error = pairwise_error(pairwise)
    if error:
        return JudgeVote(
            model=model,
            winner=None,
            stable=False,
            position_bias=False,
            error=error,
        )
    return JudgeVote(
        model=model,
        winner=pairwise.final_winner,
        stable=pairwise.stable,
        position_bias=pairwise.position_bias,
        error=None,
    )


def _ordering_majority(
    pairwises: Sequence[PairwiseResult],
    order: Ordering,
) -> Winner | None:
    winners: list[Winner] = []
    for pairwise in pairwises:
        run = pairwise.run_a_first if order is Ordering.A_FIRST else pairwise.run_b_first
        if run.verdict is not None:
            winners.append(run.verdict.winner)
    return majority_winner(winners)


def resolve_panel(
    votes: Sequence[JudgeVote],
    pairwises: Sequence[PairwiseResult],
) -> tuple[Winner, bool, bool, bool, str | None, int]:
    """Combine per-judge votes into (winner, stable, position_bias, dissent, error, n_votes).

    Panel winner is a strict majority of successful ballots. Position-bias uses
    majority of A-first mapped winners vs majority of B-first mapped winners.
    """

    ballots = [vote.winner for vote in votes if vote.winner is not None and vote.error is None]
    n_votes = len(ballots)
    a_maj = _ordering_majority(pairwises, Ordering.A_FIRST)
    b_maj = _ordering_majority(pairwises, Ordering.B_FIRST)
    _, stable, position_bias = resolve_pairwise(a_maj, b_maj)

    if n_votes < 2:
        return "tie", stable, position_bias, False, "insufficient_votes", n_votes

    panel_winner = majority_winner(ballots) or "tie"
    dissent = any(
        vote.winner is not None and vote.error is None and vote.winner != panel_winner
        for vote in votes
    )
    return panel_winner, stable, position_bias, dissent, None, n_votes


def dissent_rate(flags: Sequence[bool]) -> float | None:
    """Share of scored comparisons where any voting judge ≠ panel winner."""

    if not flags:
        return None
    return sum(1 for flag in flags if flag) / len(flags)


def compare_panel(
    question: str,
    answer_a: str,
    answer_b: str,
    *,
    evaluate_fn: EvaluateFn | None = None,
    evaluate_fns: Mapping[str, EvaluateFn] | None = None,
    settings: Settings | None = None,
    models: Sequence[str] | None = None,
) -> PanelResult:
    """Run both-orderings on each panel member, then majority-vote.

    Pass ``evaluate_fn`` to use the same injected judge for every member (tests).
    Pass ``evaluate_fns`` to inject a distinct function per model.
    """

    cfg = settings or get_settings()
    judge_models = tuple(models or cfg.judge_models)
    pairwises: list[PairwiseResult] = []
    votes: list[JudgeVote] = []
    for model in judge_models:
        if evaluate_fns is not None:
            fn = evaluate_fns.get(model)
        else:
            fn = evaluate_fn
        judge = None if fn is not None else Judge(model=model, settings=cfg)
        pairwise = compare_both_orderings(
            question,
            answer_a,
            answer_b,
            evaluate_fn=fn,
            judge=judge,
        )
        pairwises.append(pairwise)
        votes.append(vote_from_pairwise(model, pairwise))

    final_winner, stable, position_bias, dissent, error, n_votes = resolve_panel(
        votes, pairwises
    )
    logger.info(
        "panel_result winner=%s stable=%s position_bias=%s dissent=%s "
        "n_votes=%s error=%s question_chars=%s",
        final_winner,
        stable,
        position_bias,
        dissent,
        n_votes,
        error,
        len(question),
    )
    return PanelResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        votes=tuple(votes),
        final_winner=final_winner,
        stable=stable,
        position_bias=position_bias,
        dissent=dissent,
        n_votes=n_votes,
        error=error,
    )
