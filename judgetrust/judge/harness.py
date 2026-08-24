"""Both-orderings comparison harness. The only public comparison API."""

from __future__ import annotations

from collections.abc import Callable

from judgetrust.judge.chain import Judge
from judgetrust.logging_setup import get_logger
from judgetrust.models import (
    Ordering,
    OrderingRun,
    PairwiseResult,
    PresentedJudgment,
    PresentedWinner,
    Verdict,
    Winner,
)

logger = get_logger("judge.harness")

EvaluateFn = Callable[[str, str, str], PresentedJudgment]


def map_presented_winner(presented: PresentedWinner, order: Ordering) -> Winner:
    """Map a presentation-space winner (1/2/tie) back to original A/B labels."""

    if presented == "tie":
        return "tie"
    if order is Ordering.A_FIRST:
        return "A" if presented == "1" else "B"
    return "B" if presented == "1" else "A"


def resolve_pairwise(
    a_first_winner: Winner | None,
    b_first_winner: Winner | None,
) -> tuple[Winner, bool, bool]:
    """Combine two mapped winners into (final_winner, stable, position_bias).

    A decisive winner requires both orderings to succeed and agree. Disagreement
    is a position-bias hit: final winner is tie, stable is False. A missing
    verdict is not position bias — the comparison is simply non-decisive.
    """

    if a_first_winner is None or b_first_winner is None:
        return "tie", False, False
    if a_first_winner == b_first_winner:
        return a_first_winner, True, False
    return "tie", False, True


def _run_to_ordering(
    order: Ordering,
    judgment: PresentedJudgment,
) -> OrderingRun:
    if judgment.winner is None:
        return OrderingRun(
            order=order,
            raw_output=judgment.raw_output,
            verdict=None,
            error=judgment.error or "missing_winner",
        )
    mapped = map_presented_winner(judgment.winner, order)
    return OrderingRun(
        order=order,
        raw_output=judgment.raw_output,
        verdict=Verdict(
            winner=mapped,
            reason=judgment.reason,
            confidence=judgment.confidence,
            rubric=judgment.rubric,
        ),
        error=None,
    )


def compare_both_orderings(
    question: str,
    answer_a: str,
    answer_b: str,
    *,
    evaluate_fn: EvaluateFn | None = None,
    judge: Judge | None = None,
) -> PairwiseResult:
    """Compare answer A vs B by running the judge in both presentation orders.

    Later phases must call this function rather than the raw judge chain.
    Inject ``evaluate_fn`` in tests to avoid live API calls.
    """

    if evaluate_fn is None:
        bound_judge = judge or Judge()
        evaluate_fn = bound_judge.evaluate_presented

    a_first_judgment = evaluate_fn(question, answer_a, answer_b)
    b_first_judgment = evaluate_fn(question, answer_b, answer_a)

    run_a_first = _run_to_ordering(Ordering.A_FIRST, a_first_judgment)
    run_b_first = _run_to_ordering(Ordering.B_FIRST, b_first_judgment)

    a_winner = run_a_first.verdict.winner if run_a_first.verdict else None
    b_winner = run_b_first.verdict.winner if run_b_first.verdict else None
    final_winner, stable, position_bias = resolve_pairwise(a_winner, b_winner)

    logger.info(
        "pairwise_result stable=%s position_bias=%s final_winner=%s "
        "a_first=%s b_first=%s question_chars=%s",
        stable,
        position_bias,
        final_winner,
        a_winner,
        b_winner,
        len(question),
    )
    return PairwiseResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        run_a_first=run_a_first,
        run_b_first=run_b_first,
        stable=stable,
        final_winner=final_winner,
        position_bias=position_bias,
    )
