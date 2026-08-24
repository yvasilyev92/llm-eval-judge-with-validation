"""Live-mode aggregates: prompt-B win rate, cross-model agreement, consistency."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from judgetrust.models import Winner


def prompt_b_win_rate(winners: Sequence[Winner]) -> float | None:
    """Share of scored models whose final winner is B. None if none scored."""

    if not winners:
        return None
    return sum(winner == "B" for winner in winners) / len(winners)


def cross_model_agreement(winners: Sequence[Winner]) -> float | None:
    """Largest winner-cluster size / n_scored. None if none scored."""

    if not winners:
        return None
    counts = Counter(winners)
    return max(counts.values()) / len(winners)


def position_consistency(stable_flags: Sequence[bool]) -> float:
    """Share of duels whose both-orderings verdict was stable."""

    if not stable_flags:
        return 0.0
    return sum(1 for flag in stable_flags if flag) / len(stable_flags)
