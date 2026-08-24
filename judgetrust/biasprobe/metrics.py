"""Bias-probe rates. Length-bias is the headline; position-bias is the flag."""

from __future__ import annotations

from collections.abc import Sequence


def length_bias_rate(hits: Sequence[bool]) -> float | None:
    """Share of scored rows where the judge picked the longer-but-worse answer."""

    if not hits:
        return None
    return sum(1 for hit in hits if hit) / len(hits)


def position_bias_rate(flags: Sequence[bool]) -> float:
    """Share of all rows with a position-bias hit."""

    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)
