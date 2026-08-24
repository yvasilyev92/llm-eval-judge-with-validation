"""Calibration metrics: Cohen's kappa, agreement, position-consistency. No LLM calls."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

from sklearn.metrics import cohen_kappa_score

from judgetrust.config import KAPPA_BANDS
from judgetrust.models import Winner

KAPPA_LABELS: list[Winner] = ["A", "B", "tie"]
RAW_AGREEMENT_NOTE = (
    "Kappa corrects for chance; raw agreement overstates reliability."
)


def kappa_band(
    value: float,
    bands: Sequence[tuple[float, str]] = KAPPA_BANDS,
) -> str:
    """Map a kappa value onto a Landis & Koch-style band. Negative → poor."""

    for floor, label in bands:
        if value >= floor:
            return label
    return "poor"


def raw_agreement(human: Sequence[Winner], judge: Sequence[Winner]) -> float:
    """Exact-match rate. Empty input is 0.0."""

    if len(human) != len(judge):
        raise ValueError("human and judge label lists must be the same length")
    if not human:
        return 0.0
    matches = sum(left == right for left, right in zip(human, judge, strict=True))
    return matches / len(human)


def cohen_kappa(human: Sequence[Winner], judge: Sequence[Winner]) -> float | None:
    """Cohen's kappa vs humans. None if there are no scored rows.

    When sklearn returns NaN (single-class, perfect agreement), treat as 1.0.
    """

    if len(human) != len(judge):
        raise ValueError("human and judge label lists must be the same length")
    if not human:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        score = float(
            cohen_kappa_score(list(human), list(judge), labels=KAPPA_LABELS)
        )
    if math.isnan(score):
        return 1.0 if raw_agreement(human, judge) == 1.0 else 0.0
    return score


def position_consistency(stable_flags: Sequence[bool]) -> float:
    """Share of rows whose both-orderings verdict was stable."""

    if not stable_flags:
        return 0.0
    return sum(1 for flag in stable_flags if flag) / len(stable_flags)
