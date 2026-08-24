"""Kappa, band, agreement, and consistency — no API calls."""

from __future__ import annotations

import pytest
from sklearn.metrics import cohen_kappa_score

from judgetrust.calibrate.metrics import (
    KAPPA_LABELS,
    cohen_kappa,
    kappa_band,
    position_consistency,
    raw_agreement,
)


def test_kappa_band_cutoffs() -> None:
    assert kappa_band(0.81) == "almost-perfect"
    assert kappa_band(0.80) == "substantial"
    assert kappa_band(0.61) == "substantial"
    assert kappa_band(0.60) == "moderate"
    assert kappa_band(0.41) == "moderate"
    assert kappa_band(0.21) == "fair"
    assert kappa_band(0.00) == "poor"
    assert kappa_band(-0.20) == "poor"


def test_raw_agreement() -> None:
    assert raw_agreement(["A", "B", "tie"], ["A", "B", "tie"]) == 1.0
    assert raw_agreement(["A", "B"], ["A", "A"]) == 0.5
    assert raw_agreement([], []) == 0.0


def test_cohen_kappa_matches_sklearn() -> None:
    human: list = ["A", "A", "B", "B", "tie"]
    judge: list = ["A", "B", "B", "B", "tie"]
    expected = float(cohen_kappa_score(human, judge, labels=KAPPA_LABELS))
    assert cohen_kappa(human, judge) == pytest.approx(expected)


def test_cohen_kappa_empty_is_none() -> None:
    assert cohen_kappa([], []) is None


def test_cohen_kappa_perfect_single_class() -> None:
    labels: list = ["A", "A", "A"]
    assert cohen_kappa(labels, labels) == 1.0


def test_position_consistency() -> None:
    assert position_consistency([True, True, False]) == 2 / 3
    assert position_consistency([]) == 0.0
    assert position_consistency([True, True]) == 1.0
