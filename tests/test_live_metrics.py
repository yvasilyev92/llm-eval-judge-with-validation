"""Live aggregate math — no API."""

from __future__ import annotations

from judgetrust.live.metrics import (
    cross_model_agreement,
    position_consistency,
    prompt_b_win_rate,
)


def test_prompt_b_win_rate() -> None:
    assert prompt_b_win_rate(["B", "B", "B"]) == 1.0
    assert prompt_b_win_rate(["B", "A", "B"]) == 2 / 3
    assert prompt_b_win_rate(["A", "tie", "A"]) == 0.0
    assert prompt_b_win_rate([]) is None


def test_cross_model_agreement() -> None:
    assert cross_model_agreement(["B", "B", "B"]) == 1.0
    assert cross_model_agreement(["B", "B", "A"]) == 2 / 3
    assert cross_model_agreement(["A", "B", "tie"]) == 1 / 3
    assert cross_model_agreement([]) is None


def test_position_consistency() -> None:
    assert position_consistency([True, True, False]) == 2 / 3
    assert position_consistency([]) == 0.0
