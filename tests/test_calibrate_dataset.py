"""Calibration dataset schema, size, and failure-mode coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judgetrust.calibrate.dataset import (
    FAILURE_MODES,
    load_calibration_set,
    validate_rows,
)
from judgetrust.models import CalibrationRow


def test_starter_set_loads() -> None:
    rows = load_calibration_set()
    assert 30 <= len(rows) <= 40
    assert {row.failure_mode for row in rows} == FAILURE_MODES
    assert len({row.id for row in rows}) == len(rows)
    winners = {row.human_winner for row in rows}
    assert winners >= {"A", "B"}
    assert "tie" in winners


def test_each_failure_mode_has_several_rows() -> None:
    rows = load_calibration_set()
    for mode in FAILURE_MODES:
        count = sum(1 for row in rows if row.failure_mode == mode)
        assert count >= 5, mode


def test_meta_disclaimer_present() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "calibration_set.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    disclaimer = payload["meta"]["disclaimer"].lower()
    assert "not medical advice" in disclaimer
    assert "review" in disclaimer


def test_validate_rejects_duplicate_ids() -> None:
    row = CalibrationRow(
        id="x",
        question="q",
        answer_a="a",
        answer_b="b",
        human_winner="A",
        failure_mode="confidently_wrong",
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows([row, row] + _filler_rows(30))


def test_validate_rejects_wrong_size() -> None:
    with pytest.raises(ValueError, match="30–40"):
        validate_rows(_filler_rows(2))


def test_load_rejects_bad_winner(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "meta": {},
                "rows": [
                    {
                        "id": "bad-1",
                        "question": "q",
                        "answer_A": "a",
                        "answer_B": "b",
                        "human_winner": "C",
                        "failure_mode": "confidently_wrong",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="human_winner"):
        load_calibration_set(path)


def _filler_rows(n: int) -> list[CalibrationRow]:
    modes = list(FAILURE_MODES)
    rows: list[CalibrationRow] = []
    for i in range(n):
        rows.append(
            CalibrationRow(
                id=f"fill-{i}",
                question="q",
                answer_a="a",
                answer_b="b",
                human_winner="A" if i % 2 == 0 else "B",
                failure_mode=modes[i % len(modes)],
            )
        )
    return rows
