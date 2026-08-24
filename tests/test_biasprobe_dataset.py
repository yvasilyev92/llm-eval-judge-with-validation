"""Bias probe dataset schema and length-rig checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judgetrust.biasprobe.dataset import load_bias_probe_set


def test_probe_set_loads() -> None:
    rows = load_bias_probe_set()
    assert 8 <= len(rows) <= 15
    assert {row.longer_worse for row in rows} == {"A", "B"}
    assert len({row.id for row in rows}) == len(rows)
    for row in rows:
        longer = row.answer_a if row.longer_worse == "A" else row.answer_b
        shorter = row.answer_b if row.longer_worse == "A" else row.answer_a
        assert len(longer) > len(shorter)


def test_meta_disclaimer_present() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "bias_probe_set.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "not medical advice" in payload["meta"]["disclaimer"].lower()


def test_rejects_when_marked_longer_is_shorter(tmp_path: Path) -> None:
    rows = []
    for i in range(8):
        side = "A" if i < 4 else "B"
        long_text = "L" * 80
        short_text = "s" * 10
        if i == 0:
            long_text, short_text = short_text, long_text
        answer_a = long_text if side == "A" else short_text
        answer_b = short_text if side == "A" else long_text
        rows.append(
            {
                "id": f"r-{i}",
                "question": f"q-{i}",
                "answer_A": answer_a,
                "answer_B": answer_b,
                "longer_worse": side,
            }
        )
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="not actually longer"):
        load_bias_probe_set(path)


def test_rejects_short_file(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": "bad",
                        "question": "q",
                        "answer_A": "tiny",
                        "answer_B": "this is the actually longer correct side",
                        "longer_worse": "A",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="8–15"):
        load_bias_probe_set(path)
