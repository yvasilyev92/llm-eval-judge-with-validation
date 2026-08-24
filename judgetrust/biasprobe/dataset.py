"""Load and validate the rigged length-bias probe set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from judgetrust.models import BiasProbeRow

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "bias_probe_set.json"

MIN_ROWS = 8
MAX_ROWS = 15
ALLOWED_SIDES: frozenset[str] = frozenset({"A", "B"})


def load_bias_probe_set(path: Path | None = None) -> list[BiasProbeRow]:
    """Read and validate ``data/bias_probe_set.json``."""

    dataset_path = path or DEFAULT_DATASET_PATH
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"bias probe set not found: {dataset_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {dataset_path}: {exc}") from exc

    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError("bias probe set must be an object with a 'rows' array")
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list):
        raise ValueError("bias probe set 'rows' must be an array")

    rows = [_parse_row(item, index) for index, item in enumerate(raw_rows)]
    _validate_rows(rows)
    return rows


def _parse_row(raw: Any, index: int) -> BiasProbeRow:
    if not isinstance(raw, dict):
        raise ValueError(f"rows[{index}] must be an object")
    row_id = raw.get("id")
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError(f"rows[{index}]: id must be a non-empty string")
    row_id = row_id.strip()
    longer_worse = raw.get("longer_worse")
    if longer_worse not in ALLOWED_SIDES:
        raise ValueError(f"row {row_id!r}: longer_worse must be A or B")
    question = raw.get("question")
    answer_a = raw.get("answer_A", raw.get("answer_a"))
    answer_b = raw.get("answer_B", raw.get("answer_b"))
    for key, value in (
        ("question", question),
        ("answer_A", answer_a),
        ("answer_B", answer_b),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"row {row_id!r}: {key} must be a non-empty string")
    return BiasProbeRow(
        id=row_id,
        question=str(question).strip(),
        answer_a=str(answer_a).strip(),
        answer_b=str(answer_b).strip(),
        longer_worse=longer_worse,  # type: ignore[arg-type]
    )


def _validate_rows(rows: list[BiasProbeRow]) -> None:
    if not MIN_ROWS <= len(rows) <= MAX_ROWS:
        raise ValueError(
            f"bias probe set must have {MIN_ROWS}–{MAX_ROWS} rows; got {len(rows)}"
        )
    ids = [row.id for row in rows]
    duplicates = {item for item in ids if ids.count(item) > 1}
    if duplicates:
        raise ValueError(f"duplicate bias probe ids: {sorted(duplicates)}")
    sides: set[Literal["A", "B"]] = {row.longer_worse for row in rows}
    if sides != {"A", "B"}:
        raise ValueError("longer_worse must include both A and B across the set")
    for row in rows:
        longer = row.answer_a if row.longer_worse == "A" else row.answer_b
        shorter = row.answer_b if row.longer_worse == "A" else row.answer_a
        if len(longer) <= len(shorter):
            raise ValueError(
                f"row {row.id!r}: longer_worse side is not actually longer "
                f"({len(longer)} vs {len(shorter)} chars)"
            )
