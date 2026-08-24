"""Load and validate the human-labeled calibration set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from judgetrust.models import CalibrationRow, Winner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "calibration_set.json"

FAILURE_MODES: frozenset[str] = frozenset(
    {
        "confidently_wrong",
        "verbosity_trap",
        "subtle_wrong",
        "filler_dodge",
        "outdated_guidance",
    }
)
ALLOWED_WINNERS: frozenset[str] = frozenset({"A", "B", "tie"})
MIN_ROWS = 30
MAX_ROWS = 40


def _require_str(row: dict[str, Any], key: str, row_id: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {row_id!r}: {key} must be a non-empty string")
    return value.strip()


def _parse_row(raw: Any, index: int) -> CalibrationRow:
    if not isinstance(raw, dict):
        raise ValueError(f"rows[{index}] must be an object")
    row_id = raw.get("id")
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError(f"rows[{index}]: id must be a non-empty string")
    row_id = row_id.strip()

    winner = raw.get("human_winner")
    if winner not in ALLOWED_WINNERS:
        raise ValueError(f"row {row_id!r}: human_winner must be A, B, or tie")

    failure_mode = raw.get("failure_mode")
    if failure_mode not in FAILURE_MODES:
        raise ValueError(f"row {row_id!r}: unknown failure_mode {failure_mode!r}")

    return CalibrationRow(
        id=row_id,
        question=_require_str(raw, "question", row_id),
        answer_a=_require_str(raw, "answer_A", row_id)
        if "answer_A" in raw
        else _require_str(raw, "answer_a", row_id),
        answer_b=_require_str(raw, "answer_B", row_id)
        if "answer_B" in raw
        else _require_str(raw, "answer_b", row_id),
        human_winner=winner,  # type: ignore[arg-type]
        failure_mode=str(failure_mode),
    )


def validate_rows(rows: list[CalibrationRow]) -> None:
    """Enforce size, unique ids, and coverage of every failure mode."""

    if not MIN_ROWS <= len(rows) <= MAX_ROWS:
        raise ValueError(
            f"calibration set must have {MIN_ROWS}–{MAX_ROWS} rows; got {len(rows)}"
        )
    ids = [row.id for row in rows]
    duplicates = {item for item in ids if ids.count(item) > 1}
    if duplicates:
        raise ValueError(f"duplicate calibration ids: {sorted(duplicates)}")
    present = {row.failure_mode for row in rows}
    missing = FAILURE_MODES - present
    if missing:
        raise ValueError(f"missing failure modes: {sorted(missing)}")
    winners: set[Winner] = {row.human_winner for row in rows}
    if winners <= {"A"} or winners <= {"B"}:
        raise ValueError("human_winner must not be the same label on every row")


def load_calibration_set(path: Path | None = None) -> list[CalibrationRow]:
    """Read and validate ``data/calibration_set.json``."""

    dataset_path = path or DEFAULT_DATASET_PATH
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"calibration set not found: {dataset_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {dataset_path}: {exc}") from exc

    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError("calibration set must be an object with a 'rows' array")
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list):
        raise ValueError("calibration set 'rows' must be an array")

    rows = [_parse_row(item, index) for index, item in enumerate(raw_rows)]
    validate_rows(rows)
    return rows
