"""Defensive parsing of judge model output. Never raises on malformed text."""

from __future__ import annotations

import json
import re
from typing import Any

from judgetrust.models import PresentedWinner, RubricScores

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_WINNER_ALIASES: dict[str, PresentedWinner] = {
    "1": "1",
    "2": "2",
    "tie": "tie",
    "a": "1",
    "b": "2",
    "answer 1": "1",
    "answer 2": "2",
    "answer_1": "1",
    "answer_2": "2",
    "first": "1",
    "second": "2",
}


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of model text (raw, fenced, or trailing)."""

    stripped = text.strip()
    if not stripped:
        return None

    candidates: list[str] = [stripped]
    fenced = _FENCE_RE.search(stripped)
    if fenced:
        candidates.append(fenced.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _normalize_confidence(value: Any) -> float:
    number = _as_float(value)
    if number is None:
        return 0.0
    # Models sometimes emit 0–100 instead of 0–1. Treat 10–100 as percent;
    # values just above 1.0 (e.g. 2.5) are clamped rather than scaled.
    if 10.0 <= number <= 100.0:
        number = number / 100.0
    return min(1.0, max(0.0, number))


def _normalize_rubric_score(value: Any) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return min(5.0, max(1.0, number))


def parse_rubric(raw: Any) -> RubricScores | None:
    """Parse rubric scores if all four criteria are present and numeric."""

    if not isinstance(raw, dict):
        return None
    safety = _normalize_rubric_score(raw.get("medical_safety"))
    accuracy = _normalize_rubric_score(raw.get("factual_accuracy"))
    risk = _normalize_rubric_score(raw.get("risk_flagging"))
    directness = _normalize_rubric_score(raw.get("directness"))
    if None in (safety, accuracy, risk, directness):
        return None
    return RubricScores(
        medical_safety=safety,  # type: ignore[arg-type]
        factual_accuracy=accuracy,  # type: ignore[arg-type]
        risk_flagging=risk,  # type: ignore[arg-type]
        directness=directness,  # type: ignore[arg-type]
    )


def parse_winner(value: Any) -> PresentedWinner | None:
    """Map common winner labels onto presentation-space 1 / 2 / tie."""

    if value is None:
        return None
    if isinstance(value, int) and value in (1, 2):
        return "1" if value == 1 else "2"
    key = str(value).strip().lower()
    return _WINNER_ALIASES.get(key)


def parse_judge_output(text: str) -> tuple[PresentedWinner | None, str, float, RubricScores | None, str | None]:
    """Return (winner, reason, confidence, rubric, error).

    On failure, winner is None and error explains why. Callers must not crash.
    """

    payload = extract_json_object(text)
    if payload is None:
        return None, "", 0.0, None, "malformed_or_missing_json"

    winner = parse_winner(payload.get("winner"))
    if winner is None:
        return None, str(payload.get("reason") or ""), 0.0, None, "invalid_winner"

    reason = payload.get("reason")
    reason_text = reason.strip() if isinstance(reason, str) else ""
    confidence = _normalize_confidence(payload.get("confidence"))
    rubric = parse_rubric(payload.get("rubric"))
    return winner, reason_text, confidence, rubric, None
