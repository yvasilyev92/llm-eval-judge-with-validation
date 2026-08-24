"""Load unlabeled sample questions for live mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from judgetrust.models import LiveQuestion

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS_PATH = REPO_ROOT / "data" / "live_questions.json"

MIN_QUESTIONS = 5
MAX_QUESTIONS = 20


def load_live_questions(path: Path | None = None) -> list[LiveQuestion]:
    """Read and validate ``data/live_questions.json``."""

    questions_path = path or DEFAULT_QUESTIONS_PATH
    try:
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"live questions not found: {questions_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {questions_path}: {exc}") from exc

    if not isinstance(payload, dict) or "questions" not in payload:
        raise ValueError("live questions file must be an object with a 'questions' array")
    raw = payload["questions"]
    if not isinstance(raw, list):
        raise ValueError("'questions' must be an array")

    questions = [_parse_question(item, index) for index, item in enumerate(raw)]
    if not MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(
            f"live questions must have {MIN_QUESTIONS}–{MAX_QUESTIONS} items; "
            f"got {len(questions)}"
        )
    ids = [item.id for item in questions]
    duplicates = {item for item in ids if ids.count(item) > 1}
    if duplicates:
        raise ValueError(f"duplicate live question ids: {sorted(duplicates)}")
    return questions


def get_live_question(question_id: str, path: Path | None = None) -> LiveQuestion:
    """Return one sample question by id."""

    for item in load_live_questions(path):
        if item.id == question_id:
            return item
    raise KeyError(f"unknown live question id: {question_id}")


def _parse_question(raw: Any, index: int) -> LiveQuestion:
    if not isinstance(raw, dict):
        raise ValueError(f"questions[{index}] must be an object")
    question_id = raw.get("id")
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError(f"questions[{index}]: id must be a non-empty string")
    text = raw.get("question")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"question {question_id!r}: question must be a non-empty string")
    return LiveQuestion(id=question_id.strip(), question=text.strip())
