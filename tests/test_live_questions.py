"""Live sample questions load."""

from __future__ import annotations

import pytest

from judgetrust.live.questions import get_live_question, load_live_questions


def test_live_questions_load() -> None:
    questions = load_live_questions()
    assert 5 <= len(questions) <= 20
    assert len({item.id for item in questions}) == len(questions)
    assert all(item.question.strip() for item in questions)


def test_get_live_question() -> None:
    sample = get_live_question("lq-01")
    assert "acetaminophen" in sample.question.lower()
    with pytest.raises(KeyError, match="unknown"):
        get_live_question("does-not-exist")
