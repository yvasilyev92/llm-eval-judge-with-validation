"""Generator prompt template variables."""

from __future__ import annotations

from judgetrust.generators.prompts import GENERATOR_PROMPT


def test_generator_prompt_variables() -> None:
    assert set(GENERATOR_PROMPT.input_variables) == {
        "preface",
        "task_prompt",
        "question",
    }
