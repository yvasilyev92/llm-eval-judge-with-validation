"""Judge prompt template exposes the expected LCEL input variables."""

from __future__ import annotations

from judgetrust.judge.prompts import JUDGE_PROMPT
from judgetrust.models import Mode


def test_judge_prompt_variables() -> None:
    assert set(JUDGE_PROMPT.input_variables) == {"question", "answer_1", "answer_2"}


def test_modes_are_distinct() -> None:
    assert Mode.CALIBRATION is not Mode.LIVE
    assert {mode.value for mode in Mode} == {"calibration", "live", "bias_probe"}
