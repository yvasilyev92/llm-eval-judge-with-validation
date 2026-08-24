"""LCEL pairwise judge chain. One presented ordering per call."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from judgetrust.config import Settings, get_settings
from judgetrust.judge.parse import parse_judge_output
from judgetrust.judge.prompts import JUDGE_PROMPT
from judgetrust.llm import invoke_with_backoff, make_chat_model
from judgetrust.logging_setup import get_logger
from judgetrust.models import PresentedJudgment

logger = get_logger("judge.chain")


class Judge:
    """Pairwise LLM judge. Evaluates Answer 1 vs Answer 2 in presentation order."""

    def __init__(
        self,
        *,
        model: str | None = None,
        chat_model: ChatOpenAI | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model or self.settings.judge_models[0]
        self.llm = chat_model or make_chat_model(
            self.model_name,
            temperature=self.settings.judge_temperature,
            settings=self.settings,
        )
        self.chain: Runnable[dict[str, str], str] = (
            JUDGE_PROMPT | self.llm | StrOutputParser()
        )

    def evaluate_presented(
        self,
        question: str,
        answer_1: str,
        answer_2: str,
    ) -> PresentedJudgment:
        """Run the judge once on a presented (answer_1, answer_2) pair.

        Malformed output and API errors become a recorded error, not an exception.
        """

        try:
            raw = invoke_with_backoff(
                self.chain,
                {
                    "question": question,
                    "answer_1": answer_1,
                    "answer_2": answer_2,
                },
                settings=self.settings,
            )
        except Exception as exc:  # noqa: BLE001 — recorded as a non-decisive error
            logger.warning(
                "judge_invoke_failed model=%s error_type=%s",
                self.model_name,
                type(exc).__name__,
            )
            return PresentedJudgment(
                winner=None,
                reason="",
                confidence=0.0,
                rubric=None,
                raw_output="",
                error=f"invoke_failed:{type(exc).__name__}",
            )

        winner, reason, confidence, rubric, error = parse_judge_output(raw)
        logger.info(
            "judge_presented model=%s winner=%s confidence=%.2f error=%s "
            "question_chars=%s answer_1_chars=%s answer_2_chars=%s",
            self.model_name,
            winner,
            confidence,
            error,
            len(question),
            len(answer_1),
            len(answer_2),
        )
        return PresentedJudgment(
            winner=winner,
            reason=reason,
            confidence=confidence,
            rubric=rubric,
            raw_output=raw,
            error=error,
        )
