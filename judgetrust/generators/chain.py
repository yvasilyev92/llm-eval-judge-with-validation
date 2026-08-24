"""LCEL answer generators for prompt A vs prompt B."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from judgetrust.config import Settings, get_settings
from judgetrust.generators.prompts import GENERATOR_PREFACE, GENERATOR_PROMPT
from judgetrust.llm import invoke_with_backoff, make_chat_model
from judgetrust.logging_setup import get_logger

logger = get_logger("generators")

GenerateFn = Callable[[str, str, str], str]


class Generator:
    """Generate one answer for a question under a task prompt and model."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        chat_model: ChatOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model
        self._chat_model = chat_model

    def generate(self, question: str, task_prompt: str, model: str) -> str:
        """Return generated text. Raises on invoke failure (runner records it)."""

        llm = self._chat_model or make_chat_model(
            model,
            temperature=self.settings.generator_temperature,
            settings=self.settings,
        )
        chain = GENERATOR_PROMPT | llm | StrOutputParser()
        text = invoke_with_backoff(
            chain,
            {
                "preface": GENERATOR_PREFACE,
                "task_prompt": task_prompt,
                "question": question,
            },
            settings=self.settings,
        )
        logger.info(
            "generated model=%s question_chars=%s answer_chars=%s",
            model,
            len(question),
            len(text),
        )
        return text


def generate_answer(
    question: str,
    task_prompt: str,
    *,
    model: str,
    generator: Generator | None = None,
    generate_fn: GenerateFn | None = None,
) -> str:
    """Generate one answer. Prefer ``generate_fn`` in tests."""

    if generate_fn is not None:
        return generate_fn(question, task_prompt, model)
    bound = generator or Generator(model=model)
    return bound.generate(question, task_prompt, model)
