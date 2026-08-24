"""Chat model factory and invoke helper with timeout + exponential backoff."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from judgetrust.config import Settings, get_settings
from judgetrust.logging_setup import get_logger

logger = get_logger("llm")

_DOTENV_LOADED = False


def load_env() -> None:
    """Load `.env` once. Does not log file contents or key values."""

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv()
    _DOTENV_LOADED = True


def make_chat_model(
    model: str,
    *,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> ChatOpenAI:
    """Build a ChatOpenAI client with timeout and provider-level retries."""

    load_env()
    cfg = settings or get_settings()
    return ChatOpenAI(
        model=model,
        temperature=cfg.judge_temperature if temperature is None else temperature,
        timeout=cfg.llm_timeout_seconds,
        max_retries=cfg.llm_max_retries,
    )


def invoke_with_backoff(
    runnable: Runnable[Mapping[str, Any], str],
    inputs: Mapping[str, Any],
    *,
    settings: Settings | None = None,
) -> str:
    """Invoke an LCEL chain, retrying transient failures with exponential backoff.

    ChatOpenAI already retries HTTP errors. This outer loop covers remaining
    transport/runtime failures. The last exception is re-raised.
    """

    cfg = settings or get_settings()
    attempts = cfg.llm_max_retries + 1
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = runnable.invoke(dict(inputs))
            if not isinstance(result, str):
                result = str(result)
            return result
        except Exception as exc:  # noqa: BLE001 — last line of defense before judge records an error
            last_error = exc
            if attempt >= attempts - 1:
                break
            delay = cfg.llm_retry_base_delay_seconds * (2**attempt)
            logger.warning(
                "llm_invoke_retry attempt=%s/%s delay_s=%.1f error_type=%s",
                attempt + 1,
                attempts,
                delay,
                type(exc).__name__,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def missing_api_key() -> bool:
    """True when OPENAI_API_KEY is unset or empty."""

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return not key
