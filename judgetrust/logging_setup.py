"""Structured logging for Judge Trust. Never emit secrets."""

from __future__ import annotations

import logging
import re

_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*\S+"
)
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")

_CONFIGURED = False
LOGGER_NAME = "judgetrust"


def redact_secrets(text: str) -> str:
    """Strip credential-like tokens from a log string."""

    redacted = _SECRET_RE.sub(r"\1=***", text)
    return _OPENAI_KEY_RE.sub("sk-***", redacted)


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return redact_secrets(formatted)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Idempotently configure the package logger."""

    global _CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            _RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the judgetrust namespace."""

    configure_logging()
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
