"""Model-family helpers for the self-preference caveat."""

from __future__ import annotations

from collections.abc import Sequence

SELF_PREFERENCE_NOTE = (
    "The judge shares a model family with at least one generator, so "
    "self-preference bias is possible. Point JUDGE_MODEL at a different "
    "family (Anthropic or Google) when that API key is available."
)


def model_family(model_name: str) -> str:
    """Return openai / anthropic / google / unknown from a model id."""

    name = model_name.lower().strip()
    if name.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")) or "openai" in name:
        return "openai"
    if "claude" in name:
        return "anthropic"
    if "gemini" in name:
        return "google"
    return "unknown"


def has_self_preference(judge_model: str, generator_models: Sequence[str]) -> bool:
    """True when the judge family overlaps any generator family."""

    judge_fam = model_family(judge_model)
    if judge_fam == "unknown":
        return False
    return judge_fam in {model_family(name) for name in generator_models}
