"""Model-family helpers for the self-preference caveat."""

from __future__ import annotations

from collections.abc import Sequence

SELF_PREFERENCE_NOTE = (
    "The judge panel shares a model family with at least one generator, so "
    "self-preference bias is possible. A same-family panel does not cancel that "
    "risk. Point JUDGE_MODELS at a different family (Anthropic or Google) when "
    "that API key is available."
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


def has_self_preference(
    judge_models: str | Sequence[str],
    generator_models: Sequence[str],
) -> bool:
    """True when any judge family overlaps any generator family."""

    judges = (judge_models,) if isinstance(judge_models, str) else tuple(judge_models)
    gen_fams = {model_family(name) for name in generator_models}
    for name in judges:
        fam = model_family(name)
        if fam != "unknown" and fam in gen_fams:
            return True
    return False
