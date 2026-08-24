"""Tunable settings: models, prompts, metric bands, and LLM robustness."""

from __future__ import annotations

from dataclasses import dataclass

# Default generators: three OpenAI models so the app runs on a single key.
GENERATOR_MODELS: tuple[str, ...] = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-mini",
)

# Prefer a different tier than the generators. Point this at a non-OpenAI
# model (e.g. claude-sonnet or gemini) when that provider key is available.
JUDGE_MODEL: str = "gpt-4.1"

PROMPT_A: str = "Answer the user's health question."
PROMPT_B: str = (
    "Answer the user's health question. Flag any risks and note when to see a doctor."
)

# Landis & Koch-style bands. First matching lower bound (descending) wins.
# Negative kappa maps to "poor".
KAPPA_BANDS: tuple[tuple[float, str], ...] = (
    (0.81, "almost-perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.00, "poor"),
)

# Trust Report / UI color cutoffs (Phase 4). Lower length-bias is better.
TRUST_THRESHOLDS: dict[str, float] = {
    "kappa_green": 0.61,
    "kappa_amber": 0.41,
    "position_consistency_green": 0.85,
    "position_consistency_amber": 0.70,
    "length_bias_green": 0.10,
    "length_bias_amber": 0.20,
}

LLM_TIMEOUT_SECONDS: float = 60.0
LLM_MAX_RETRIES: int = 3
LLM_RETRY_BASE_DELAY_SECONDS: float = 1.0
JUDGE_TEMPERATURE: float = 0.0
GENERATOR_TEMPERATURE: float = 0.2


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime config. Later phases can construct variants."""

    generator_models: tuple[str, ...] = GENERATOR_MODELS
    judge_model: str = JUDGE_MODEL
    prompt_a: str = PROMPT_A
    prompt_b: str = PROMPT_B
    kappa_bands: tuple[tuple[float, str], ...] = KAPPA_BANDS
    trust_thresholds: dict[str, float] | None = None
    llm_timeout_seconds: float = LLM_TIMEOUT_SECONDS
    llm_max_retries: int = LLM_MAX_RETRIES
    llm_retry_base_delay_seconds: float = LLM_RETRY_BASE_DELAY_SECONDS
    judge_temperature: float = JUDGE_TEMPERATURE
    generator_temperature: float = GENERATOR_TEMPERATURE

    def __post_init__(self) -> None:
        if self.trust_thresholds is None:
            object.__setattr__(self, "trust_thresholds", dict(TRUST_THRESHOLDS))


def get_settings() -> Settings:
    """Return the default settings snapshot."""

    return Settings()
