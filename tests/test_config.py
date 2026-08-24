"""Config loads and has the expected Phase 1 shape."""

from __future__ import annotations

from judgetrust.config import (
    GENERATOR_MODELS,
    JUDGE_MODELS,
    KAPPA_BANDS,
    PROMPT_A,
    PROMPT_B,
    TRUST_THRESHOLDS,
    Settings,
    get_settings,
)


def test_generator_list_has_three_models() -> None:
    assert len(GENERATOR_MODELS) == 3
    assert all(isinstance(name, str) and name for name in GENERATOR_MODELS)


def test_judge_panel_has_three_models() -> None:
    assert len(JUDGE_MODELS) == 3
    assert all(isinstance(name, str) and name for name in JUDGE_MODELS)
    assert len(set(JUDGE_MODELS)) == 3


def test_prompts_are_distinct() -> None:
    assert "health" in PROMPT_A.lower()
    assert "health" in PROMPT_B.lower()
    assert PROMPT_A != PROMPT_B
    assert "risk" in PROMPT_B.lower() or "doctor" in PROMPT_B.lower()


def test_kappa_bands_are_descending() -> None:
    floors = [floor for floor, _ in KAPPA_BANDS]
    assert floors == sorted(floors, reverse=True)
    labels = [label for _, label in KAPPA_BANDS]
    assert labels == ["almost-perfect", "substantial", "moderate", "fair", "poor"]


def test_trust_thresholds_present() -> None:
    for key in (
        "kappa_green",
        "kappa_amber",
        "position_consistency_green",
        "length_bias_green",
    ):
        assert key in TRUST_THRESHOLDS
        assert 0.0 <= TRUST_THRESHOLDS[key] <= 1.0


def test_get_settings_snapshot() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.judge_models == JUDGE_MODELS
    assert settings.generator_models == GENERATOR_MODELS
    assert settings.llm_timeout_seconds > 0
    assert settings.llm_max_retries >= 0
    assert 0.0 <= settings.generator_temperature <= 1.0
    assert settings.generator_temperature != settings.judge_temperature
