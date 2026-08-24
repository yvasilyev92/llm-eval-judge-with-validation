"""Trust Report assembler, colors, and self-preference — no API."""

from __future__ import annotations

from judgetrust.config import Settings
from judgetrust.report.assemble import assemble_trust_report, build_verdict
from judgetrust.report.colors import overall_color, signal_colors, traffic_light
from judgetrust.report.family import has_self_preference, model_family


def _calibration(**overrides: object) -> dict:
    payload: dict = {
        "mode": "calibration",
        "judge_model": "gpt-4.1",
        "n": 35,
        "n_scored": 35,
        "n_errors": 0,
        "kappa": 0.61,
        "kappa_band": "substantial",
        "raw_agreement": 0.78,
        "raw_agreement_note": "Kappa corrects for chance; raw agreement overstates reliability.",
        "position_consistency": 0.90,
        "disagreements": [],
        "rows": [],
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _duel(model: str, winner: str, *, stable: bool = True) -> dict:
    return {
        "model": model,
        "answer_a": "a",
        "answer_b": "b",
        "winner": winner,
        "stable": stable,
        "position_bias": not stable,
        "error": None,
    }


def _live(**overrides: object) -> dict:
    duels = [
        _duel("gpt-4o", "B"),
        _duel("gpt-4o-mini", "B"),
        _duel("gpt-4.1-mini", "B"),
    ]
    payload: dict = {
        "mode": "live",
        "question": "Does sunscreen matter on a cloudy day?",
        "question_id": "lq-05",
        "judge_model": "gpt-4.1",
        "generator_models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"],
        "duels": duels,
        "n": 3,
        "n_scored": 3,
        "n_errors": 0,
        "prompt_b_win_rate": 1.0,
        "cross_model_agreement": 1.0,
        "position_consistency": 0.88,
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _probe(**overrides: object) -> dict:
    payload: dict = {
        "mode": "bias_probe",
        "judge_model": "gpt-4.1",
        "n": 10,
        "n_scored": 10,
        "n_errors": 0,
        "length_bias_rate": 0.08,
        "position_bias_rate": 0.10,
        "rows": [],
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def test_model_family() -> None:
    assert model_family("gpt-4.1") == "openai"
    assert model_family("gpt-4o-mini") == "openai"
    assert model_family("claude-sonnet-4-5") == "anthropic"
    assert model_family("gemini-2.5-pro") == "google"
    assert model_family("mystery-7b") == "unknown"


def test_self_preference_defaults() -> None:
    settings = Settings()
    assert has_self_preference(settings.judge_model, settings.generator_models) is True
    assert has_self_preference("claude-sonnet-4-5", settings.generator_models) is False


def test_traffic_lights() -> None:
    assert traffic_light(0.61, 0.61, 0.41, higher_is_better=True) == "green"
    assert traffic_light(0.50, 0.61, 0.41, higher_is_better=True) == "amber"
    assert traffic_light(0.20, 0.61, 0.41, higher_is_better=True) == "red"
    assert traffic_light(0.08, 0.10, 0.20, higher_is_better=False) == "green"
    assert traffic_light(0.15, 0.10, 0.20, higher_is_better=False) == "amber"
    assert traffic_light(0.40, 0.10, 0.20, higher_is_better=False) == "red"
    assert traffic_light(None, 0.61, 0.41, higher_is_better=True) == "gray"
    assert overall_color("green", "amber", "gray") == "amber"
    assert overall_color("gray", "gray") == "gray"


def test_assemble_empty() -> None:
    report = assemble_trust_report(load_missing_from_disk=False)
    assert report.missing == ("calibration", "live", "bias_probe")
    assert report.overall_color == "gray"
    assert "No live comparison yet" in report.verdict
    assert "missing calibration, live, bias_probe" in report.verdict


def test_assemble_supported_win() -> None:
    report = assemble_trust_report(
        calibration=_calibration(),
        live=_live(),
        probe=_probe(),
        load_missing_from_disk=False,
    )
    assert report.missing == ()
    assert report.n_b_wins == 3
    assert report.n_models == 3
    assert report.prompt_b_win_rate == 1.0
    assert report.kappa == 0.61
    assert report.position_consistency == 0.88
    assert report.position_consistency_source == "live"
    assert report.length_bias_rate == 0.08
    assert report.overall_color == "green"
    assert report.self_preference is True
    assert report.self_preference_note is not None
    assert "Prompt B wins on 3/3 models" in report.verdict
    assert "SUBSTANTIAL" in report.verdict
    assert "reasonably supported" in report.verdict
    assert ", and trust is" in report.verdict
    assert report.live_question == "Does sunscreen matter on a cloudy day?"
    assert report.live_question_id == "lq-05"
    assert report.calibration_n == 35
    assert report.calibration_n_scored == 35
    assert report.disagreement_ids == ()
    assert report.probe_n == 10
    assert report.probe_n_scored == 10


def test_assemble_disagreement_ids() -> None:
    report = assemble_trust_report(
        calibration=_calibration(
            disagreements=[
                {"id": "fd-07", "human_winner": "tie", "judge_winner": "A"},
                {"id": "og-07", "human_winner": "tie", "judge_winner": "A"},
            ]
        ),
        live=_live(),
        probe=_probe(),
        load_missing_from_disk=False,
    )
    assert report.disagreement_ids == ("fd-07", "og-07")


def test_limits_text_names_sample_sizes() -> None:
    from judgetrust.ui.trust_card import _limits_text

    report = assemble_trust_report(
        calibration=_calibration(
            disagreements=[{"id": "fd-07", "human_winner": "tie", "judge_winner": "A"}]
        ),
        live=_live(),
        probe=_probe(),
        load_missing_from_disk=False,
    )
    text = _limits_text(report)
    assert "one question" in text
    assert "35 starter human labels" in text
    assert "fd-07" in text or "1 row" in text
    assert "10 rigged pairs" in text


def test_assemble_suggestive_when_length_bias_high() -> None:
    report = assemble_trust_report(
        calibration=_calibration(),
        live=_live(),
        probe=_probe(length_bias_rate=0.30),
        load_missing_from_disk=False,
    )
    assert report.length_bias_color == "red"
    assert report.overall_color == "red"
    assert "suggestive, not proven" in report.verdict
    assert ", but trust is" in report.verdict


def test_consistency_prefers_live() -> None:
    report = assemble_trust_report(
        calibration=_calibration(position_consistency=0.99),
        live=_live(position_consistency=0.50),
        probe=_probe(),
        load_missing_from_disk=False,
    )
    assert report.position_consistency == 0.50
    assert report.position_consistency_source == "live"


def test_consistency_falls_back_to_calibration() -> None:
    report = assemble_trust_report(
        calibration=_calibration(position_consistency=0.77),
        live=None,
        probe=_probe(),
        load_missing_from_disk=False,
    )
    assert report.position_consistency == 0.77
    assert report.position_consistency_source == "calibration"
    assert "No live comparison yet" in report.verdict
    assert "missing live" in report.verdict


def test_signal_colors_kappa() -> None:
    kappa_c, cons_c, len_c, overall = signal_colors(
        kappa=0.50,
        position_consistency=0.90,
        length_bias_rate=0.05,
    )
    assert kappa_c == "amber"
    assert cons_c == "green"
    assert len_c == "green"
    assert overall == "amber"


def test_build_verdict_partial() -> None:
    text = build_verdict(
        n_b_wins=None,
        n_models=None,
        kappa=0.4,
        kappa_band="fair",
        position_consistency=None,
        length_bias_rate=None,
        overall_color="gray",
        missing=("live", "bias_probe"),
    )
    assert text.startswith("No live comparison yet")
    assert "FAIR" in text
    assert "missing live, bias_probe" in text
