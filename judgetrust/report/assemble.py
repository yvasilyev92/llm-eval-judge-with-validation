"""Assemble a TrustReport from persisted calibration, live, and probe JSON."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from judgetrust.calibrate.metrics import RAW_AGREEMENT_NOTE
from judgetrust.config import Settings, get_settings
from judgetrust.logging_setup import get_logger
from judgetrust.models import ModelDuelSummary, TrustReport, Winner
from judgetrust.report.colors import signal_colors
from judgetrust.report.family import (
    SELF_PREFERENCE_NOTE,
    has_self_preference,
)

logger = get_logger("report.assemble")

Loader = Callable[[], dict[str, Any]]


def _optional_load(loader: Loader, label: str) -> dict[str, Any] | None:
    try:
        return loader()
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "trust_source_unreadable source=%s error_type=%s",
            label,
            type(exc).__name__,
        )
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_winner(value: Any) -> Winner | None:
    if value in ("A", "B", "tie"):
        return value
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def _as_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _disagreement_ids(calibration: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not calibration:
        return ()
    raw = calibration.get("disagreements")
    if not isinstance(raw, list):
        return ()
    ids: list[str] = []
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return tuple(ids)


def _per_model(live: Mapping[str, Any] | None) -> tuple[ModelDuelSummary, ...]:
    if not live:
        return ()
    raw = live.get("duels")
    if not isinstance(raw, list):
        return ()
    chips: list[ModelDuelSummary] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        if not isinstance(model, str) or not model:
            continue
        chips.append(
            ModelDuelSummary(
                model=model,
                winner=_as_winner(item.get("winner")),
                stable=bool(item.get("stable")),
                error=item.get("error") if isinstance(item.get("error"), str) else None,
            )
        )
    return tuple(chips)


def build_verdict(
    *,
    n_b_wins: int | None,
    n_models: int | None,
    kappa: float | None,
    kappa_band: str | None,
    position_consistency: float | None,
    length_bias_rate: float | None,
    overall_color: str,
    missing: tuple[str, ...],
) -> str:
    """Plain-language Trust Report line."""

    if n_b_wins is not None and n_models:
        win_clause = f"Prompt B wins on {n_b_wins}/{n_models} models"
    else:
        win_clause = "No live comparison yet"

    if "calibration" in missing or kappa is None or kappa_band is None:
        band = "UNKNOWN"
    else:
        band = kappa_band.upper().replace("-", " ")

    have_all = not missing and kappa is not None and position_consistency is not None
    have_all = have_all and length_bias_rate is not None and n_models is not None
    if have_all:
        pos_bias = 1.0 - position_consistency
        supported = overall_color == "green"
        joiner = "and" if supported else "but"
        tail = (
            "treat the win as reasonably supported."
            if supported
            else "treat the win as suggestive, not proven."
        )
        return (
            f"{win_clause}, {joiner} trust is {band}: judge–human kappa {kappa:.2f}, "
            f"{pos_bias:.0%} position bias, {length_bias_rate:.0%} length bias — {tail}"
        )

    parts = [win_clause]
    if kappa is not None and kappa_band is not None:
        parts.append(f"judge–human kappa {kappa:.2f} ({band})")
    if missing:
        parts.append("missing " + ", ".join(missing) + " — run those modes to complete the report")
    else:
        parts.append("signals incomplete")
    return ". ".join(parts) + "."


def assemble_trust_report(
    *,
    calibration: dict[str, Any] | None = None,
    live: dict[str, Any] | None = None,
    probe: dict[str, Any] | None = None,
    settings: Settings | None = None,
    load_missing_from_disk: bool = True,
) -> TrustReport:
    """Combine the three result files. Missing sources yield a partial report."""

    cfg = settings or get_settings()
    if load_missing_from_disk:
        if calibration is None:
            from judgetrust.calibrate.runner import load_report as load_calibration

            calibration = _optional_load(load_calibration, "calibration")
        if live is None:
            from judgetrust.live.runner import load_report as load_live

            live = _optional_load(load_live, "live")
        if probe is None:
            from judgetrust.biasprobe.runner import load_report as load_probe

            probe = _optional_load(load_probe, "bias_probe")

    missing: list[str] = []
    if calibration is None:
        missing.append("calibration")
    if live is None:
        missing.append("live")
    if probe is None:
        missing.append("bias_probe")

    chips = _per_model(live)
    scored = [chip for chip in chips if chip.winner is not None]
    n_models = len(scored) if scored else None
    n_b_wins = sum(1 for chip in scored if chip.winner == "B") if scored else None
    b_rate = _as_float(live.get("prompt_b_win_rate")) if live else None
    live_question = _as_str(live.get("question")) if live else None
    live_question_id = _as_str(live.get("question_id")) if live else None
    calibration_n = _as_int(calibration.get("n")) if calibration else None
    calibration_n_scored = (
        _as_int(calibration.get("n_scored")) if calibration else None
    )
    disagreement_ids = _disagreement_ids(calibration)
    probe_n = _as_int(probe.get("n")) if probe else None
    probe_n_scored = _as_int(probe.get("n_scored")) if probe else None

    kappa = _as_float(calibration.get("kappa")) if calibration else None
    kappa_band = calibration.get("kappa_band") if calibration else None
    if not isinstance(kappa_band, str):
        kappa_band = None
    raw_agreement = _as_float(calibration.get("raw_agreement")) if calibration else None
    raw_note = calibration.get("raw_agreement_note") if calibration else None
    if not isinstance(raw_note, str):
        raw_note = RAW_AGREEMENT_NOTE if calibration else None

    live_consistency = _as_float(live.get("position_consistency")) if live else None
    cal_consistency = (
        _as_float(calibration.get("position_consistency")) if calibration else None
    )
    if live_consistency is not None:
        position_consistency = live_consistency
        consistency_source = "live"
    elif cal_consistency is not None:
        position_consistency = cal_consistency
        consistency_source = "calibration"
    else:
        position_consistency = None
        consistency_source = None

    length_bias = _as_float(probe.get("length_bias_rate")) if probe else None
    probe_pos = _as_float(probe.get("position_bias_rate")) if probe else None

    judge_model = cfg.judge_model
    generators = tuple(cfg.generator_models)
    if live:
        live_judge = live.get("judge_model")
        if isinstance(live_judge, str) and live_judge and live_judge != "injected":
            judge_model = live_judge
        live_gens = live.get("generator_models")
        if isinstance(live_gens, list) and all(isinstance(item, str) for item in live_gens):
            generators = tuple(live_gens)

    self_pref = has_self_preference(judge_model, generators)
    kappa_color, consistency_color, length_color, overall = signal_colors(
        kappa=kappa,
        position_consistency=position_consistency,
        length_bias_rate=length_bias,
        thresholds=cfg.trust_thresholds,
    )
    verdict = build_verdict(
        n_b_wins=n_b_wins,
        n_models=n_models,
        kappa=kappa,
        kappa_band=kappa_band,
        position_consistency=position_consistency,
        length_bias_rate=length_bias,
        overall_color=overall,
        missing=tuple(missing),
    )
    return TrustReport(
        prompt_b_win_rate=b_rate,
        n_models=n_models,
        n_b_wins=n_b_wins,
        per_model=chips,
        kappa=kappa,
        kappa_band=kappa_band,
        raw_agreement=raw_agreement,
        raw_agreement_note=raw_note,
        position_consistency=position_consistency,
        position_consistency_source=consistency_source,
        length_bias_rate=length_bias,
        probe_position_bias_rate=probe_pos,
        self_preference=self_pref,
        self_preference_note=SELF_PREFERENCE_NOTE if self_pref else None,
        verdict=verdict,
        overall_color=overall,
        kappa_color=kappa_color,
        consistency_color=consistency_color,
        length_bias_color=length_color,
        missing=tuple(missing),
        judge_model=judge_model,
        generator_models=generators,
        live_question=live_question,
        live_question_id=live_question_id,
        calibration_n=calibration_n,
        calibration_n_scored=calibration_n_scored,
        disagreement_ids=disagreement_ids,
        probe_n=probe_n,
        probe_n_scored=probe_n_scored,
    )
