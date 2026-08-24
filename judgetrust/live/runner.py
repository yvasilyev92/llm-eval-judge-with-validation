"""Live runner: generate A vs B per model, then judge. Never runs calibration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from judgetrust.config import Settings, get_settings
from judgetrust.generators.chain import GenerateFn, generate_answer
from judgetrust.judge.harness import EvaluateFn
from judgetrust.judge.panel import compare_panel, dissent_rate
from judgetrust.live.metrics import (
    cross_model_agreement,
    position_consistency,
    prompt_b_win_rate,
)
from judgetrust.logging_setup import get_logger
from judgetrust.models import LiveDuel, LiveReport, Mode, Winner

logger = get_logger("live.runner")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = REPO_ROOT / "data" / "results" / "live.json"


def run_live(
    question: str,
    *,
    question_id: str | None = None,
    generate_fn: GenerateFn | None = None,
    evaluate_fn: EvaluateFn | None = None,
    evaluate_fns: Mapping[str, EvaluateFn] | None = None,
    settings: Settings | None = None,
    persist: bool = False,
    results_path: Path | None = None,
) -> LiveReport:
    """Generate prompt A vs B on each generator model, then panel-judge both orderings.

    Inject ``generate_fn`` and ``evaluate_fn`` / ``evaluate_fns`` in tests.
    Does not load calibration.
    """

    if not question.strip():
        raise ValueError("question must be a non-empty string")

    cfg = settings or get_settings()
    logger.info(
        "live_start mode=%s n_models=%s judge_models=%s question_chars=%s",
        Mode.LIVE.value,
        len(cfg.generator_models),
        ",".join(cfg.judge_models),
        len(question),
    )

    duels: list[LiveDuel] = []
    for model in cfg.generator_models:
        duel = _run_one_model(
            question,
            model=model,
            prompt_a=cfg.prompt_a,
            prompt_b=cfg.prompt_b,
            generate_fn=generate_fn,
            evaluate_fn=evaluate_fn,
            evaluate_fns=evaluate_fns,
            settings=cfg,
        )
        duels.append(duel)

    scored_winners: list[Winner] = [
        duel.winner for duel in duels if duel.winner is not None
    ]
    scored = [duel for duel in duels if duel.winner is not None]
    report = LiveReport(
        mode=Mode.LIVE.value,
        question=question,
        question_id=question_id,
        judge_models=tuple(cfg.judge_models),
        generator_models=tuple(cfg.generator_models),
        duels=tuple(duels),
        n=len(duels),
        n_scored=len(scored_winners),
        n_errors=sum(1 for duel in duels if duel.error),
        prompt_b_win_rate=prompt_b_win_rate(scored_winners),
        cross_model_agreement=cross_model_agreement(scored_winners),
        position_consistency=position_consistency([duel.stable for duel in duels]),
        panel_dissent_rate=dissent_rate([duel.dissent for duel in scored]),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "live_done b_win_rate=%s agreement=%s consistency=%.3f dissent=%s errors=%s",
        report.prompt_b_win_rate,
        report.cross_model_agreement,
        report.position_consistency,
        report.panel_dissent_rate,
        report.n_errors,
    )
    if persist:
        persist_report(report, results_path)
    return report


def _run_one_model(
    question: str,
    *,
    model: str,
    prompt_a: str,
    prompt_b: str,
    generate_fn: GenerateFn | None,
    evaluate_fn: EvaluateFn | None,
    evaluate_fns: Mapping[str, EvaluateFn] | None,
    settings: Settings,
) -> LiveDuel:
    try:
        answer_a = generate_answer(
            question, prompt_a, model=model, generate_fn=generate_fn
        )
        answer_b = generate_answer(
            question, prompt_b, model=model, generate_fn=generate_fn
        )
    except Exception as exc:  # noqa: BLE001 — recorded on the duel
        logger.warning(
            "live_generate_failed model=%s error_type=%s",
            model,
            type(exc).__name__,
        )
        return LiveDuel(
            model=model,
            answer_a="",
            answer_b="",
            winner=None,
            stable=False,
            position_bias=False,
            error=f"generate_failed:{type(exc).__name__}",
        )

    panel = compare_panel(
        question,
        answer_a,
        answer_b,
        evaluate_fn=evaluate_fn,
        evaluate_fns=evaluate_fns,
        settings=settings,
    )
    winner: Winner | None = None if panel.error else panel.final_winner
    return LiveDuel(
        model=model,
        answer_a=answer_a,
        answer_b=answer_b,
        winner=winner,
        stable=panel.stable,
        position_bias=panel.position_bias,
        error=panel.error,
        votes=panel.votes,
        dissent=panel.dissent,
    )


def persist_report(report: LiveReport, path: Path | None = None) -> Path:
    """Write the live report JSON for the Trust Report to load later."""

    output = path or DEFAULT_RESULTS_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("live_persisted path=%s", output)
    return output


def load_report(path: Path | None = None) -> dict[str, object]:
    """Load a previously persisted live report."""

    target = path or DEFAULT_RESULTS_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def format_report(report: LiveReport) -> str:
    """Human-readable stdout summary."""

    b_rate = (
        f"{report.prompt_b_win_rate:.0%}"
        if report.prompt_b_win_rate is not None
        else "n/a"
    )
    agreement = (
        f"{report.cross_model_agreement:.0%}"
        if report.cross_model_agreement is not None
        else "n/a"
    )
    dissent = (
        f"{report.panel_dissent_rate:.0%}"
        if report.panel_dissent_rate is not None
        else "n/a"
    )
    qid = f" ({report.question_id})" if report.question_id else ""
    lines = [
        "Live report",
        f"  Question{qid}: {report.question}",
        f"  Panel: {', '.join(report.judge_models)}",
        f"  n={report.n}  scored={report.n_scored}  errors={report.n_errors}",
        f"  Prompt B win rate: {b_rate}",
        f"  Cross-model agreement: {agreement}",
        f"  Position consistency: {report.position_consistency:.0%}",
        f"  Panel dissent: {dissent}",
    ]
    for duel in report.duels:
        votes = ", ".join(
            f"{vote.model}={vote.winner or vote.error}" for vote in duel.votes
        )
        lines.append(
            f"    {duel.model}  panel={duel.winner}  stable={duel.stable}  "
            f"dissent={duel.dissent}  votes=[{votes}]  error={duel.error}"
        )
    return "\n".join(lines)
