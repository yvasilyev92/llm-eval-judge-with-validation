"""Calibration runner: judge pre-written pairs only. Never generates answers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from judgetrust.calibrate.dataset import DEFAULT_DATASET_PATH, load_calibration_set
from judgetrust.calibrate.metrics import (
    RAW_AGREEMENT_NOTE,
    cohen_kappa,
    kappa_band,
    position_consistency,
    raw_agreement,
)
from judgetrust.config import Settings, get_settings
from judgetrust.judge.harness import EvaluateFn
from judgetrust.judge.panel import compare_panel, dissent_rate
from judgetrust.llm import missing_api_key  # re-exported for CLI and tests
from judgetrust.logging_setup import get_logger
from judgetrust.models import (
    CalibrationReport,
    CalibrationRow,
    CalibrationRowResult,
    JudgeKappa,
    Mode,
    Winner,
)

logger = get_logger("calibrate.runner")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = REPO_ROOT / "data" / "results" / "calibration.json"


def run_calibration(
    rows: list[CalibrationRow] | None = None,
    *,
    evaluate_fn: EvaluateFn | None = None,
    evaluate_fns: Mapping[str, EvaluateFn] | None = None,
    settings: Settings | None = None,
    dataset_path: Path | None = None,
    persist: bool = False,
    results_path: Path | None = None,
) -> CalibrationReport:
    """Run the judge panel on every calibration row and compute trust metrics.

    Inject ``evaluate_fn`` (all members) or ``evaluate_fns`` (per model) in tests.
    This function never calls generators.
    """

    cfg = settings or get_settings()
    loaded = rows if rows is not None else load_calibration_set(dataset_path)
    injected = evaluate_fn is not None or evaluate_fns is not None
    judge_models = ("injected",) * len(cfg.judge_models) if injected else tuple(cfg.judge_models)

    logger.info(
        "calibration_start mode=%s n=%s judge_models=%s dataset=%s",
        Mode.CALIBRATION.value,
        len(loaded),
        ",".join(judge_models),
        str(dataset_path or DEFAULT_DATASET_PATH),
    )

    row_results: list[CalibrationRowResult] = []
    for row in loaded:
        panel = compare_panel(
            row.question,
            row.answer_a,
            row.answer_b,
            evaluate_fn=evaluate_fn,
            evaluate_fns=evaluate_fns,
            settings=cfg,
        )
        judge_winner: Winner | None = None if panel.error else panel.final_winner
        row_results.append(
            CalibrationRowResult(
                id=row.id,
                human_winner=row.human_winner,
                judge_winner=judge_winner,
                failure_mode=row.failure_mode,
                stable=panel.stable,
                position_bias=panel.position_bias,
                error=panel.error,
                votes=panel.votes,
                dissent=panel.dissent,
            )
        )

    scored = [item for item in row_results if item.judge_winner is not None]
    human_labels = [item.human_winner for item in scored]
    judge_labels = [item.judge_winner for item in scored if item.judge_winner is not None]
    kappa = cohen_kappa(human_labels, judge_labels)
    agreement = raw_agreement(human_labels, judge_labels) if scored else None
    consistency = position_consistency([item.stable for item in row_results])
    disagreements = tuple(
        item
        for item in scored
        if item.judge_winner != item.human_winner
    )
    names = tuple(cfg.judge_models)
    report = CalibrationReport(
        mode=Mode.CALIBRATION.value,
        judge_models=tuple(cfg.judge_models),
        n=len(row_results),
        n_scored=len(scored),
        n_errors=len(row_results) - len(scored),
        kappa=kappa,
        kappa_band=kappa_band(kappa) if kappa is not None else None,
        raw_agreement=agreement,
        raw_agreement_note=RAW_AGREEMENT_NOTE,
        position_consistency=consistency,
        panel_dissent_rate=dissent_rate([item.dissent for item in scored]),
        judge_kappas=_per_judge_kappas(row_results, names),
        disagreements=disagreements,
        rows=tuple(row_results),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "calibration_done kappa=%s band=%s agreement=%s consistency=%.3f "
        "dissent=%s errors=%s",
        report.kappa,
        report.kappa_band,
        report.raw_agreement,
        report.position_consistency,
        report.panel_dissent_rate,
        report.n_errors,
    )
    if persist:
        persist_report(report, results_path)
    return report


def _per_judge_kappas(
    rows: list[CalibrationRowResult],
    models: tuple[str, ...],
) -> tuple[JudgeKappa, ...]:
    out: list[JudgeKappa] = []
    for model in models:
        human: list[Winner] = []
        judged: list[Winner] = []
        for row in rows:
            vote = next((item for item in row.votes if item.model == model), None)
            if vote is None or vote.winner is None:
                continue
            human.append(row.human_winner)
            judged.append(vote.winner)
        score = cohen_kappa(human, judged) if human else None
        out.append(
            JudgeKappa(
                model=model,
                kappa=score,
                kappa_band=kappa_band(score) if score is not None else None,
                n_scored=len(human),
            )
        )
    return tuple(out)


def persist_report(
    report: CalibrationReport,
    path: Path | None = None,
) -> Path:
    """Write the calibration report JSON for the Trust Report to load later."""

    output = path or DEFAULT_RESULTS_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("calibration_persisted path=%s", output)
    return output


def load_report(path: Path | None = None) -> dict[str, object]:
    """Load a previously persisted calibration report."""

    target = path or DEFAULT_RESULTS_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def format_report(report: CalibrationReport) -> str:
    """Human-readable stdout summary. Kappa is the headline."""

    kappa_text = (
        f"{report.kappa:.2f}  ({report.kappa_band})"
        if report.kappa is not None and report.kappa_band is not None
        else "n/a"
    )
    agreement_text = (
        f"{report.raw_agreement:.0%}" if report.raw_agreement is not None else "n/a"
    )
    dissent_text = (
        f"{report.panel_dissent_rate:.0%}"
        if report.panel_dissent_rate is not None
        else "n/a"
    )
    lines = [
        "Calibration report",
        f"  Panel: {', '.join(report.judge_models)}",
        f"  n={report.n}  scored={report.n_scored}  errors={report.n_errors}",
        f"  Cohen's kappa (panel): {kappa_text}",
        f"  Raw agreement: {agreement_text}  ({report.raw_agreement_note})",
        f"  Position consistency: {report.position_consistency:.0%}",
        f"  Panel dissent: {dissent_text}",
        f"  Disagreements: {len(report.disagreements)}",
    ]
    for item in report.judge_kappas:
        k = f"{item.kappa:.2f}" if item.kappa is not None else "n/a"
        band = f" ({item.kappa_band})" if item.kappa_band else ""
        lines.append(f"    {item.model}  kappa={k}{band}  n={item.n_scored}")
    for row in report.disagreements:
        lines.append(
            f"    {row.id}  human={row.human_winner}  panel={row.judge_winner}  "
            f"mode={row.failure_mode}  stable={row.stable}  dissent={row.dissent}"
        )
    return "\n".join(lines)
