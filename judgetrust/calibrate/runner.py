"""Calibration runner: judge pre-written pairs only. Never generates answers."""

from __future__ import annotations

import json
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
from judgetrust.judge.chain import Judge
from judgetrust.judge.harness import EvaluateFn, compare_both_orderings
from judgetrust.llm import missing_api_key  # re-exported for CLI and tests
from judgetrust.logging_setup import get_logger
from judgetrust.models import (
    CalibrationReport,
    CalibrationRow,
    CalibrationRowResult,
    Mode,
    Winner,
)

logger = get_logger("calibrate.runner")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = REPO_ROOT / "data" / "results" / "calibration.json"


def _row_error(result) -> str | None:
    a_err = result.run_a_first.error
    b_err = result.run_b_first.error
    if result.run_a_first.verdict is None and result.run_b_first.verdict is None:
        return a_err or b_err or "both_orderings_failed"
    return None


def run_calibration(
    rows: list[CalibrationRow] | None = None,
    *,
    evaluate_fn: EvaluateFn | None = None,
    judge: Judge | None = None,
    settings: Settings | None = None,
    dataset_path: Path | None = None,
    persist: bool = False,
    results_path: Path | None = None,
) -> CalibrationReport:
    """Run the judge on every calibration row and compute trust metrics.

    Inject ``evaluate_fn`` in tests. This function never calls generators.
    """

    cfg = settings or get_settings()
    loaded = rows if rows is not None else load_calibration_set(dataset_path)
    judge_model = "injected" if evaluate_fn is not None else cfg.judge_model

    logger.info(
        "calibration_start mode=%s n=%s judge_model=%s dataset=%s",
        Mode.CALIBRATION.value,
        len(loaded),
        judge_model,
        str(dataset_path or DEFAULT_DATASET_PATH),
    )

    row_results: list[CalibrationRowResult] = []
    for row in loaded:
        pairwise = compare_both_orderings(
            row.question,
            row.answer_a,
            row.answer_b,
            evaluate_fn=evaluate_fn,
            judge=judge,
        )
        error = _row_error(pairwise)
        judge_winner: Winner | None = None if error else pairwise.final_winner
        row_results.append(
            CalibrationRowResult(
                id=row.id,
                human_winner=row.human_winner,
                judge_winner=judge_winner,
                failure_mode=row.failure_mode,
                stable=pairwise.stable,
                position_bias=pairwise.position_bias,
                error=error,
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

    report = CalibrationReport(
        mode=Mode.CALIBRATION.value,
        judge_model=judge_model,
        n=len(row_results),
        n_scored=len(scored),
        n_errors=len(row_results) - len(scored),
        kappa=kappa,
        kappa_band=kappa_band(kappa) if kappa is not None else None,
        raw_agreement=agreement,
        raw_agreement_note=RAW_AGREEMENT_NOTE,
        position_consistency=consistency,
        disagreements=disagreements,
        rows=tuple(row_results),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "calibration_done kappa=%s band=%s agreement=%s consistency=%.3f errors=%s",
        report.kappa,
        report.kappa_band,
        report.raw_agreement,
        report.position_consistency,
        report.n_errors,
    )
    if persist:
        persist_report(report, results_path)
    return report


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
    lines = [
        "Calibration report",
        f"  Judge: {report.judge_model}",
        f"  n={report.n}  scored={report.n_scored}  errors={report.n_errors}",
        f"  Cohen's kappa: {kappa_text}",
        f"  Raw agreement: {agreement_text}  ({report.raw_agreement_note})",
        f"  Position consistency: {report.position_consistency:.0%}",
        f"  Disagreements: {len(report.disagreements)}",
    ]
    for row in report.disagreements:
        lines.append(
            f"    {row.id}  human={row.human_winner}  judge={row.judge_winner}  "
            f"mode={row.failure_mode}  stable={row.stable}"
        )
    return "\n".join(lines)
