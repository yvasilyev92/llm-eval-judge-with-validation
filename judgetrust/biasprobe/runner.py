"""Bias probe runner: judge rigged pairs only. Never generates answers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from judgetrust.biasprobe.dataset import DEFAULT_DATASET_PATH, load_bias_probe_set
from judgetrust.biasprobe.metrics import length_bias_rate, position_bias_rate
from judgetrust.config import Settings, get_settings
from judgetrust.judge.chain import Judge
from judgetrust.judge.harness import EvaluateFn, compare_both_orderings
from judgetrust.logging_setup import get_logger
from judgetrust.models import (
    BiasProbeReport,
    BiasProbeRow,
    BiasProbeRowResult,
    Mode,
    Winner,
)

logger = get_logger("biasprobe.runner")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = REPO_ROOT / "data" / "results" / "bias_probe.json"


def _row_error(result) -> str | None:
    if result.run_a_first.verdict is None and result.run_b_first.verdict is None:
        return (
            result.run_a_first.error
            or result.run_b_first.error
            or "both_orderings_failed"
        )
    return None


def run_bias_probe(
    rows: list[BiasProbeRow] | None = None,
    *,
    evaluate_fn: EvaluateFn | None = None,
    judge: Judge | None = None,
    settings: Settings | None = None,
    dataset_path: Path | None = None,
    persist: bool = False,
    results_path: Path | None = None,
) -> BiasProbeReport:
    """Run the judge on every probe row. Does not call generators or calibration."""

    cfg = settings or get_settings()
    loaded = rows if rows is not None else load_bias_probe_set(dataset_path)
    judge_model = "injected" if evaluate_fn is not None else cfg.judge_model
    logger.info(
        "bias_probe_start mode=%s n=%s judge_model=%s dataset=%s",
        Mode.BIAS_PROBE.value,
        len(loaded),
        judge_model,
        str(dataset_path or DEFAULT_DATASET_PATH),
    )

    results: list[BiasProbeRowResult] = []
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
        hit = (
            judge_winner is not None and judge_winner == row.longer_worse
        )
        results.append(
            BiasProbeRowResult(
                id=row.id,
                longer_worse=row.longer_worse,
                judge_winner=judge_winner,
                length_bias_hit=hit,
                stable=pairwise.stable,
                position_bias=pairwise.position_bias,
                error=error,
            )
        )

    scored = [item for item in results if item.judge_winner is not None]
    report = BiasProbeReport(
        mode=Mode.BIAS_PROBE.value,
        judge_model=judge_model,
        n=len(results),
        n_scored=len(scored),
        n_errors=len(results) - len(scored),
        length_bias_rate=length_bias_rate([item.length_bias_hit for item in scored]),
        position_bias_rate=position_bias_rate(
            [item.position_bias for item in results]
        ),
        rows=tuple(results),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "bias_probe_done length_bias=%s position_bias=%.3f errors=%s",
        report.length_bias_rate,
        report.position_bias_rate,
        report.n_errors,
    )
    if persist:
        persist_report(report, results_path)
    return report


def persist_report(
    report: BiasProbeReport,
    path: Path | None = None,
) -> Path:
    """Write the bias-probe report JSON for the Trust Report to load later."""

    output = path or DEFAULT_RESULTS_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("bias_probe_persisted path=%s", output)
    return output


def load_report(path: Path | None = None) -> dict[str, object]:
    """Load a previously persisted bias-probe report."""

    target = path or DEFAULT_RESULTS_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def format_report(report: BiasProbeReport) -> str:
    """Human-readable stdout summary."""

    length_text = (
        f"{report.length_bias_rate:.0%}"
        if report.length_bias_rate is not None
        else "n/a"
    )
    lines = [
        "Bias probe report",
        f"  Judge: {report.judge_model}",
        f"  n={report.n}  scored={report.n_scored}  errors={report.n_errors}",
        f"  Length-bias rate: {length_text}",
        f"  Position-bias rate: {report.position_bias_rate:.0%}",
    ]
    for row in report.rows:
        if row.length_bias_hit or row.position_bias or row.error:
            lines.append(
                f"    {row.id}  longer_worse={row.longer_worse}  "
                f"judge={row.judge_winner}  length_hit={row.length_bias_hit}  "
                f"position_bias={row.position_bias}  error={row.error}"
            )
    return "\n".join(lines)
