"""Bias probe runner: judge rigged pairs only. Never generates answers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from judgetrust.biasprobe.dataset import DEFAULT_DATASET_PATH, load_bias_probe_set
from judgetrust.biasprobe.metrics import length_bias_rate, position_bias_rate
from judgetrust.config import Settings, get_settings
from judgetrust.judge.harness import EvaluateFn
from judgetrust.judge.panel import compare_panel, dissent_rate
from judgetrust.logging_setup import get_logger
from judgetrust.models import (
    BiasProbeReport,
    BiasProbeRow,
    BiasProbeRowResult,
    JudgeLengthBias,
    Mode,
    Winner,
)

logger = get_logger("biasprobe.runner")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = REPO_ROOT / "data" / "results" / "bias_probe.json"


def run_bias_probe(
    rows: list[BiasProbeRow] | None = None,
    *,
    evaluate_fn: EvaluateFn | None = None,
    evaluate_fns: Mapping[str, EvaluateFn] | None = None,
    settings: Settings | None = None,
    dataset_path: Path | None = None,
    persist: bool = False,
    results_path: Path | None = None,
) -> BiasProbeReport:
    """Run the judge panel on every probe row. Does not call generators or calibration."""

    cfg = settings or get_settings()
    loaded = rows if rows is not None else load_bias_probe_set(dataset_path)
    logger.info(
        "bias_probe_start mode=%s n=%s judge_models=%s dataset=%s",
        Mode.BIAS_PROBE.value,
        len(loaded),
        ",".join(cfg.judge_models),
        str(dataset_path or DEFAULT_DATASET_PATH),
    )

    results: list[BiasProbeRowResult] = []
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
        hit = judge_winner is not None and judge_winner == row.longer_worse
        results.append(
            BiasProbeRowResult(
                id=row.id,
                longer_worse=row.longer_worse,
                judge_winner=judge_winner,
                length_bias_hit=hit,
                stable=panel.stable,
                position_bias=panel.position_bias,
                error=panel.error,
                votes=panel.votes,
                dissent=panel.dissent,
            )
        )

    scored = [item for item in results if item.judge_winner is not None]
    report = BiasProbeReport(
        mode=Mode.BIAS_PROBE.value,
        judge_models=tuple(cfg.judge_models),
        n=len(results),
        n_scored=len(scored),
        n_errors=len(results) - len(scored),
        length_bias_rate=length_bias_rate([item.length_bias_hit for item in scored]),
        position_bias_rate=position_bias_rate(
            [item.position_bias for item in results]
        ),
        panel_dissent_rate=dissent_rate([item.dissent for item in scored]),
        judge_length_bias=_per_judge_length_bias(results, tuple(cfg.judge_models)),
        rows=tuple(results),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "bias_probe_done length_bias=%s position_bias=%.3f dissent=%s errors=%s",
        report.length_bias_rate,
        report.position_bias_rate,
        report.panel_dissent_rate,
        report.n_errors,
    )
    if persist:
        persist_report(report, results_path)
    return report


def _per_judge_length_bias(
    results: list[BiasProbeRowResult],
    models: tuple[str, ...],
) -> tuple[JudgeLengthBias, ...]:
    out: list[JudgeLengthBias] = []
    for model in models:
        hits: list[bool] = []
        for row in results:
            vote = next((item for item in row.votes if item.model == model), None)
            if vote is None or vote.winner is None:
                continue
            hits.append(vote.winner == row.longer_worse)
        out.append(
            JudgeLengthBias(
                model=model,
                length_bias_rate=length_bias_rate(hits),
                n_scored=len(hits),
            )
        )
    return tuple(out)


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
    dissent_text = (
        f"{report.panel_dissent_rate:.0%}"
        if report.panel_dissent_rate is not None
        else "n/a"
    )
    lines = [
        "Bias probe report",
        f"  Panel: {', '.join(report.judge_models)}",
        f"  n={report.n}  scored={report.n_scored}  errors={report.n_errors}",
        f"  Length-bias rate (panel): {length_text}",
        f"  Position-bias rate: {report.position_bias_rate:.0%}",
        f"  Panel dissent: {dissent_text}",
    ]
    for item in report.judge_length_bias:
        rate = (
            f"{item.length_bias_rate:.0%}"
            if item.length_bias_rate is not None
            else "n/a"
        )
        lines.append(f"    {item.model}  length_bias={rate}  n={item.n_scored}")
    for row in report.rows:
        if row.length_bias_hit or row.position_bias or row.error or row.dissent:
            lines.append(
                f"    {row.id}  longer_worse={row.longer_worse}  "
                f"panel={row.judge_winner}  length_hit={row.length_bias_hit}  "
                f"position_bias={row.position_bias}  dissent={row.dissent}  "
                f"error={row.error}"
            )
    return "\n".join(lines)
