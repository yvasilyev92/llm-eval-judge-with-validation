"""Shared data models for verdicts, pairwise runs, and mutually exclusive modes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Winner = Literal["A", "B", "tie"]
PresentedWinner = Literal["1", "2", "tie"]


class Mode(Enum):
    """Eval modes. Only one mode runs at a time — never mix in a single action."""

    CALIBRATION = "calibration"
    LIVE = "live"
    BIAS_PROBE = "bias_probe"


class Ordering(Enum):
    """Presentation order of the two answers shown to the judge."""

    A_FIRST = "A_first"
    B_FIRST = "B_first"


@dataclass(frozen=True)
class RubricScores:
    """Decomposed 1–5 scores. Safety dominates when combining into a winner."""

    medical_safety: float
    factual_accuracy: float
    risk_flagging: float
    directness: float


@dataclass(frozen=True)
class Verdict:
    """Judge verdict in original A/B space (after flip-back from presentation order)."""

    winner: Winner
    reason: str
    confidence: float
    rubric: RubricScores | None = None


@dataclass(frozen=True)
class PresentedJudgment:
    """Raw judge output in presentation space (answer 1 vs 2), before A/B mapping."""

    winner: PresentedWinner | None
    reason: str
    confidence: float
    rubric: RubricScores | None
    raw_output: str
    error: str | None = None


@dataclass(frozen=True)
class OrderingRun:
    """One judge call at a single presentation order."""

    order: Ordering
    raw_output: str
    verdict: Verdict | None
    error: str | None = None


@dataclass(frozen=True)
class PairwiseResult:
    """Both-orderings comparison. Decisive only when the two mapped winners agree."""

    question: str
    answer_a: str
    answer_b: str
    run_a_first: OrderingRun
    run_b_first: OrderingRun
    stable: bool
    final_winner: Winner
    position_bias: bool


@dataclass(frozen=True)
class CalibrationRow:
    """One human-labeled pairwise example from the calibration set."""

    id: str
    question: str
    answer_a: str
    answer_b: str
    human_winner: Winner
    failure_mode: str


@dataclass(frozen=True)
class CalibrationRowResult:
    """Judge outcome for one calibration row."""

    id: str
    human_winner: Winner
    judge_winner: Winner | None
    failure_mode: str
    stable: bool
    position_bias: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "human_winner": self.human_winner,
            "judge_winner": self.judge_winner,
            "failure_mode": self.failure_mode,
            "stable": self.stable,
            "position_bias": self.position_bias,
            "error": self.error,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregated calibration metrics for the Trust Report."""

    mode: str
    judge_model: str
    n: int
    n_scored: int
    n_errors: int
    kappa: float | None
    kappa_band: str | None
    raw_agreement: float | None
    raw_agreement_note: str
    position_consistency: float
    disagreements: tuple[CalibrationRowResult, ...]
    rows: tuple[CalibrationRowResult, ...]
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "judge_model": self.judge_model,
            "n": self.n,
            "n_scored": self.n_scored,
            "n_errors": self.n_errors,
            "kappa": self.kappa,
            "kappa_band": self.kappa_band,
            "raw_agreement": self.raw_agreement,
            "raw_agreement_note": self.raw_agreement_note,
            "position_consistency": self.position_consistency,
            "disagreements": [row.to_dict() for row in self.disagreements],
            "rows": [row.to_dict() for row in self.rows],
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class LiveQuestion:
    """Unlabeled sample question for live mode."""

    id: str
    question: str


@dataclass(frozen=True)
class LiveDuel:
    """One generator model's A vs B comparison for a live question."""

    model: str
    answer_a: str
    answer_b: str
    winner: Winner | None
    stable: bool
    position_bias: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "answer_a": self.answer_a,
            "answer_b": self.answer_b,
            "winner": self.winner,
            "stable": self.stable,
            "position_bias": self.position_bias,
            "error": self.error,
        }


@dataclass(frozen=True)
class LiveReport:
    """Per-question live comparison across generator models."""

    mode: str
    question: str
    question_id: str | None
    judge_model: str
    generator_models: tuple[str, ...]
    duels: tuple[LiveDuel, ...]
    n: int
    n_scored: int
    n_errors: int
    prompt_b_win_rate: float | None
    cross_model_agreement: float | None
    position_consistency: float
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "question": self.question,
            "question_id": self.question_id,
            "judge_model": self.judge_model,
            "generator_models": list(self.generator_models),
            "duels": [duel.to_dict() for duel in self.duels],
            "n": self.n,
            "n_scored": self.n_scored,
            "n_errors": self.n_errors,
            "prompt_b_win_rate": self.prompt_b_win_rate,
            "cross_model_agreement": self.cross_model_agreement,
            "position_consistency": self.position_consistency,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class BiasProbeRow:
    """Rigged pair: one side is longer-but-worse, the other shorter-but-correct."""

    id: str
    question: str
    answer_a: str
    answer_b: str
    longer_worse: Literal["A", "B"]


@dataclass(frozen=True)
class BiasProbeRowResult:
    """Judge outcome for one bias-probe row."""

    id: str
    longer_worse: Literal["A", "B"]
    judge_winner: Winner | None
    length_bias_hit: bool
    stable: bool
    position_bias: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "longer_worse": self.longer_worse,
            "judge_winner": self.judge_winner,
            "length_bias_hit": self.length_bias_hit,
            "stable": self.stable,
            "position_bias": self.position_bias,
            "error": self.error,
        }


@dataclass(frozen=True)
class BiasProbeReport:
    """Length-bias and position-bias flags for the Trust Report."""

    mode: str
    judge_model: str
    n: int
    n_scored: int
    n_errors: int
    length_bias_rate: float | None
    position_bias_rate: float
    rows: tuple[BiasProbeRowResult, ...]
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "judge_model": self.judge_model,
            "n": self.n,
            "n_scored": self.n_scored,
            "n_errors": self.n_errors,
            "length_bias_rate": self.length_bias_rate,
            "position_bias_rate": self.position_bias_rate,
            "rows": [row.to_dict() for row in self.rows],
            "generated_at": self.generated_at,
        }


ColorName = Literal["green", "amber", "red", "gray"]


@dataclass(frozen=True)
class ModelDuelSummary:
    """Per-generator chip on the Trust Report."""

    model: str
    winner: Winner | None
    stable: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "winner": self.winner,
            "stable": self.stable,
            "error": self.error,
        }


@dataclass(frozen=True)
class TrustReport:
    """Assembled scoreboard from calibration, live, and bias-probe JSON."""

    prompt_b_win_rate: float | None
    n_models: int | None
    n_b_wins: int | None
    per_model: tuple[ModelDuelSummary, ...]
    kappa: float | None
    kappa_band: str | None
    raw_agreement: float | None
    raw_agreement_note: str | None
    position_consistency: float | None
    position_consistency_source: str | None
    length_bias_rate: float | None
    probe_position_bias_rate: float | None
    self_preference: bool
    self_preference_note: str | None
    verdict: str
    overall_color: ColorName
    kappa_color: ColorName
    consistency_color: ColorName
    length_bias_color: ColorName
    missing: tuple[str, ...]
    judge_model: str
    generator_models: tuple[str, ...]
    live_question: str | None = None
    live_question_id: str | None = None
    calibration_n: int | None = None
    calibration_n_scored: int | None = None
    disagreement_ids: tuple[str, ...] = ()
    probe_n: int | None = None
    probe_n_scored: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_b_win_rate": self.prompt_b_win_rate,
            "n_models": self.n_models,
            "n_b_wins": self.n_b_wins,
            "per_model": [item.to_dict() for item in self.per_model],
            "kappa": self.kappa,
            "kappa_band": self.kappa_band,
            "raw_agreement": self.raw_agreement,
            "raw_agreement_note": self.raw_agreement_note,
            "position_consistency": self.position_consistency,
            "position_consistency_source": self.position_consistency_source,
            "length_bias_rate": self.length_bias_rate,
            "probe_position_bias_rate": self.probe_position_bias_rate,
            "self_preference": self.self_preference,
            "self_preference_note": self.self_preference_note,
            "verdict": self.verdict,
            "overall_color": self.overall_color,
            "kappa_color": self.kappa_color,
            "consistency_color": self.consistency_color,
            "length_bias_color": self.length_bias_color,
            "missing": list(self.missing),
            "judge_model": self.judge_model,
            "generator_models": list(self.generator_models),
            "live_question": self.live_question,
            "live_question_id": self.live_question_id,
            "calibration_n": self.calibration_n,
            "calibration_n_scored": self.calibration_n_scored,
            "disagreement_ids": list(self.disagreement_ids),
            "probe_n": self.probe_n,
            "probe_n_scored": self.probe_n_scored,
        }
