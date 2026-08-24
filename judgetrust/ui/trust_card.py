"""Prominent Trust Report card — the final assembled scoreboard."""

from __future__ import annotations

import streamlit as st

from judgetrust.models import TrustReport
from judgetrust.ui.styles import COLOR_HEX, color_span


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}"


def _num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _n_caption(n: int | None, scored: int | None, noun: str) -> str:
    if n is None:
        return f"No {noun} yet"
    if scored is not None and scored != n:
        return f"n={scored} scored / {n} {noun}"
    return f"n={n} {noun}"


def render_trust_card(report: TrustReport) -> None:
    """Always-on scoreboard assembled from persisted results."""

    accent = COLOR_HEX[report.overall_color]
    st.markdown(
        f'<div class="jt-card" style="--jt-accent:{accent}">'
        '<div style="font-size:0.8rem;font-weight:700;letter-spacing:0.06em;color:#6b7280;">'
        "JUDGE TRUST REPORT</div>"
        f'<p class="jt-verdict">{report.verdict}</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            color_span(
                "Prompt B win rate",
                report.overall_color if report.prompt_b_win_rate is not None else "gray",
            ),
            unsafe_allow_html=True,
        )
        win = _pct(report.prompt_b_win_rate)
        if report.n_b_wins is not None and report.n_models:
            win = f"{win}  ({report.n_b_wins}/{report.n_models})"
        st.markdown(f"### {win}")
        st.caption(
            _n_caption(report.n_models, None, "models")
            if report.n_models
            else "No live comparison yet"
        )
    with col2:
        st.markdown(color_span("Cohen's kappa", report.kappa_color), unsafe_allow_html=True)
        band = f" ({report.kappa_band})" if report.kappa_band else ""
        st.markdown(f"### {_num(report.kappa)}{band}")
        st.caption(_n_caption(report.calibration_n, report.calibration_n_scored, "labeled pairs"))
    with col3:
        st.markdown(
            color_span("Position consistency", report.consistency_color),
            unsafe_allow_html=True,
        )
        src = (
            f" · {report.position_consistency_source}"
            if report.position_consistency_source
            else ""
        )
        st.markdown(f"### {_pct(report.position_consistency)}{src}")
        st.caption("Share of comparisons stable across A-then-B and B-then-A.")
    with col4:
        st.markdown(color_span("Length bias", report.length_bias_color), unsafe_allow_html=True)
        st.markdown(f"### {_pct(report.length_bias_rate)}")
        st.caption(_n_caption(report.probe_n, report.probe_n_scored, "rigged pairs"))

    if report.live_question:
        qid = f" ({report.live_question_id})" if report.live_question_id else ""
        st.markdown(f"**Live question{qid}:** {report.live_question}")

    if report.per_model:
        chips = []
        for duel in report.per_model:
            stable = "stable" if duel.stable else "unstable"
            chips.append(
                f'<span class="jt-chip"><b>{duel.model}</b> · {duel.winner or "error"} · {stable}</span>'
            )
        st.markdown("".join(chips), unsafe_allow_html=True)

    if report.disagreement_ids:
        st.caption(
            "Judge disagreed with humans on "
            + ", ".join(report.disagreement_ids)
            + "."
        )

    if report.self_preference and report.self_preference_note:
        st.markdown(
            f'<div class="jt-caveat"><b>Self-preference caveat.</b> {report.self_preference_note}</div>',
            unsafe_allow_html=True,
        )

    limits = _limits_text(report)
    if limits:
        st.markdown(f'<div class="jt-limits"><b>How to read this.</b> {limits}</div>', unsafe_allow_html=True)

    if report.missing:
        st.caption("Missing sources: " + ", ".join(report.missing) + ".")
    elif report.raw_agreement is not None and report.raw_agreement_note:
        st.caption(
            f"Raw agreement {_pct(report.raw_agreement)}. {report.raw_agreement_note}"
        )


def _limits_text(report: TrustReport) -> str:
    """Plain-language bounds so the verdict is not taken as a proof."""

    parts: list[str] = []
    if report.n_models:
        parts.append(
            "Live is one question on "
            f"{report.n_models} generator model"
            f"{'s' if report.n_models != 1 else ''} — not a general claim that prompt B always wins."
        )
    if report.calibration_n:
        n_miss = len(report.disagreement_ids)
        miss = (
            f" The judge disagreed with humans on {n_miss} row"
            f"{'s' if n_miss != 1 else ''}."
            if n_miss
            else ""
        )
        parts.append(
            f"Kappa is vs {report.calibration_n} starter human labels."
            f"{miss} Review those labels; they are the ground truth."
        )
    if report.probe_n:
        parts.append(
            f"Length-bias is from {report.probe_n} rigged pairs. "
            "A 0% rate here is encouraging, not a guarantee."
        )
    if not parts:
        return ""
    return " ".join(parts)
