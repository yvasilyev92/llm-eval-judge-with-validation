"""Mutually exclusive mode tabs. Each action has its own run button."""

from __future__ import annotations

import streamlit as st

from judgetrust.biasprobe.runner import persist_report as persist_probe
from judgetrust.biasprobe.runner import run_bias_probe
from judgetrust.calibrate.runner import persist_report as persist_calibration
from judgetrust.calibrate.runner import run_calibration
from judgetrust.live.questions import load_live_questions
from judgetrust.live.runner import persist_report as persist_live
from judgetrust.live.runner import run_live
from judgetrust.llm import load_env, missing_api_key
from judgetrust.report.assemble import assemble_trust_report


def refresh_trust() -> None:
    """Reload the Trust Report from disk into session state."""

    st.session_state["trust"] = assemble_trust_report()


def _api_ready() -> bool:
    load_env()
    if missing_api_key():
        st.error("Missing OPENAI_API_KEY. Copy `.env.example` to `.env` and add your key.")
        return False
    return True


def _run_guarded(label: str, action) -> None:
    if not _api_ready():
        return
    try:
        with st.spinner(label):
            action()
        refresh_trust()
        st.rerun()
    except Exception as exc:  # noqa: BLE001 — UI must not dump a traceback
        st.error(f"{type(exc).__name__}: {exc}")


def render_calibrate_tab() -> None:
    st.subheader("Calibrate judge")
    st.write(
        "Run the 3-judge panel on the human-labeled set. This never generates answers. "
        "Cohen's kappa is the headline — it corrects for chance, unlike raw agreement. "
        "About 210 judge calls (3 judges × both orderings × ~35 rows)."
    )
    if st.button("Run calibration", type="primary"):
        def _go() -> None:
            report = run_calibration()
            persist_calibration(report)
            st.session_state["calibration"] = report

        _run_guarded("Calibrating 3-judge panel (both orderings on every labeled row)…", _go)

    report = st.session_state.get("calibration")
    if report is None:
        st.caption("No calibration in this session yet. Persisted results still feed the Trust Report above.")
        return
    cols = st.columns(3)
    kappa_label = f"{report.kappa:.2f}" if report.kappa is not None else "n/a"
    cols[0].metric("Cohen's kappa (panel)", kappa_label, report.kappa_band)
    cols[1].metric(
        "Raw agreement",
        f"{report.raw_agreement:.0%}" if report.raw_agreement is not None else "n/a",
    )
    cols[2].metric("Position consistency", f"{report.position_consistency:.0%}")
    st.caption(report.raw_agreement_note)
    dissent = (
        f"{report.panel_dissent_rate:.0%}"
        if report.panel_dissent_rate is not None
        else "n/a"
    )
    st.caption(f"Panel dissent: {dissent} · disagreements with humans: {len(report.disagreements)}")
    if report.judge_kappas:
        st.dataframe(
            [item.to_dict() for item in report.judge_kappas],
            hide_index=True,
            use_container_width=True,
        )
    if report.disagreements:
        st.dataframe(
            [row.to_dict() for row in report.disagreements],
            hide_index=True,
            use_container_width=True,
        )


def render_live_tab() -> None:
    st.subheader("Compare prompts (live)")
    st.write(
        "Generate answer A (plain prompt) vs answer B (safety-tuned prompt) on each "
        "generator model, then the 3-judge panel compares both orderings "
        "(6 generator + 18 judge calls)."
    )
    samples = load_live_questions()
    options = ["(type your own)"] + [f"{item.id} — {item.question}" for item in samples]
    choice = st.selectbox("Sample question", options)
    if choice == "(type your own)":
        question = st.text_area("Health question", height=80)
        question_id = None
    else:
        selected = samples[options.index(choice) - 1]
        question = selected.question
        question_id = selected.id
        st.info(question)

    if st.button("Compare prompts", type="primary"):
        if not question.strip():
            st.warning("Enter a question or pick a sample.")
        else:
            def _go() -> None:
                live_report = run_live(question.strip(), question_id=question_id)
                persist_live(live_report)
                st.session_state["live"] = live_report

            _run_guarded("Generating answers and judging (6 generator + 18 judge calls)…", _go)

    live_report = st.session_state.get("live")
    if live_report is None:
        st.caption("No live comparison in this session yet.")
        return
    agg = st.columns(4)
    b_rate = (
        f"{live_report.prompt_b_win_rate:.0%}"
        if live_report.prompt_b_win_rate is not None
        else "n/a"
    )
    agree = (
        f"{live_report.cross_model_agreement:.0%}"
        if live_report.cross_model_agreement is not None
        else "n/a"
    )
    dissent = (
        f"{live_report.panel_dissent_rate:.0%}"
        if live_report.panel_dissent_rate is not None
        else "n/a"
    )
    agg[0].metric("Prompt B win rate", b_rate)
    agg[1].metric("Cross-model agreement", agree)
    agg[2].metric("Position consistency", f"{live_report.position_consistency:.0%}")
    agg[3].metric("Panel dissent", dissent)

    cols = st.columns(len(live_report.duels) or 1)
    for column, duel in zip(cols, live_report.duels, strict=False):
        with column:
            st.markdown(f"**{duel.model}**")
            st.write(f"Panel: `{duel.winner}` · stable: `{duel.stable}` · dissent: `{duel.dissent}`")
            if duel.votes:
                st.caption(
                    "Votes: "
                    + " · ".join(
                        f"{vote.model}={vote.winner or vote.error}" for vote in duel.votes
                    )
                )
            if duel.error:
                st.error(duel.error)
            with st.expander("Answer A (plain)"):
                st.write(duel.answer_a or "(empty)")
            with st.expander("Answer B (safety-tuned)"):
                st.write(duel.answer_b or "(empty)")


def render_probe_tab() -> None:
    st.subheader("Run bias probe")
    st.write(
        "Fixed rigged pairs: longer-but-worse vs shorter-but-correct. "
        "Length-bias is the share of times the panel picked the long worse answer. "
        "This never generates answers. About 60 judge calls (3 judges × both orderings × 10 pairs)."
    )
    if st.button("Run bias probe", type="primary"):
        def _go() -> None:
            probe_report = run_bias_probe()
            persist_probe(probe_report)
            st.session_state["probe"] = probe_report

        _run_guarded("Running bias probe (3-judge panel, both orderings on each rigged pair)…", _go)

    probe = st.session_state.get("probe")
    if probe is None:
        st.caption("No bias probe in this session yet.")
        return
    cols = st.columns(3)
    length = (
        f"{probe.length_bias_rate:.0%}" if probe.length_bias_rate is not None else "n/a"
    )
    dissent = (
        f"{probe.panel_dissent_rate:.0%}"
        if probe.panel_dissent_rate is not None
        else "n/a"
    )
    cols[0].metric("Length-bias rate (panel)", length, help="Lower is better.")
    cols[1].metric("Position-bias rate", f"{probe.position_bias_rate:.0%}")
    cols[2].metric("Panel dissent", dissent)
    if probe.judge_length_bias:
        st.dataframe(
            [item.to_dict() for item in probe.judge_length_bias],
            hide_index=True,
            use_container_width=True,
        )
    hits = [row.to_dict() for row in probe.rows if row.length_bias_hit or row.position_bias]
    if hits:
        st.dataframe(hits, hide_index=True, use_container_width=True)
