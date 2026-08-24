"""Streamlit app: Trust Report plus three mutually exclusive run tabs."""

from __future__ import annotations

import streamlit as st

from judgetrust.config import get_settings
from judgetrust.llm import load_env
from judgetrust.logging_setup import configure_logging
from judgetrust.report.assemble import assemble_trust_report
from judgetrust.ui.styles import CSS, DISCLAIMER
from judgetrust.ui.trust_card import render_trust_card
from judgetrust.ui.views import (
    render_calibrate_tab,
    render_live_tab,
    render_probe_tab,
)


def run() -> None:
    """Render the Judge Trust UI. Called from the repo-root ``app.py``."""

    configure_logging()
    load_env()
    st.set_page_config(page_title="Judge Trust", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    settings = get_settings()
    with st.sidebar:
        st.header("Config")
        st.caption("Edit `judgetrust/config.py` to change models or prompts.")
        st.write("**Judge**")
        st.code(settings.judge_model, language=None)
        st.write("**Generators**")
        for model in settings.generator_models:
            st.code(model, language=None)
        st.write("**Prompt A**")
        st.write(settings.prompt_a)
        st.write("**Prompt B**")
        st.write(settings.prompt_b)

    st.title("Judge Trust")
    st.caption(
        "Compare a plain health prompt vs a safety-tuned one across several LLMs, "
        "then prove how much the separate judge can be trusted."
    )
    st.markdown(f'<div class="jt-banner">{DISCLAIMER}</div>', unsafe_allow_html=True)

    if "trust" not in st.session_state:
        st.session_state["trust"] = assemble_trust_report()
    render_trust_card(st.session_state["trust"])

    calibrate_tab, live_tab, probe_tab = st.tabs(
        ["Calibrate judge", "Compare prompts (live)", "Run bias probe"]
    )
    with calibrate_tab:
        render_calibrate_tab()
    with live_tab:
        render_live_tab()
    with probe_tab:
        render_probe_tab()
