"""Shared Streamlit styling for Judge Trust."""

from __future__ import annotations

from judgetrust.models import ColorName

COLOR_HEX: dict[ColorName, str] = {
    "green": "#15803d",
    "amber": "#b45309",
    "red": "#b91c1c",
    "gray": "#6b7280",
}

DISCLAIMER = (
    "This app exists to <b>test the judge</b>, not to give medical advice. "
    "Answers are generated only as evaluation inputs. Keep questions to general, "
    "well-known safety facts — never personalized dosing. Starter human labels "
    "in the calibration set <b>must be reviewed and expanded</b>; they are the ground truth."
)

CSS = """
<style>
.jt-banner {
    background: #fff7ed;
    border: 1px solid #fdba74;
    color: #7c2d12;
    padding: 0.85rem 1.1rem;
    border-radius: 10px;
    margin: 0 0 1.25rem 0;
    font-size: 0.95rem;
    line-height: 1.45;
}
.jt-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 6px solid var(--jt-accent, #6b7280);
    border-radius: 12px;
    padding: 1.2rem 1.4rem 1.35rem 1.4rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.jt-verdict {
    font-size: 1.18rem;
    font-weight: 650;
    line-height: 1.5;
    margin: 0.35rem 0 1rem 0;
    color: #111827;
}
.jt-chip {
    display: inline-block;
    border: 1px solid #e5e7eb;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    margin: 0.15rem 0.35rem 0.15rem 0;
    font-size: 0.85rem;
    background: #f9fafb;
}
.jt-muted { color: #6b7280; font-size: 0.9rem; }
.jt-caveat {
    background: #fef3c7;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin-top: 0.85rem;
    color: #78350f;
}
.jt-limits {
    background: #f3f4f6;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin-top: 0.85rem;
    color: #374151;
    font-size: 0.95rem;
    line-height: 1.45;
}
</style>
"""


def color_span(label: str, color: ColorName) -> str:
    hex_color = COLOR_HEX[color]
    return f'<span style="color:{hex_color};font-weight:700;">{label}</span>'
