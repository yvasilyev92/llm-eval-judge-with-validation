"""CLI: ``uv run python -m judgetrust.biasprobe``."""

from __future__ import annotations

import sys

from judgetrust.biasprobe.runner import format_report, persist_report, run_bias_probe
from judgetrust.llm import load_env, missing_api_key
from judgetrust.logging_setup import configure_logging


def main() -> int:
    """Run the bias probe, print the report, and persist JSON."""

    configure_logging()
    load_env()
    if missing_api_key():
        print(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1
    try:
        report = run_bias_probe()
    except Exception as exc:  # noqa: BLE001 — CLI must not dump a traceback
        print(f"Bias probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    path = persist_report(report)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
