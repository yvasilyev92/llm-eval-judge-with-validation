"""CLI: ``uv run python -m judgetrust.live --question "..."`` or ``--sample <id>``."""

from __future__ import annotations

import argparse
import sys

from judgetrust.live.questions import get_live_question
from judgetrust.live.runner import format_report, persist_report, run_live
from judgetrust.llm import load_env, missing_api_key
from judgetrust.logging_setup import configure_logging


def main(argv: list[str] | None = None) -> int:
    """Run one live comparison, print the report, and persist JSON."""

    configure_logging()
    parser = argparse.ArgumentParser(
        description="Compare prompt A vs B across generator models (live mode)."
    )
    parser.add_argument("--question", help="Health question to compare prompts on.")
    parser.add_argument(
        "--sample",
        help="id from data/live_questions.json (instead of --question).",
    )
    args = parser.parse_args(argv)
    if bool(args.question) == bool(args.sample):
        parser.error("provide exactly one of --question or --sample")

    question_id: str | None = None
    if args.sample:
        try:
            sample = get_live_question(args.sample)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        question = sample.question
        question_id = sample.id
    else:
        question = args.question

    load_env()
    if missing_api_key():
        print(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1
    try:
        report = run_live(question, question_id=question_id)
    except Exception as exc:  # noqa: BLE001 — CLI must not dump a traceback
        print(f"Live run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    path = persist_report(report)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
