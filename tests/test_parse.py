"""Defensive JSON parsing for judge output."""

from __future__ import annotations

from judgetrust.judge.parse import extract_json_object, parse_judge_output, parse_winner
from judgetrust.logging_setup import redact_secrets


def test_extract_clean_json() -> None:
    payload = extract_json_object('{"winner": "1", "reason": "safer"}')
    assert payload == {"winner": "1", "reason": "safer"}


def test_extract_fenced_json() -> None:
    text = 'Here you go:\n```json\n{"winner": "2", "reason": "ok"}\n```\n'
    payload = extract_json_object(text)
    assert payload is not None
    assert payload["winner"] == "2"


def test_extract_prose_then_json() -> None:
    text = 'Reasoning: answer 1 is safer.\n{"winner": "1", "reason": "safety", "confidence": 0.8}'
    payload = extract_json_object(text)
    assert payload is not None
    assert payload["winner"] == "1"


def test_extract_empty_and_garbage() -> None:
    assert extract_json_object("") is None
    assert extract_json_object("no json here") is None
    assert extract_json_object("{not json") is None


def test_parse_valid_verdict() -> None:
    raw = """
    {
      "winner": "1",
      "reason": "Safer guidance",
      "confidence": 0.9,
      "rubric": {
        "medical_safety": 5,
        "factual_accuracy": 4,
        "risk_flagging": 5,
        "directness": 4
      }
    }
    """
    winner, reason, confidence, rubric, error = parse_judge_output(raw)
    assert error is None
    assert winner == "1"
    assert reason == "Safer guidance"
    assert confidence == 0.9
    assert rubric is not None
    assert rubric.medical_safety == 5


def test_parse_malformed_is_error_not_exception() -> None:
    winner, reason, confidence, rubric, error = parse_judge_output("the winner is 1")
    assert winner is None
    assert error == "malformed_or_missing_json"
    assert confidence == 0.0
    assert rubric is None
    assert reason == ""


def test_parse_invalid_winner() -> None:
    winner, _, _, _, error = parse_judge_output('{"winner": "C", "reason": "nope"}')
    assert winner is None
    assert error == "invalid_winner"


def test_parse_winner_aliases() -> None:
    assert parse_winner("A") == "1"
    assert parse_winner("b") == "2"
    assert parse_winner(1) == "1"
    assert parse_winner("tie") == "tie"
    assert parse_winner("maybe") is None


def test_confidence_percent_and_clamp() -> None:
    _, _, confidence, _, error = parse_judge_output(
        '{"winner": "tie", "reason": "close", "confidence": 80}'
    )
    assert error is None
    assert confidence == 0.8

    _, _, clipped, _, err2 = parse_judge_output(
        '{"winner": "1", "reason": "x", "confidence": 2.5}'
    )
    assert err2 is None
    assert clipped == 1.0

    _, _, already_unit, _, err3 = parse_judge_output(
        '{"winner": "1", "reason": "x", "confidence": 0.4}'
    )
    assert err3 is None
    assert already_unit == 0.4


def test_redact_secrets() -> None:
    text = "OPENAI_API_KEY=sk-abcDEF1234567890 authorization: Bearer secret"
    redacted = redact_secrets(text)
    assert "sk-abcDEF1234567890" not in redacted
    assert "Bearer secret" not in redacted
    assert "***" in redacted
