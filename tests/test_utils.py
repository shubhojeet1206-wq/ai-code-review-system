"""Tests for file helpers and secret redaction."""

from io import BytesIO

import pytest

from utils.file_utils import (
    InvalidSourceFileError,
    UnsupportedFileTypeError,
    language_from_filename,
    read_uploaded_source,
    with_line_numbers,
)
from utils.logging_config import redact_secrets


def test_language_from_filename() -> None:
    assert language_from_filename("app.py") == "Python"
    assert language_from_filename("Main.java") == "Java"
    assert language_from_filename("query.SQL") == "SQL"
    assert language_from_filename("notes.txt") is None


def test_with_line_numbers() -> None:
    numbered = with_line_numbers("a\nb")
    assert numbered.splitlines()[0].endswith("| a")
    assert "1" in numbered.splitlines()[0]
    assert numbered.splitlines()[1].endswith("| b")


def test_read_uploaded_source_success() -> None:
    handle = BytesIO(b"print('ok')\n")
    text = read_uploaded_source(handle, "demo.py")
    assert "print" in text


def test_read_uploaded_rejects_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        read_uploaded_source(BytesIO(b"x"), "notes.txt")


def test_read_uploaded_rejects_empty() -> None:
    with pytest.raises(InvalidSourceFileError):
        read_uploaded_source(BytesIO(b""), "demo.py")


def test_read_uploaded_rejects_whitespace() -> None:
    with pytest.raises(InvalidSourceFileError):
        read_uploaded_source(BytesIO(b"  \n  "), "demo.py")


def test_redact_google_api_key_pattern() -> None:
    text = redact_secrets("GOOGLE_API_KEY=AIzaSyDummyKeyValueThatLooksLongEnough")
    assert "AIza" not in text or "[REDACTED]" in text
    assert "[REDACTED]" in text


def test_parse_retry_after_from_google_message() -> None:
    from llm.client import _parse_retry_after, sanitize_error_message

    exc = RuntimeError(
        "429 Resource exhausted. Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 20, model: gemini-3.6-flash Please retry in 42.858s."
    )
    assert _parse_retry_after(exc) == pytest.approx(42.858)
    message = sanitize_error_message(exc)
    assert "20 requests/day" in message
    assert "retry in" in message
