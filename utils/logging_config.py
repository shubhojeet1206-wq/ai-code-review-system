"""Application logging without secrets or full source dumps."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

_CONFIGURED = False

_SECRET_PATTERNS = (
    re.compile(r"(GOOGLE_API_KEY\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(Bearer\s+)\S+", re.IGNORECASE),
    re.compile(r"(AIza[0-9A-Za-z\-_]{20,})"),
)


def redact_secrets(text: str) -> str:
    """Remove API keys and similar secrets from log or UI error text."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(r"\1[REDACTED]", redacted)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key in redacted:
        redacted = redacted.replace(api_key, "[REDACTED]")
    return redacted


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once. Safe to call from Streamlit reruns."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
