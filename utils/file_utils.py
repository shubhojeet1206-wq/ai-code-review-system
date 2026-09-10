"""Source-file validation and language helpers."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

MAX_SOURCE_BYTES = 200_000
MAX_SOURCE_CHARS = 80_000

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C++",
    "C",
    "Go",
    "SQL",
)

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "Python",
    ".java": "Java",
    ".js": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".c": "C",
    ".h": "C",
    ".go": "Go",
    ".sql": "SQL",
}

ALLOWED_EXTENSIONS = frozenset(_EXTENSION_TO_LANGUAGE.keys())


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file is not a supported source type."""


class InvalidSourceFileError(ValueError):
    """Raised when file contents cannot be decoded as source code."""


def language_from_filename(filename: str) -> str | None:
    """Map a filename to a supported language, or None if unknown."""
    suffix = Path(filename).suffix.lower()
    return _EXTENSION_TO_LANGUAGE.get(suffix)


def with_line_numbers(source_code: str) -> str:
    """Prefix each line with a 1-based line number for agent prompts."""
    lines = source_code.splitlines()
    width = len(str(len(lines) or 1))
    return "\n".join(f"{index:>{width}} | {line}" for index, line in enumerate(lines, start=1))


def read_uploaded_source(uploaded_file: BinaryIO, filename: str) -> str:
    """Read and validate an uploaded source file.

    Args:
        uploaded_file: File-like object from Streamlit.
        filename: Original filename used for extension checks.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or 'unknown'}'. Allowed extensions: {allowed}."
        )

    raw = uploaded_file.read()
    if not raw:
        raise InvalidSourceFileError("The uploaded file is empty.")
    if len(raw) > MAX_SOURCE_BYTES:
        raise InvalidSourceFileError(
            f"File is too large ({len(raw)} bytes). Maximum allowed is {MAX_SOURCE_BYTES} bytes."
        )

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise InvalidSourceFileError("Could not decode the file as text.")

    if not text.strip():
        raise InvalidSourceFileError("The uploaded file contains only whitespace.")
    if len(text) > MAX_SOURCE_CHARS:
        raise InvalidSourceFileError(
            f"Source is too large ({len(text)} characters). Maximum allowed is {MAX_SOURCE_CHARS}."
        )
    return text
