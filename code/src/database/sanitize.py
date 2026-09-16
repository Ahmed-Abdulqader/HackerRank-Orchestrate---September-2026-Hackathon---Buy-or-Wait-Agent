import logging
import re

logger = logging.getLogger(__name__)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"[ \t]+")


def sanitize_text(value: object) -> object:
    """
    Basic, dependency-free text cleaner for untrusted CSV fields.

    - Leaves None/numbers unchanged.
    - Strips control characters and NUL bytes.
    - Collapses runs of spaces/tabs.
    - Trims leading/trailing whitespace.

    Does NOT attempt semantic prompt-injection defense; it only normalizes
    raw text so downstream parsers and the LLM see clean, inert input.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    text = _CONTROL_CHARS.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def sanitize_row(row: dict) -> dict:
    """Applies sanitize_text to every string field of a row dict."""
    if not isinstance(row, dict):
        return row
    cleaned = {}
    for key, val in row.items():
        cleaned[key] = sanitize_text(val)
    return cleaned