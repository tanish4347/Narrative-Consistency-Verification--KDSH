"""General JSON and text utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""

    return " ".join(str(text).split())


def stable_hash(obj: Any) -> str:
    """Return a deterministic SHA-256 hash for a JSON-like object."""

    raw = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def safe_json_loads(text: str) -> Any:
    """Parse JSON and raise a compact, caller-friendly error on failure."""

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first balanced JSON object in text."""

    parsed = _extract_json_value(text, "{", "}")
    if not isinstance(parsed, dict):
        raise ValueError("Extracted JSON value is not an object")
    return parsed


def extract_json_list(text: str) -> list[Any]:
    """Extract and parse the first balanced JSON list in text."""

    parsed = _extract_json_value(text, "[", "]")
    if not isinstance(parsed, list):
        raise ValueError("Extracted JSON value is not a list")
    return parsed


def _extract_json_value(text: str, opener: str, closer: str) -> Any:
    source = _strip_markdown_fences(str(text))
    starts = [index for index, char in enumerate(source) if char == opener]
    if not starts:
        raise ValueError(f"No JSON value starting with `{opener}` found")

    last_error: Exception | None = None
    for start in starts:
        end = _find_balanced_end(source, start, opener, closer)
        if end is None:
            last_error = ValueError(f"Malformed JSON: no balanced `{closer}` found")
            continue

        candidate = source[start : end + 1]
        try:
            return safe_json_loads(candidate)
        except ValueError as exc:
            last_error = exc

    if last_error is not None:
        raise ValueError(f"No valid JSON value found: {last_error}") from last_error
    raise ValueError(f"Malformed JSON: no balanced `{closer}` found")


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text


def _find_balanced_end(text: str, start: int, opener: str, closer: str) -> int | None:
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    return None
