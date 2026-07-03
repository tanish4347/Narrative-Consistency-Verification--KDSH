"""CSV dataset loading and normalization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Optional

from .schemas import BackstoryExample


ALLOWED_LABELS = {"consistent", "contradict"}

COLUMN_ALIASES = {
    "id": ("id", "story_id", "example_id"),
    "book_name": ("book_name",),
    "character": ("character", "characters", "char"),
    "backstory": ("backstory", "content"),
    "caption": ("caption",),
    "label": ("label",),
}


def load_train_examples(path: str | Path) -> list[BackstoryExample]:
    """Load labeled training examples from CSV into the normalized schema."""

    return _load_examples(path, require_label=True)


def load_test_examples(path: str | Path) -> list[BackstoryExample]:
    """Load test examples from CSV; labels are optional and may be absent."""

    return _load_examples(path, require_label=False)


def _load_examples(path: str | Path, *, require_label: bool) -> list[BackstoryExample]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: CSV file is missing a header row")

        field_lookup = _field_lookup(reader.fieldnames)
        required = ("book_name", "character", "backstory")
        if require_label:
            required = (*required, "label")
        _validate_required_columns(csv_path, field_lookup, required)

        examples: list[BackstoryExample] = []
        for row_index, row in enumerate(reader, start=2):
            examples.append(_row_to_example(csv_path, row, field_lookup, row_index, require_label=require_label))
        return examples


def _field_lookup(fieldnames: list[str]) -> dict[str, str]:
    normalized_to_actual: dict[str, str] = {}
    for field in fieldnames:
        key = _normalize_column_name(field)
        if key and key not in normalized_to_actual:
            normalized_to_actual[key] = field
    return normalized_to_actual


def _normalize_column_name(name: str | None) -> str:
    return (name or "").strip().lstrip("\ufeff").lower()


def _resolve_column(field_lookup: Mapping[str, str], logical_name: str) -> Optional[str]:
    for alias in COLUMN_ALIASES[logical_name]:
        actual = field_lookup.get(alias)
        if actual is not None:
            return actual
    return None


def _validate_required_columns(path: Path, field_lookup: Mapping[str, str], required: tuple[str, ...]) -> None:
    missing = [name for name in required if _resolve_column(field_lookup, name) is None]
    if not missing:
        return

    available = ", ".join(sorted(field_lookup)) or "<none>"
    details = "; ".join(
        f"{name} (accepted: {', '.join(COLUMN_ALIASES[name])})"
        for name in missing
    )
    raise ValueError(f"{path}: missing required column(s): {details}. Available columns: {available}")


def _row_to_example(
    path: Path,
    row: Mapping[str, str],
    field_lookup: Mapping[str, str],
    row_index: int,
    *,
    require_label: bool,
) -> BackstoryExample:
    example_id = _optional_value(row, field_lookup, "id") or str(row_index - 1)
    book_name = _required_value(path, row, field_lookup, row_index, "book_name")
    character = _required_value(path, row, field_lookup, row_index, "character")
    backstory = _required_value(path, row, field_lookup, row_index, "backstory")
    caption = _optional_value(row, field_lookup, "caption")
    label = _load_label(path, row, field_lookup, row_index, require_label=require_label)

    return BackstoryExample(
        id=example_id,
        book_name=book_name,
        character=character,
        content=backstory,
        label=label,
        caption=caption,
        metadata={"source_path": str(path), "row_number": row_index},
    )


def _required_value(
    path: Path,
    row: Mapping[str, str],
    field_lookup: Mapping[str, str],
    row_index: int,
    logical_name: str,
) -> str:
    value = _optional_value(row, field_lookup, logical_name)
    if value is None:
        aliases = ", ".join(COLUMN_ALIASES[logical_name])
        raise ValueError(f"{path}: row {row_index} has empty required field `{logical_name}` ({aliases})")
    return value


def _optional_value(row: Mapping[str, str], field_lookup: Mapping[str, str], logical_name: str) -> Optional[str]:
    column = _resolve_column(field_lookup, logical_name)
    if column is None:
        return None
    raw = row.get(column)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _load_label(
    path: Path,
    row: Mapping[str, str],
    field_lookup: Mapping[str, str],
    row_index: int,
    *,
    require_label: bool,
) -> Optional[str]:
    label = _optional_value(row, field_lookup, "label")
    if label is None:
        if require_label:
            raise ValueError(f"{path}: row {row_index} has empty required field `label`")
        return None

    normalized = label.lower()
    if normalized not in ALLOWED_LABELS:
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        raise ValueError(f"{path}: row {row_index} has invalid label `{label}`; allowed labels: {allowed}")
    return normalized

