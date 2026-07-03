"""Lightweight character timeline extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .ingest import TEMPORAL_MARKERS
from .schemas import EventFrame, EvidenceChunk


CAPITALIZED_PHRASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
LOCATION_AFTER_PREP_RE = re.compile(r"\b(?:at|in|to|from|near)\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)")

INCOMPATIBLE_EVENT_ALIASES = {
    "movement": {"voyage", "movement"},
    "escape_release": {"escape", "rescue", "escape_release"},
    "public_appearance": {"public_appearance", "voyage", "family_relationship"},
    "marriage_relationship": {"family_relationship", "marriage_relationship"},
    "knowledge_secret": {"identity_secret", "knowledge_secret"},
    "rescue_help": {"rescue", "rescue_help"},
    "loyalty_help": {"rescue", "loyalty_help"},
    "death": {"violence", "death"},
    "custody": {"captivity", "custody"},
}


def build_timeline(chunks: list[EvidenceChunk]) -> dict[str, list[EventFrame]]:
    timeline: dict[str, list[EventFrame]] = {}
    for chunk in chunks:
        for character in chunk.character_mentions:
            for event_type in chunk.event_mentions or ("mention",):
                frame = EventFrame(
                    frame_id=f"{chunk.chunk_id}:{_slug(character)}:{_slug(event_type)}",
                    book_name=chunk.book_name,
                    character=character,
                    event_type=event_type,
                    surface_text=_surface_text(chunk.text, character),
                    chunk_id=chunk.chunk_id,
                    chapter_id=chunk.chapter_id,
                    chunk_index_global=chunk.chunk_index_global,
                    temporal_markers=detect_temporal_markers(chunk.text),
                    location_markers=detect_location_markers(chunk.text),
                    confidence=0.7 if event_type != "mention" else 0.45,
                )
                timeline.setdefault(character.casefold(), []).append(frame)
    for frames in timeline.values():
        frames.sort(key=lambda frame: (frame.chunk_index_global, frame.frame_id))
    return timeline


def get_events_for_character(timeline: dict[str, list[EventFrame]], character: str) -> list[EventFrame]:
    return list(timeline.get(character.casefold(), ()))


def find_events_after(timeline: dict[str, list[EventFrame]], character: str, chunk_index: int) -> list[EventFrame]:
    return [frame for frame in get_events_for_character(timeline, character) if frame.chunk_index_global > chunk_index]


def find_incompatible_events(
    timeline: dict[str, list[EventFrame]],
    character: str,
    incompatible_event_types: tuple[str, ...] | list[str],
) -> list[EventFrame]:
    aliases: set[str] = set()
    for event_type in incompatible_event_types:
        aliases.update(INCOMPATIBLE_EVENT_ALIASES.get(str(event_type), {str(event_type)}))
    aliases = {alias.casefold() for alias in aliases}
    return [frame for frame in get_events_for_character(timeline, character) if frame.event_type.casefold() in aliases]


def save_timeline_json(path: str | Path, timeline: dict[str, list[EventFrame]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {character: [frame.to_dict() for frame in frames] for character, frames in timeline.items()}
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return output_path


def load_timeline_json(path: str | Path) -> dict[str, list[EventFrame]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        character: [
            EventFrame(
                frame_id=item["frame_id"],
                book_name=item["book_name"],
                character=item["character"],
                event_type=item["event_type"],
                surface_text=item["surface_text"],
                chunk_id=item["chunk_id"],
                chapter_id=item.get("chapter_id"),
                chunk_index_global=item["chunk_index_global"],
                temporal_markers=tuple(item.get("temporal_markers") or ()),
                location_markers=tuple(item.get("location_markers") or ()),
                confidence=item.get("confidence", 0.5),
            )
            for item in frames
        ]
        for character, frames in raw.items()
    }


def detect_temporal_markers(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    return tuple(marker for marker in TEMPORAL_MARKERS if marker in lowered)


def detect_location_markers(text: str) -> tuple[str, ...]:
    markers: list[str] = []
    for match in LOCATION_AFTER_PREP_RE.findall(text):
        if match not in markers:
            markers.append(match)
    for phrase in CAPITALIZED_PHRASE_RE.findall(text):
        if phrase not in markers:
            markers.append(phrase)
    return tuple(markers[:12])


def _surface_text(text: str, character: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if character.casefold() in sentence.casefold():
            return sentence[:320]
    return text.strip()[:320]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "x"

