"""Deterministic local book ingestion and chunk indexing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .schemas import EvidenceChunk


KNOWN_CHARACTERS = (
    "Thalcave",
    "Faria",
    "Kai-Koumou",
    "Noirtier",
    "Tom Ayrton",
    "Ben Joyce",
    "Jacques Paganel",
)

EVENT_ONTOLOGY: Mapping[str, tuple[str, ...]] = {
    "custody": (
        "arrest",
        "arrested",
        "captured",
        "captive",
        "captivity",
        "confined",
        "imprisoned",
        "imprisonment",
        "prison",
        "dungeon",
        "cell",
        "detained",
        "prisoner",
    ),
    "captivity": (
        "arrest",
        "arrested",
        "captured",
        "captive",
        "imprisoned",
        "imprisonment",
        "prison",
        "dungeon",
        "cell",
        "detained",
    ),
    "escape": ("escape", "escaped", "fled", "flight", "slipped away", "liberated"),
    "escape_release": ("escape", "escaped", "released", "freed", "liberated", "no longer prisoner"),
    "movement": (
        "travelled",
        "traveled",
        "journeyed",
        "sailed",
        "arrived",
        "departed",
        "left",
        "went",
        "returned",
        "came",
        "rode",
        "walked",
        "marched",
    ),
    "public_appearance": ("appeared", "attended", "joined", "met", "visited", "presented", "assembly", "public"),
    "later_action": ("later", "afterwards", "subsequently", "returned", "spoke", "wrote", "confessed", "appeared"),
    "death": ("died", "dead", "death", "killed", "murdered", "poisoned", "executed", "slain"),
    "betrayal": ("betray", "betrayed", "treason", "traitor", "denounced", "informer"),
    "violence": ("attack", "attacked", "killed", "murdered", "stabbed", "shot", "wounded", "poisoned"),
    "rescue": ("rescue", "rescued", "saved", "delivered", "freed"),
    "voyage": ("voyage", "sailed", "ship", "aboard", "coast", "sea", "journey", "travelled", "traveled"),
    "political_intrigue": ("conspiracy", "plot", "royalist", "bonapartist", "minister", "procureur", "spy"),
    "family_relationship": ("father", "mother", "son", "daughter", "sister", "brother", "married", "betrothed"),
    "identity_secret": ("disguise", "alias", "secret", "hidden", "unknown", "revealed", "confessed"),
    "wealth_property": ("treasure", "fortune", "inheritance", "property", "estate", "money"),
}

TEMPORAL_MARKERS = (
    "before",
    "after",
    "later",
    "earlier",
    "then",
    "when",
    "while",
    "during",
    "soon",
    "once",
    "years",
    "months",
    "days",
    "night",
    "morning",
    "evening",
)

CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:chapter)\s+([0-9]+|[ivxlcdm]+)\.?\s*(?:[-:.]\s*)?(.*\S)?\s*$",
    re.IGNORECASE,
)
GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9']+")
CAPITALIZED_PHRASE_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+(?:of|the|de|du|da|del|[A-Z][a-z]+))*\b"
)
DIALOGUE_MARKER_RE = re.compile(r'["“”]|\b(?:said|replied|asked|answered|cried|exclaimed|whispered)\b', re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class BookText:
    """A loaded source novel."""

    book_name: str
    path: Path
    text: str


@dataclass(frozen=True, slots=True)
class Chapter:
    """A detected chapter span."""

    chapter_index: int
    chapter_title: str | None
    text: str


def read_book_texts(books_dir: str | Path) -> list[BookText]:
    """Read all `.txt` books from a directory in stable filename order."""

    root = Path(books_dir)
    if not root.exists():
        raise FileNotFoundError(f"Books directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Books path is not a directory: {root}")

    books: list[BookText] = []
    for path in sorted(root.glob("*.txt"), key=lambda item: item.name.casefold()):
        text = path.read_text(encoding="utf-8", errors="ignore")
        books.append(BookText(book_name=path.stem, path=path, text=strip_gutenberg_boilerplate(text)))
    return books


def split_into_chapters(text: str) -> list[Chapter]:
    """Split a book into chapters using conservative chapter-heading detection."""

    cleaned = strip_gutenberg_boilerplate(text)
    lines = cleaned.splitlines()
    markers: list[tuple[int, str | None]] = []

    for index, line in enumerate(lines):
        match = CHAPTER_HEADING_RE.match(line.strip())
        if not match:
            continue
        trailing_title = (match.group(2) or "").strip(" .:-")
        chapter_title = trailing_title or line.strip()
        markers.append((index, chapter_title))

    if not markers:
        body = cleaned.strip()
        return [Chapter(chapter_index=1, chapter_title=None, text=body)] if body else []

    chapters: list[Chapter] = []
    for chapter_index, (line_index, chapter_title) in enumerate(markers, start=1):
        next_line_index = markers[chapter_index][0] if chapter_index < len(markers) else len(lines)
        body_start = line_index + 1
        if _is_generic_chapter_title(chapter_title):
            detected_title, title_line_index = _detect_following_title(lines, body_start, next_line_index)
            if detected_title is not None and title_line_index is not None:
                chapter_title = detected_title
                body_start = title_line_index + 1
        body = "\n".join(lines[body_start:next_line_index]).strip()
        if not body:
            continue
        chapters.append(Chapter(chapter_index=len(chapters) + 1, chapter_title=chapter_title, text=body))

    return chapters or ([Chapter(chapter_index=1, chapter_title=None, text=cleaned.strip())] if cleaned.strip() else [])


def chunk_chapter(chapter_text: str, chunk_size: int = 400, overlap: int = 100) -> list[str]:
    """Split a chapter into deterministic overlapping word chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = chapter_text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def build_chunks(
    books_dir: str | Path,
    *,
    chunk_size: int = 400,
    overlap: int = 100,
) -> list[EvidenceChunk]:
    """Build deterministic evidence chunks with metadata for all books."""

    chunks: list[EvidenceChunk] = []
    global_index = 0
    for book in read_book_texts(books_dir):
        book_slug = slugify(book.book_name)
        chapters = split_into_chapters(book.text)
        for chapter in chapters:
            chapter_id = f"{book_slug}:ch{chapter.chapter_index:04d}"
            for chunk_index_in_chapter, chunk_text in enumerate(
                chunk_chapter(chapter.text, chunk_size=chunk_size, overlap=overlap),
                start=1,
            ):
                global_index += 1
                character_mentions = detect_character_mentions(chunk_text)
                event_mentions = detect_event_mentions(chunk_text)
                density = plot_density_score(chunk_text)
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"{chapter_id}:chunk{chunk_index_in_chapter:04d}",
                        book_name=book.book_name,
                        chunk_text=chunk_text,
                        source_path=str(book.path),
                        chapter_id=chapter_id,
                        chapter_title=chapter.chapter_title,
                        chunk_index_global=global_index,
                        chunk_index_in_chapter=chunk_index_in_chapter,
                        character_mentions=character_mentions,
                        event_mentions=event_mentions,
                        plot_density_score=density,
                        metadata=plot_density_features(chunk_text),
                    )
                )
    return chunks


def detect_character_mentions(text: str) -> tuple[str, ...]:
    """Return known character names mentioned in text, preserving ontology order."""

    mentions: list[str] = []
    for character in KNOWN_CHARACTERS:
        if _count_phrase(text, character) > 0:
            mentions.append(character)
    return tuple(mentions)


def detect_event_mentions(text: str) -> tuple[str, ...]:
    """Return event ontology categories detected in text."""

    lowered = text.casefold()
    mentions: list[str] = []
    for event_name, forms in EVENT_ONTOLOGY.items():
        if any(_phrase_in_text(lowered, form.casefold()) for form in forms):
            mentions.append(event_name)
    return tuple(mentions)


def plot_density_score(text: str) -> float:
    """Score how narratively dense a chunk is using transparent heuristics."""

    features = plot_density_features(text)
    return round(
        2.0 * features["character_mention_count"]
        + 1.5 * features["event_mention_count"]
        + 1.0 * features["temporal_marker_count"]
        + 0.5 * features["capitalized_phrase_count"]
        + 0.75 * features["dialogue_marker_count"],
        3,
    )


def plot_density_features(text: str) -> dict[str, int]:
    """Return component counts used by `plot_density_score`."""

    return {
        "character_mention_count": count_character_mentions(text),
        "event_mention_count": count_event_mentions(text),
        "temporal_marker_count": count_temporal_markers(text),
        "capitalized_phrase_count": count_capitalized_phrases(text),
        "dialogue_marker_count": count_dialogue_markers(text),
    }


def count_character_mentions(text: str) -> int:
    return sum(_count_phrase(text, character) for character in KNOWN_CHARACTERS)


def count_event_mentions(text: str) -> int:
    lowered = text.casefold()
    total = 0
    for forms in EVENT_ONTOLOGY.values():
        total += sum(_count_phrase(lowered, form.casefold()) for form in forms)
    return total


def count_temporal_markers(text: str) -> int:
    lowered = text.casefold()
    return sum(_count_phrase(lowered, marker) for marker in TEMPORAL_MARKERS)


def count_capitalized_phrases(text: str) -> int:
    phrases = CAPITALIZED_PHRASE_RE.findall(text)
    return sum(1 for phrase in phrases if len(phrase) > 1)


def count_dialogue_markers(text: str) -> int:
    return len(DIALOGUE_MARKER_RE.findall(text))


def write_chunks_jsonl(chunks: Iterable[EvidenceChunk], output_path: str | Path) -> Path:
    """Write chunk index records to JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk_to_json_record(chunk), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def chunk_to_json_record(chunk: EvidenceChunk) -> dict[str, object]:
    """Serialize an `EvidenceChunk` with stable public JSON keys."""

    return {
        "chunk_id": chunk.chunk_id,
        "book_name": chunk.book_name,
        "chapter_id": chunk.chapter_id,
        "chapter_title": chunk.chapter_title,
        "chunk_index_global": chunk.chunk_index_global,
        "chunk_index_in_chapter": chunk.chunk_index_in_chapter,
        "text": chunk.text,
        "character_mentions": list(chunk.character_mentions),
        "event_mentions": list(chunk.event_mentions),
        "plot_density_score": chunk.plot_density_score,
        "source_path": chunk.source_path,
        "metadata": dict(chunk.metadata),
    }


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg boilerplate when start/end markers exist."""

    start_match = GUTENBERG_START_RE.search(text)
    end_match = GUTENBERG_END_RE.search(text)
    if start_match:
        text = text[start_match.end() :]
    if end_match:
        text = text[: end_match.start()]
    return text.strip()


def slugify(value: str) -> str:
    """Create a stable lowercase identifier segment."""

    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "book"


def normalize_book_name(name: str) -> str:
    """Normalize book names used across CSVs, text files, and vector stores."""

    text = Path(str(name).strip().strip('"')).stem
    return " ".join(text.split()).casefold()


def _is_generic_chapter_title(title: str | None) -> bool:
    return bool(title and CHAPTER_HEADING_RE.match(title.strip()))


def _detect_following_title(lines: list[str], start: int, end: int) -> tuple[str | None, int | None]:
    for index in range(start, end):
        candidate = lines[index].strip()
        if not candidate:
            continue
        if _looks_like_chapter_title(candidate):
            return candidate, index
        return None, None
    return None, None


def _looks_like_chapter_title(value: str) -> bool:
    words = WORD_RE.findall(value)
    if not words or len(words) > 12 or len(value) > 100:
        return False
    letters = [char for char in value if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase_ratio >= 0.75 or value.istitle()


def split_chapters(text: str) -> list[tuple[str, str]]:
    """Backward-compatible wrapper around `split_into_chapters`."""

    return [((chapter.chapter_title or f"chapter_{chapter.chapter_index}"), chapter.text) for chapter in split_into_chapters(text)]


def chunk_words(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Backward-compatible wrapper around `chunk_chapter`."""

    return chunk_chapter(text, chunk_size=chunk_size, overlap=chunk_overlap)


def chunks_from_book(path: str | Path, *, chunk_size: int, chunk_overlap: int) -> list[EvidenceChunk]:
    """Build chunks for a single book path."""

    book_path = Path(path)
    text = book_path.read_text(encoding="utf-8", errors="ignore")
    chunks: list[EvidenceChunk] = []
    book_slug = slugify(book_path.stem)
    global_index = 0
    for chapter in split_into_chapters(text):
        chapter_id = f"{book_slug}:ch{chapter.chapter_index:04d}"
        for chunk_index_in_chapter, chunk_text in enumerate(
            chunk_chapter(chapter.text, chunk_size=chunk_size, overlap=chunk_overlap),
            start=1,
        ):
            global_index += 1
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{chapter_id}:chunk{chunk_index_in_chapter:04d}",
                    book_name=book_path.stem,
                    chunk_text=chunk_text,
                    source_path=str(book_path),
                    chapter_id=chapter_id,
                    chapter_title=chapter.chapter_title,
                    chunk_index_global=global_index,
                    chunk_index_in_chapter=chunk_index_in_chapter,
                    character_mentions=detect_character_mentions(chunk_text),
                    event_mentions=detect_event_mentions(chunk_text),
                    plot_density_score=plot_density_score(chunk_text),
                    metadata=plot_density_features(chunk_text),
                )
            )
    return chunks


def iter_book_paths(books_dir: str | Path) -> Iterable[Path]:
    """Yield `.txt` book files from a directory in stable order."""

    return sorted(Path(books_dir).glob("*.txt"), key=lambda item: item.name.casefold())


def _phrase_in_text(lowered_text: str, lowered_phrase: str) -> bool:
    return _count_phrase(lowered_text, lowered_phrase) > 0


def _count_phrase(text: str, phrase: str) -> int:
    if not phrase:
        return 0
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))
