"""Validated dataclass schemas for narrative consistency verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


SUPPORTED_CLAIM_VERDICTS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "ERROR"}
SUPPORTED_BACKSTORY_VERDICTS = {"CONSISTENT", "INCONSISTENT", "UNKNOWN", "ERROR"}
SUPPORTED_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
SUPPORTED_QUERY_TYPES = {"support", "contradiction", "implied_state", "timeline"}


def _require_text(name: str, value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tuple_of_text(name: str, values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    out = tuple(str(v).strip() for v in values if str(v).strip())
    return out


def _copy_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(values or {})


@dataclass(frozen=True, slots=True)
class BackstoryExample:
    """One generated backstory to verify against a source novel."""

    id: str
    book_name: str
    character: str
    content: str
    label: Optional[str] = None
    caption: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "book_name", _require_text("book_name", self.book_name))
        object.__setattr__(self, "character", _require_text("character", self.character))
        object.__setattr__(self, "content", _require_text("content", self.content))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "caption", _optional_text(self.caption))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @property
    def backstory(self) -> str:
        """Normalized alias for the source CSV `content` column."""

        return self.content


@dataclass(frozen=True, slots=True)
class Claim:
    """A factual, verifiable statement extracted from a backstory."""

    claim_id: int
    claim_text: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    source: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.claim_id) <= 0:
            raise ValueError("claim_id must be positive")
        object.__setattr__(self, "claim_id", int(self.claim_id))
        object.__setattr__(self, "claim_text", _require_text("claim_text", self.claim_text))
        object.__setattr__(self, "keywords", _tuple_of_text("keywords", self.keywords))
        object.__setattr__(self, "source", _optional_text(self.source))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ClaimPlan:
    """A collection of explicit claims and implied states for one backstory."""

    example_id: str
    claims: tuple[Claim, ...]
    implied_states: tuple[Claim, ...] = field(default_factory=tuple)
    notes: Optional[str] = None
    claim_id: Optional[int] = None
    claim_text: Optional[str] = None
    book_name: Optional[str] = None
    character: Optional[str] = None
    claim_type: str = "generic"
    expected_event_types: tuple[str, ...] = field(default_factory=tuple)
    incompatible_event_types: tuple[str, ...] = field(default_factory=tuple)
    support_queries: tuple[str, ...] = field(default_factory=tuple)
    incompatible_states: tuple[str, ...] = field(default_factory=tuple)
    contradiction_queries: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _require_text("example_id", self.example_id))
        claims = tuple(self.claims or ())
        implied_states = tuple(self.implied_states or ())
        if not claims:
            raise ValueError("claims must contain at least one Claim")
        if not all(isinstance(claim, Claim) for claim in claims):
            raise TypeError("claims must contain only Claim instances")
        if not all(isinstance(claim, Claim) for claim in implied_states):
            raise TypeError("implied_states must contain only Claim instances")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "implied_states", implied_states)
        object.__setattr__(self, "notes", _optional_text(self.notes))

        primary_claim = claims[0]
        claim_id = self.claim_id if self.claim_id is not None else primary_claim.claim_id
        if int(claim_id) <= 0:
            raise ValueError("claim_id must be positive")
        object.__setattr__(self, "claim_id", int(claim_id))
        object.__setattr__(self, "claim_text", _optional_text(self.claim_text) or primary_claim.claim_text)
        object.__setattr__(self, "book_name", _optional_text(self.book_name))
        object.__setattr__(self, "character", _optional_text(self.character))
        object.__setattr__(self, "claim_type", _require_text("claim_type", self.claim_type))
        object.__setattr__(self, "expected_event_types", _tuple_of_text("expected_event_types", self.expected_event_types))
        object.__setattr__(self, "incompatible_event_types", _tuple_of_text("incompatible_event_types", self.incompatible_event_types))
        object.__setattr__(self, "support_queries", _tuple_of_text("support_queries", self.support_queries))
        object.__setattr__(self, "incompatible_states", _tuple_of_text("incompatible_states", self.incompatible_states))
        object.__setattr__(self, "contradiction_queries", _tuple_of_text("contradiction_queries", self.contradiction_queries))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @property
    def primary_claim(self) -> Claim:
        return self.claims[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "book_name": self.book_name,
            "character": self.character,
            "claim_type": self.claim_type,
            "expected_event_types": list(self.expected_event_types),
            "incompatible_event_types": list(self.incompatible_event_types),
            "support_queries": list(self.support_queries),
            "incompatible_states": list(self.incompatible_states),
            "contradiction_queries": list(self.contradiction_queries),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """One evidence search query produced from a claim or implied state."""

    claim_id: int
    query_text: str
    query_type: str
    book_name: Optional[str] = None
    character: Optional[str] = None
    filters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.claim_id) <= 0:
            raise ValueError("claim_id must be positive")
        query_type = _require_text("query_type", self.query_type).lower()
        if query_type not in SUPPORTED_QUERY_TYPES:
            raise ValueError(f"query_type must be one of {sorted(SUPPORTED_QUERY_TYPES)}")
        object.__setattr__(self, "claim_id", int(self.claim_id))
        object.__setattr__(self, "query_text", _require_text("query_text", self.query_text))
        object.__setattr__(self, "query_type", query_type)
        object.__setattr__(self, "book_name", _optional_text(self.book_name))
        object.__setattr__(self, "character", _optional_text(self.character))
        object.__setattr__(self, "filters", _copy_mapping(self.filters))


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    """A retrieved quote-bearing passage from a source novel."""

    chunk_id: str
    book_name: str
    chunk_text: str
    score: float = 0.0
    source_path: Optional[str] = None
    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None
    chunk_index_global: int = 0
    chunk_index_in_chapter: int = 0
    character_mentions: tuple[str, ...] = field(default_factory=tuple)
    event_mentions: tuple[str, ...] = field(default_factory=tuple)
    plot_density_score: float = 0.0
    character_hits: tuple[str, ...] = field(default_factory=tuple)
    event_hits: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _require_text("chunk_id", self.chunk_id))
        object.__setattr__(self, "book_name", _require_text("book_name", self.book_name))
        object.__setattr__(self, "chunk_text", _require_text("chunk_text", self.chunk_text))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "source_path", _optional_text(self.source_path))
        object.__setattr__(self, "chapter_id", _optional_text(self.chapter_id))
        object.__setattr__(self, "chapter_title", _optional_text(self.chapter_title))
        if int(self.chunk_index_global) < 0:
            raise ValueError("chunk_index_global cannot be negative")
        if int(self.chunk_index_in_chapter) < 0:
            raise ValueError("chunk_index_in_chapter cannot be negative")
        object.__setattr__(self, "chunk_index_global", int(self.chunk_index_global))
        object.__setattr__(self, "chunk_index_in_chapter", int(self.chunk_index_in_chapter))

        character_mentions = _tuple_of_text("character_mentions", self.character_mentions)
        event_mentions = _tuple_of_text("event_mentions", self.event_mentions)
        character_hits = _tuple_of_text("character_hits", self.character_hits) or character_mentions
        event_hits = _tuple_of_text("event_hits", self.event_hits) or event_mentions
        character_mentions = character_mentions or character_hits
        event_mentions = event_mentions or event_hits
        object.__setattr__(self, "character_mentions", character_mentions)
        object.__setattr__(self, "event_mentions", event_mentions)
        object.__setattr__(self, "plot_density_score", float(self.plot_density_score))
        object.__setattr__(self, "character_hits", character_hits)
        object.__setattr__(self, "event_hits", event_hits)
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @property
    def text(self) -> str:
        """Normalized alias used by the JSONL chunk index."""

        return self.chunk_text


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """Verifier decision for one claim."""

    claim_id: int
    verdict: str
    evidence: tuple[EvidenceChunk, ...] = field(default_factory=tuple)
    confidence: str = "LOW"
    justification: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.claim_id) <= 0:
            raise ValueError("claim_id must be positive")
        verdict = _require_text("verdict", self.verdict).upper()
        confidence = _require_text("confidence", self.confidence).upper()
        if verdict not in SUPPORTED_CLAIM_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(SUPPORTED_CLAIM_VERDICTS)}")
        if confidence not in SUPPORTED_CONFIDENCE:
            raise ValueError(f"confidence must be one of {sorted(SUPPORTED_CONFIDENCE)}")
        evidence = tuple(self.evidence or ())
        if not all(isinstance(chunk, EvidenceChunk) for chunk in evidence):
            raise TypeError("evidence must contain only EvidenceChunk instances")
        object.__setattr__(self, "claim_id", int(self.claim_id))
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "confidence", confidence)
        if verdict == "SUPPORTED" and not evidence:
            raise ValueError("SUPPORTED verdicts require evidence")
        if verdict == "CONTRADICTED" and confidence in {"HIGH", "MEDIUM"} and not evidence:
            raise ValueError("HIGH/MEDIUM CONTRADICTED verdicts require evidence")
        object.__setattr__(self, "justification", _require_text("justification", self.justification))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class BackstoryVerdict:
    """Aggregated decision for a complete backstory."""

    example_id: str
    verdict: str
    claim_verdicts: tuple[ClaimVerdict, ...]
    summary: Mapping[str, Any] = field(default_factory=dict)
    final_label: Optional[str] = None
    internal_status: Optional[str] = None

    def __post_init__(self) -> None:
        verdict = _require_text("verdict", self.verdict).upper()
        if verdict not in SUPPORTED_BACKSTORY_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(SUPPORTED_BACKSTORY_VERDICTS)}")
        claim_verdicts = tuple(self.claim_verdicts or ())
        if not claim_verdicts:
            raise ValueError("claim_verdicts must contain at least one ClaimVerdict")
        if not all(isinstance(item, ClaimVerdict) for item in claim_verdicts):
            raise TypeError("claim_verdicts must contain only ClaimVerdict instances")
        object.__setattr__(self, "example_id", _require_text("example_id", self.example_id))
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "claim_verdicts", claim_verdicts)
        object.__setattr__(self, "summary", _copy_mapping(self.summary))
        object.__setattr__(self, "internal_status", (_optional_text(self.internal_status) or verdict).upper())
        object.__setattr__(self, "final_label", _optional_text(self.final_label))


@dataclass(frozen=True, slots=True)
class ContradictionFamily:
    """A family of evidence patterns that can make a claim false."""

    family_name: str
    natural_language_query: str
    event_types: tuple[str, ...]
    explanation: str
    claim_id: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_name", _require_text("family_name", self.family_name))
        object.__setattr__(self, "natural_language_query", _require_text("natural_language_query", self.natural_language_query))
        object.__setattr__(self, "event_types", _tuple_of_text("event_types", self.event_types))
        object.__setattr__(self, "explanation", _require_text("explanation", self.explanation))
        if self.claim_id is not None and int(self.claim_id) <= 0:
            raise ValueError("claim_id must be positive when provided")
        object.__setattr__(self, "claim_id", None if self.claim_id is None else int(self.claim_id))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "natural_language_query": self.natural_language_query,
            "event_types": list(self.event_types),
            "explanation": self.explanation,
            "claim_id": self.claim_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieved chunk plus query/channel provenance and explainable scores."""

    chunk_id: str
    book_name: str
    chapter_id: Optional[str]
    text: str
    query: str
    channel: str
    final_score: float
    score_breakdown: Mapping[str, Any]
    source: str = "retrieval"
    claim_id: Optional[int] = None
    rank: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _require_text("chunk_id", self.chunk_id))
        object.__setattr__(self, "book_name", _require_text("book_name", self.book_name))
        object.__setattr__(self, "chapter_id", _optional_text(self.chapter_id))
        object.__setattr__(self, "text", _require_text("text", self.text))
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "channel", _require_text("channel", self.channel))
        object.__setattr__(self, "final_score", float(self.final_score))
        object.__setattr__(self, "score_breakdown", _copy_mapping(self.score_breakdown))
        object.__setattr__(self, "source", _require_text("source", self.source))
        if self.claim_id is not None and int(self.claim_id) <= 0:
            raise ValueError("claim_id must be positive when provided")
        if self.rank is not None and int(self.rank) <= 0:
            raise ValueError("rank must be positive when provided")
        object.__setattr__(self, "claim_id", None if self.claim_id is None else int(self.claim_id))
        object.__setattr__(self, "rank", None if self.rank is None else int(self.rank))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "book_name": self.book_name,
            "chapter_id": self.chapter_id,
            "text": self.text,
            "query": self.query,
            "channel": self.channel,
            "final_score": self.final_score,
            "score_breakdown": dict(self.score_breakdown),
            "source": self.source,
            "claim_id": self.claim_id,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EventFrame:
    """Lightweight timeline event frame extracted from an evidence chunk."""

    frame_id: str
    book_name: str
    character: str
    event_type: str
    surface_text: str
    chunk_id: str
    chapter_id: Optional[str]
    chunk_index_global: int
    temporal_markers: tuple[str, ...] = field(default_factory=tuple)
    location_markers: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _require_text("frame_id", self.frame_id))
        object.__setattr__(self, "book_name", _require_text("book_name", self.book_name))
        object.__setattr__(self, "character", _require_text("character", self.character))
        object.__setattr__(self, "event_type", _require_text("event_type", self.event_type))
        object.__setattr__(self, "surface_text", _require_text("surface_text", self.surface_text))
        object.__setattr__(self, "chunk_id", _require_text("chunk_id", self.chunk_id))
        object.__setattr__(self, "chapter_id", _optional_text(self.chapter_id))
        if int(self.chunk_index_global) < 0:
            raise ValueError("chunk_index_global cannot be negative")
        object.__setattr__(self, "chunk_index_global", int(self.chunk_index_global))
        object.__setattr__(self, "temporal_markers", _tuple_of_text("temporal_markers", self.temporal_markers))
        object.__setattr__(self, "location_markers", _tuple_of_text("location_markers", self.location_markers))
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "book_name": self.book_name,
            "character": self.character,
            "event_type": self.event_type,
            "surface_text": self.surface_text,
            "chunk_id": self.chunk_id,
            "chapter_id": self.chapter_id,
            "chunk_index_global": self.chunk_index_global,
            "temporal_markers": list(self.temporal_markers),
            "location_markers": list(self.location_markers),
            "confidence": self.confidence,
        }
