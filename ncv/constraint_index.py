"""Novel-derived constraint index and feature extraction."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .anchors import KNOWN_CHARACTERS, AnchorSet, extract_anchors
from .claim_planner import plan_claims
from .contradiction_planner import generate_contradiction_families
from .retrieval import SimpleBM25Retriever
from .schemas import BackstoryExample, EvidenceChunk


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def build_constraint_index(chunks: Iterable[EvidenceChunk]) -> dict[str, Any]:
    chunk_list = list(chunks)
    profiles: dict[str, dict[str, Any]] = {}
    for character in _canonical_targets():
        mentions = [chunk for chunk in chunk_list if _mentions_character(chunk, character)]
        profiles[character] = _profile_for_character(character, mentions, chunk_list)

    return {
        "characters": profiles,
        "cooccurrence": _cooccurrence_graph(chunk_list),
        "num_chunks": len(chunk_list),
    }


def save_constraint_index(index: Mapping[str, Any], path: str | Path = "outputs/constraint_index.json") -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_constraint_index(path: str | Path = "outputs/constraint_index.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_character_constraints(path: str | Path = "data/character_constraints.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data.get("characters") or {})


def extract_constraint_features(
    example: BackstoryExample,
    chunks: Iterable[EvidenceChunk],
    constraint_index: Mapping[str, Any],
) -> dict[str, float]:
    anchors = extract_anchors(example)
    target = _canonical_character(example.character)
    profile = dict((constraint_index.get("characters") or {}).get(target) or {})
    cooccurrence = constraint_index.get("cooccurrence") or {}
    chunk_list = list(chunks)
    book_chunks = [chunk for chunk in chunk_list if chunk.book_name.casefold() == example.book_name.casefold()]
    target_chunks = [chunk for chunk in book_chunks if _mentions_character(chunk, target)]

    same_chunk = 0
    same_chapter = 0
    missing_secondary = 0
    for secondary in anchors.secondary_characters:
        key = _pair_key(target, _canonical_character(secondary))
        pair = cooccurrence.get(key, {})
        same_chunk += int(pair.get("same_chunk", 0))
        same_chapter += int(pair.get("same_chapter", 0))
        if not pair.get("same_chunk") and not pair.get("same_chapter"):
            missing_secondary += 1

    claimed_dates = set(anchors.dates)
    claimed_locations = {_norm(item) for item in anchors.locations}
    claimed_roles = {_norm(item) for item in anchors.roles}
    profile_dates = set(profile.get("associated_dates", []))
    profile_locations = {_norm(item) for item in profile.get("associated_locations", [])}
    profile_roles = {_norm(item) for item in profile.get("associated_roles", [])}

    support_score = _max_query_score([example.backstory], target_chunks or book_chunks)
    contradiction_queries = _contradiction_queries(example)
    contradiction_score = _max_query_score(contradiction_queries, target_chunks or book_chunks)

    location_missing = _missing_ratio(claimed_locations, profile_locations)
    role_missing = _missing_ratio(claimed_roles, profile_roles)
    date_missing = _missing_ratio(claimed_dates, profile_dates)
    unsupported = _unsupported_specific_anchor_score(anchors, missing_secondary, location_missing, role_missing, date_missing)

    return {
        "num_secondary_anchors": float(len(anchors.secondary_characters)),
        "num_dates": float(len(anchors.dates)),
        "num_locations": float(len(anchors.locations)),
        "num_roles": float(len(anchors.roles)),
        "target_secondary_same_chunk_count": float(same_chunk),
        "target_secondary_same_chapter_count": float(same_chapter),
        "missing_secondary_anchor_count": float(missing_secondary),
        "claimed_date_mentioned_near_target": float(bool(claimed_dates & profile_dates)),
        "claimed_location_mentioned_near_target": float(bool(claimed_locations & profile_locations)),
        "claimed_role_mentioned_near_target": float(bool(claimed_roles & profile_roles)),
        "unsupported_specific_anchor_score": float(unsupported),
        "contradiction_query_max_score": float(contradiction_score),
        "support_query_max_score": float(support_score),
        "support_minus_contradiction_score": float(support_score - contradiction_score),
        "first_meeting_conflict_score": float(_first_meeting_conflict(example, anchors, same_chunk, same_chapter)),
        "role_mismatch_score": float(role_missing),
        "temporal_mismatch_score": float(date_missing),
        "location_mismatch_score": float(location_missing),
    }


def constraint_rule_score(
    example: BackstoryExample,
    anchors: AnchorSet | None = None,
    character_constraints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    anchors = anchors or extract_anchors(example)
    constraints = character_constraints or load_character_constraints()
    target = _canonical_character(example.character)
    row = constraints.get(target) or {}
    text = " ".join(part for part in (example.caption or "", example.backstory) if part).casefold()
    hard_hits = []
    soft_hits = []
    for item in row.get("hard_constraints", []) or []:
        patterns = [str(pattern).casefold() for pattern in item.get("patterns", [])]
        if patterns and all(pattern in text for pattern in patterns):
            hard_hits.append(str(item.get("description") or "constraint hit"))
    for term in row.get("soft_terms", []) or []:
        if str(term).casefold() in text:
            soft_hits.append(str(term))
    risk = min(1.0, 0.5 * len(hard_hits) + 0.1 * len(soft_hits))
    return {
        "hard_constraint_hits": len(hard_hits),
        "soft_constraint_hits": len(soft_hits),
        "matched_constraints": hard_hits + soft_hits,
        "constraint_risk_score": risk,
    }


def _profile_for_character(character: str, mentions: list[EvidenceChunk], all_chunks: list[EvidenceChunk]) -> dict[str, Any]:
    first = min(mentions, key=lambda chunk: chunk.chunk_index_global, default=None)
    co_chars: Counter[str] = Counter()
    locations: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    dates: Counter[str] = Counter()
    events: Counter[str] = Counter()
    for chunk in mentions:
        anchors = extract_anchors(chunk.text, target_character=character)
        co_chars.update(_canonical_character(item) for item in anchors.secondary_characters if _canonical_character(item) != character)
        locations.update(anchors.locations)
        roles.update(anchors.roles)
        dates.update(anchors.dates)
        events.update(chunk.event_mentions)
    high_signal = sorted(mentions, key=lambda chunk: chunk.plot_density_score, reverse=True)[:8]
    return {
        "first_mention_chunk": first.chunk_id if first else None,
        "first_mention_chapter": first.chapter_id if first else None,
        "high_signal_chunks": [chunk.chunk_id for chunk in high_signal],
        "co_mentioned_characters": [item for item, _ in co_chars.most_common(12)],
        "associated_locations": [item for item, _ in locations.most_common(12)],
        "associated_roles": [item for item, _ in roles.most_common(12)],
        "associated_dates": [item for item, _ in dates.most_common(12)],
        "important_event_types": [item for item, _ in events.most_common(12)],
        "mention_count": len(mentions),
    }


def _cooccurrence_graph(chunks: list[EvidenceChunk]) -> dict[str, dict[str, int]]:
    graph: dict[str, dict[str, int]] = defaultdict(lambda: {"same_chunk": 0, "same_chapter": 0, "nearby_chunks": 0})
    by_chapter: dict[tuple[str, str | None], list[EvidenceChunk]] = defaultdict(list)
    for chunk in chunks:
        by_chapter[(chunk.book_name, chunk.chapter_id)].append(chunk)
        chars = [_canonical_character(item) for item in chunk.character_mentions]
        for left in chars:
            for right in chars:
                if left < right:
                    graph[_pair_key(left, right)]["same_chunk"] += 1
    for chapter_chunks in by_chapter.values():
        chapter_chars = set()
        for chunk in chapter_chunks:
            chapter_chars.update(_canonical_character(item) for item in chunk.character_mentions)
        for left in chapter_chars:
            for right in chapter_chars:
                if left < right:
                    graph[_pair_key(left, right)]["same_chapter"] += 1
        ordered = sorted(chapter_chunks, key=lambda chunk: chunk.chunk_index_in_chapter)
        for index, chunk in enumerate(ordered):
            chars = {_canonical_character(item) for item in chunk.character_mentions}
            nearby = ordered[max(0, index - 2) : index + 3]
            near_chars = set()
            for other in nearby:
                near_chars.update(_canonical_character(item) for item in other.character_mentions)
            for left in chars:
                for right in near_chars:
                    if left < right:
                        graph[_pair_key(left, right)]["nearby_chunks"] += 1
    return dict(graph)


def _max_query_score(queries: Iterable[str], chunks: list[EvidenceChunk]) -> float:
    if not chunks:
        return 0.0
    retriever = SimpleBM25Retriever(chunks)
    best = 0.0
    for query in queries:
        scores = retriever.score_query(query)
        best = max(best, max(scores, default=0.0))
    return round(best, 6)


def _contradiction_queries(example: BackstoryExample) -> list[str]:
    queries = []
    for plan in plan_claims(example, llm=None):
        queries.extend(plan.incompatible_states)
        queries.extend(family.natural_language_query for family in generate_contradiction_families(plan))
    return queries or [example.backstory]


def _unsupported_specific_anchor_score(
    anchors: AnchorSet,
    missing_secondary: int,
    location_missing: float,
    role_missing: float,
    date_missing: float,
) -> float:
    specific_count = len(anchors.secondary_characters) + len(anchors.locations) + len(anchors.roles) + len(anchors.dates)
    if specific_count == 0:
        return 0.0
    raw = missing_secondary + location_missing + role_missing + date_missing
    return min(raw / max(specific_count, 1), 1.0)


def _first_meeting_conflict(example: BackstoryExample, anchors: AnchorSet, same_chunk: int, same_chapter: int) -> float:
    text = example.backstory.casefold()
    if not anchors.secondary_characters:
        return 0.0
    if "first" in text or "met" in text or "hired" in text:
        return 0.0 if same_chunk or same_chapter else 1.0
    return 0.0


def _missing_ratio(claimed: set[str], observed: set[str]) -> float:
    if not claimed:
        return 0.0
    return len(claimed - observed) / len(claimed)


def _mentions_character(chunk: EvidenceChunk, character: str) -> bool:
    target = _canonical_character(character).casefold()
    mentions = {_canonical_character(item).casefold() for item in chunk.character_mentions}
    return target in mentions or target in chunk.text.casefold()


def _canonical_targets() -> tuple[str, ...]:
    return tuple(dict.fromkeys(_canonical_character(item) for item in KNOWN_CHARACTERS))


def _canonical_character(value: str) -> str:
    text = " ".join(str(value).replace("/", " ").split())
    aliases = {
        "Ayrton": "Tom Ayrton",
        "Tom Ayrton Ben Joyce": "Tom Ayrton/Ben Joyce",
        "Ben Joyce": "Tom Ayrton/Ben Joyce",
        "Paganel": "Jacques Paganel",
        "Edmond": "Edmond Dantès",
        "Edmond Dantes": "Edmond Dantès",
        "Mercedes": "Mercédès",
    }
    return aliases.get(text, text)


def _pair_key(left: str, right: str) -> str:
    first, second = sorted((_canonical_character(left), _canonical_character(right)))
    return f"{first}||{second}"


def _norm(value: str) -> str:
    return " ".join(str(value).casefold().split())
