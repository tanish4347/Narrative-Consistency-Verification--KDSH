"""Evidence reranking after broad retrieval."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Iterable

from .config import DEFAULT_CONFIG, NCVConfig
from .retrieval import EVENT_TYPE_TO_CHUNK_SIGNALS, tokenize
from .schemas import ClaimPlan, ContradictionFamily, RetrievedChunk


TEMPORAL_MARKERS = {"before", "after", "later", "earlier", "during", "then", "returned", "subsequently"}


@dataclass(frozen=True, slots=True)
class HeuristicReranker:
    """Deterministic evidence reranker using transparent feature weights."""

    def rerank(
        self,
        claim_plan: ClaimPlan,
        contradiction_families: Iterable[ContradictionFamily],
        retrieved_chunks: Iterable[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        family_event_types = tuple(event for family in contradiction_families for event in family.event_types)
        reranked: list[RetrievedChunk] = []
        for original_index, chunk in enumerate(retrieved_chunks):
            breakdown = self.score_chunk(claim_plan, family_event_types, chunk)
            score_breakdown = dict(chunk.score_breakdown)
            score_breakdown["rerank_score"] = breakdown["rerank_score"]
            score_breakdown["rerank_breakdown"] = breakdown
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    book_name=chunk.book_name,
                    chapter_id=chunk.chapter_id,
                    text=chunk.text,
                    query=chunk.query,
                    channel=chunk.channel,
                    final_score=round(chunk.final_score + breakdown["rerank_score"], 6),
                    score_breakdown=score_breakdown,
                    source=chunk.source,
                    claim_id=chunk.claim_id,
                    rank=chunk.rank,
                    metadata={**dict(chunk.metadata), "_original_order": original_index},
                )
            )

        sorted_chunks = sorted(
            reranked,
            key=lambda item: (-item.final_score, int(item.metadata.get("_original_order", 0)), item.chunk_id),
        )
        return [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                book_name=item.book_name,
                chapter_id=item.chapter_id,
                text=item.text,
                query=item.query,
                channel=item.channel,
                final_score=item.final_score,
                score_breakdown=item.score_breakdown,
                source=item.source,
                claim_id=item.claim_id,
                rank=rank,
                metadata={key: value for key, value in dict(item.metadata).items() if key != "_original_order"},
            )
            for rank, item in enumerate(sorted_chunks, start=1)
        ]

    def score_chunk(
        self,
        claim_plan: ClaimPlan,
        family_event_types: tuple[str, ...],
        chunk: RetrievedChunk,
    ) -> dict[str, float]:
        text = chunk.text.casefold()
        tokens = set(tokenize(chunk.text))
        metadata = dict(chunk.metadata)
        character_mentions = {str(item).casefold() for item in metadata.get("character_mentions", [])}
        event_mentions = {str(item).casefold() for item in metadata.get("event_mentions", [])}
        character = (claim_plan.character or "").casefold()
        exact_character = 1.0 if character and (character in character_mentions or character in text) else 0.0
        alias_overlap = _alias_overlap(character, character_mentions, text)
        claim_keywords = set(claim_plan.primary_claim.keywords) or set(tokenize(claim_plan.claim_text or ""))
        claim_keyword_overlap = _overlap_score(claim_keywords, tokens)
        expected_event_overlap = _event_overlap(claim_plan.expected_event_types, event_mentions, text)
        incompatible_event_overlap = _event_overlap(
            tuple(claim_plan.incompatible_event_types) + tuple(family_event_types),
            event_mentions,
            text,
        )
        temporal_marker_overlap = min(len(tokens & TEMPORAL_MARKERS) / 2.0, 1.0)
        plot_density = min(float(metadata.get("plot_density_score", chunk.score_breakdown.get("plot_density_score", 0))) / 50.0, 1.0)
        query_source = _query_source_score(chunk)
        proximity = 0.4 if chunk.source == "neighbor_expansion" else 0.0

        rerank_score = (
            0.22 * exact_character
            + 0.08 * alias_overlap
            + 0.16 * claim_keyword_overlap
            + 0.13 * expected_event_overlap
            + 0.18 * incompatible_event_overlap
            + 0.08 * temporal_marker_overlap
            + 0.08 * plot_density
            + 0.05 * query_source
            + 0.02 * proximity
        )
        return {
            "exact_character_mention": round(exact_character, 6),
            "alias_name_overlap": round(alias_overlap, 6),
            "claim_keyword_overlap": round(claim_keyword_overlap, 6),
            "expected_event_overlap": round(expected_event_overlap, 6),
            "incompatible_event_overlap": round(incompatible_event_overlap, 6),
            "temporal_marker_overlap": round(temporal_marker_overlap, 6),
            "plot_density": round(plot_density, 6),
            "query_source_type": round(query_source, 6),
            "chunk_proximity": round(proximity, 6),
            "rerank_score": round(rerank_score, 6),
        }


@dataclass(frozen=True, slots=True)
class CrossEncoderReranker:
    """Optional cross-encoder reranker; gracefully no-ops if unavailable."""

    model_name: str

    def rerank(
        self,
        claim_plan: ClaimPlan,
        contradiction_families: Iterable[ContradictionFamily],
        retrieved_chunks: Iterable[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        chunks = list(retrieved_chunks)
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            warnings.warn(f"CrossEncoder unavailable; using heuristic reranker: {exc}", RuntimeWarning)
            return HeuristicReranker().rerank(claim_plan, contradiction_families, chunks)

        try:
            model = CrossEncoder(self.model_name)
            pairs = [(claim_plan.claim_text or "", chunk.text) for chunk in chunks]
            scores = [float(score) for score in model.predict(pairs)]
        except Exception as exc:
            warnings.warn(f"CrossEncoder failed; using heuristic reranker: {exc}", RuntimeWarning)
            return HeuristicReranker().rerank(claim_plan, contradiction_families, chunks)

        adjusted: list[RetrievedChunk] = []
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        denom = (max_score - min_score) or 1.0
        for chunk, score in zip(chunks, scores):
            normalized = (score - min_score) / denom
            score_breakdown = dict(chunk.score_breakdown)
            score_breakdown["rerank_score"] = round(normalized, 6)
            score_breakdown["rerank_breakdown"] = {"cross_encoder_score": score, "rerank_score": round(normalized, 6)}
            adjusted.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    book_name=chunk.book_name,
                    chapter_id=chunk.chapter_id,
                    text=chunk.text,
                    query=chunk.query,
                    channel=chunk.channel,
                    final_score=round(chunk.final_score + normalized, 6),
                    score_breakdown=score_breakdown,
                    source=chunk.source,
                    claim_id=chunk.claim_id,
                    rank=chunk.rank,
                    metadata=chunk.metadata,
                )
            )
        return sorted(adjusted, key=lambda item: (-item.final_score, item.chunk_id))


def rerank_evidence(
    claim_plan: ClaimPlan,
    contradiction_families: Iterable[ContradictionFamily],
    retrieved_chunks: Iterable[RetrievedChunk],
    config: NCVConfig = DEFAULT_CONFIG,
) -> list[RetrievedChunk]:
    chunks = list(retrieved_chunks)
    if not config.use_rerank:
        return chunks
    if config.use_cross_encoder:
        return CrossEncoderReranker(config.cross_encoder_model).rerank(claim_plan, contradiction_families, chunks)
    return HeuristicReranker().rerank(claim_plan, contradiction_families, chunks)


def _alias_overlap(character: str, mentions: set[str], text: str) -> float:
    if not character:
        return 0.0
    parts = [part for part in re.split(r"[\s/.-]+", character) if len(part) > 2]
    if not parts:
        return 0.0
    hits = sum(1 for part in parts if part in text or part in mentions)
    return hits / len(parts)


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return min(len(left & right) / max(len(left), 1), 1.0)


def _event_overlap(event_types: Iterable[str], event_mentions: set[str], text: str) -> float:
    desired = {event.casefold() for event in event_types}
    if not desired:
        return 0.0
    hits = 0
    for event in desired:
        signals = EVENT_TYPE_TO_CHUNK_SIGNALS.get(event, {"categories": (event,), "terms": (event,)})
        if event in event_mentions or any(category in event_mentions for category in signals["categories"]):
            hits += 1
        elif any(term in text for term in signals["terms"]):
            hits += 1
    return min(hits / len(desired), 1.0)


def _query_source_score(chunk: RetrievedChunk) -> float:
    if "contradiction_family" in chunk.channel:
        return 1.0
    if "incompatible_state" in chunk.channel:
        return 0.9
    if "support" in chunk.channel:
        return 0.7
    if "timeline" in chunk.channel:
        return 0.6
    return 0.4
