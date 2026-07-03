"""Hybrid retrieval for support and contradiction evidence."""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .config import DEFAULT_CONFIG, NCVConfig
from .ingest import normalize_book_name
from .schemas import ClaimPlan, ContradictionFamily, EvidenceChunk, RetrievedChunk, RetrievalQuery


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
TEMPORAL_TERMS = {
    "before",
    "after",
    "later",
    "earlier",
    "then",
    "during",
    "afterwards",
    "subsequently",
    "returned",
}

EVENT_TYPE_TO_CHUNK_SIGNALS: Mapping[str, dict[str, tuple[str, ...]]] = {
    "custody": {"categories": ("custody", "captivity"), "terms": ("imprisoned", "prison", "cell", "dungeon", "captive", "prisoner")},
    "captivity": {"categories": ("custody", "captivity"), "terms": ("imprisoned", "prison", "cell", "dungeon", "captive", "prisoner")},
    "escape": {"categories": ("escape", "escape_release"), "terms": ("escape", "escaped", "fled", "flight")},
    "escape_release": {
        "categories": ("escape", "escape_release", "rescue"),
        "terms": ("escaped", "released", "freed", "liberated", "no longer prisoner"),
    },
    "movement": {
        "categories": ("movement", "voyage"),
        "terms": ("travelled", "traveled", "arrived", "departed", "sailed", "went", "journey", "returned", "rode"),
    },
    "voyage": {"categories": ("movement", "voyage"), "terms": ("voyage", "sailed", "ship", "aboard", "journey")},
    "public_appearance": {
        "categories": ("public_appearance",),
        "terms": ("appeared", "attended", "joined", "met", "visited", "public", "assembly"),
    },
    "later_action": {
        "categories": ("later_action", "public_appearance", "movement"),
        "terms": ("later", "afterwards", "subsequently", "spoke", "met", "wrote", "returned", "appeared"),
    },
    "death": {"categories": ("death", "violence"), "terms": ("died", "dead", "death", "killed", "murdered", "poisoned", "executed")},
    "violence": {"categories": ("death", "violence"), "terms": ("attack", "attacked", "killed", "murdered", "wounded", "poisoned")},
    "marriage_relationship": {
        "categories": ("family_relationship",),
        "terms": ("married", "wife", "husband", "wedding", "betrothed", "marriage"),
    },
    "betrayal_deception": {"categories": ("betrayal", "political_intrigue"), "terms": ("betrayed", "treason", "traitor", "denounced", "spy")},
    "rescue_help": {"categories": ("rescue",), "terms": ("rescued", "saved", "helped", "protected", "warned")},
    "loyalty_help": {"categories": ("rescue",), "terms": ("loyal", "helped", "protected", "saved", "warned")},
    "knowledge_secret": {"categories": ("identity_secret",), "terms": ("secret", "learned", "discovered", "revealed", "confessed")},
    "later_discovery": {"categories": ("identity_secret",), "terms": ("later", "discovered", "learned", "revealed")},
    "identity_alias": {"categories": ("identity_secret",), "terms": ("alias", "disguise", "identity", "unknown")},
    "possession_document": {"categories": ("wealth_property",), "terms": ("document", "letter", "map", "dossier", "paper", "treasure")},
    "document_conflict": {"categories": ("wealth_property",), "terms": ("lost", "found", "document", "letter", "map", "dossier")},
    "incompatible_location": {"categories": ("voyage",), "terms": ("elsewhere", "place", "arrived", "departed", "at the same time")},
    "incompatible_time": {"categories": (), "terms": ("before", "after", "later", "earlier", "subsequently")},
    "direct_negation": {"categories": (), "terms": ("not", "never", "false", "instead", "contrary")},
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization with very short tokens removed."""

    return [token.casefold() for token in TOKEN_RE.findall(text) if len(token) >= 3]


def lexical_score(query: str, document: str) -> float:
    """Dependency-free BM25-style lexical score for compatibility."""

    retriever = SimpleBM25Retriever([_chunk_for_text(document)])
    return retriever.score_query(query)[0]


class SimpleBM25Retriever:
    """BM25 retriever using `rank_bm25` when available, else local BM25."""

    def __init__(self, chunks: Iterable[EvidenceChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.tokenized_docs = [tokenize(chunk.text) for chunk in self.chunks]
        self.k1 = k1
        self.b = b
        self._rank_bm25 = None
        try:
            from rank_bm25 import BM25Okapi

            self._rank_bm25 = BM25Okapi(self.tokenized_docs)
        except Exception:
            self._rank_bm25 = None
            self._doc_freqs = self._build_doc_freqs()
            self._avg_doc_len = (
                sum(len(doc) for doc in self.tokenized_docs) / len(self.tokenized_docs)
                if self.tokenized_docs
                else 0.0
            )

    def score_query(self, query: str) -> list[float]:
        tokens = tokenize(query)
        if not self.chunks:
            return []
        if not tokens:
            return [0.0 for _ in self.chunks]
        if self._rank_bm25 is not None:
            return [float(score) for score in self._rank_bm25.get_scores(tokens)]
        return [self._local_score(tokens, doc) for doc in self.tokenized_docs]

    def rank(self, query: str, *, top_k: int | None = None) -> list[tuple[int, float]]:
        scores = self.score_query(query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        if top_k is not None:
            return ranked[:top_k]
        return [item for item in ranked if item[1] > 0]

    def _build_doc_freqs(self) -> Counter[str]:
        freqs: Counter[str] = Counter()
        for doc in self.tokenized_docs:
            freqs.update(set(doc))
        return freqs

    def _local_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        counts = Counter(doc_tokens)
        score = 0.0
        num_docs = len(self.tokenized_docs)
        doc_len = len(doc_tokens)
        for token in query_tokens:
            tf = counts.get(token, 0)
            if not tf:
                continue
            df = self._doc_freqs.get(token, 0)
            idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_doc_len or 1))
            score += idf * (tf * (self.k1 + 1) / denom)
        return float(score)


class EmbeddingRetriever:
    """Optional semantic retriever with mock and graceful lexical-only modes."""

    def __init__(
        self,
        chunks: Iterable[EvidenceChunk],
        *,
        model_name: str = "all-MiniLM-L6-v2",
        mode: str = "auto",
        enabled: bool = False,
    ) -> None:
        self.chunks = list(chunks)
        self.model_name = model_name
        self.mode = mode
        self.enabled = enabled
        self._model = None
        self._chunk_embeddings: list[list[float]] | None = None

    def score_query(self, query: str) -> list[float]:
        if not self.enabled:
            return [0.0 for _ in self.chunks]
        if self.mode == "mock":
            query_embedding = _mock_embed(query)
            return [_cosine(query_embedding, embedding) for embedding in self._mock_chunk_embeddings()]
        try:
            model = self._load_model()
        except Exception as exc:
            warnings.warn(f"sentence-transformers unavailable; semantic retrieval disabled: {exc}", RuntimeWarning)
            return [0.0 for _ in self.chunks]
        query_embedding = model.encode(query).tolist()
        if self._chunk_embeddings is None:
            self._chunk_embeddings = [model.encode(chunk.text).tolist() for chunk in self.chunks]
        return [_cosine(query_embedding, embedding) for embedding in self._chunk_embeddings]

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _mock_chunk_embeddings(self) -> list[list[float]]:
        if self._chunk_embeddings is None:
            self._chunk_embeddings = [_mock_embed(chunk.text) for chunk in self.chunks]
        return self._chunk_embeddings


@dataclass(frozen=True, slots=True)
class RetrievalWorkItem:
    query: str
    channel: str
    event_types: tuple[str, ...] = ()
    family_name: str | None = None


class HybridRetriever:
    """Hybrid retriever with explainable score components."""

    def __init__(self, chunks: Iterable[EvidenceChunk], config: NCVConfig = DEFAULT_CONFIG) -> None:
        self.chunks = list(chunks)
        self.config = config
        self.bm25 = SimpleBM25Retriever(self.chunks)
        self.embedding = EmbeddingRetriever(
            self.chunks,
            model_name=config.embedding_model,
            mode=config.embedding_mode,
            enabled=config.use_embeddings,
        )
        self._neighbor_index = {
            (chunk.book_name, chunk.chapter_id, chunk.chunk_index_in_chapter): chunk
            for chunk in self.chunks
            if chunk.chapter_id and chunk.chunk_index_in_chapter
        }

    def retrieve_for_claim(
        self,
        claim_plan: ClaimPlan,
        contradiction_families: Iterable[ContradictionFamily],
    ) -> list[RetrievedChunk]:
        work_items = build_retrieval_work_items(claim_plan, contradiction_families, self.config)
        best_by_chunk: dict[str, RetrievedChunk] = {}

        for item in work_items:
            bm25_scores = self.bm25.score_query(item.query) if self.config.use_lexical else [0.0 for _ in self.chunks]
            semantic_scores = self.embedding.score_query(item.query)
            bm25_norm = _normalize_scores(bm25_scores)
            semantic_norm = _normalize_scores(semantic_scores)

            for index, chunk in enumerate(self.chunks):
                if not _passes_character_filter(claim_plan, chunk, self.config.strict_character_filter):
                    continue
                breakdown = score_chunk(
                    claim_plan=claim_plan,
                    chunk=chunk,
                    work_item=item,
                    bm25_score=bm25_norm[index] if index < len(bm25_norm) else 0.0,
                    semantic_score=semantic_norm[index] if index < len(semantic_norm) else 0.0,
                    config=self.config,
                )
                if breakdown["final_score"] <= 0:
                    continue
                retrieved = _retrieved_from_chunk(
                    chunk,
                    claim_plan=claim_plan,
                    work_item=item,
                    score_breakdown=breakdown,
                    source="retrieval",
                )
                existing = best_by_chunk.get(chunk.chunk_id)
                if existing is None or retrieved.final_score > existing.final_score:
                    best_by_chunk[chunk.chunk_id] = retrieved

        top = sorted(best_by_chunk.values(), key=lambda item: item.final_score, reverse=True)[: self.config.top_k_retrieval]
        expanded = self._expand_neighbors(top, {item.chunk_id for item in top})
        combined = _dedupe_retrieved([*top, *expanded])
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
                metadata=item.metadata,
            )
            for rank, item in enumerate(sorted(combined, key=lambda item: item.final_score, reverse=True), start=1)
        ]

    def _expand_neighbors(self, retrieved: list[RetrievedChunk], seen_chunk_ids: set[str]) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for item in retrieved:
            chunk = self._chunk_by_id(item.chunk_id)
            if chunk is None or not chunk.chapter_id:
                continue
            for neighbor_index in (chunk.chunk_index_in_chapter - 1, chunk.chunk_index_in_chapter + 1):
                neighbor = self._neighbor_index.get((chunk.book_name, chunk.chapter_id, neighbor_index))
                if neighbor is None or neighbor.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(neighbor.chunk_id)
                out.append(
                    RetrievedChunk(
                        chunk_id=neighbor.chunk_id,
                        book_name=neighbor.book_name,
                        chapter_id=neighbor.chapter_id,
                        text=neighbor.text,
                        query=item.query,
                        channel=f"{item.channel}:neighbor",
                        final_score=round(item.final_score * 0.65, 6),
                        score_breakdown={
                            "final_score": round(item.final_score * 0.65, 6),
                            "neighbor_of": item.chunk_id,
                        },
                        source="neighbor_expansion",
                        claim_id=item.claim_id,
                        metadata={
                            "chapter_title": neighbor.chapter_title,
                            "chunk_index_in_chapter": neighbor.chunk_index_in_chapter,
                        },
                    )
                )
        return out

    def _chunk_by_id(self, chunk_id: str) -> EvidenceChunk | None:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None


def retrieve_for_claim(
    claim_plan: ClaimPlan,
    contradiction_families: Iterable[ContradictionFamily],
    chunks: Iterable[EvidenceChunk],
    config: NCVConfig = DEFAULT_CONFIG,
) -> list[RetrievedChunk]:
    return HybridRetriever(chunks, config).retrieve_for_claim(claim_plan, contradiction_families)


def build_retrieval_work_items(
    claim_plan: ClaimPlan,
    contradiction_families: Iterable[ContradictionFamily],
    config: NCVConfig = DEFAULT_CONFIG,
) -> list[RetrievalWorkItem]:
    items = [
        RetrievalWorkItem(query=claim_plan.claim_text or claim_plan.primary_claim.claim_text, channel="support_claim_query"),
    ]
    for query in claim_plan.support_queries:
        items.append(RetrievalWorkItem(query=query, channel="support_keyword_query", event_types=claim_plan.expected_event_types))
    if config.use_contradiction_families:
        for family in contradiction_families:
            items.append(
                RetrievalWorkItem(
                    query=family.natural_language_query,
                    channel="contradiction_family_query",
                    event_types=family.event_types,
                    family_name=family.family_name,
                )
            )
        for state in claim_plan.incompatible_states:
            items.append(
                RetrievalWorkItem(
                    query=f"{claim_plan.character or ''} {state}",
                    channel="incompatible_state_query",
                    event_types=claim_plan.incompatible_event_types,
                )
            )
    if config.use_timeline_features:
        items.append(
            RetrievalWorkItem(
                query=f"{claim_plan.character or ''} before after later timeline {claim_plan.claim_text}",
                channel="timeline_sensitive_query",
                event_types=("incompatible_time",),
            )
        )
    return items


def score_chunk(
    *,
    claim_plan: ClaimPlan,
    chunk: EvidenceChunk,
    work_item: RetrievalWorkItem,
    bm25_score: float,
    semantic_score: float,
    config: NCVConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    character_match_score = _character_match_score(claim_plan, chunk) if config.use_character_scoring else 0.0
    query_specific_event_score = (
        _query_specific_event_score(work_item.query, work_item.event_types, chunk)
        if config.use_event_scoring
        else 0.0
    )
    incompatible_event_score = (
        _query_specific_event_score(work_item.query, claim_plan.incompatible_event_types, chunk)
        if config.use_event_scoring and work_item.channel.startswith(("contradiction", "incompatible"))
        else 0.0
    )
    plot_density = min(chunk.plot_density_score / 50.0, 1.0)
    temporal_cue_score = _temporal_cue_score(chunk) if config.use_timeline_features else 0.0

    final_score = (
        0.38 * bm25_score
        + 0.18 * semantic_score
        + 0.15 * character_match_score
        + 0.14 * query_specific_event_score
        + 0.07 * incompatible_event_score
        + 0.05 * plot_density
        + 0.03 * temporal_cue_score
    )
    return {
        "bm25_score": round(bm25_score, 6),
        "semantic_score": round(semantic_score, 6),
        "character_match_score": round(character_match_score, 6),
        "query_specific_event_score": round(query_specific_event_score, 6),
        "incompatible_event_score": round(incompatible_event_score, 6),
        "plot_density_score": round(plot_density, 6),
        "temporal_cue_score": round(temporal_cue_score, 6),
        "final_score": round(final_score, 6),
    }


def load_chunks_jsonl(path: str | Path) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunks.append(
                EvidenceChunk(
                    chunk_id=row["chunk_id"],
                    book_name=row["book_name"],
                    chunk_text=row.get("text") or row.get("chunk_text"),
                    source_path=row.get("source_path"),
                    chapter_id=row.get("chapter_id"),
                    chapter_title=row.get("chapter_title"),
                    chunk_index_global=row.get("chunk_index_global", 0),
                    chunk_index_in_chapter=row.get("chunk_index_in_chapter", 0),
                    character_mentions=tuple(row.get("character_mentions") or ()),
                    event_mentions=tuple(row.get("event_mentions") or ()),
                    plot_density_score=row.get("plot_density_score", 0.0),
                    metadata=row.get("metadata") or {},
                )
            )
    return chunks


def save_retrieval_trace(
    path: str | Path,
    example_id: str,
    claim_plan: ClaimPlan,
    retrieved_chunks: Iterable[RetrievedChunk],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for item in retrieved_chunks:
            row = {
                "example_id": example_id,
                "claim_id": claim_plan.claim_id,
                "claim_text": claim_plan.claim_text,
                "query": item.query,
                "channel": item.channel,
                "chunk_id": item.chunk_id,
                "score_breakdown": dict(item.score_breakdown),
                "text_preview": item.text[:240],
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return output_path


def chunk_matches_query_filters(query: RetrievalQuery, chunk: EvidenceChunk) -> bool:
    if query.book_name and normalize_book_name(query.book_name) != normalize_book_name(chunk.book_name):
        return False
    if query.character:
        character = query.character.casefold()
        hits = {hit.casefold() for hit in chunk.character_mentions}
        if hits and character not in hits:
            return False
    return True


def retrieve(query: RetrievalQuery, chunks: Iterable[EvidenceChunk], *, top_k: int) -> list[EvidenceChunk]:
    """Backward-compatible lexical retrieval returning EvidenceChunk objects."""

    chunk_list = list(chunks)
    bm25 = SimpleBM25Retriever(chunk_list)
    ranked = bm25.rank(query.query_text, top_k=top_k)
    out: list[EvidenceChunk] = []
    for index, score in ranked:
        chunk = chunk_list[index]
        if not chunk_matches_query_filters(query, chunk):
            continue
        out.append(
            EvidenceChunk(
                chunk_id=chunk.chunk_id,
                book_name=chunk.book_name,
                chunk_text=chunk.chunk_text,
                score=score,
                source_path=chunk.source_path,
                chapter_id=chunk.chapter_id,
                chapter_title=chunk.chapter_title,
                chunk_index_global=chunk.chunk_index_global,
                chunk_index_in_chapter=chunk.chunk_index_in_chapter,
                character_mentions=chunk.character_mentions,
                event_mentions=chunk.event_mentions,
                plot_density_score=chunk.plot_density_score,
                metadata=chunk.metadata,
            )
        )
    return out


def _retrieved_from_chunk(
    chunk: EvidenceChunk,
    *,
    claim_plan: ClaimPlan,
    work_item: RetrievalWorkItem,
    score_breakdown: Mapping[str, float],
    source: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        book_name=chunk.book_name,
        chapter_id=chunk.chapter_id,
        text=chunk.text,
        query=work_item.query,
        channel=work_item.channel,
        final_score=score_breakdown["final_score"],
        score_breakdown=dict(score_breakdown),
        source=source,
        claim_id=claim_plan.claim_id,
        metadata={
            "chapter_title": chunk.chapter_title,
            "chunk_index_global": chunk.chunk_index_global,
            "chunk_index_in_chapter": chunk.chunk_index_in_chapter,
            "character_mentions": list(chunk.character_mentions),
            "event_mentions": list(chunk.event_mentions),
            "family_name": work_item.family_name,
        },
    )


def _passes_character_filter(claim_plan: ClaimPlan, chunk: EvidenceChunk, strict: bool) -> bool:
    if not strict:
        return True
    if not claim_plan.character:
        return True
    return _character_match_score(claim_plan, chunk) > 0


def _character_match_score(claim_plan: ClaimPlan, chunk: EvidenceChunk) -> float:
    if not claim_plan.character:
        return 0.0
    target = claim_plan.character.casefold()
    mentions = {mention.casefold() for mention in chunk.character_mentions}
    if target in mentions:
        return 1.0
    if "/" in target:
        parts = [part.strip() for part in target.split("/") if part.strip()]
        if any(part in mentions for part in parts):
            return 1.0
    if target in chunk.text.casefold():
        return 0.8
    return 0.0


def _query_specific_event_score(query: str, event_types: Iterable[str], chunk: EvidenceChunk) -> float:
    desired = tuple(event_types) or _infer_event_types_from_query(query)
    if not desired:
        return 0.0
    text = chunk.text.casefold()
    chunk_events = {event.casefold() for event in chunk.event_mentions}
    raw = 0.0
    for event_type in desired:
        normalized = event_type.casefold()
        signals = EVENT_TYPE_TO_CHUNK_SIGNALS.get(normalized, {"categories": (normalized,), "terms": (normalized,)})
        if normalized in chunk_events:
            raw += 1.0
        elif any(category in chunk_events for category in signals["categories"]):
            raw += 1.0
        if any(term in text for term in signals["terms"]):
            raw += 0.6
    return min(raw / max(len(desired), 1), 1.0)


def _infer_event_types_from_query(query: str) -> tuple[str, ...]:
    lowered = query.casefold()
    inferred: list[str] = []
    for event_type, signals in EVENT_TYPE_TO_CHUNK_SIGNALS.items():
        if any(term in lowered for term in signals["terms"]):
            inferred.append(event_type)
    return tuple(inferred)


def _temporal_cue_score(chunk: EvidenceChunk) -> float:
    tokens = set(tokenize(chunk.text))
    return min(sum(1 for term in TEMPORAL_TERMS if term in tokens) / 3.0, 1.0)


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]
    return [max(score / max_score, 0.0) for score in scores]


def _dedupe_retrieved(items: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    best: dict[str, RetrievedChunk] = {}
    for item in items:
        existing = best.get(item.chunk_id)
        if existing is None or item.final_score > existing.final_score:
            best[item.chunk_id] = item
    return list(best.values())


def _mock_embed(text: str, *, dims: int = 32) -> list[float]:
    vector = [0.0] * dims
    for token in tokenize(text):
        token = _semantic_bucket(token)
        bucket = hash(token) % dims
        vector[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _semantic_bucket(token: str) -> str:
    buckets = {
        "imprisoned": "custody",
        "imprisonment": "custody",
        "prison": "custody",
        "dungeon": "custody",
        "cell": "custody",
        "captive": "custody",
        "captivity": "custody",
        "confined": "custody",
        "confinement": "custody",
        "escaped": "escape",
        "escape": "escape",
        "released": "escape",
        "freed": "escape",
        "travelled": "movement",
        "traveled": "movement",
        "arrived": "movement",
        "departed": "movement",
        "returned": "movement",
        "appeared": "appearance",
        "public": "appearance",
        "met": "appearance",
    }
    return buckets.get(token, token)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    numerator = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def _chunk_for_text(text: str) -> EvidenceChunk:
    return EvidenceChunk(chunk_id="doc", book_name="doc", chunk_text=text)
