"""Evidence-grounded claim verification with exact quote validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Protocol

from .cache import LLMCache
from .config import DEFAULT_CONFIG, NCVConfig
from .llm import CachedLLMClient, ChatClient, LLMClient, MockLLMClient
from .schemas import Claim, ClaimPlan, ClaimVerdict, EvidenceChunk, RetrievedChunk
from .utils import extract_json_object


VERIFIER_SYSTEM_PROMPT = """You are a strict narrative consistency verifier.
Use only the supplied chunks. Do not use external knowledge.
Output only valid JSON with:
{
  "claim_id": int,
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT",
  "evidence": [{"chunk_id": "...", "quote": "..."}],
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning_short": "..."
}
Quotes must be exact substrings from the provided chunks. Silence is not contradiction.
"""

LOCAL_VERIFIER_SYSTEM_PROMPT = """Use only the chunks. Return JSON only:
{"claim_id": int, "verdict": "SUPPORTED|CONTRADICTED|INSUFFICIENT", "evidence": [{"chunk_id": "...", "quote": "..."}], "confidence": "HIGH|MEDIUM|LOW", "reasoning_short": "..."}
Quotes must be exact substrings. If evidence is not decisive, use INSUFFICIENT.
"""


class Verifier(Protocol):
    def verify_claim(
        self,
        claim_plan: ClaimPlan,
        retrieved_chunks: Iterable[RetrievedChunk],
        *,
        config: NCVConfig = DEFAULT_CONFIG,
    ) -> ClaimVerdict:
        ...


@dataclass(frozen=True, slots=True)
class MockVerifier:
    """Deterministic verifier used by tests and mock pipeline mode."""

    def verify_claim(
        self,
        claim_plan: ClaimPlan,
        retrieved_chunks: Iterable[RetrievedChunk],
        *,
        config: NCVConfig = DEFAULT_CONFIG,
    ) -> ClaimVerdict:
        chunks = list(retrieved_chunks)
        for chunk in chunks:
            if "CONTRADICTION_FIXTURE" in chunk.text:
                quote = "CONTRADICTION_FIXTURE"
                return ClaimVerdict(
                    claim_id=claim_plan.claim_id or claim_plan.primary_claim.claim_id,
                    verdict="CONTRADICTED",
                    evidence=(_evidence_from_retrieved(chunk),),
                    confidence="HIGH",
                    justification="Mock contradiction marker found in retrieved evidence.",
                    metadata={
                        "reasoning_short": "Mock contradiction marker found.",
                        "evidence_quotes": [{"chunk_id": chunk.chunk_id, "quote": quote}],
                        "verifier": "mock",
                    },
                )

        claim_terms = {token for token in (claim_plan.primary_claim.keywords or ()) if len(token) > 3}
        for chunk in chunks:
            lowered = chunk.text.casefold()
            if claim_terms and sum(1 for token in claim_terms if token.casefold() in lowered) >= 2:
                quote = _quote_window(chunk.text, next(iter(claim_terms)))
                return ClaimVerdict(
                    claim_id=claim_plan.claim_id or claim_plan.primary_claim.claim_id,
                    verdict="SUPPORTED",
                    evidence=(_evidence_from_retrieved(chunk),),
                    confidence="MEDIUM",
                    justification="Mock verifier found overlapping claim terms in retrieved evidence.",
                    metadata={
                        "reasoning_short": "Mock overlap support.",
                        "evidence_quotes": [{"chunk_id": chunk.chunk_id, "quote": quote}],
                        "verifier": "mock",
                    },
                )

        return ClaimVerdict(
            claim_id=claim_plan.claim_id or claim_plan.primary_claim.claim_id,
            verdict="INSUFFICIENT",
            evidence=(),
            confidence="LOW",
            justification="Mock verifier found no decisive evidence.",
            metadata={"reasoning_short": "No decisive mock evidence.", "evidence_quotes": [], "verifier": "mock"},
        )


@dataclass(slots=True)
class LLMVerifier:
    """LLM verifier with strict JSON and exact-quote validation."""

    llm: LLMClient
    max_retries: int = 2
    confirm_contradictions: bool = True

    def verify_claim(
        self,
        claim_plan: ClaimPlan,
        retrieved_chunks: Iterable[RetrievedChunk],
        *,
        config: NCVConfig = DEFAULT_CONFIG,
    ) -> ClaimVerdict:
        chunks = list(retrieved_chunks)
        if not chunks:
            return _insufficient(claim_plan, "No retrieved chunks were provided.")

        client = _cache_client_if_needed(self.llm, config)
        model = _model_for_client(client, config)
        messages = _verifier_messages(claim_plan, chunks, config=config, provider=getattr(client, "provider", ""))
        parsed: dict | None = None
        last_reason = "Verifier failed to produce valid output."
        for attempt in range(self.max_retries):
            try:
                attempt_messages = messages
                if attempt:
                    attempt_messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "Retry with JSON only. Use this shape exactly: "
                                '{"claim_id": 1, "verdict": "INSUFFICIENT", "evidence": [], '
                                '"confidence": "LOW", "reasoning_short": "brief reason"}. '
                                "Any quote must be copied exactly from a provided chunk."
                            ),
                        },
                    ]
                raw = client.generate(
                    attempt_messages,
                    model=model,
                    temperature=config.llm_temperature,
                    max_tokens=config.llm_max_tokens,
                    timeout_s=config.llm_timeout_s,
                )
                parsed = extract_json_object(raw)
                claim_verdict = _claim_verdict_from_json(claim_plan, chunks, parsed)
                if claim_verdict.verdict == "CONTRADICTED" and self.confirm_contradictions:
                    if not self._confirm_contradiction(client, claim_plan, claim_verdict, chunks, config):
                        return _insufficient(claim_plan, "Contradiction was not confirmed by second pass.")
                return claim_verdict
            except Exception as exc:
                if _is_local_provider_failure(exc):
                    raise
                last_reason = str(exc)
                parsed = None

        return _insufficient(claim_plan, f"Verifier invalid after retries: {last_reason}", raw=parsed)

    def _confirm_contradiction(
        self,
        client: LLMClient,
        claim_plan: ClaimPlan,
        first_verdict: ClaimVerdict,
        chunks: list[RetrievedChunk],
        config: NCVConfig,
    ) -> bool:
        quote_rows = first_verdict.metadata.get("evidence_quotes", [])
        evidence_text = "\n".join(f"[{row['chunk_id']}] {row['quote']}" for row in quote_rows)
        model = _model_for_client(client, config)
        messages = [
            {"role": "system", "content": _system_prompt_for_provider(getattr(client, "provider", ""))},
            {
                "role": "user",
                "content": (
                    "Confirm the contradiction using only these exact quoted snippets.\n"
                    f"Claim: {claim_plan.claim_text}\nEvidence:\n{evidence_text}"
                ),
            },
        ]
        try:
            raw = client.generate(
                messages,
                model=model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                timeout_s=config.llm_timeout_s,
            )
            parsed = extract_json_object(raw)
            confirmed = _claim_verdict_from_json(claim_plan, chunks, parsed)
            return confirmed.verdict == "CONTRADICTED" and bool(confirmed.evidence)
        except Exception as exc:
            if _is_local_provider_failure(exc):
                raise
            return False


def verify_claim(
    claim_plan: ClaimPlan,
    retrieved_chunks: Iterable[RetrievedChunk],
    llm: LLMClient | Verifier | None = None,
    config: NCVConfig = DEFAULT_CONFIG,
) -> ClaimVerdict:
    """Verify one claim with a mock verifier by default."""

    if llm is None:
        return MockVerifier().verify_claim(claim_plan, retrieved_chunks, config=config)
    if hasattr(llm, "verify_claim"):
        return llm.verify_claim(claim_plan, retrieved_chunks, config=config)  # type: ignore[union-attr]
    if isinstance(llm, MockLLMClient):
        return LLMVerifier(llm).verify_claim(claim_plan, retrieved_chunks, config=config)
    return LLMVerifier(llm).verify_claim(claim_plan, retrieved_chunks, config=config)


def local_verify_claim(claim: Claim, evidence: Iterable[EvidenceChunk]) -> ClaimVerdict:
    """Dependency-free conservative verifier used for dry-runs."""

    chunks = tuple(evidence)
    if not chunks:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="INSUFFICIENT",
            evidence=(),
            confidence="LOW",
            justification="No evidence chunks were provided.",
        )
    return ClaimVerdict(
        claim_id=claim.claim_id,
        verdict="INSUFFICIENT",
        evidence=(),
        confidence="LOW",
        justification="Local verifier does not infer support or contradiction.",
        metadata={"num_evidence_chunks": len(chunks), "evidence_quotes": []},
    )


def verify_claim_with_llm(client: ChatClient, claim: Claim, evidence: Iterable[EvidenceChunk]) -> ClaimVerdict:
    """Backward-compatible helper used by earlier tests."""

    chunks = tuple(evidence)
    evidence_text = "\n".join(f"[{chunk.chunk_id}] {chunk.chunk_text}" for chunk in chunks)
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Claim: {claim.claim_text}\n\nEvidence:\n{evidence_text}"},
    ]
    raw = client.complete(messages, temperature=0.0, max_tokens=512)
    parsed = json.loads(raw)
    verdict = parsed.get("verdict", "INSUFFICIENT")
    confidence = parsed.get("confidence", "LOW")
    justification = parsed.get("justification") or parsed.get("reasoning_short") or "LLM returned no justification."
    evidence_for_verdict = chunks if verdict in {"SUPPORTED", "CONTRADICTED"} else ()
    return ClaimVerdict(
        claim_id=claim.claim_id,
        verdict=verdict,
        evidence=evidence_for_verdict,
        confidence=confidence,
        justification=justification,
        metadata={"raw": parsed, "evidence_quotes": parsed.get("evidence", [])},
    )


def _verifier_messages(
    claim_plan: ClaimPlan,
    chunks: list[RetrievedChunk],
    *,
    config: NCVConfig = DEFAULT_CONFIG,
    provider: str = "",
) -> list[dict]:
    if _provider_is_local(provider, config):
        chunk_listing = "\n\n".join(f"[{chunk.chunk_id}] {_compact_chunk_text(chunk.text)}" for chunk in chunks[:6])
        return [
            {"role": "system", "content": LOCAL_VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"claim_id={claim_plan.claim_id}\nclaim={claim_plan.claim_text}\nchunks:\n{chunk_listing}",
            },
        ]

    chunk_listing = "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.text}" for chunk in chunks[:8])
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Claim id: {claim_plan.claim_id}\n"
                f"Claim: {claim_plan.claim_text}\n\n"
                f"Chunks:\n{chunk_listing}"
            ),
        },
    ]


def _claim_verdict_from_json(
    claim_plan: ClaimPlan,
    chunks: list[RetrievedChunk],
    parsed: dict,
) -> ClaimVerdict:
    claim_id = int(parsed.get("claim_id", claim_plan.claim_id or claim_plan.primary_claim.claim_id))
    verdict = str(parsed.get("verdict", "INSUFFICIENT")).upper()
    confidence = str(parsed.get("confidence", "LOW")).upper()
    reasoning = str(parsed.get("reasoning_short") or parsed.get("justification") or "No reasoning provided.")
    evidence_rows = parsed.get("evidence") or []
    if not isinstance(evidence_rows, list):
        raise ValueError("Verifier evidence field must be a list")

    evidence_chunks: list[EvidenceChunk] = []
    validated_quotes: list[dict[str, str]] = []
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for row in evidence_rows:
        if not isinstance(row, dict):
            raise ValueError("Evidence item must be an object")
        chunk_id = str(row.get("chunk_id", ""))
        quote = str(row.get("quote", ""))
        chunk = by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Evidence references unknown chunk_id `{chunk_id}`")
        if not quote or quote not in chunk.text:
            raise ValueError(f"Evidence quote is not an exact substring of chunk `{chunk_id}`")
        evidence_chunks.append(_evidence_from_retrieved(chunk))
        validated_quotes.append({"chunk_id": chunk_id, "quote": quote})

    if verdict in {"SUPPORTED", "CONTRADICTED"} and not evidence_chunks:
        raise ValueError(f"{verdict} verdict requires exact quoted evidence")
    if verdict == "INSUFFICIENT":
        evidence_chunks = []
        validated_quotes = []

    return ClaimVerdict(
        claim_id=claim_id,
        verdict=verdict,
        evidence=tuple(evidence_chunks),
        confidence=confidence,
        justification=reasoning,
        metadata={"reasoning_short": reasoning, "evidence_quotes": validated_quotes, "raw": parsed, "verifier": "llm"},
    )


def _cache_client_if_needed(llm: LLMClient, config: NCVConfig) -> LLMClient:
    if isinstance(llm, CachedLLMClient):
        return llm
    if isinstance(llm, MockLLMClient) and not config.use_llm_cache:
        return llm
    return CachedLLMClient(llm, LLMCache(config.llm_cache_path, enabled=True))


def _model_for_client(client: LLMClient, config: NCVConfig) -> str:
    provider = getattr(client, "provider", "")
    if provider in {"mock", "local", "openai"}:
        return config.model_for_provider(provider)
    return config.model_for_provider(config.llm_provider)


def _system_prompt_for_provider(provider: str) -> str:
    return LOCAL_VERIFIER_SYSTEM_PROMPT if provider == "local" else VERIFIER_SYSTEM_PROMPT


def _provider_is_local(provider: str, config: NCVConfig) -> bool:
    return provider == "local" or (not provider and config.llm_provider == "local")


def _compact_chunk_text(text: str, *, max_chars: int = 900) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit(" ", 1)[0] + " ..."


def _is_local_provider_failure(exc: Exception) -> bool:
    return "Local LLM provider" in str(exc)


def _evidence_from_retrieved(chunk: RetrievedChunk) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk.chunk_id,
        book_name=chunk.book_name,
        chunk_text=chunk.text,
        chapter_id=chunk.chapter_id,
        chapter_title=chunk.metadata.get("chapter_title"),
        chunk_index_global=chunk.metadata.get("chunk_index_global", 0),
        chunk_index_in_chapter=chunk.metadata.get("chunk_index_in_chapter", 0),
        character_mentions=tuple(chunk.metadata.get("character_mentions", ())),
        event_mentions=tuple(chunk.metadata.get("event_mentions", ())),
        plot_density_score=chunk.metadata.get("plot_density_score", 0.0),
        metadata=chunk.metadata,
    )


def _insufficient(claim_plan: ClaimPlan, reason: str, raw=None) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=claim_plan.claim_id or claim_plan.primary_claim.claim_id,
        verdict="INSUFFICIENT",
        evidence=(),
        confidence="LOW",
        justification=reason,
        metadata={"reasoning_short": reason, "evidence_quotes": [], "raw": raw, "verifier": "llm"},
    )


def _quote_window(text: str, token: str, width: int = 120) -> str:
    lowered = text.casefold()
    index = lowered.find(str(token).casefold())
    if index < 0:
        return text[:width]
    start = max(index - width // 2, 0)
    end = min(index + width // 2, len(text))
    return text[start:end]
