"""Claim planning helpers.

A backstory sentence becomes a retrieval plan: claim type, expected evidence,
and families of incompatible narrative states to search for.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from .cache import LLMCache
from .config import DEFAULT_CONFIG, NCVConfig
from .llm import CachedLLMClient, LLMClient, MockLLMClient
from .schemas import BackstoryExample, Claim, ClaimPlan
from .utils import extract_json_list, normalize_whitespace


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|;\s+")
WORD_RE = re.compile(r"[A-Za-z0-9'-]+")

CUSTODY_TERMS = {
    "imprisoned",
    "imprison",
    "prison",
    "captive",
    "captivity",
    "confined",
    "cell",
    "dungeon",
    "arrested",
    "re-arrested",
}
DEATH_TERMS = {"died", "dead", "death", "killed", "murdered", "slain", "poisoned", "executed"}
MOVEMENT_TERMS = {
    "travelled",
    "traveled",
    "journeyed",
    "sailed",
    "arrived",
    "departed",
    "left",
    "went",
    "moved",
    "lived",
    "stayed",
    "returned",
    "appeared",
    "attended",
    "joined",
    "met",
}
MARRIAGE_TERMS = {"married", "marriage", "wife", "husband", "betrothed", "wedding", "spouse"}
BETRAYAL_TERMS = {"betrayed", "betrayal", "treason", "traitor", "deceived", "denounced", "spy", "informer"}
IDENTITY_TERMS = {"alias", "disguise", "identity", "secretly known", "assumed name", "renamed"}
KNOWLEDGE_TERMS = {"knew", "learned", "discovered", "secret", "revealed", "confessed", "knowledge"}
DOCUMENT_TERMS = {"document", "letter", "map", "dossier", "will", "paper", "message", "note", "treasure"}
RESCUE_TERMS = {"rescued", "saved", "helped", "aided", "protected", "freed"}


def split_candidate_claims(text: str) -> list[str]:
    """Heuristically split a backstory into factual sentence candidates."""

    parts = [normalize_whitespace(part.strip(" ;\n\t")) for part in SENTENCE_RE.split(text.strip())]
    return [part for part in parts if _is_usable_claim(part)]


def plan_claims(
    example: BackstoryExample,
    llm: LLMClient | None = None,
    config: NCVConfig = DEFAULT_CONFIG,
) -> list[ClaimPlan]:
    """Plan claims using an optional LLM, with deterministic fallback."""

    if llm is None:
        return fallback_plan_claims(example, max_claims=config.max_claims)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a literary fact extraction system for narrative consistency retrieval. "
                "Output strict JSON list only. "
                "Each item must include claim_id, claim_text, claim_type, expected_event_types, "
                "incompatible_event_types, support_queries, incompatible_states, and contradiction_queries."
            ),
        },
        {
            "role": "user",
            "content": f"Book: {example.book_name}\nCharacter: {example.character}\nBackstory:\n{example.backstory}",
        },
    ]

    client = _cache_client_if_needed(llm, config)
    model = _model_for_client(client, config)

    for attempt in range(2):
        attempt_messages = messages
        if attempt:
            attempt_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Retry with JSON only. Return a list of objects. No markdown, no prose, "
                        "no comments."
                    ),
                },
            ]
        try:
            raw = client.generate(
                attempt_messages,
                model=model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                timeout_s=config.llm_timeout_s,
            )
            return _plans_from_llm_json(example, extract_json_list(raw), max_claims=config.max_claims)
        except Exception:
            continue
    return fallback_plan_claims(example, max_claims=config.max_claims)


def fallback_plan_claims(example: BackstoryExample, *, max_claims: int = 6) -> list[ClaimPlan]:
    """Deterministic planner used when no LLM is available."""

    sentences = split_candidate_claims(example.backstory)
    if not sentences:
        sentences = [normalize_whitespace(example.backstory)]
    sentences = sentences[:max_claims]
    plans: list[ClaimPlan] = []
    for claim_id, claim_text in enumerate(sentences, start=1):
        plans.append(_plan_single_claim(example, claim_id, claim_text))
    return plans


def plan_claims_from_backstory(example: BackstoryExample, *, max_claims: int | None = None) -> ClaimPlan:
    """Backward-compatible aggregate claim plan from the earlier package scaffold."""

    plans = fallback_plan_claims(example, max_claims=max_claims or DEFAULT_CONFIG.max_claims)
    claims = tuple(plan.primary_claim for plan in plans)
    return ClaimPlan(example_id=example.id, claims=claims)


def claims_from_json(example_id: str, claims_json: str) -> ClaimPlan:
    """Parse the repository's generated `claims` JSON column."""

    raw = json.loads(claims_json)
    if not isinstance(raw, list):
        raise ValueError("claims_json must decode to a list")
    claims = tuple(
        Claim(
            claim_id=item["claim_id"],
            claim_text=item["claim_text"],
            keywords=tuple(item.get("keywords", ())),
            source="generated",
        )
        for item in raw
    )
    return ClaimPlan(example_id=example_id, claims=claims)


def merge_claims(example_id: str, claims: Iterable[Claim], implied_states: Iterable[Claim] = ()) -> ClaimPlan:
    """Build a claim plan from already-created claim objects."""

    return ClaimPlan(example_id=example_id, claims=tuple(claims), implied_states=tuple(implied_states))


def _plan_single_claim(example: BackstoryExample, claim_id: int, claim_text: str) -> ClaimPlan:
    claim_type = identify_claim_type(claim_text)
    expected_event_types, incompatible_event_types = event_types_for_claim_type(claim_type)
    incompatible_states = incompatible_states_for_claim(claim_type, example.character)
    support_queries = support_queries_for_claim(claim_text, claim_type, example.character, example.book_name)
    contradiction_queries = contradiction_queries_for_claim(claim_text, incompatible_states, example.character, example.book_name)
    claim = Claim(
        claim_id=claim_id,
        claim_text=claim_text,
        keywords=tuple(_keywords(claim_text)),
        source="fallback",
    )
    return ClaimPlan(
        example_id=example.id,
        claims=(claim,),
        claim_id=claim_id,
        claim_text=claim_text,
        book_name=example.book_name,
        character=example.character,
        claim_type=claim_type,
        expected_event_types=expected_event_types,
        incompatible_event_types=incompatible_event_types,
        support_queries=support_queries,
        incompatible_states=incompatible_states,
        contradiction_queries=contradiction_queries,
        metadata={"planner": "fallback"},
    )


def identify_claim_type(claim_text: str) -> str:
    lowered = claim_text.casefold()
    tokens = set(WORD_RE.findall(lowered))
    if _contains_any(lowered, tokens, DEATH_TERMS):
        return "death_status"
    if _contains_any(lowered, tokens, CUSTODY_TERMS):
        return "custody_status"
    if _contains_any(lowered, tokens, MARRIAGE_TERMS):
        return "relationship_marriage"
    if _contains_any(lowered, tokens, BETRAYAL_TERMS):
        return "betrayal_deception"
    if _contains_any(lowered, tokens, IDENTITY_TERMS):
        return "identity_alias"
    if _contains_any(lowered, tokens, KNOWLEDGE_TERMS):
        return "knowledge_secret"
    if _contains_any(lowered, tokens, DOCUMENT_TERMS):
        return "possession_document"
    if _contains_any(lowered, tokens, RESCUE_TERMS):
        return "rescue_help"
    if _contains_any(lowered, tokens, MOVEMENT_TERMS):
        return "movement_location"
    return "generic"


def event_types_for_claim_type(claim_type: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapping = {
        "custody_status": (
            ("custody", "captivity"),
            ("escape_release", "escape", "movement", "voyage", "public_appearance", "later_action"),
        ),
        "death_status": (
            ("death", "violence"),
            ("movement", "voyage", "public_appearance", "later_action", "marriage_relationship", "knowledge_secret"),
        ),
        "movement_location": (("movement", "voyage", "public_appearance"), ("incompatible_location", "incompatible_time")),
        "relationship_marriage": (("marriage_relationship", "family_relationship"), ("incompatible_relationship",)),
        "betrayal_deception": (("betrayal_deception",), ("rescue_help", "loyalty_help")),
        "identity_alias": (("identity_alias",), ("public_identity_conflict",)),
        "knowledge_secret": (("knowledge_secret",), ("prior_knowledge", "later_discovery")),
        "possession_document": (("possession_document",), ("document_conflict",)),
        "rescue_help": (("rescue_help",), ("betrayal_deception", "abandonment")),
    }
    return mapping.get(claim_type, (("generic",), ("direct_negation",)))


def incompatible_states_for_claim(claim_type: str, character: str) -> tuple[str, ...]:
    if claim_type == "custody_status":
        return (
            f"{character} escapes confinement",
            f"{character} is released",
            f"{character} travels freely",
            f"{character} appears publicly elsewhere",
            f"{character} attends or joins events",
            f"{character} lives outside prison",
        )
    if claim_type == "death_status":
        return (
            f"{character} speaks later",
            f"{character} travels later",
            f"{character} meets someone later",
            f"{character} appears publicly later",
            f"{character} returns later",
            f"{character} performs actions after supposed death",
        )
    if claim_type == "relationship_marriage":
        return (
            f"{character} is unmarried",
            f"{character} marries someone else",
            f"{character} rejects or breaks the engagement",
        )
    if claim_type == "betrayal_deception":
        return (
            f"{character} remains loyal",
            f"{character} rescues or helps the alleged victim",
            f"{character} protects rather than betrays",
        )
    if claim_type == "knowledge_secret":
        return (
            f"{character} learns the secret later",
            f"{character} is surprised by a later discovery",
            f"{character} lacks prior knowledge",
        )
    if claim_type == "possession_document":
        return (
            f"{character} lacks the document",
            f"someone else possesses the document",
            f"the document is discovered elsewhere",
        )
    if claim_type == "movement_location":
        return (
            f"{character} is somewhere incompatible at the same time",
            f"{character} arrives or departs on a conflicting timeline",
        )
    return (f"evidence contradicts the claim about {character}",)


def support_queries_for_claim(claim_text: str, claim_type: str, character: str, book_name: str) -> tuple[str, ...]:
    if claim_type == "custody_status":
        return (
            f"{character} imprisoned prison cell dungeon",
            f"{character} captivity confined",
        )
    if claim_type == "death_status":
        return (
            f"{character} died dead killed death",
            f"{character} murder execution poisoned",
        )
    if claim_type == "movement_location":
        return (
            f"{character} travelled arrived departed returned",
            f"{character} appeared attended met joined",
        )
    if claim_type == "relationship_marriage":
        return (
            f"{character} married wife husband betrothed wedding",
            f"{character} relationship marriage {book_name}",
        )
    if claim_type == "betrayal_deception":
        return (
            f"{character} betrayed treason deception denounced",
            f"{character} traitor spy informer",
        )
    return (
        f"{character} {claim_text}",
        f"{book_name} {character} {' '.join(_keywords(claim_text)[:6])}",
    )


def contradiction_queries_for_claim(
    claim_text: str,
    incompatible_states: tuple[str, ...],
    character: str,
    book_name: str,
) -> tuple[str, ...]:
    queries = [f"{book_name} {state}" for state in incompatible_states]
    queries.append(f"{character} incompatible evidence against claim: {claim_text}")
    return tuple(queries)


def _plans_from_llm_json(example: BackstoryExample, raw: list, *, max_claims: int) -> list[ClaimPlan]:
    plans: list[ClaimPlan] = []
    for index, item in enumerate(raw[:max_claims], start=1):
        if not isinstance(item, dict):
            raise ValueError("LLM claim plan item must be an object")
        claim_text = str(item.get("claim_text") or "").strip()
        if not claim_text:
            raise ValueError("LLM claim plan item missing claim_text")
        claim_id = int(item.get("claim_id") or index)
        claim = Claim(claim_id=claim_id, claim_text=claim_text, keywords=tuple(_keywords(claim_text)), source="llm")
        plans.append(
            ClaimPlan(
                example_id=example.id,
                claims=(claim,),
                claim_id=claim_id,
                claim_text=claim_text,
                book_name=example.book_name,
                character=example.character,
                claim_type=str(item.get("claim_type") or identify_claim_type(claim_text)),
                expected_event_types=tuple(item.get("expected_event_types") or ()),
                incompatible_event_types=tuple(item.get("incompatible_event_types") or ()),
                support_queries=tuple(item.get("support_queries") or ()),
                incompatible_states=tuple(item.get("incompatible_states") or ()),
                contradiction_queries=tuple(item.get("contradiction_queries") or ()),
                metadata={"planner": "llm"},
            )
        )
    if not plans:
        raise ValueError("LLM returned no claim plans")
    return plans


def _is_usable_claim(text: str) -> bool:
    if len(text.split()) < 3:
        return False
    if len(text) < 12:
        return False
    return any(char.isalpha() for char in text)


def _keywords(text: str) -> list[str]:
    stop = {"the", "and", "that", "with", "from", "into", "were", "was", "his", "her", "for", "but"}
    out: list[str] = []
    for token in WORD_RE.findall(text.casefold()):
        if len(token) > 2 and token not in stop and token not in out:
            out.append(token)
    return out[:10]


def _contains_any(lowered: str, tokens: set[str], terms: set[str]) -> bool:
    for term in terms:
        if " " in term and term in lowered:
            return True
        if term in tokens:
            return True
    return False


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
