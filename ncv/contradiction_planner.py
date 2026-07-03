"""Contradiction-family planning.

Contradictions are represented as evidence families rather than only direct
sentence negations.
"""

from __future__ import annotations

import json

from .schemas import Claim, ClaimPlan, ContradictionFamily, RetrievalQuery


def generate_contradiction_families(claim_plan: ClaimPlan) -> list[ContradictionFamily]:
    """Generate deterministic contradiction families for a claim plan."""

    character = claim_plan.character or "the character"
    claim_id = claim_plan.claim_id
    families = [
        ContradictionFamily(
            claim_id=claim_id,
            family_name="direct_negation",
            natural_language_query=f"{character} evidence directly contradicts: {claim_plan.claim_text}",
            event_types=tuple(dict.fromkeys((*claim_plan.incompatible_event_types, "direct_negation"))),
            explanation="Searches for explicit evidence that the claim is false.",
        )
    ]

    if claim_plan.claim_type == "custody_status":
        families.append(
            ContradictionFamily(
                claim_id=claim_id,
                family_name="free_activity_during_custody",
                natural_language_query=(
                    f"{character} escaped released travelled arrived departed attended appeared met "
                    "lived outside joined others"
                ),
                event_types=("escape_release", "movement", "public_appearance"),
                explanation="A supposedly imprisoned character may be contradicted by free movement or public activity.",
            )
        )
        families.append(
            ContradictionFamily(
                claim_id=claim_id,
                family_name="incompatible_status",
                natural_language_query=f"{character} free released no longer prisoner outside prison",
                event_types=("escape_release", "status_free"),
                explanation="Searches for status evidence incompatible with ongoing custody.",
            )
        )

    elif claim_plan.claim_type == "death_status":
        families.append(
            ContradictionFamily(
                claim_id=claim_id,
                family_name="later_alive_after_death",
                natural_language_query=f"{character} spoke met travelled appeared married confessed wrote returned",
                event_types=("movement", "public_appearance", "later_action", "knowledge_secret"),
                explanation="A supposedly dead character may be contradicted by later speech, travel, meetings, or actions.",
            )
        )
        families.append(
            ContradictionFamily(
                claim_id=claim_id,
                family_name="incompatible_time",
                natural_language_query=f"{character} later afterwards subsequently returned appeared alive",
                event_types=("incompatible_time", "later_action"),
                explanation="Searches for timeline evidence after the alleged death.",
            )
        )

    elif claim_plan.claim_type == "movement_location":
        families.extend(
            [
                ContradictionFamily(
                    claim_id=claim_id,
                    family_name="incompatible_location",
                    natural_language_query=f"{character} elsewhere another place same time location",
                    event_types=("movement", "incompatible_location"),
                    explanation="Searches for evidence placing the character somewhere else.",
                ),
                ContradictionFamily(
                    claim_id=claim_id,
                    family_name="incompatible_time",
                    natural_language_query=f"{character} before after later earlier arrived departed timeline",
                    event_types=("movement", "incompatible_time"),
                    explanation="Searches for timeline evidence inconsistent with the movement claim.",
                ),
            ]
        )

    elif claim_plan.claim_type == "relationship_marriage":
        families.append(
            ContradictionFamily(
                claim_id=claim_id,
                family_name="incompatible_relationship",
                natural_language_query=f"{character} unmarried married someone else engagement broken refused",
                event_types=("marriage_relationship", "incompatible_relationship"),
                explanation="Searches for relationship evidence incompatible with the claimed marriage or engagement.",
            )
        )

    elif claim_plan.claim_type == "knowledge_secret":
        families.append(
            ContradictionFamily(
                claim_id=claim_id,
                family_name="prior_knowledge_vs_later_discovery",
                natural_language_query=f"{character} discovered learned later surprised secret revealed confessed",
                event_types=("knowledge_secret", "later_discovery"),
                explanation="Searches for evidence that knowledge was gained later rather than already possessed.",
            )
        )

    elif claim_plan.claim_type == "betrayal_deception":
        families.append(
            ContradictionFamily(
                claim_id=claim_id,
                family_name="loyalty_vs_betrayal",
                natural_language_query=f"{character} loyal helped rescued protected saved warned",
                event_types=("rescue_help", "loyalty_help"),
                explanation="Searches for loyalty or help that conflicts with a betrayal claim.",
            )
        )

    elif claim_plan.claim_type == "possession_document":
        families.append(
            ContradictionFamily(
                claim_id=claim_id,
                family_name="possession_or_document_conflict",
                natural_language_query=f"{character} document letter map dossier lost found by someone else",
                event_types=("possession_document", "document_conflict"),
                explanation="Searches for document possession evidence incompatible with the claim.",
            )
        )

    return families


def build_support_query(claim: Claim, *, book_name: str | None = None, character: str | None = None) -> RetrievalQuery:
    return RetrievalQuery(
        claim_id=claim.claim_id,
        query_text=claim.claim_text,
        query_type="support",
        book_name=book_name,
        character=character,
    )


def build_contradiction_query(
    claim: Claim,
    *,
    book_name: str | None = None,
    character: str | None = None,
) -> RetrievalQuery:
    query = f"Evidence that contradicts: {claim.claim_text}"
    return RetrievalQuery(
        claim_id=claim.claim_id,
        query_text=query,
        query_type="contradiction",
        book_name=book_name,
        character=character,
    )


def build_queries_for_claim(
    claim: Claim,
    *,
    book_name: str | None = None,
    character: str | None = None,
) -> tuple[RetrievalQuery, ...]:
    """Return default support and contradiction queries for a claim."""

    return (
        build_support_query(claim, book_name=book_name, character=character),
        build_contradiction_query(claim, book_name=book_name, character=character),
    )


def contradiction_queries_from_json(
    claim: Claim,
    contradictions_json: str,
    *,
    book_name: str | None = None,
    character: str | None = None,
) -> tuple[RetrievalQuery, ...]:
    """Parse contradiction strings for a claim into retrieval queries."""

    raw = json.loads(contradictions_json)
    if not isinstance(raw, dict):
        raise ValueError("contradictions_json must decode to an object")
    texts = raw.get(str(claim.claim_id), [])
    return tuple(
        RetrievalQuery(
            claim_id=claim.claim_id,
            query_text=text,
            query_type="contradiction",
            book_name=book_name,
            character=character,
        )
        for text in texts
    )
