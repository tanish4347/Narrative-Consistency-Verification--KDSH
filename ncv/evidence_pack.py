"""Structured evidence packing for optional LLM rationale support."""

from __future__ import annotations

from typing import Iterable, Mapping

from .anchors import extract_anchors
from .constraint_index import constraint_rule_score
from .schemas import BackstoryExample, ClaimPlan, ContradictionFamily, RetrievedChunk


def build_evidence_pack(
    example: BackstoryExample,
    claim_plan: ClaimPlan,
    contradiction_families: Iterable[ContradictionFamily],
    retrieved_chunks: Iterable[RetrievedChunk],
    *,
    character_constraints: Mapping[str, object] | None = None,
    max_items_per_section: int = 4,
) -> dict[str, object]:
    anchors = extract_anchors(example)
    families = {family.family_name: family for family in contradiction_families}
    support = []
    contradiction = []
    for chunk in retrieved_chunks:
        item = {
            "chunk_id": chunk.chunk_id,
            "preview": _preview(chunk.text),
            "why_retrieved": chunk.channel,
            "score": chunk.final_score,
        }
        if "contradiction" in chunk.channel or "incompatible" in chunk.channel:
            family_name = str(chunk.metadata.get("family_name") or "")
            item["incompatible_state"] = family_name or ", ".join(claim_plan.incompatible_event_types)
            item["family_explanation"] = families.get(family_name).explanation if family_name in families else ""
            contradiction.append(item)
        else:
            support.append(item)
    constraints = constraint_rule_score(example, anchors, character_constraints or {})
    return {
        "claim": {
            "claim_id": claim_plan.claim_id,
            "claim_text": claim_plan.claim_text,
            "claim_type": claim_plan.claim_type,
        },
        "anchors": anchors.to_dict(),
        "support_candidates": support[:max_items_per_section],
        "contradiction_candidates": contradiction[:max_items_per_section],
        "canonical_constraints": {
            "constraint_risk_score": constraints["constraint_risk_score"],
            "matched_constraints": constraints["matched_constraints"],
        },
    }


def _preview(text: str, limit: int = 420) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rsplit(" ", 1)[0] + " ..."
