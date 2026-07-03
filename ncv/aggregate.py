"""Backstory-level aggregation from claim verdicts."""

from __future__ import annotations

from typing import Iterable

from .schemas import BackstoryExample, BackstoryVerdict, ClaimVerdict


def aggregate_backstory(
    example: BackstoryExample,
    claim_verdicts: Iterable[ClaimVerdict],
    *,
    binary_output: bool = True,
) -> BackstoryVerdict:
    """Convert claim-level verdicts into dataset labels and internal status."""

    verdicts = tuple(claim_verdicts)
    if not verdicts:
        raise ValueError("claim_verdicts must not be empty")

    strong_contradictions = [
        item
        for item in verdicts
        if item.verdict == "CONTRADICTED" and item.confidence in {"MEDIUM", "HIGH"} and bool(item.evidence)
    ]
    supported = [item for item in verdicts if item.verdict == "SUPPORTED"]
    insufficient = [item for item in verdicts if item.verdict == "INSUFFICIENT"]

    if strong_contradictions:
        internal_status = "INCONSISTENT"
        final_label = "contradict"
    elif supported:
        internal_status = "CONSISTENT"
        final_label = "consistent"
    else:
        internal_status = "UNKNOWN"
        final_label = "consistent" if binary_output else "UNKNOWN"

    summary = {
        "num_claims": len(verdicts),
        "num_supported": len(supported),
        "num_contradicted": sum(1 for item in verdicts if item.verdict == "CONTRADICTED"),
        "num_insufficient": len(insufficient),
        "strongest_evidence": _strongest_evidence(verdicts),
        "contradiction_claim_ids": [item.claim_id for item in strong_contradictions],
    }

    return BackstoryVerdict(
        example_id=example.id,
        verdict=internal_status,
        internal_status=internal_status,
        final_label=final_label,
        claim_verdicts=verdicts,
        summary=summary,
    )


def aggregate_claim_verdicts(example_id: str, claim_verdicts: Iterable[ClaimVerdict]) -> BackstoryVerdict:
    """Backward-compatible aggregation helper from the initial package pass."""

    example = BackstoryExample(
        id=example_id,
        book_name="unknown",
        character="unknown",
        content="unknown",
    )
    return aggregate_backstory(example, claim_verdicts, binary_output=False)


def _strongest_evidence(verdicts: tuple[ClaimVerdict, ...]) -> dict | None:
    confidence_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    candidates = [item for item in verdicts if item.evidence]
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (confidence_rank.get(item.confidence, 0), len(item.evidence)))
    quotes = best.metadata.get("evidence_quotes", [])
    return {
        "claim_id": best.claim_id,
        "verdict": best.verdict,
        "confidence": best.confidence,
        "quotes": quotes,
    }

