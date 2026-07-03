"""Run retrieval diagnostics without any LLM calls."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.claim_planner import plan_claims
from ncv.config import NCVConfig
from ncv.contradiction_planner import generate_contradiction_families
from ncv.data import load_train_examples
from ncv.ingest import build_chunks, write_chunks_jsonl
from ncv.rerank import rerank_evidence
from ncv.retrieval import HybridRetriever, load_chunks_jsonl
from ncv.schemas import BackstoryExample, ClaimPlan, RetrievedChunk


CUSTODY_INCOMPATIBLE_EVENTS = {"escape", "rescue", "voyage", "movement", "escape_release", "public_appearance"}
CUSTODY_INCOMPATIBLE_TERMS = {
    "escaped",
    "escape",
    "released",
    "freed",
    "travelled",
    "traveled",
    "arrived",
    "departed",
    "appeared",
    "attended",
    "joined",
    "met",
}
DEATH_INCOMPATIBLE_EVENTS = {"voyage", "movement", "public_appearance", "later_action", "knowledge_secret"}
DEATH_INCOMPATIBLE_TERMS = {
    "later",
    "afterwards",
    "subsequently",
    "spoke",
    "met",
    "travelled",
    "traveled",
    "appeared",
    "returned",
    "wrote",
}


@dataclass(frozen=True, slots=True)
class DiagnosticsResult:
    metrics: dict[str, object]
    examples: list[dict[str, object]]
    claim_rows: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-LLM retrieval diagnostics.")
    parser.add_argument("--dataset", default="outputs/dev_split.csv")
    parser.add_argument("--books-dir", default="Dataset/Books")
    parser.add_argument("--chunks", default="outputs/chunks.jsonl")
    parser.add_argument("--output", default="outputs/retrieval_diagnostics.jsonl")
    parser.add_argument("--report", default="outputs/retrieval_diagnostics_report.md")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--use-embeddings", default="false")
    parser.add_argument("--embedding-mode", default="auto", choices=("auto", "mock"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_diagnostics(
        dataset=args.dataset,
        books_dir=args.books_dir,
        chunks_path=args.chunks,
        max_rows=args.max_rows,
        use_embeddings=_as_bool(args.use_embeddings),
        embedding_mode=args.embedding_mode,
    )
    write_diagnostics_jsonl(args.output, result.claim_rows)
    write_diagnostics_report(args.report, result)
    print(format_summary(result.metrics))
    print(f"Wrote diagnostics rows to {args.output}")
    print(f"Wrote diagnostics report to {args.report}")


def run_diagnostics(
    *,
    dataset: str | Path,
    books_dir: str | Path,
    chunks_path: str | Path,
    max_rows: int | None = None,
    use_embeddings: bool = False,
    embedding_mode: str = "auto",
) -> DiagnosticsResult:
    examples = load_train_examples(dataset)
    if max_rows is not None:
        examples = examples[:max_rows]

    chunks_file = Path(chunks_path)
    if chunks_file.exists():
        chunks = load_chunks_jsonl(chunks_file)
    else:
        chunks = build_chunks(books_dir)
        write_chunks_jsonl(chunks, chunks_file)

    config = NCVConfig(books_dir=Path(books_dir), use_embeddings=use_embeddings, embedding_mode=embedding_mode)
    retriever = HybridRetriever(chunks, config)
    claim_rows: list[dict[str, object]] = []
    readable_examples: list[dict[str, object]] = []

    for example in examples:
        plans = plan_claims(example, llm=None, config=config)
        for plan in plans:
            families = generate_contradiction_families(plan)
            retrieved = rerank_evidence(plan, families, retriever.retrieve_for_claim(plan, families), config)
            row = _claim_diagnostics_row(example, plan, families, retrieved)
            claim_rows.append(row)
            if len(readable_examples) < 5:
                readable_examples.append(_readable_example(row, retrieved))

    metrics = _compute_diagnostic_metrics(examples, claim_rows, semantic_enabled=config.use_embeddings)
    return DiagnosticsResult(metrics=metrics, examples=readable_examples, claim_rows=claim_rows)


def write_diagnostics_jsonl(path: str | Path, rows: list[dict[str, object]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return output_path


def write_diagnostics_report(path: str | Path, result: DiagnosticsResult) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_report(result), encoding="utf-8")
    return output_path


def format_summary(metrics: dict[str, object]) -> str:
    return "\n".join(
        [
            f"dev_examples: {metrics['num_dev_examples']}",
            f"claims: {metrics['num_claims']}",
            f"avg_claims_per_example: {metrics['avg_claims_per_example']:.3f}",
            f"avg_retrieved_chunks_per_claim: {metrics['avg_retrieved_chunks_per_claim']:.3f}",
            f"target_character_hit_rate: {metrics['target_character_hit_rate']:.3f}",
            f"event_match_hit_rate: {metrics['event_match_hit_rate']:.3f}",
            f"semantic_enabled: {metrics['semantic_enabled']}",
        ]
    )


def format_report(result: DiagnosticsResult) -> str:
    metrics = result.metrics
    lines = [
        "# Retrieval Diagnostics Report",
        "",
        "No LLM calls are used in this report. Claims come from the deterministic fallback planner.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "num_dev_examples",
        "num_claims",
        "avg_claims_per_example",
        "avg_retrieved_chunks_per_claim",
        "target_character_hit_rate",
        "event_match_hit_rate",
        "semantic_enabled",
        "semantic_nonzero_score_rate",
        "custody_claims",
        "custody_incompatible_candidate_rate",
        "death_claims",
        "death_incompatible_candidate_rate",
    ):
        value = metrics[key]
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| `{key}` | {rendered} |")

    lines.extend(["", "## Readable Examples", ""])
    for index, example in enumerate(result.examples, start=1):
        lines.extend(
            [
                f"### Example {index}",
                "",
                f"- id: `{example['id']}`",
                f"- character: `{example['character']}`",
                f"- claim: {example['claim']}",
                f"- contradiction families: {', '.join(example['contradiction_families'])}",
                f"- top chunk IDs: {', '.join(example['top_chunk_ids'])}",
                "- top score breakdown:",
                "```json",
                json.dumps(example["top_score_breakdown"], indent=2, sort_keys=True),
                "```",
                f"- preview: {example['preview']}",
                "",
            ]
        )
    return "\n".join(lines)


def _claim_diagnostics_row(
    example: BackstoryExample,
    plan: ClaimPlan,
    families,
    retrieved: list[RetrievedChunk],
) -> dict[str, object]:
    return {
        "id": example.id,
        "book_name": example.book_name,
        "character": example.character,
        "claim_id": plan.claim_id,
        "claim": plan.claim_text,
        "claim_type": plan.claim_type,
        "expected_event_types": list(plan.expected_event_types),
        "incompatible_event_types": list(plan.incompatible_event_types),
        "contradiction_families": [family.family_name for family in families],
        "num_retrieved_chunks": len(retrieved),
        "has_target_character_chunk": any(_has_target_character(example.character, chunk) for chunk in retrieved),
        "has_event_matching_chunk": any(_has_event_match(plan, chunk) for chunk in retrieved),
        "has_custody_incompatible_candidate": (
            any(_has_custody_incompatible_candidate(chunk) for chunk in retrieved)
            if plan.claim_type == "custody_status"
            else None
        ),
        "has_death_incompatible_candidate": (
            any(_has_death_incompatible_candidate(chunk) for chunk in retrieved)
            if plan.claim_type == "death_status"
            else None
        ),
        "top_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "channel": chunk.channel,
                "final_score": chunk.final_score,
                "event_mentions": list(chunk.metadata.get("event_mentions", [])),
                "character_mentions": list(chunk.metadata.get("character_mentions", [])),
                "score_breakdown": dict(chunk.score_breakdown),
                "preview": chunk.text[:260],
            }
            for chunk in retrieved[:5]
        ],
    }


def _readable_example(row: dict[str, object], retrieved: list[RetrievedChunk]) -> dict[str, object]:
    top = retrieved[0] if retrieved else None
    return {
        "id": str(row["id"]),
        "character": str(row["character"]),
        "claim": str(row["claim"]),
        "contradiction_families": list(row["contradiction_families"]),
        "top_chunk_ids": [chunk.chunk_id for chunk in retrieved[:5]],
        "top_score_breakdown": dict(top.score_breakdown) if top else {},
        "preview": (top.text[:360] if top else "No chunks retrieved."),
    }


def _compute_diagnostic_metrics(
    examples: list[BackstoryExample],
    claim_rows: list[dict[str, object]],
    *,
    semantic_enabled: bool,
) -> dict[str, object]:
    num_claims = len(claim_rows)
    retrieved_counts = [int(row["num_retrieved_chunks"]) for row in claim_rows]
    custody_rows = [row for row in claim_rows if row["claim_type"] == "custody_status"]
    death_rows = [row for row in claim_rows if row["claim_type"] == "death_status"]
    semantic_scores = [
        float(chunk.get("score_breakdown", {}).get("semantic_score", 0.0))
        for row in claim_rows
        for chunk in row.get("top_chunks", [])
    ]
    return {
        "num_dev_examples": len(examples),
        "num_claims": num_claims,
        "avg_claims_per_example": (num_claims / len(examples)) if examples else 0.0,
        "avg_retrieved_chunks_per_claim": mean(retrieved_counts) if retrieved_counts else 0.0,
        "target_character_hit_rate": _rate(row["has_target_character_chunk"] for row in claim_rows),
        "event_match_hit_rate": _rate(row["has_event_matching_chunk"] for row in claim_rows),
        "semantic_enabled": semantic_enabled,
        "semantic_nonzero_score_rate": _rate(score > 0 for score in semantic_scores),
        "custody_claims": len(custody_rows),
        "custody_incompatible_candidate_rate": _rate(
            row["has_custody_incompatible_candidate"] for row in custody_rows
        ),
        "death_claims": len(death_rows),
        "death_incompatible_candidate_rate": _rate(row["has_death_incompatible_candidate"] for row in death_rows),
    }


def _has_target_character(character: str, chunk: RetrievedChunk) -> bool:
    target = character.casefold()
    mentions = {str(item).casefold() for item in chunk.metadata.get("character_mentions", [])}
    if target in mentions or target in chunk.text.casefold():
        return True
    if "/" in target:
        return any(part.strip() in mentions or part.strip() in chunk.text.casefold() for part in target.split("/"))
    return False


def _has_event_match(plan: ClaimPlan, chunk: RetrievedChunk) -> bool:
    desired = {event.casefold() for event in (*plan.expected_event_types, *plan.incompatible_event_types)}
    if not desired:
        return False
    events = {str(event).casefold() for event in chunk.metadata.get("event_mentions", [])}
    text = chunk.text.casefold()
    return bool(desired & events) or any(event in text for event in desired)


def _has_custody_incompatible_candidate(chunk: RetrievedChunk) -> bool:
    events = {str(event).casefold() for event in chunk.metadata.get("event_mentions", [])}
    text = chunk.text.casefold()
    return bool(events & CUSTODY_INCOMPATIBLE_EVENTS) or any(term in text for term in CUSTODY_INCOMPATIBLE_TERMS)


def _has_death_incompatible_candidate(chunk: RetrievedChunk) -> bool:
    events = {str(event).casefold() for event in chunk.metadata.get("event_mentions", [])}
    text = chunk.text.casefold()
    return bool(events & DEATH_INCOMPATIBLE_EVENTS) or any(term in text for term in DEATH_INCOMPATIBLE_TERMS)


def _rate(values) -> float:
    items = [bool(value) for value in values]
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
