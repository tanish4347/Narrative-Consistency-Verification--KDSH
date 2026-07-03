"""Run the NCV pipeline end to end.

Example:
    python scripts/run_pipeline.py --dataset Dataset/train.csv --books-dir Dataset/Books --llm-provider mock --max-rows 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.aggregate import aggregate_backstory
from ncv.claim_planner import plan_claims
from ncv.config import NCVConfig
from ncv.contradiction_planner import generate_contradiction_families
from ncv.data import load_test_examples, load_train_examples
from ncv.ingest import build_chunks, write_chunks_jsonl
from ncv.llm import get_llm_client
from ncv.anchors import extract_anchors
from ncv.constraint_index import constraint_rule_score, load_character_constraints
from ncv.meta_classifier import load_meta_artifact, predict_meta_examples
from ncv.rerank import rerank_evidence
from ncv.retrieval import HybridRetriever, load_chunks_jsonl
from ncv.schemas import ClaimPlan, ClaimVerdict
from ncv.supervised import load_supervised_artifact, predict_examples_with_artifact
from ncv.timeline import build_timeline
from ncv.verifier import LLMVerifier, MockVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Narrative Consistency Verification.")
    parser.add_argument("--dataset", default="Dataset/train.csv")
    parser.add_argument("--books-dir", default="Dataset/Books")
    parser.add_argument("--chunks", default="outputs/chunks.jsonl")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--mode", default="train_eval", choices=("train_eval", "predict"))
    parser.add_argument("--llm-provider", default=None, choices=("mock", "openai", "local"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-cache", default="true")
    parser.add_argument("--use-lexical", default="true")
    parser.add_argument("--use-embeddings", default="false")
    parser.add_argument("--embedding-mode", default="auto", choices=("auto", "mock"))
    parser.add_argument("--use-character-scoring", default="true")
    parser.add_argument("--use-event-scoring", default="true")
    parser.add_argument("--use-contradiction-families", default="true")
    parser.add_argument("--use-timeline-features", default="true")
    parser.add_argument("--rerank", default="true")
    parser.add_argument("--strict-character-filter", default="false")
    parser.add_argument("--binary-output", default="true")
    parser.add_argument("--write-trace", default="true")
    parser.add_argument("--max-claims-per-example", type=int, default=4)
    parser.add_argument("--max-families-per-claim", type=int, default=4)
    parser.add_argument("--verifier-top-k", type=int, default=6)
    parser.add_argument("--disable-second-pass", default="false")
    parser.add_argument("--write-incremental", default="true")
    parser.add_argument("--row-timeout-s", type=int, default=180)
    parser.add_argument("--claim-timeout-s", type=int, default=90)
    parser.add_argument("--decision-mode", default="hybrid", choices=("verifier", "meta", "supervised", "rules", "hybrid"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "pipeline_trace.jsonl"
    write_trace = _as_bool(args.write_trace)
    if write_trace and trace_path.exists():
        trace_path.unlink()

    base_config = NCVConfig.from_env()
    provider = args.llm_provider or base_config.llm_provider
    real_provider = _is_real_provider(provider)
    max_claims_per_example = _arg(args, "max_claims_per_example", 4)
    max_claims = max_claims_per_example if max_claims_per_example > 0 else base_config.max_claims
    llm_timeout_s = base_config.llm_timeout_s
    claim_timeout_s = _arg(args, "claim_timeout_s", 90)
    if real_provider and claim_timeout_s > 0:
        llm_timeout_s = min(base_config.llm_timeout_s, claim_timeout_s)
    config = NCVConfig(
        books_dir=Path(args.books_dir),
        train_path=base_config.train_path,
        test_path=base_config.test_path,
        output_dir=output_dir,
        embedding_model=base_config.embedding_model,
        chunk_size=base_config.chunk_size,
        chunk_overlap=base_config.chunk_overlap,
        top_k_retrieval=args.top_k or base_config.top_k_retrieval,
        llm_provider=provider,
        max_claims=max_claims,
        use_llm_cache=_as_bool(args.use_cache),
        strict_character_filter=_as_bool(args.strict_character_filter),
        llm_cache_path=base_config.llm_cache_path,
        local_llm_model=base_config.local_llm_model,
        local_llm_base_url=base_config.local_llm_base_url,
        local_llm_api_key=base_config.local_llm_api_key,
        openai_model=base_config.openai_model,
        llm_temperature=base_config.llm_temperature,
        llm_max_tokens=base_config.llm_max_tokens,
        llm_timeout_s=llm_timeout_s,
        use_rerank=_as_bool(args.rerank),
        use_lexical=_as_bool(args.use_lexical),
        use_embeddings=_as_bool(args.use_embeddings),
        embedding_mode=args.embedding_mode,
        use_character_scoring=_as_bool(args.use_character_scoring),
        use_event_scoring=_as_bool(args.use_event_scoring),
        use_contradiction_families=_as_bool(args.use_contradiction_families),
        use_timeline_features=_as_bool(args.use_timeline_features),
    )

    examples = _load_examples(Path(args.dataset), args.mode)
    if args.max_rows is not None:
        examples = examples[: args.max_rows]

    chunks_path = Path(args.chunks)
    if chunks_path.exists():
        chunks = load_chunks_jsonl(chunks_path)
    else:
        chunks = build_chunks(args.books_dir, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        write_chunks_jsonl(chunks, chunks_path)

    timeline = build_timeline(chunks)
    verifier = _build_verifier(provider, config, disable_second_pass=_as_bool(_arg(args, "disable_second_pass", "false")))
    retriever = HybridRetriever(chunks, config)
    decision_context = _load_decision_context(_arg(args, "decision_mode", "hybrid"), output_dir)
    predictions: list[dict[str, object]] = []
    write_incremental = _as_bool(_arg(args, "write_incremental", "true"))
    row_timeout_s = _arg(args, "row_timeout_s", 180)
    max_families_per_claim = _arg(args, "max_families_per_claim", 4)
    verifier_top_k = _arg(args, "verifier_top_k", 6)

    print(f"Loaded {len(examples)} examples and {len(chunks)} chunks.")
    print(f"Timeline characters: {len(timeline)}")

    for row_index, example in enumerate(examples, start=1):
        print(f"[{row_index}/{len(examples)}] {example.id} | {example.character}")
        row_start = time.monotonic()
        row_error = None
        claim_plans = plan_claims(example, llm=None, config=config)
        claim_verdicts = []
        trace_claims = []
        for claim_plan in claim_plans:
            if _timed_out(row_start, row_timeout_s):
                row_error = f"row_timeout_s exceeded before claim {claim_plan.claim_id}"
                break
            claim_start = time.monotonic()
            families = _cap_items(generate_contradiction_families(claim_plan), max_families_per_claim)
            retrieved = []
            reranked = []
            verifier_chunks = []
            try:
                retrieved = retriever.retrieve_for_claim(claim_plan, families)
                reranked = rerank_evidence(claim_plan, families, retrieved, config)
                verifier_chunks = _evidence_for_verifier(reranked, verifier_top_k, real_provider)
                claim_verdict = verifier.verify_claim(claim_plan, verifier_chunks, config=config)
            except Exception as exc:
                claim_verdict = _insufficient_verdict(claim_plan, f"Claim failed or timed out: {exc}")
            claim_elapsed = time.monotonic() - claim_start
            claim_verdicts.append(claim_verdict)
            if write_trace:
                trace_claims.append(
                    {
                        "claim_plan": claim_plan.to_dict(),
                        "contradiction_families": [family.to_dict() for family in families],
                        "num_retrieved_chunks": len(reranked),
                        "num_chunks_sent_to_verifier": len(verifier_chunks),
                        "verifier_chunk_ids": [item.chunk_id for item in verifier_chunks],
                        "prompt_char_estimate": _estimate_verifier_prompt_chars(claim_plan, verifier_chunks, provider),
                        "claim_elapsed_s": round(claim_elapsed, 3),
                        "retrieved_chunks": [item.to_dict() for item in reranked],
                        "verifier_output": _claim_verdict_to_dict(claim_verdict),
                    }
                )
            if _timed_out(row_start, row_timeout_s):
                row_error = f"row_timeout_s exceeded after claim {claim_plan.claim_id}"
                break

        if claim_verdicts:
            final = aggregate_backstory(example, claim_verdicts, binary_output=_as_bool(args.binary_output))
            decision = _decide_final_label(
                example,
                final.final_label or "consistent",
                claim_verdicts,
                chunks,
                decision_context,
                _arg(args, "decision_mode", "hybrid"),
            )
            prediction = {
                "id": example.id,
                "book_name": example.book_name,
                "character": example.character,
                "predicted_label": decision["predicted_label"],
                "internal_status": "ERROR" if row_error else final.internal_status,
                "num_claims": final.summary["num_claims"],
                "num_supported": final.summary["num_supported"],
                "num_contradicted": final.summary["num_contradicted"],
                "num_insufficient": final.summary["num_insufficient"],
                "error": row_error or "",
                "decision_mode": decision["decision_mode"],
                "decision_score": decision.get("score_contradict", ""),
            }
        else:
            final = None
            prediction = _error_prediction(example, row_error or "No claims were verified.")
        predictions.append(prediction)
        if write_incremental:
            _write_predictions(output_path, predictions)
        if write_trace:
            _append_trace(
                trace_path,
                {
                    "example": {
                        "id": example.id,
                        "book_name": example.book_name,
                        "character": example.character,
                        "label": example.label,
                        "caption": example.caption,
                        "backstory": example.backstory,
                    },
                        "claims": trace_claims,
                        "row_elapsed_s": round(time.monotonic() - row_start, 3),
                        "row_error": row_error,
                        "decision_mode": prediction.get("decision_mode"),
                        "decision_score": prediction.get("decision_score"),
                        "final_aggregation": _backstory_verdict_to_dict(final) if final is not None else None,
                    },
            )

    _write_predictions(output_path, predictions)
    summary = _prediction_summary(predictions)
    print(f"Wrote predictions to {output_path}")
    if write_trace:
        print(f"Wrote trace to {trace_path}")
    print(f"Summary: {summary}")
    return {
        "predictions": predictions,
        "summary": summary,
        "output_path": str(output_path),
        "trace_path": str(trace_path) if write_trace else None,
        "num_examples": len(examples),
    }


def _load_examples(dataset_path: Path, mode: str):
    if mode == "train_eval":
        return load_train_examples(dataset_path)
    return load_test_examples(dataset_path)


def _build_verifier(provider: str, config: NCVConfig, *, disable_second_pass: bool = False):
    if provider == "mock":
        return MockVerifier()
    max_retries = 1 if provider == "local" else 2
    return LLMVerifier(
        get_llm_client(provider, config),
        max_retries=max_retries,
        confirm_contradictions=not disable_second_pass,
    )


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "book_name",
        "character",
        "predicted_label",
        "internal_status",
        "num_claims",
        "num_supported",
        "num_contradicted",
        "num_insufficient",
        "error",
        "decision_mode",
        "decision_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_trace(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _claim_verdict_to_dict(verdict) -> dict:
    return {
        "claim_id": verdict.claim_id,
        "verdict": verdict.verdict,
        "confidence": verdict.confidence,
        "justification": verdict.justification,
        "evidence_chunk_ids": [chunk.chunk_id for chunk in verdict.evidence],
        "metadata": dict(verdict.metadata),
    }


def _backstory_verdict_to_dict(verdict) -> dict:
    return {
        "example_id": verdict.example_id,
        "final_label": verdict.final_label,
        "internal_status": verdict.internal_status,
        "summary": dict(verdict.summary),
    }


def _prediction_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        label = str(row.get("predicted_label"))
        summary[label] = summary.get(label, 0) + 1
    return summary


def _is_real_provider(provider: str) -> bool:
    return provider in {"local", "openai"}


def _arg(args: argparse.Namespace, name: str, default):
    return getattr(args, name, default)


def _timed_out(start: float, timeout_s: int | None) -> bool:
    return bool(timeout_s and timeout_s > 0 and (time.monotonic() - start) > timeout_s)


def _cap_items(items, limit: int):
    values = list(items)
    if limit <= 0:
        return values
    return values[:limit]


def _evidence_for_verifier(reranked, verifier_top_k: int, real_provider: bool):
    values = list(reranked)
    if not real_provider or verifier_top_k <= 0:
        return values
    return values[:verifier_top_k]


def _estimate_verifier_prompt_chars(claim_plan: ClaimPlan, chunks, provider: str) -> int:
    if provider == "local":
        return len(str(claim_plan.claim_text or "")) + sum(min(len(item.text), 900) for item in chunks)
    return len(str(claim_plan.claim_text or "")) + sum(len(item.text) for item in chunks[:8])


def _insufficient_verdict(claim_plan: ClaimPlan, reason: str) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=claim_plan.claim_id or claim_plan.primary_claim.claim_id,
        verdict="INSUFFICIENT",
        evidence=(),
        confidence="LOW",
        justification=reason[:500],
        metadata={"reasoning_short": reason[:500], "evidence_quotes": [], "verifier": "pipeline_guard"},
    )


def _error_prediction(example, error: str) -> dict[str, object]:
    return {
        "id": example.id,
        "book_name": example.book_name,
        "character": example.character,
        "predicted_label": "consistent",
        "internal_status": "ERROR",
        "num_claims": 0,
        "num_supported": 0,
        "num_contradicted": 0,
        "num_insufficient": 0,
        "error": error[:500],
        "decision_mode": "error",
        "decision_score": "",
    }


def _load_decision_context(decision_mode: str, output_dir: Path) -> dict[str, object]:
    context: dict[str, object] = {}
    if decision_mode in {"meta", "hybrid"}:
        meta_path = output_dir / "meta_model.joblib"
        if meta_path.exists():
            context["meta"] = load_meta_artifact(meta_path)
    if decision_mode in {"supervised", "hybrid"} or "meta" not in context:
        supervised_path = output_dir / "supervised_model.joblib"
        if supervised_path.exists():
            context["supervised"] = load_supervised_artifact(supervised_path)
    if decision_mode in {"rules", "hybrid"}:
        constraints_path = Path("data/character_constraints.yaml")
        if constraints_path.exists():
            context["constraints"] = load_character_constraints(constraints_path)
    return context


def _decide_final_label(
    example,
    verifier_label: str,
    claim_verdicts: list[ClaimVerdict],
    chunks,
    context: dict[str, object],
    decision_mode: str,
) -> dict[str, object]:
    if decision_mode == "verifier":
        return {"predicted_label": verifier_label, "decision_mode": "verifier", "score_contradict": ""}
    if decision_mode == "rules":
        return _rules_decision(example, context)
    if decision_mode == "supervised":
        return _supervised_decision(example, context) or {"predicted_label": verifier_label, "decision_mode": "verifier_fallback", "score_contradict": ""}
    if decision_mode == "meta":
        return _meta_decision(example, chunks, context) or _supervised_decision(example, context) or {
            "predicted_label": verifier_label,
            "decision_mode": "verifier_fallback",
            "score_contradict": "",
        }

    if any(verdict.verdict == "CONTRADICTED" and verdict.confidence == "HIGH" for verdict in claim_verdicts):
        return {"predicted_label": "contradict", "decision_mode": "hybrid_verifier_high_confidence", "score_contradict": 1.0}
    return _meta_decision(example, chunks, context) or _supervised_decision(example, context) or _rules_decision(example, context) or {
        "predicted_label": verifier_label,
        "decision_mode": "verifier_fallback",
        "score_contradict": "",
    }


def _meta_decision(example, chunks, context: dict[str, object]) -> dict[str, object] | None:
    artifact = context.get("meta")
    if not artifact:
        return None
    if artifact.get("selected_system") == "supervised" and context.get("supervised"):
        result = _supervised_decision(example, context)
        if result:
            result["decision_mode"] = "hybrid_supervised_selected"
        return result
    prediction = predict_meta_examples(artifact, [example], chunks)[0]
    return {
        "predicted_label": prediction["predicted_label"],
        "decision_mode": "meta",
        "score_contradict": prediction["score_contradict"],
    }


def _supervised_decision(example, context: dict[str, object]) -> dict[str, object] | None:
    artifact = context.get("supervised")
    if not artifact:
        return None
    prediction = predict_examples_with_artifact(artifact, [example])[0]
    return {
        "predicted_label": prediction["predicted_label"],
        "decision_mode": "supervised",
        "score_contradict": prediction["score_contradict"],
    }


def _rules_decision(example, context: dict[str, object]) -> dict[str, object] | None:
    constraints = context.get("constraints")
    if not constraints:
        return None
    score = constraint_rule_score(example, extract_anchors(example), constraints)
    return {
        "predicted_label": "contradict" if float(score["constraint_risk_score"]) >= 0.5 else "consistent",
        "decision_mode": "rules",
        "score_contradict": float(score["constraint_risk_score"]),
    }


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
