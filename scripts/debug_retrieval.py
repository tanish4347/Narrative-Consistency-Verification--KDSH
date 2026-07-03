"""Readable retrieval trace for a few dataset rows.

Example:
    python scripts/debug_retrieval.py --dataset Dataset/train.csv --books-dir Dataset/Books --max-rows 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.claim_planner import plan_claims
from ncv.config import NCVConfig
from ncv.contradiction_planner import generate_contradiction_families
from ncv.data import load_test_examples, load_train_examples
from ncv.ingest import build_chunks, write_chunks_jsonl
from ncv.retrieval import load_chunks_jsonl, retrieve_for_claim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug NCV retrieval on local data.")
    parser.add_argument("--dataset", default="Dataset/train.csv")
    parser.add_argument("--books-dir", default="Dataset/Books")
    parser.add_argument("--chunks", default="outputs/chunks.jsonl")
    parser.add_argument("--max-rows", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-mode", default="fallback", choices=("fallback", "mock"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    chunks_path = Path(args.chunks)

    try:
        examples = load_train_examples(dataset_path)
    except ValueError:
        examples = load_test_examples(dataset_path)

    if chunks_path.exists():
        chunks = load_chunks_jsonl(chunks_path)
    else:
        chunks = build_chunks(args.books_dir)
        write_chunks_jsonl(chunks, chunks_path)

    config = NCVConfig(
        books_dir=Path(args.books_dir),
        top_k_retrieval=args.top_k,
        use_embeddings=False,
        max_claims=6,
    )

    for example in examples[: args.max_rows]:
        print("=" * 88)
        print(f"Backstory ID: {example.id} | Character: {example.character} | Book: {example.book_name}")
        plans = plan_claims(example, llm=None, config=config)
        for plan in plans:
            print("-" * 88)
            print(f"Claim {plan.claim_id} [{plan.claim_type}]: {plan.claim_text}")
            families = generate_contradiction_families(plan)
            for family in families:
                print(f"  Family: {family.family_name} -> {family.natural_language_query}")
            retrieved = retrieve_for_claim(plan, families, chunks, config)
            for item in retrieved[: args.top_k]:
                breakdown = ", ".join(
                    f"{key}={value}" for key, value in item.score_breakdown.items() if key.endswith("_score")
                )
                preview = item.text.replace("\n", " ")[:180]
                print(f"  [{item.rank}] {item.chunk_id} | {item.channel} | {breakdown}")
                print(f"      {preview}")


if __name__ == "__main__":
    main()
