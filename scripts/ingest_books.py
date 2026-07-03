"""Build the local chunk index.

Run from the repository root:

    python -m scripts.ingest_books
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.config import NCVConfig
from ncv.ingest import build_chunks, write_chunks_jsonl


def main() -> None:
    args = parse_args()
    config = NCVConfig.from_env()
    books_dir = Path(args.books_dir) if args.books_dir else config.books_dir
    output_path = Path(args.output) if args.output else config.output_dir / "chunks.jsonl"
    chunks = build_chunks(
        books_dir,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    write_chunks_jsonl(chunks, output_path)
    print(f"Wrote {len(chunks)} chunks to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local NCV book chunk index.")
    parser.add_argument("--books-dir", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
