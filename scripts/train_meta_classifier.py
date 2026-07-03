"""Train the hybrid meta-classifier and write final test predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.meta_classifier import train_meta_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NCV hybrid meta-classifier.")
    parser.add_argument("--train", default="Dataset/train.csv")
    parser.add_argument("--test", default="Dataset/test.csv")
    parser.add_argument("--chunks", default="outputs/chunks.jsonl")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--trace", default=None, help="Optional pipeline trace JSONL for verifier-derived meta features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_meta_classifier(
        train_path=args.train,
        test_path=args.test,
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        trace_path=args.trace,
        root_results_path="results.csv",
    )
    best = result.model_reports[result.best_meta_model]
    print(f"Selected final system: {result.selected_system}")
    print(f"Best meta model: {result.best_meta_model}")
    print(f"meta_accuracy: {best['accuracy']:.6f}")
    print(f"meta_macro_f1: {best['macro_f1']:.6f}")
    print(f"meta_recall_contradict: {best['recall_contradict']:.6f}")
    print(f"Wrote report to {Path(args.output_dir) / 'meta_classifier_report.md'}")
    print(f"Wrote test predictions to {Path(args.output_dir) / 'results.csv'}")
    print("Wrote root test predictions to results.csv")


if __name__ == "__main__":
    main()
