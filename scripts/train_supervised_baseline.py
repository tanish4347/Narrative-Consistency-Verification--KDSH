"""Train supervised text baselines and write reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.supervised import train_supervised_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised NCV text baselines.")
    parser.add_argument("--train", default="Dataset/train.csv")
    parser.add_argument("--test", default="Dataset/test.csv")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_supervised_baselines(train_path=args.train, test_path=args.test, output_dir=args.output_dir)
    best = result.model_reports[result.best_model_name]
    print(f"Selected supervised baseline: {result.best_model_name}")
    print(f"accuracy: {best['accuracy']:.6f}")
    print(f"macro_f1: {best['macro_f1']:.6f}")
    print(f"recall_contradict: {best['recall_contradict']:.6f}")
    print(f"Wrote report to {Path(args.output_dir) / 'supervised_baseline_report.md'}")


if __name__ == "__main__":
    main()
