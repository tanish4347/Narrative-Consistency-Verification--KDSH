"""Evaluate prediction CSVs against labeled gold data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.evaluate import (
    compute_metrics,
    load_gold_labels,
    load_predictions,
    print_metrics,
    save_confusion_matrix_csv,
    save_eval_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NCV prediction labels.")
    parser.add_argument("--pred", default="outputs/predictions.csv")
    parser.add_argument("--gold", default="Dataset/train.csv")
    parser.add_argument("--report", default="outputs/eval_report.md")
    parser.add_argument("--confusion", default="outputs/confusion_matrix.csv")
    parser.add_argument("--allow-subset", default="false")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        predictions = load_predictions(args.pred)
        gold = load_gold_labels(args.gold)
        metrics = compute_metrics(
            predictions,
            gold,
            allow_subset=_as_bool(args.allow_subset),
            allow_partial=args.allow_partial,
        )
    except ValueError as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc

    save_eval_report(metrics, args.report)
    save_confusion_matrix_csv(metrics["confusion_matrix"], args.confusion)
    print(print_metrics(metrics))
    print(f"Wrote report to {args.report}")
    print(f"Wrote confusion matrix to {args.confusion}")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
