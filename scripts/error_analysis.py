"""Analyze Gemma/verifier prediction failures against labeled gold rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.evaluate import compute_metrics, load_gold_labels, load_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NCV prediction errors.")
    parser.add_argument("--gold", default="outputs/dev_split.csv")
    parser.add_argument("--pred", default="outputs/predictions_gemma4_dev_10_fast.csv")
    parser.add_argument("--trace", default=None, help="Optional pipeline trace JSONL for verifier-layer diagnostics.")
    parser.add_argument("--output-csv", default="outputs/error_analysis_gemma4_dev_10.csv")
    parser.add_argument("--output-md", default="outputs/error_analysis_gemma4_dev_10.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = analyze_errors(args.gold, args.pred, args.trace)
    write_error_csv(args.output_csv, analysis["rows"])
    write_error_report(args.output_md, analysis)
    print(f"Rows evaluated: {analysis['metrics']['num_examples']}")
    print(f"False negatives: {analysis['false_negatives']}")
    print(f"False positives: {analysis['false_positives']}")
    print(f"Wrote CSV to {args.output_csv}")
    print(f"Wrote report to {args.output_md}")


def analyze_errors(gold_path: str | Path, pred_path: str | Path, trace_path: str | Path | None = None) -> dict:
    gold = load_gold_labels(gold_path)
    pred = load_predictions(pred_path)
    overlap = {example_id: label for example_id, label in pred.items() if example_id in gold}
    metrics = compute_metrics(overlap, gold, allow_partial=True)
    prediction_rows = _load_prediction_rows(pred_path)
    trace = _load_trace(trace_path)
    rows = []
    for example_id, predicted in sorted(overlap.items(), key=lambda item: _sort_key(item[0])):
        actual = gold[example_id]
        pred_row = prediction_rows.get(example_id, {})
        trace_row = trace.get(example_id, {})
        row = {
            "id": example_id,
            "gold_label": actual,
            "predicted_label": predicted,
            "error_type": _error_type(actual, predicted),
            "num_claims": pred_row.get("num_claims", ""),
            "internal_status": pred_row.get("internal_status", ""),
            "num_supported": pred_row.get("num_supported", ""),
            "num_contradicted": pred_row.get("num_contradicted", ""),
            "num_insufficient": pred_row.get("num_insufficient", ""),
            "likely_failure_layer": _likely_failure_layer(actual, predicted, pred_row, trace_row),
        }
        rows.append(row)
    trace_stats = _trace_stats(trace)
    return {
        "metrics": metrics,
        "rows": rows,
        "false_negatives": sum(1 for row in rows if row["error_type"] == "false_negative"),
        "false_positives": sum(1 for row in rows if row["error_type"] == "false_positive"),
        "internal_status_distribution": Counter(row["internal_status"] for row in rows),
        "verdict_counts": trace_stats["verdict_counts"],
        "insufficient_count": trace_stats["insufficient_count"],
        "json_failures": trace_stats["json_failures"],
        "local_provider_failures": trace_stats["local_provider_failures"],
        "second_pass_downgrades": trace_stats["second_pass_downgrades"],
        "failed_contradiction_rows": _failed_contradiction_rows(rows, trace),
    }


def write_error_csv(path: str | Path, rows: list[dict]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_error_report(path: str | Path, analysis: dict) -> Path:
    metrics = analysis["metrics"]
    matrix = metrics["confusion_matrix"]
    lines = [
        "# Gemma4 Dev-10 Error Analysis",
        "",
        "This report evaluates only overlapping labeled rows. It does not use `Dataset/test.csv`.",
        "",
        "## Summary",
        "",
        f"- total rows evaluated: `{metrics['num_examples']}`",
        f"- false negatives: `{analysis['false_negatives']}`",
        f"- false positives: `{analysis['false_positives']}`",
        f"- accuracy: `{metrics['accuracy']:.6f}`",
        f"- macro-F1: `{metrics['macro_f1']:.6f}`",
        f"- contradict recall: `{metrics['recall_contradict']:.6f}`",
        "",
        "**Current Gemma path predicts all evaluated rows as `consistent`; contradiction recall is zero.**",
        "",
        "## Confusion Matrix",
        "",
        "| Actual | Predicted consistent | Predicted contradict |",
        "| --- | ---: | ---: |",
        f"| consistent | {matrix['consistent']['consistent']} | {matrix['consistent']['contradict']} |",
        f"| contradict | {matrix['contradict']['consistent']} | {matrix['contradict']['contradict']} |",
        "",
        "## Internal Status Distribution",
        "",
    ]
    for status, count in analysis["internal_status_distribution"].items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Verifier Trace Counts",
            "",
            f"- verifier verdicts: `{dict(analysis['verdict_counts'])}`",
            f"- INSUFFICIENT count: `{analysis['insufficient_count']}`",
            f"- JSON failures: `{analysis['json_failures']}`",
            f"- local provider/memory failures: `{analysis['local_provider_failures']}`",
            f"- second-pass contradiction downgrades: `{analysis['second_pass_downgrades']}`",
            "",
            "## Per-Row Errors",
            "",
            "| id | gold | pred | claims | supported | contradicted | insufficient | failure layer |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in analysis["rows"]:
        lines.append(
            f"| {row['id']} | {row['gold_label']} | {row['predicted_label']} | {row['num_claims']} | "
            f"{row['num_supported']} | {row['num_contradicted']} | {row['num_insufficient']} | "
            f"{row['likely_failure_layer']} |"
        )
    lines.extend(["", "## Failed Contradiction Rows", ""])
    for row in analysis["failed_contradiction_rows"]:
        lines.extend(
            [
                f"### id `{row['id']}`",
                "",
                f"- claim: {row['claim']}",
                f"- verifier verdict: `{row['verdict']}`",
                "- retrieved previews:",
            ]
        )
        for preview in row["previews"]:
            lines.append(f"  - `{preview['chunk_id']}`: {preview['preview']}")
        lines.append("")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _load_prediction_rows(path: str | Path) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def _load_trace(path: str | Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    trace_path = Path(path)
    if not trace_path.exists():
        return {}
    rows = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example", {}).get("id", ""))
            if example_id:
                rows[example_id] = row
    return rows


def _trace_stats(trace: dict[str, dict]) -> dict:
    verdicts = Counter()
    json_failures = 0
    provider_failures = 0
    second_pass = 0
    for row in trace.values():
        for claim in row.get("claims", []):
            output = claim.get("verifier_output", {})
            verdict = output.get("verdict", "")
            if verdict:
                verdicts[verdict] += 1
            text = " ".join(str(value) for value in (output.get("justification"), output.get("metadata", {})))
            lowered = text.casefold()
            if "json" in lowered or "no json" in lowered:
                json_failures += 1
            if "local llm provider" in lowered or "memory" in lowered:
                provider_failures += 1
            if "not confirmed by second pass" in lowered:
                second_pass += 1
    return {
        "verdict_counts": verdicts,
        "insufficient_count": verdicts.get("INSUFFICIENT", 0),
        "json_failures": json_failures,
        "local_provider_failures": provider_failures,
        "second_pass_downgrades": second_pass,
    }


def _failed_contradiction_rows(rows: list[dict], trace: dict[str, dict]) -> list[dict]:
    out = []
    for row in rows:
        if row["error_type"] != "false_negative":
            continue
        trace_row = trace.get(row["id"], {})
        for claim in trace_row.get("claims", [])[:2]:
            previews = [
                {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "preview": " ".join(str(chunk.get("text", ""))[:240].split()),
                }
                for chunk in claim.get("retrieved_chunks", [])[:3]
            ]
            out.append(
                {
                    "id": row["id"],
                    "claim": claim.get("claim_plan", {}).get("claim_text", ""),
                    "verdict": claim.get("verifier_output", {}).get("verdict", ""),
                    "previews": previews,
                }
            )
    return out


def _error_type(actual: str, predicted: str) -> str:
    if actual == predicted:
        return "correct"
    if actual == "contradict" and predicted == "consistent":
        return "false_negative"
    if actual == "consistent" and predicted == "contradict":
        return "false_positive"
    return "wrong"


def _likely_failure_layer(actual: str, predicted: str, pred_row: dict, trace_row: dict) -> str:
    if actual == predicted:
        return "none"
    if pred_row.get("num_contradicted") in {"0", 0} and pred_row.get("num_insufficient") == pred_row.get("num_claims"):
        return "verifier_all_insufficient"
    if not trace_row:
        return "missing_trace"
    return "retrieval_or_verifier_missed_contradiction"


def _sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    main()
