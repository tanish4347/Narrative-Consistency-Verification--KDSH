"""Run retrieval/planning ablations for the NCV pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncv.evaluate import compute_metrics, load_gold_labels, load_predictions
from scripts.run_pipeline import run_pipeline


ABLATION_CONFIGS: dict[str, dict[str, object]] = {
    "lexical_only": {
        "use_lexical": True,
        "use_embeddings": False,
        "use_character_scoring": False,
        "use_event_scoring": False,
        "use_contradiction_families": False,
        "use_timeline_features": False,
        "rerank": False,
        "notes": "BM25 lexical retrieval only; no special character/event/timeline/rerank features.",
    },
    "semantic_only": {
        "use_lexical": False,
        "use_embeddings": True,
        "embedding_mode": "mock",
        "use_character_scoring": False,
        "use_event_scoring": False,
        "use_contradiction_families": False,
        "use_timeline_features": False,
        "rerank": False,
        "notes": "Mock semantic retrieval only for local debugging; no model download.",
    },
    "lexical_plus_semantic": {
        "use_lexical": True,
        "use_embeddings": True,
        "embedding_mode": "mock",
        "use_character_scoring": False,
        "use_event_scoring": False,
        "use_contradiction_families": False,
        "use_timeline_features": False,
        "rerank": False,
        "notes": "Lexical retrieval plus mock semantic score.",
    },
    "plus_character_scoring": {
        "use_lexical": True,
        "use_embeddings": True,
        "embedding_mode": "mock",
        "use_character_scoring": True,
        "use_event_scoring": False,
        "use_contradiction_families": False,
        "use_timeline_features": False,
        "rerank": False,
        "notes": "Adds explicit character mention scoring.",
    },
    "plus_event_scoring": {
        "use_lexical": True,
        "use_embeddings": True,
        "embedding_mode": "mock",
        "use_character_scoring": True,
        "use_event_scoring": True,
        "use_contradiction_families": False,
        "use_timeline_features": False,
        "rerank": False,
        "notes": "Adds ontology/event-aware scoring.",
    },
    "plus_contradiction_families": {
        "use_lexical": True,
        "use_embeddings": True,
        "embedding_mode": "mock",
        "use_character_scoring": True,
        "use_event_scoring": True,
        "use_contradiction_families": True,
        "use_timeline_features": False,
        "rerank": False,
        "notes": "Adds contradiction-family and incompatible-state queries.",
    },
    "plus_timeline_features": {
        "use_lexical": True,
        "use_embeddings": True,
        "embedding_mode": "mock",
        "use_character_scoring": True,
        "use_event_scoring": True,
        "use_contradiction_families": True,
        "use_timeline_features": True,
        "rerank": False,
        "notes": "Adds timeline-sensitive query/features.",
    },
    "plus_reranker": {
        "use_lexical": True,
        "use_embeddings": True,
        "embedding_mode": "mock",
        "use_character_scoring": True,
        "use_event_scoring": True,
        "use_contradiction_families": True,
        "use_timeline_features": True,
        "rerank": True,
        "notes": "Adds deterministic heuristic reranking.",
    },
}


@dataclass(frozen=True, slots=True)
class AblationResult:
    config: str
    status: str
    prediction_path: str | None
    metrics: dict[str, object] | None
    num_examples: int
    notes: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "status": self.status,
            "prediction_path": self.prediction_path,
            "metrics": self.metrics,
            "num_examples": self.num_examples,
            "notes": self.notes,
            "error": self.error,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NCV ablation configurations.")
    parser.add_argument("--dataset", default="outputs/dev_split.csv")
    parser.add_argument("--books-dir", default="Dataset/Books")
    parser.add_argument("--chunks", default="outputs/chunks.jsonl")
    parser.add_argument("--llm-provider", default="mock", choices=("mock", "openai", "local"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--fail-fast", default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_ablations(args)
    report_path, metrics_path = save_ablation_outputs(results, args.output_dir)
    print(format_ablation_report(results))
    print(f"Wrote ablation report to {report_path}")
    print(f"Wrote ablation metrics to {metrics_path}")


def run_ablations(args: argparse.Namespace) -> list[AblationResult]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fail_fast = _as_bool(args.fail_fast)
    gold = _try_load_gold(args.dataset)
    results: list[AblationResult] = []

    for config_name, config in ABLATION_CONFIGS.items():
        prediction_path = output_dir / f"{config_name}_predictions.csv"
        notes = str(config.get("notes", ""))
        try:
            pipeline_args = _pipeline_args(args, prediction_path, config)
            run_result = run_pipeline(pipeline_args)
            predictions = load_predictions(prediction_path)
            metrics = _compute_metrics_if_possible(predictions, gold, args.max_rows)
            results.append(
                AblationResult(
                    config=config_name,
                    status="ok",
                    prediction_path=str(prediction_path),
                    metrics=metrics,
                    num_examples=int(run_result["num_examples"]),
                    notes=_metrics_note(notes, gold is not None, args.max_rows),
                )
            )
        except Exception as exc:
            result = AblationResult(
                config=config_name,
                status="failed",
                prediction_path=str(prediction_path),
                metrics=None,
                num_examples=0,
                notes=notes,
                error=str(exc),
            )
            results.append(result)
            if fail_fast:
                break
    return results


def save_ablation_outputs(results: list[AblationResult], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_text = format_ablation_report(results)
    metrics_json = json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True)

    report_path = output_path / "ablation_report.md"
    metrics_path = output_path / "ablation_metrics.json"
    report_path.write_text(report_text, encoding="utf-8")
    metrics_path.write_text(metrics_json + "\n", encoding="utf-8")

    if output_path.name == "ablations":
        parent_report = output_path.parent / "ablation_report.md"
        parent_metrics = output_path.parent / "ablation_metrics.json"
        parent_report.write_text(report_text, encoding="utf-8")
        parent_metrics.write_text(metrics_json + "\n", encoding="utf-8")
        return parent_report, parent_metrics
    return report_path, metrics_path


def format_ablation_report(results: list[AblationResult]) -> str:
    lines = [
        "# Ablation Report",
        "",
        "Metrics are computed only when gold labels are available for the exact predicted rows.",
        "",
        "| Config | Status | Accuracy | Macro F1 | Num examples | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        if result.metrics:
            accuracy = _fmt(result.metrics.get("accuracy"))
            macro_f1 = _fmt(result.metrics.get("macro_f1"))
        else:
            accuracy = ""
            macro_f1 = ""
        notes = result.notes
        if result.error:
            notes = f"FAILED: {result.error}"
        lines.append(
            f"| {result.config} | {result.status} | {accuracy} | {macro_f1} | {result.num_examples} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def _pipeline_args(args: argparse.Namespace, prediction_path: Path, config: dict[str, object]) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=args.dataset,
        books_dir=args.books_dir,
        chunks=args.chunks,
        output=str(prediction_path),
        mode="train_eval" if _has_label_column(args.dataset) else "predict",
        llm_provider=args.llm_provider,
        max_rows=args.max_rows,
        top_k=None,
        use_cache="true",
        use_lexical=str(config.get("use_lexical", True)).lower(),
        use_embeddings=str(config.get("use_embeddings", False)).lower(),
        embedding_mode=str(config.get("embedding_mode", "auto")),
        use_character_scoring=str(config.get("use_character_scoring", True)).lower(),
        use_event_scoring=str(config.get("use_event_scoring", True)).lower(),
        use_contradiction_families=str(config.get("use_contradiction_families", True)).lower(),
        use_timeline_features=str(config.get("use_timeline_features", True)).lower(),
        rerank=str(config.get("rerank", True)).lower(),
        strict_character_filter="false",
        binary_output="true",
        write_trace="false",
    )


def _compute_metrics_if_possible(
    predictions: dict[str, str],
    gold: dict[str, str] | None,
    max_rows: int | None,
) -> dict[str, object] | None:
    if gold is None:
        return None
    missing = sorted(set(predictions) - set(gold))
    if missing:
        raise ValueError(f"prediction IDs missing from gold: {', '.join(missing[:8])}")
    gold_subset = {example_id: gold[example_id] for example_id in predictions}
    metrics = compute_metrics(predictions, gold_subset)
    if max_rows is not None:
        metrics["evaluation_scope"] = f"first {len(predictions)} predicted rows from max_rows={max_rows}"
    return metrics


def _try_load_gold(path: str | Path) -> dict[str, str] | None:
    if not _has_label_column(path):
        return None
    return load_gold_labels(path)


def _has_label_column(path: str | Path) -> bool:
    dataset_path = Path(path)
    try:
        first_line = dataset_path.read_text(encoding="utf-8-sig").splitlines()[0]
    except IndexError:
        return False
    columns = {column.strip().casefold() for column in first_line.split(",")}
    return "label" in columns


def _metrics_note(base_note: str, has_gold: bool, max_rows: int | None) -> str:
    parts = [base_note]
    if not has_gold:
        parts.append("No gold labels found; metrics not computed.")
    elif max_rows is not None:
        parts.append(f"Metrics are for the explicit max_rows subset ({max_rows}).")
    return " ".join(part for part in parts if part)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
