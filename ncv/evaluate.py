"""Evaluation metrics for backstory-level labels."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


VALID_LABELS = ("consistent", "contradict")
PREDICTION_LABEL_COLUMNS = ("predicted_label", "prediction", "label", "final_label", "pred_label")
GOLD_LABEL_COLUMNS = ("label", "gold_label", "actual_label")


def load_predictions(path: str | Path) -> dict[str, str]:
    """Load prediction CSVs as `id -> normalized label`."""

    return _load_label_mapping(
        path,
        label_columns=PREDICTION_LABEL_COLUMNS,
        kind="prediction",
    )


def load_gold_labels(path: str | Path) -> dict[str, str]:
    """Load gold labels from a labeled CSV as `id -> normalized label`."""

    return _load_label_mapping(
        path,
        label_columns=GOLD_LABEL_COLUMNS,
        kind="gold",
    )


def align_predictions_with_gold(
    predictions: Mapping[str, str],
    gold: Mapping[str, str],
    *,
    allow_subset: bool = False,
    allow_partial: bool = False,
) -> list[tuple[str, str, str]]:
    """Return aligned `(id, actual, predicted)` rows after strict ID checks."""

    predicted = _normalize_label_mapping(predictions, kind="prediction")
    actual = _normalize_label_mapping(gold, kind="gold")
    prediction_ids = set(predicted)
    gold_ids = set(actual)

    extra_prediction_ids = sorted(prediction_ids - gold_ids, key=_sort_key)
    missing_prediction_ids = sorted(gold_ids - prediction_ids, key=_sort_key)
    if allow_partial:
        overlap_ids = prediction_ids & gold_ids
        if not overlap_ids:
            raise ValueError("Partial evaluation requires at least one overlapping prediction/gold id")
        return [(example_id, actual[example_id], predicted[example_id]) for example_id in sorted(overlap_ids, key=_sort_key)]
    if extra_prediction_ids:
        raise ValueError(f"prediction IDs missing from gold: {_preview_ids(extra_prediction_ids)}")
    if missing_prediction_ids and not allow_subset:
        details: list[str] = []
        details.append(f"gold IDs missing from predictions: {_preview_ids(missing_prediction_ids)}")
        raise ValueError("Prediction/gold IDs must match exactly; " + "; ".join(details))

    ids_to_score = prediction_ids if allow_subset else gold_ids
    return [(example_id, actual[example_id], predicted[example_id]) for example_id in sorted(ids_to_score, key=_sort_key)]


def compute_metrics(
    predictions: Mapping[str, str],
    gold: Mapping[str, str],
    *,
    allow_subset: bool = False,
    allow_partial: bool = False,
) -> dict[str, float | int | str | dict[str, dict[str, int]]]:
    """Compute accuracy, per-class precision/recall/F1, macro metrics, and matrix."""

    aligned = align_predictions_with_gold(
        predictions,
        gold,
        allow_subset=allow_subset,
        allow_partial=allow_partial,
    )
    if not aligned:
        raise ValueError("At least one aligned prediction is required")

    matrix = _confusion_from_aligned(aligned)
    total = len(aligned)
    correct = sum(1 for _, actual, predicted in aligned if actual == predicted)
    per_class = {label: _class_metrics(label, matrix) for label in VALID_LABELS}

    macro_precision = sum(per_class[label]["precision"] for label in VALID_LABELS) / len(VALID_LABELS)
    macro_recall = sum(per_class[label]["recall"] for label in VALID_LABELS) / len(VALID_LABELS)
    macro_f1 = sum(per_class[label]["f1"] for label in VALID_LABELS) / len(VALID_LABELS)

    metrics: dict[str, float | int | str | dict[str, dict[str, int]]] = {
        "num_examples": total,
        "num_gold_examples": len(gold),
        "num_prediction_examples": len(predictions),
        "subset_evaluation": bool(allow_subset),
        "partial_evaluation": bool(allow_partial),
        "accuracy": correct / total,
        "precision_consistent": per_class["consistent"]["precision"],
        "recall_consistent": per_class["consistent"]["recall"],
        "f1_consistent": per_class["consistent"]["f1"],
        "precision_contradict": per_class["contradict"]["precision"],
        "recall_contradict": per_class["contradict"]["recall"],
        "f1_contradict": per_class["contradict"]["f1"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix,
    }
    if allow_partial:
        metrics["evaluation_scope"] = (
            f"Partial evaluation: {len(aligned)} overlapping rows; "
            f"{len(predictions)} prediction rows; {len(gold)} gold rows. "
            "Mark this metric as partial."
        )
    elif allow_subset and len(predictions) != len(gold):
        metrics["evaluation_scope"] = f"Subset evaluation: {len(predictions)} predicted rows out of {len(gold)} gold rows"
    else:
        metrics["evaluation_scope"] = f"Exact evaluation: {len(aligned)} predicted rows matched {len(gold)} gold rows"
    return metrics


def confusion_matrix(
    predictions: Mapping[str, str],
    gold: Mapping[str, str],
    *,
    allow_subset: bool = False,
    allow_partial: bool = False,
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix with rows as actual labels and columns as predictions."""

    return _confusion_from_aligned(
        align_predictions_with_gold(predictions, gold, allow_subset=allow_subset, allow_partial=allow_partial)
    )


def classification_report(
    predictions: Mapping[str, str],
    gold: Mapping[str, str],
    *,
    allow_subset: bool = False,
    allow_partial: bool = False,
) -> str:
    """Return a markdown classification report for aligned predictions."""

    return format_eval_report(compute_metrics(predictions, gold, allow_subset=allow_subset, allow_partial=allow_partial))


def save_eval_report(metrics: Mapping[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_eval_report(metrics), encoding="utf-8")
    return output_path


def save_confusion_matrix_csv(matrix: Mapping[str, Mapping[str, int]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual", "predicted_consistent", "predicted_contradict"])
        for actual in VALID_LABELS:
            row = matrix.get(actual, {})
            writer.writerow([actual, int(row.get("consistent", 0)), int(row.get("contradict", 0))])
    return output_path


def format_eval_report(metrics: Mapping[str, object]) -> str:
    matrix = metrics.get("confusion_matrix", {})
    lines = [
        "# Evaluation Report",
        "",
        "These metrics are computed only from aligned labeled rows. The scope is shown below.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "num_examples",
        "num_gold_examples",
        "num_prediction_examples",
        "accuracy",
        "precision_consistent",
        "recall_consistent",
        "f1_consistent",
        "precision_contradict",
        "recall_contradict",
        "f1_contradict",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        value = metrics.get(key)
        if isinstance(value, float):
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        lines.append(f"| `{key}` | {rendered} |")
    scope = metrics.get("evaluation_scope")
    if scope:
        lines.extend(["", f"**Scope:** {scope}", ""])

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "Rows are actual labels. Columns are predicted labels.",
            "",
            "| Actual | Predicted consistent | Predicted contradict |",
            "| --- | ---: | ---: |",
        ]
    )
    if isinstance(matrix, Mapping):
        for actual in VALID_LABELS:
            row = matrix.get(actual, {})
            if isinstance(row, Mapping):
                lines.append(f"| {actual} | {int(row.get('consistent', 0))} | {int(row.get('contradict', 0))} |")
    lines.append("")
    return "\n".join(lines)


def print_metrics(metrics: Mapping[str, object]) -> str:
    """Return terminal-friendly metric lines."""

    keys = (
        "num_examples",
        "accuracy",
        "precision_consistent",
        "recall_consistent",
        "f1_consistent",
        "precision_contradict",
        "recall_contradict",
        "f1_contradict",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    lines = []
    for key in keys:
        value = metrics[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def normalize_label(label: object, *, kind: str = "label") -> str:
    value = str(label).strip().casefold()
    if value in {"consistent"}:
        return "consistent"
    if value in {"contradict"}:
        return "contradict"
    allowed = ", ".join(VALID_LABELS)
    raise ValueError(f"Invalid {kind} label `{label}`; allowed labels: {allowed}")


def normalize_gold_label(label: str) -> str:
    """Backward-compatible normalization from the earlier evaluator."""

    normalized = normalize_label(label, kind="gold")
    return "CONSISTENT" if normalized == "consistent" else "INCONSISTENT"


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Backward-compatible binary report for the `contradict` class."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: Mapping[str, int]


def evaluate_predictions(gold_labels: Iterable[str], predicted_labels: Iterable[str]) -> EvaluationReport:
    """Backward-compatible list-based evaluator for legacy callers."""

    gold = [normalize_label(label, kind="gold") for label in gold_labels]
    predicted = [normalize_label(label, kind="prediction") for label in predicted_labels]
    if len(gold) != len(predicted):
        raise ValueError("gold_labels and predicted_labels must have the same length")
    if not gold:
        raise ValueError("at least one prediction is required")

    actual_by_id = {str(index): label for index, label in enumerate(gold)}
    predicted_by_id = {str(index): label for index, label in enumerate(predicted)}
    metrics = compute_metrics(predicted_by_id, actual_by_id)
    matrix = metrics["confusion_matrix"]
    assert isinstance(matrix, Mapping)
    consistent_row = matrix["consistent"]
    contradict_row = matrix["contradict"]
    tp = int(contradict_row["contradict"])
    tn = int(consistent_row["consistent"])
    fp = int(consistent_row["contradict"])
    fn = int(contradict_row["consistent"])

    return EvaluationReport(
        accuracy=float(metrics["accuracy"]),
        precision=float(metrics["precision_contradict"]),
        recall=float(metrics["recall_contradict"]),
        f1=float(metrics["f1_contradict"]),
        confusion_matrix={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )


def _load_label_mapping(path: str | Path, *, label_columns: tuple[str, ...], kind: str) -> dict[str, str]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: CSV file is missing a header row")
        id_column = _resolve_column(reader.fieldnames, ("id", "example_id", "story_id"))
        label_column = _resolve_column(reader.fieldnames, label_columns)
        if id_column is None:
            raise ValueError(f"{csv_path}: missing required id column")
        if label_column is None:
            accepted = ", ".join(label_columns)
            raise ValueError(f"{csv_path}: missing required {kind} label column; accepted columns: {accepted}")

        labels: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            example_id = str(row.get(id_column, "")).strip()
            if not example_id:
                raise ValueError(f"{csv_path}: row {row_number} has an empty id")
            if example_id in labels:
                raise ValueError(f"{csv_path}: duplicate id `{example_id}` at row {row_number}")
            labels[example_id] = normalize_label(row.get(label_column), kind=kind)
    return labels


def _resolve_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {field.strip().lstrip("\ufeff").casefold(): field for field in fieldnames}
    for candidate in candidates:
        actual = lookup.get(candidate.casefold())
        if actual is not None:
            return actual
    return None


def _normalize_label_mapping(labels: Mapping[str, str], *, kind: str) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for example_id, label in labels.items():
        key = str(example_id).strip()
        if not key:
            raise ValueError(f"{kind} labels contain an empty id")
        if key in normalized:
            raise ValueError(f"{kind} labels contain duplicate id `{key}`")
        normalized[key] = normalize_label(label, kind=kind)
    return normalized


def _confusion_from_aligned(aligned: Iterable[tuple[str, str, str]]) -> dict[str, dict[str, int]]:
    matrix = {actual: {predicted: 0 for predicted in VALID_LABELS} for actual in VALID_LABELS}
    for _, actual, predicted in aligned:
        matrix[actual][predicted] += 1
    return matrix


def _class_metrics(label: str, matrix: Mapping[str, Mapping[str, int]]) -> dict[str, float]:
    tp = int(matrix[label][label])
    fp = sum(int(matrix[actual][label]) for actual in VALID_LABELS if actual != label)
    fn = sum(int(matrix[label][predicted]) for predicted in VALID_LABELS if predicted != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _preview_ids(ids: list[str], *, limit: int = 8) -> str:
    rendered = ", ".join(ids[:limit])
    if len(ids) > limit:
        rendered += f", ... ({len(ids)} total)"
    return rendered or "<none>"


def _sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def metrics_to_json(metrics: Mapping[str, object]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)
