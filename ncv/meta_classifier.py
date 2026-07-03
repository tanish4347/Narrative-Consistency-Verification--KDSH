"""Hybrid meta-classifier features and training utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .anchors import extract_anchors
from .constraint_index import (
    build_constraint_index,
    constraint_rule_score,
    extract_constraint_features,
    load_character_constraints,
    save_constraint_index,
)
from .data import load_test_examples, load_train_examples
from .retrieval import load_chunks_jsonl
from .schemas import BackstoryExample, EvidenceChunk
from .supervised import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    SupervisedTrainingResult,
    metrics_from_predictions,
    predict_examples_with_artifact,
    predict_scores,
    train_supervised_baselines,
    tune_threshold,
)


@dataclass(slots=True)
class MetaTrainingResult:
    selected_system: str
    best_meta_model: str
    model_reports: dict[str, dict[str, Any]]
    text_metrics: dict[str, Any]
    all_consistent_metrics: dict[str, Any]
    feature_names: list[str]
    oof_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    artifact: dict[str, Any]


def train_meta_classifier(
    *,
    train_path: str | Path = "Dataset/train.csv",
    test_path: str | Path = "Dataset/test.csv",
    chunks_path: str | Path = "outputs/chunks.jsonl",
    output_dir: str | Path = "outputs",
    trace_path: str | Path | None = None,
    root_results_path: str | Path | None = None,
) -> MetaTrainingResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_examples = load_train_examples(train_path)
    test_examples = load_test_examples(test_path)
    chunks = load_chunks_jsonl(chunks_path)
    constraint_index = build_constraint_index(chunks)
    save_constraint_index(constraint_index, output / "constraint_index.json")
    constraints = load_character_constraints()

    supervised = train_supervised_baselines(train_path=train_path, test_path=test_path, output_dir=output)
    supervised_oof_scores = {str(row["id"]): float(row["score_contradict"]) for row in supervised.oof_rows}
    y = np.array([LABEL_TO_INT[example.label or "consistent"] for example in train_examples])
    train_features = build_meta_features(
        train_examples,
        chunks,
        constraint_index,
        constraints,
        supervised_scores=supervised_oof_scores,
        trace_features=_trace_features_by_id(trace_path) if trace_path is not None else {},
    )
    feature_names = list(train_features.columns)
    reports, best_name, best_scores, best_predictions, best_threshold = cross_validate_meta_models(train_features, y)
    all_consistent = metrics_from_predictions(y, np.zeros_like(y))
    text_metrics = supervised.model_reports[supervised.best_model_name]

    selected_system = "meta" if reports[best_name]["macro_f1"] > text_metrics["macro_f1"] else "supervised"
    final_model = clone(_meta_candidates()[best_name])
    final_model.fit(train_features, y)

    supervised_test = predict_examples_with_artifact(supervised.artifact, test_examples)
    supervised_test_scores = {str(row["id"]): float(row["score_contradict"]) for row in supervised_test}
    test_features = build_meta_features(
        test_examples,
        chunks,
        constraint_index,
        constraints,
        supervised_scores=supervised_test_scores,
        trace_features={},
    )
    test_scores = _predict_scores(final_model, test_features)
    test_predictions = (test_scores >= best_threshold).astype(int)
    meta_test_rows = [
        {
            "id": example.id,
            "predicted_label": INT_TO_LABEL[int(pred)],
            "score_contradict": float(score),
            "selected_system": selected_system,
        }
        for example, pred, score in zip(test_examples, test_predictions, test_scores)
    ]

    if selected_system == "supervised":
        final_test_rows = [
            {"id": row["id"], "predicted_label": row["predicted_label"], "score_contradict": row["score_contradict"]}
            for row in supervised_test
        ]
    else:
        final_test_rows = meta_test_rows

    artifact = {
        "model": final_model,
        "model_name": best_name,
        "threshold": float(best_threshold),
        "feature_names": feature_names,
        "selected_system": selected_system,
        "supervised_artifact": supervised.artifact,
        "constraint_index": constraint_index,
        "character_constraints": constraints,
        "reports": reports,
        "text_metrics": text_metrics,
        "trace_path": str(trace_path) if trace_path is not None else None,
    }
    joblib.dump(artifact, output / "meta_model.joblib")

    oof_rows = [
        {
            "id": example.id,
            "gold_label": example.label,
            "predicted_label": INT_TO_LABEL[int(pred)],
            "score_contradict": float(score),
            "model": best_name,
        }
        for example, pred, score in zip(train_examples, best_predictions, best_scores)
    ]
    write_rows(output / "meta_oof_predictions.csv", oof_rows)
    write_rows(output / "meta_test_predictions.csv", meta_test_rows)
    write_rows(output / "results.csv", final_test_rows)
    if root_results_path is not None:
        write_rows(root_results_path, final_test_rows)
    write_feature_importance(output / "meta_feature_importance.csv", final_model, feature_names)
    (output / "meta_classifier_report.md").write_text(
        format_meta_report(all_consistent, text_metrics, reports, best_name, selected_system),
        encoding="utf-8",
    )
    return MetaTrainingResult(
        selected_system=selected_system,
        best_meta_model=best_name,
        model_reports=reports,
        text_metrics=text_metrics,
        all_consistent_metrics=all_consistent,
        feature_names=feature_names,
        oof_rows=oof_rows,
        test_rows=final_test_rows,
        artifact=artifact,
    )


def build_meta_features(
    examples: Iterable[BackstoryExample],
    chunks: Iterable[EvidenceChunk],
    constraint_index: Mapping[str, Any],
    constraints: Mapping[str, Any],
    *,
    supervised_scores: Mapping[str, float],
    trace_features: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    chunk_list = list(chunks)
    rows = []
    trace_features = trace_features or {}
    for example in examples:
        anchors = extract_anchors(example)
        constraint_features = extract_constraint_features(example, chunk_list, constraint_index)
        rule = constraint_rule_score(example, anchors, constraints)
        profile = (constraint_index.get("characters") or {}).get(example.character, {})
        event_hit = float(bool(set(anchors.event_types) & set(profile.get("important_event_types", []))))
        trace = trace_features.get(example.id, {})
        row = {
            "supervised_score_contradict": float(supervised_scores.get(example.id, 0.0)),
            "anchor_num_secondary": float(len(anchors.secondary_characters)),
            "anchor_num_dates": float(len(anchors.dates)),
            "anchor_num_locations": float(len(anchors.locations)),
            "anchor_num_roles": float(len(anchors.roles)),
            "anchor_num_objects": float(len(anchors.named_objects)),
            "constraint_hard_hits": float(rule["hard_constraint_hits"]),
            "constraint_soft_hits": float(rule["soft_constraint_hits"]),
            "constraint_risk_score": float(rule["constraint_risk_score"]),
            "target_character_hit": float(bool(profile.get("mention_count", 0))),
            "event_hit": event_hit,
            "top_contradiction_family_score": float(constraint_features["contradiction_query_max_score"]),
            "any_contradicted": float(trace.get("any_contradicted", 0.0)),
            "any_supported": float(trace.get("any_supported", 0.0)),
            "num_insufficient": float(trace.get("num_insufficient", 0.0)),
            "verifier_failed": float(trace.get("verifier_failed", 0.0)),
            **constraint_features,
        }
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def predict_meta_examples(
    artifact: Mapping[str, Any],
    examples: Iterable[BackstoryExample],
    chunks: Iterable[EvidenceChunk],
) -> list[dict[str, Any]]:
    examples = list(examples)
    supervised_predictions = predict_examples_with_artifact(artifact["supervised_artifact"], examples)
    supervised_scores = {str(row["id"]): float(row["score_contradict"]) for row in supervised_predictions}
    features = build_meta_features(
        examples,
        chunks,
        artifact["constraint_index"],
        artifact["character_constraints"],
        supervised_scores=supervised_scores,
        trace_features={},
    )
    scores = _predict_scores(artifact["model"], features)
    preds = (scores >= float(artifact["threshold"])).astype(int)
    return [
        {"id": example.id, "predicted_label": INT_TO_LABEL[int(pred)], "score_contradict": float(score)}
        for example, pred, score in zip(examples, preds, scores)
    ]


def load_meta_artifact(path: str | Path = "outputs/meta_model.joblib") -> dict[str, Any]:
    return joblib.load(path)


def cross_validate_meta_models(frame: pd.DataFrame, y: np.ndarray) -> tuple[dict[str, dict[str, Any]], str, np.ndarray, np.ndarray, float]:
    reports = {}
    best_name = ""
    best_macro = -1.0
    best_scores = np.zeros(len(y))
    best_preds = np.zeros(len(y), dtype=int)
    best_threshold = 0.5
    for name, model in _meta_candidates().items():
        scores = np.zeros(len(y), dtype=float)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, valid_idx in splitter.split(frame, y):
            fold_model = clone(model)
            fold_model.fit(frame.iloc[train_idx], y[train_idx])
            scores[valid_idx] = _predict_scores(fold_model, frame.iloc[valid_idx])
        threshold, metrics = tune_threshold(y, scores)
        recall_threshold, recall_metrics = tune_threshold(y, scores, min_contradict_recall=0.30)
        preds = (scores >= threshold).astype(int)
        reports[name] = {
            **metrics,
            "selected_threshold": threshold,
            "recall30_threshold": recall_threshold,
            "recall30_macro_f1": recall_metrics["macro_f1"],
            "recall30_contradict_recall": recall_metrics["recall_contradict"],
        }
        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            best_name = name
            best_scores = scores
            best_preds = preds
            best_threshold = threshold
    return reports, best_name, best_scores, best_preds, float(best_threshold)


def format_meta_report(
    all_consistent: Mapping[str, Any],
    text_metrics: Mapping[str, Any],
    meta_reports: Mapping[str, Mapping[str, Any]],
    best_name: str,
    selected_system: str,
) -> str:
    lines = [
        "# Meta-Classifier Report",
        "",
        "Scope: stratified 5-fold cross-validation on all 80 labeled training rows. No `Dataset/test.csv` labels are used.",
        "",
        "| System | Accuracy | Macro F1 | Contradict Recall | Consistent Recall | CV-safe? | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        _metric_row("all_consistent", all_consistent, "Yes", "baseline"),
        _metric_row("supervised_text", text_metrics, "Yes", "text-only baseline"),
    ]
    for name, metrics in meta_reports.items():
        note = "selected meta model" if name == best_name else ""
        lines.append(_metric_row(f"meta_{name}", metrics, "Yes", note))
    lines.extend(
        [
            "",
            f"Selected final system: `{selected_system}`.",
            "",
            "The selected system is based on cross-validated training metrics only. Test predictions are written to `results.csv`, but no test metrics are claimed.",
        ]
    )
    return "\n".join(lines)


def write_feature_importance(path: str | Path, model: Any, feature_names: list[str]) -> Path:
    values = None
    estimator = model.named_steps.get("model") if isinstance(model, Pipeline) else model
    if hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    if values is None:
        rows = [{"feature": name, "importance": ""} for name in feature_names]
    else:
        rows = [
            {"feature": name, "importance": float(value)}
            for name, value in sorted(zip(feature_names, values), key=lambda item: abs(item[1]), reverse=True)
        ]
    return write_rows(path, rows)


def write_rows(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _meta_candidates() -> dict[str, Any]:
    return {
        "LogisticRegression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, max_iter=100),
    }


def _predict_scores(model: Any, frame: pd.DataFrame) -> np.ndarray:
    estimator = model.named_steps.get("model") if isinstance(model, Pipeline) else model
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)
        classes = list(estimator.classes_)
        return probs[:, classes.index(1)]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(frame)
        return np.asarray(decision, dtype=float)
    preds = model.predict(frame)
    return np.asarray(preds, dtype=float)


def _trace_features_by_id(path: str | Path) -> dict[str, dict[str, float]]:
    trace_path = Path(path)
    if not trace_path.exists():
        return {}
    out = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            import json

            row = json.loads(line)
            example_id = str(row.get("example", {}).get("id", ""))
            verdicts = [claim.get("verifier_output", {}).get("verdict") for claim in row.get("claims", [])]
            out[example_id] = {
                "any_contradicted": float("CONTRADICTED" in verdicts),
                "any_supported": float("SUPPORTED" in verdicts),
                "num_insufficient": float(sum(1 for verdict in verdicts if verdict == "INSUFFICIENT")),
                "verifier_failed": float(bool(row.get("row_error"))),
            }
    return out


def _metric_row(name: str, metrics: Mapping[str, Any], cv_safe: str, notes: str) -> str:
    return (
        f"| {name} | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | "
        f"{metrics['recall_contradict']:.6f} | {metrics['recall_consistent']:.6f} | {cv_safe} | {notes} |"
    )
