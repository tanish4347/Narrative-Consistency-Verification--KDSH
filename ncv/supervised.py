"""Supervised text baselines for final label prediction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from .data import load_test_examples, load_train_examples
from .schemas import BackstoryExample


LABEL_TO_INT = {"consistent": 0, "contradict": 1}
INT_TO_LABEL = {0: "consistent", 1: "contradict"}


@dataclass(slots=True)
class SupervisedTrainingResult:
    best_model_name: str
    best_threshold: float
    best_score_type: str
    model_reports: dict[str, dict[str, Any]]
    all_consistent_metrics: dict[str, Any]
    oof_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    artifact: dict[str, Any]


def examples_to_frame(examples: Iterable[BackstoryExample]) -> pd.DataFrame:
    rows = []
    for example in examples:
        rows.append(
            {
                "id": example.id,
                "book_name": example.book_name,
                "character": example.character,
                "caption": example.caption or "",
                "content": example.backstory,
                "combined_text": " ".join(
                    part
                    for part in (example.book_name, example.character, example.caption or "", example.backstory)
                    if part
                ),
                "label": example.label,
            }
        )
    return pd.DataFrame(rows)


def make_feature_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("word_tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True), "combined_text"),
            ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True), "combined_text"),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["book_name", "character"]),
        ],
        sparse_threshold=1.0,
    )


def make_model_candidates() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            [
                ("features", make_feature_transformer()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
            ]
        ),
        "LinearSVC": Pipeline(
            [
                ("features", make_feature_transformer()),
                ("model", LinearSVC(class_weight="balanced", random_state=42)),
            ]
        ),
        "ComplementNB": Pipeline(
            [
                ("features", make_feature_transformer()),
                ("model", ComplementNB()),
            ]
        ),
    }


def train_supervised_baselines(
    *,
    train_path: str | Path = "Dataset/train.csv",
    test_path: str | Path = "Dataset/test.csv",
    output_dir: str | Path = "outputs",
) -> SupervisedTrainingResult:
    train_examples = load_train_examples(train_path)
    test_examples = load_test_examples(test_path)
    train_frame = examples_to_frame(train_examples)
    test_frame = examples_to_frame(test_examples)
    y = np.array([LABEL_TO_INT[str(label)] for label in train_frame["label"]])
    candidates = make_model_candidates()
    n_splits = _n_splits(y)
    model_reports: dict[str, dict[str, Any]] = {}
    best_name = ""
    best_macro_f1 = -1.0
    best_threshold = 0.5
    best_score_type = "probability"
    best_oof_scores: np.ndarray | None = None
    best_oof_pred: np.ndarray | None = None

    for name, pipeline in candidates.items():
        oof_scores, score_type = cross_val_scores(pipeline, train_frame, y, n_splits=n_splits)
        threshold, metrics = tune_threshold(y, oof_scores)
        constrained_threshold, constrained_metrics = tune_threshold(y, oof_scores, min_contradict_recall=0.30)
        predictions = (oof_scores >= threshold).astype(int)
        report = {
            **metrics,
            "score_type": score_type,
            "selected_threshold": threshold,
            "recall30_threshold": constrained_threshold,
            "recall30_macro_f1": constrained_metrics["macro_f1"],
            "recall30_contradict_recall": constrained_metrics["recall_contradict"],
        }
        model_reports[name] = report
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_name = name
            best_threshold = threshold
            best_score_type = score_type
            best_oof_scores = oof_scores
            best_oof_pred = predictions

    assert best_oof_scores is not None and best_oof_pred is not None
    final_pipeline = clone(candidates[best_name])
    final_pipeline.fit(train_frame, y)
    test_scores = predict_scores(final_pipeline, test_frame)
    test_pred = (test_scores >= best_threshold).astype(int)
    all_consistent_metrics = metrics_from_predictions(y, np.zeros_like(y))

    oof_rows = [
        {
            "id": row.id,
            "gold_label": row.label,
            "predicted_label": INT_TO_LABEL[int(pred)],
            "score_contradict": float(score),
            "model": best_name,
        }
        for row, pred, score in zip(train_examples, best_oof_pred, best_oof_scores)
    ]
    test_rows = [
        {
            "id": row.id,
            "predicted_label": INT_TO_LABEL[int(pred)],
            "score_contradict": float(score),
            "model": best_name,
        }
        for row, pred, score in zip(test_examples, test_pred, test_scores)
    ]
    artifact = {
        "model": final_pipeline,
        "model_name": best_name,
        "threshold": float(best_threshold),
        "score_type": best_score_type,
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "reports": model_reports,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out / "supervised_model.joblib")
    write_rows(out / "supervised_oof_predictions.csv", oof_rows)
    write_rows(out / "supervised_test_predictions.csv", test_rows)
    (out / "supervised_baseline_report.md").write_text(
        format_supervised_report(model_reports, all_consistent_metrics, best_name),
        encoding="utf-8",
    )
    return SupervisedTrainingResult(
        best_model_name=best_name,
        best_threshold=float(best_threshold),
        best_score_type=best_score_type,
        model_reports=model_reports,
        all_consistent_metrics=all_consistent_metrics,
        oof_rows=oof_rows,
        test_rows=test_rows,
        artifact=artifact,
    )


def cross_val_scores(pipeline: Pipeline, frame: pd.DataFrame, y: np.ndarray, *, n_splits: int) -> tuple[np.ndarray, str]:
    scores = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    score_type = "probability"
    for train_idx, valid_idx in splitter.split(frame, y):
        model = clone(pipeline)
        model.fit(frame.iloc[train_idx], y[train_idx])
        fold_scores = predict_scores(model, frame.iloc[valid_idx])
        if not hasattr(model.named_steps["model"], "predict_proba"):
            score_type = "decision"
        scores[valid_idx] = fold_scores
    return scores, score_type


def predict_scores(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["model"]
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba(frame)
        classes = list(classifier.classes_)
        return probabilities[:, classes.index(1)]
    decision = model.decision_function(frame)
    if np.ndim(decision) > 1:
        decision = decision[:, 1]
    return np.asarray(decision, dtype=float)


def tune_threshold(y_true: np.ndarray, scores: np.ndarray, *, min_contradict_recall: float | None = None) -> tuple[float, dict[str, Any]]:
    thresholds = _candidate_thresholds(scores)
    best_threshold = thresholds[0]
    best_metrics: dict[str, Any] | None = None
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        metrics = metrics_from_predictions(y_true, pred)
        if min_contradict_recall is not None and metrics["recall_contradict"] < min_contradict_recall:
            continue
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_threshold = float(threshold)
    if best_metrics is None:
        best_threshold = float(thresholds[0])
        best_metrics = metrics_from_predictions(y_true, (scores >= best_threshold).astype(int))
    return best_threshold, best_metrics


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_consistent": float(precision[0]),
        "recall_consistent": float(recall[0]),
        "f1_consistent": float(f1[0]),
        "precision_contradict": float(precision[1]),
        "recall_contradict": float(recall[1]),
        "f1_contradict": float(f1[1]),
        "confusion_matrix": {
            "consistent": {"consistent": int(matrix[0, 0]), "contradict": int(matrix[0, 1])},
            "contradict": {"consistent": int(matrix[1, 0]), "contradict": int(matrix[1, 1])},
        },
    }


def predict_examples_with_artifact(artifact: Mapping[str, Any], examples: Iterable[BackstoryExample]) -> list[dict[str, Any]]:
    frame = examples_to_frame(examples)
    scores = predict_scores(artifact["model"], frame)
    threshold = float(artifact.get("threshold", 0.5))
    preds = (scores >= threshold).astype(int)
    return [
        {"id": example.id, "predicted_label": INT_TO_LABEL[int(pred)], "score_contradict": float(score)}
        for example, pred, score in zip(examples, preds, scores)
    ]


def load_supervised_artifact(path: str | Path = "outputs/supervised_model.joblib") -> dict[str, Any]:
    return joblib.load(path)


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


def format_supervised_report(
    reports: Mapping[str, Mapping[str, Any]],
    all_consistent: Mapping[str, Any],
    best_name: str,
) -> str:
    lines = [
        "# Supervised Baseline Report",
        "",
        "Scope: stratified cross-validation on the 80 labeled training rows. No test labels are used.",
        "",
        "## Model Comparison",
        "",
        "| Model | Accuracy | Macro F1 | Contradict Recall | Consistent Recall | Threshold | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| all_consistent | {all_consistent['accuracy']:.6f} | {all_consistent['macro_f1']:.6f} | "
            f"{all_consistent['recall_contradict']:.6f} | {all_consistent['recall_consistent']:.6f} | N/A | baseline |"
        ),
    ]
    for name, metrics in reports.items():
        marker = "new primary label baseline" if name == best_name and metrics["macro_f1"] > all_consistent["macro_f1"] else ""
        lines.append(
            f"| {name} | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | "
            f"{metrics['recall_contradict']:.6f} | {metrics['recall_consistent']:.6f} | "
            f"{metrics['selected_threshold']:.6f} | {marker} |"
        )
    lines.extend(
        [
            "",
            f"Selected primary supervised baseline: `{best_name}`.",
            "",
            "## Threshold Notes",
            "",
            "For models exposing probabilities or decision scores, thresholds are selected on out-of-fold scores to maximize macro-F1. A secondary threshold constrained to contradiction recall >= 0.30 is also computed and stored in the report data.",
        ]
    )
    return "\n".join(lines)


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if len(unique) <= 100:
        return unique
    quantiles = np.linspace(0.0, 1.0, 101)
    return np.unique(np.quantile(scores, quantiles))


def _n_splits(y: np.ndarray) -> int:
    counts = np.bincount(y)
    positive_counts = [count for count in counts if count > 0]
    return max(2, min(5, min(positive_counts)))
