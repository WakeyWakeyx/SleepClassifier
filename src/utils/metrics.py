"""Evaluation metric helpers for sleep stage classification."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .utilities import ensure_directory, save_csv_rows


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    label_names: Sequence[str],
    minority_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute aggregate and per-class classification metrics."""

    label_indices = list(range(len(label_names)))
    accuracy = accuracy_score(targets, predictions)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=label_indices,
        average="weighted",
        zero_division=0,
    )
    class_precision, class_recall, class_f1, class_support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=label_indices,
            average=None,
            zero_division=0,
        )
    )
    supported_classes = class_support > 0
    balanced_accuracy = (
        float(class_recall[supported_classes].mean()) if np.any(supported_classes) else 0.0
    )
    matrix = confusion_matrix(targets, predictions, labels=label_indices)

    target_counts = Counter(targets)
    prediction_counts = Counter(predictions)
    total_targets = len(targets)
    total_predictions = len(predictions)

    per_class = {
        label_name: {
            "precision": float(class_precision[index]),
            "recall": float(class_recall[index]),
            "f1": float(class_f1[index]),
            "support": int(class_support[index]),
            "support_fraction": (
                float(class_support[index]) / float(total_targets) if total_targets else 0.0
            ),
            "predicted_count": int(prediction_counts.get(index, 0)),
            "predicted_fraction": (
                float(prediction_counts.get(index, 0)) / float(total_predictions)
                if total_predictions
                else 0.0
            ),
            "prediction_minus_target_count": int(
                prediction_counts.get(index, 0) - class_support[index]
            ),
            "prediction_minus_target_fraction": (
                (
                    float(prediction_counts.get(index, 0)) / float(total_predictions)
                    if total_predictions
                    else 0.0
                )
                - (
                    float(class_support[index]) / float(total_targets)
                    if total_targets
                    else 0.0
                )
            ),
        }
        for index, label_name in enumerate(label_names)
    }
    minority_label_set = set(minority_labels or ())
    minority_scores = [
        per_class[label_name]["f1"]
        for label_name in label_names
        if label_name in minority_label_set
    ]
    minority_macro_f1 = (
        float(sum(minority_scores) / len(minority_scores))
        if minority_scores
        else float(macro_f1)
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "minority_macro_f1": minority_macro_f1,
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "target_distribution": _build_distribution(
            counts=target_counts,
            total=total_targets,
            label_names=label_names,
        ),
        "prediction_distribution": _build_distribution(
            counts=prediction_counts,
            total=total_predictions,
            label_names=label_names,
        ),
        "distribution_shift": _build_distribution_shift(
            target_counts=target_counts,
            prediction_counts=prediction_counts,
            total_targets=total_targets,
            total_predictions=total_predictions,
            label_names=label_names,
        ),
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix_figure(
    matrix: Sequence[Sequence[int]],
    label_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    """Save a confusion matrix heatmap without adding extra dependencies."""

    ensure_directory(output_path.parent)
    values = np.asarray(matrix, dtype=np.int64)

    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = values.max() / 2.0 if values.size else 0.0
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]}",
                ha="center",
                va="center",
                color="white" if values[row_index, column_index] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_csv(
    matrix: Sequence[Sequence[int]],
    label_names: Sequence[str],
    output_path: Path,
) -> None:
    """Save the confusion matrix as a CSV table."""

    rows = []
    for row_label, row_values in zip(label_names, matrix, strict=True):
        row = {"true_label": row_label}
        row.update(
            {
                predicted_label: int(count)
                for predicted_label, count in zip(label_names, row_values, strict=True)
            }
        )
        rows.append(row)

    save_csv_rows(
        rows=rows,
        path=output_path,
        fieldnames=["true_label", *label_names],
    )


def save_per_class_metrics_csv(
    per_class_metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Save per-class metrics and prediction frequencies as CSV."""

    rows = []
    for label_name, metrics in per_class_metrics.items():
        rows.append(
            {
                "label": label_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "support": metrics["support"],
                "support_fraction": metrics["support_fraction"],
                "predicted_count": metrics["predicted_count"],
                "predicted_fraction": metrics["predicted_fraction"],
                "prediction_minus_target_count": metrics["prediction_minus_target_count"],
                "prediction_minus_target_fraction": metrics["prediction_minus_target_fraction"],
            }
        )

    save_csv_rows(
        rows=rows,
        path=output_path,
        fieldnames=[
            "label",
            "precision",
            "recall",
            "f1",
            "support",
            "support_fraction",
            "predicted_count",
            "predicted_fraction",
            "prediction_minus_target_count",
            "prediction_minus_target_fraction",
        ],
    )


def save_summary_metrics_csv(metrics: dict[str, Any], output_path: Path) -> None:
    """Save the scalar evaluation metrics as a single-row CSV."""

    save_csv_rows(
        rows=[
            {
                "split": metrics["split"],
                "num_examples": metrics["num_examples"],
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "minority_macro_f1": metrics["minority_macro_f1"],
                "weighted_precision": metrics["weighted_precision"],
                "weighted_recall": metrics["weighted_recall"],
                "weighted_f1": metrics["weighted_f1"],
            }
        ],
        path=output_path,
        fieldnames=[
            "split",
            "num_examples",
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "minority_macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ],
    )


def save_distribution_shift_csv(
    distribution_shift: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Persist prediction-vs-target distribution deltas for quick inspection."""

    rows = []
    for label_name, stats in distribution_shift.items():
        rows.append(
            {
                "label": label_name,
                "target_count": stats["target_count"],
                "prediction_count": stats["prediction_count"],
                "count_delta": stats["count_delta"],
                "target_fraction": stats["target_fraction"],
                "prediction_fraction": stats["prediction_fraction"],
                "fraction_delta": stats["fraction_delta"],
            }
        )

    save_csv_rows(
        rows=rows,
        path=output_path,
        fieldnames=[
            "label",
            "target_count",
            "prediction_count",
            "count_delta",
            "target_fraction",
            "prediction_fraction",
            "fraction_delta",
        ],
    )


def _build_distribution(
    counts: Counter[int],
    total: int,
    label_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Build count and fraction dictionaries for label frequencies."""

    distribution: dict[str, dict[str, float]] = {}
    for index, label_name in enumerate(label_names):
        count = int(counts.get(index, 0))
        distribution[label_name] = {
            "count": count,
            "fraction": (float(count) / float(total)) if total else 0.0,
        }
    return distribution


def _build_distribution_shift(
    target_counts: Counter[int],
    prediction_counts: Counter[int],
    total_targets: int,
    total_predictions: int,
    label_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compare predicted frequency against true frequency for each class."""

    shift: dict[str, dict[str, float]] = {}
    for index, label_name in enumerate(label_names):
        target_count = int(target_counts.get(index, 0))
        prediction_count = int(prediction_counts.get(index, 0))
        target_fraction = (float(target_count) / float(total_targets)) if total_targets else 0.0
        prediction_fraction = (
            float(prediction_count) / float(total_predictions)
            if total_predictions
            else 0.0
        )
        shift[label_name] = {
            "target_count": target_count,
            "prediction_count": prediction_count,
            "count_delta": prediction_count - target_count,
            "target_fraction": target_fraction,
            "prediction_fraction": prediction_fraction,
            "fraction_delta": prediction_fraction - target_fraction,
        }
    return shift
