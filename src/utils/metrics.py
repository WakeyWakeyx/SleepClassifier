"""Evaluation metric helpers for sleep stage classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .utilities import ensure_directory


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    label_names: Sequence[str],
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
    class_precision, class_recall, class_f1, class_support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=label_indices,
            average=None,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(targets, predictions, labels=label_indices)

    per_class = {
        label_name: {
            "precision": float(class_precision[index]),
            "recall": float(class_recall[index]),
            "f1": float(class_f1[index]),
            "support": int(class_support[index]),
        }
        for index, label_name in enumerate(label_names)
    }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix_figure(
    matrix: Sequence[Sequence[int]],
    label_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    """Save a simple confusion matrix heatmap without adding seaborn."""

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
