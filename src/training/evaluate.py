"""Model evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import (
    compute_classification_metrics,
    ensure_directory,
    save_confusion_matrix_csv,
    save_confusion_matrix_figure,
    save_json,
    save_per_class_metrics_csv,
    save_summary_metrics_csv,
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: Sequence[str],
    split_name: str,
) -> dict[str, Any]:
    """Evaluate a model and return a rich metric dictionary."""

    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        predictions = logits.argmax(dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    if total_examples == 0:
        raise ValueError(f"No evaluation samples were available for split '{split_name}'.")

    metrics = compute_classification_metrics(
        targets=all_targets,
        predictions=all_predictions,
        label_names=label_names,
    )
    metrics["loss"] = total_loss / total_examples
    metrics["num_examples"] = total_examples
    metrics["split"] = split_name
    return metrics


def save_evaluation_artifacts(
    metrics: dict[str, Any],
    label_names: Sequence[str],
    output_dir: Path,
    artifact_prefix: str,
    title: str,
) -> None:
    """Persist JSON, CSV, and figure outputs for one evaluation run."""

    ensure_directory(output_dir)
    save_json(metrics, output_dir / f"{artifact_prefix}_metrics.json")
    save_summary_metrics_csv(metrics, output_dir / f"{artifact_prefix}_summary.csv")
    save_per_class_metrics_csv(
        metrics["per_class"],
        output_dir / f"{artifact_prefix}_per_class_metrics.csv",
    )
    save_confusion_matrix_csv(
        matrix=metrics["confusion_matrix"],
        label_names=label_names,
        output_path=output_dir / f"{artifact_prefix}_confusion_matrix.csv",
    )
    save_confusion_matrix_figure(
        matrix=metrics["confusion_matrix"],
        label_names=label_names,
        output_path=output_dir / f"{artifact_prefix}_confusion_matrix.png",
        title=title,
    )
