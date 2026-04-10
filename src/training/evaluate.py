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


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: Sequence[str],
    split_name: str,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Evaluate a model and return a rich metric dictionary."""

    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False)
    for batch in progress:
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        target_start_indices = batch["target_start_idx"].to(device, non_blocking=True)
        target_end_indices = batch["target_end_idx"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(
                inputs,
                target_start_indices=target_start_indices,
                target_end_indices=target_end_indices,
            )
            raw_loss = criterion(logits, targets)
        if raw_loss.ndim > 0:
            loss = raw_loss.mean()
        elif getattr(criterion, "reduction", None) == "sum":
            loss = raw_loss / max(targets.size(0), 1)
        else:
            loss = raw_loss

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
