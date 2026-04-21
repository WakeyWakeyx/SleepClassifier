"""Reusable evaluation helpers shared by training and standalone evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sleep_classifier.utils import compute_classification_metrics, plot_confusion_matrix, save_json


@dataclass(slots=True)
class EvaluationResult:
    """Structured evaluation outputs."""

    split_name: str
    metrics: dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    report_dict: dict[str, Any]
    report_text: str
    confusion_matrix: np.ndarray


def create_dataloader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
    sampler: torch.utils.data.Sampler[int] | None = None,
) -> DataLoader:
    """Create a DataLoader with the project's default runtime options."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def evaluate_model(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module | None,
    device: torch.device,
    use_mixed_precision: bool,
    class_names: list[str],
    batch_size: int,
    num_workers: int,
    split_name: str = "evaluation",
    metrics_output_path: Path | None = None,
    report_output_path: Path | None = None,
    confusion_output_path: Path | None = None,
    report_text_output_path: Path | None = None,
) -> EvaluationResult:
    """Run inference, compute metrics, and optionally save evaluation artifacts."""

    data_loader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=False,
    )

    model.eval()
    running_loss = 0.0
    total_examples = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                logits = model(inputs)
                loss = criterion(logits, labels) if criterion is not None else None

            batch_size_value = labels.size(0)
            if loss is not None:
                running_loss += float(loss.item()) * batch_size_value
            total_examples += int(batch_size_value)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())

    if total_examples == 0:
        raise ValueError(f"{split_name} loader produced zero examples.")

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)
    if criterion is not None:
        metrics["loss"] = running_loss / total_examples

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    if metrics_output_path is not None:
        save_json(metrics, metrics_output_path)
    if report_output_path is not None:
        save_json(report_dict, report_output_path)
    if report_text_output_path is not None:
        report_text_output_path.parent.mkdir(parents=True, exist_ok=True)
        report_text_output_path.write_text(report_text, encoding="utf-8")
    if confusion_output_path is not None:
        plot_confusion_matrix(
            confusion_matrix=confusion,
            class_names=class_names,
            output_path=confusion_output_path,
            title=f"{split_name.title()} Confusion Matrix",
        )

    return EvaluationResult(
        split_name=split_name,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        report_dict=report_dict,
        report_text=report_text,
        confusion_matrix=confusion,
    )
