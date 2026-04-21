"""Reusable evaluation helpers for epoch-sequence sleep-stage models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from sleep_classifier.data.epoch_dataset import PreparedSplitData, SleepWindowDataset
from sleep_classifier.experiment_config import ExperimentConfig
from sleep_classifier.utils import compute_classification_metrics, plot_confusion_matrix, save_json


@dataclass(slots=True)
class EvaluationResult:
    """Structured evaluation outputs."""

    split_name: str
    metrics: dict[str, Any]
    smoothed_metrics: dict[str, Any] | None
    y_true: np.ndarray
    raw_y_pred: np.ndarray
    smoothed_y_pred: np.ndarray | None
    raw_probabilities: np.ndarray
    report_dict: dict[str, Any]
    report_text: str
    confusion_matrix: np.ndarray


def create_dataloader(
    dataset: SleepWindowDataset,
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


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a batched sample dictionary to the target device."""

    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def _group_sequence_indices(metadata: list[dict[str, Any]]) -> list[np.ndarray]:
    groups: list[list[int]] = []
    current_group: list[int] = []
    previous_participant: str | None = None
    previous_epoch_index: int | None = None
    previous_stride: int | None = None

    for index, row in enumerate(metadata):
        participant_id = str(row["participant_id"])
        center_epoch_index = int(row["center_epoch_index"])
        stride_epochs = int(row["stride_epochs"])
        is_contiguous = (
            previous_participant == participant_id
            and previous_epoch_index is not None
            and previous_stride is not None
            and center_epoch_index - previous_epoch_index == previous_stride
        )
        if not current_group or is_contiguous:
            current_group.append(index)
        else:
            groups.append(current_group)
            current_group = [index]
        previous_participant = participant_id
        previous_epoch_index = center_epoch_index
        previous_stride = stride_epochs

    if current_group:
        groups.append(current_group)
    return [np.asarray(group, dtype=np.int64) for group in groups]


def _median_smooth_sequence(predictions: np.ndarray, window_size: int) -> np.ndarray:
    if len(predictions) == 0:
        return predictions.copy()
    radius = window_size // 2
    smoothed = np.empty_like(predictions)
    for index in range(len(predictions)):
        start = max(0, index - radius)
        end = min(len(predictions), index + radius + 1)
        smoothed[index] = int(np.median(predictions[start:end]))
    return smoothed


def _viterbi_decode(
    probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    class_priors: np.ndarray,
) -> np.ndarray:
    if len(probabilities) == 0:
        return np.empty((0,), dtype=np.int64)

    log_transition = np.log(np.clip(transition_matrix, 1e-8, 1.0))
    log_prior = np.log(np.clip(class_priors, 1e-8, 1.0))
    log_emission = np.log(np.clip(probabilities, 1e-8, 1.0))

    steps, num_classes = log_emission.shape
    scores = np.empty((steps, num_classes), dtype=np.float64)
    backpointers = np.empty((steps, num_classes), dtype=np.int64)
    scores[0] = log_prior + log_emission[0]
    backpointers[0] = 0

    for time_index in range(1, steps):
        previous_scores = scores[time_index - 1][:, None] + log_transition
        best_previous = np.argmax(previous_scores, axis=0)
        scores[time_index] = previous_scores[best_previous, np.arange(num_classes)] + log_emission[time_index]
        backpointers[time_index] = best_previous

    decoded = np.empty((steps,), dtype=np.int64)
    decoded[-1] = int(np.argmax(scores[-1]))
    for time_index in range(steps - 2, -1, -1):
        decoded[time_index] = backpointers[time_index + 1, decoded[time_index + 1]]
    return decoded


def smooth_predictions(
    metadata: list[dict[str, Any]],
    raw_predictions: np.ndarray,
    raw_probabilities: np.ndarray,
    config: ExperimentConfig,
    transition_matrix: list[list[float]] | None,
    class_priors: list[float] | None,
) -> np.ndarray | None:
    """Apply inference-time smoothing to a prediction sequence."""

    if config.smoothing_mode == "none":
        return None
    if config.smoothing_mode == "viterbi" and (transition_matrix is None or class_priors is None):
        return None

    grouped_indices = _group_sequence_indices(metadata)
    smoothed_predictions = raw_predictions.copy()
    for indices in grouped_indices:
        if config.smoothing_mode == "median":
            smoothed_predictions[indices] = _median_smooth_sequence(
                raw_predictions[indices],
                config.median_filter_size,
            )
            continue

        if config.smoothing_mode == "viterbi" and transition_matrix is not None and class_priors is not None:
            smoothed_predictions[indices] = _viterbi_decode(
                raw_probabilities[indices],
                np.asarray(transition_matrix, dtype=np.float64),
                np.asarray(class_priors, dtype=np.float64),
            )

    return smoothed_predictions


def _build_report_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, Any], str]:
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
    return report_dict, report_text


def evaluate_model(
    model: nn.Module,
    prepared_split: PreparedSplitData,
    criterion: nn.Module | None,
    config: ExperimentConfig,
    device: torch.device,
    use_mixed_precision: bool,
    class_names: list[str],
    batch_size: int,
    num_workers: int,
    transition_matrix: list[list[float]] | None = None,
    class_priors: list[float] | None = None,
    split_name: str = "evaluation",
    metrics_output_path: Path | None = None,
    report_output_path: Path | None = None,
    confusion_output_path: Path | None = None,
    report_text_output_path: Path | None = None,
) -> EvaluationResult:
    """Run inference, compute metrics, and optionally save evaluation artifacts."""

    data_loader = create_dataloader(
        dataset=prepared_split.dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=False,
    )

    model.eval()
    running_total_loss = 0.0
    running_final_loss = 0.0
    running_raw_loss = 0.0
    total_examples = 0
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                outputs = model(batch["features"])
                loss_components: dict[str, Any] | None = None
                if criterion is not None:
                    _, loss_components = criterion(outputs, batch)
                final_logits = outputs["final_logits"]
                if final_logits is None:
                    raise ValueError("Model outputs are missing final_logits.")
                batch_probabilities = torch.softmax(final_logits, dim=1)

            batch_size_value = batch["final_target"].size(0)
            if loss_components is not None:
                running_total_loss += float(loss_components["total_loss"]) * batch_size_value
                running_final_loss += float(loss_components["final_loss"]) * batch_size_value
                running_raw_loss += float(loss_components["raw_loss"]) * batch_size_value
            total_examples += int(batch_size_value)
            predictions.append(torch.argmax(final_logits, dim=1).cpu().numpy())
            probabilities.append(batch_probabilities.cpu().numpy())
            targets.append(batch["final_target"].cpu().numpy())

    if total_examples == 0:
        raise ValueError(f"{split_name} loader produced zero examples.")

    y_true = np.concatenate(targets)
    raw_y_pred = np.concatenate(predictions)
    raw_probabilities = np.concatenate(probabilities, axis=0)
    metrics = compute_classification_metrics(y_true=y_true, y_pred=raw_y_pred, class_names=class_names)
    if criterion is not None:
        metrics["loss"] = running_total_loss / total_examples
        metrics["final_loss"] = running_final_loss / total_examples
        metrics["raw_loss"] = running_raw_loss / total_examples

    smoothed_y_pred = smooth_predictions(
        metadata=prepared_split.metadata,
        raw_predictions=raw_y_pred,
        raw_probabilities=raw_probabilities,
        config=config,
        transition_matrix=transition_matrix,
        class_priors=class_priors,
    )
    smoothed_metrics = None
    if smoothed_y_pred is not None:
        smoothed_metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=smoothed_y_pred,
            class_names=class_names,
        )

    report_dict, report_text = _build_report_bundle(y_true, raw_y_pred, class_names)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    if metrics_output_path is not None:
        payload = {
            "raw": metrics,
            "smoothed": smoothed_metrics,
            "smoothing_mode": config.smoothing_mode,
        }
        save_json(payload, metrics_output_path)
    if report_output_path is not None:
        smoothed_report_dict = (
            _build_report_bundle(y_true, smoothed_y_pred, class_names)[0]
            if smoothed_y_pred is not None
            else None
        )
        save_json(
            {
                "raw": report_dict,
                "smoothed": smoothed_report_dict,
                "smoothing_mode": config.smoothing_mode,
            },
            report_output_path,
        )
    if report_text_output_path is not None:
        report_text_output_path.parent.mkdir(parents=True, exist_ok=True)
        report_sections = ["RAW\n", report_text]
        if smoothed_y_pred is not None:
            _, smoothed_report_text = _build_report_bundle(y_true, smoothed_y_pred, class_names)
            report_sections.extend(["\nSMOOTHED\n", smoothed_report_text])
        report_text_output_path.write_text("\n".join(report_sections), encoding="utf-8")
    if confusion_output_path is not None:
        plot_confusion_matrix(
            confusion_matrix=confusion,
            class_names=class_names,
            output_path=confusion_output_path,
            title=f"{split_name.title()} Raw Confusion Matrix",
        )
        if smoothed_y_pred is not None:
            smoothed_confusion = np.asarray(
                smoothed_metrics["confusion_matrix"],
                dtype=np.int64,
            )
            plot_confusion_matrix(
                confusion_matrix=smoothed_confusion,
                class_names=class_names,
                output_path=confusion_output_path.with_name(
                    f"{confusion_output_path.stem}_smoothed{confusion_output_path.suffix}"
                ),
                title=f"{split_name.title()} Smoothed Confusion Matrix",
            )

    return EvaluationResult(
        split_name=split_name,
        metrics=metrics,
        smoothed_metrics=smoothed_metrics,
        y_true=y_true,
        raw_y_pred=raw_y_pred,
        smoothed_y_pred=smoothed_y_pred,
        raw_probabilities=raw_probabilities,
        report_dict=report_dict,
        report_text=report_text,
        confusion_matrix=confusion,
    )
