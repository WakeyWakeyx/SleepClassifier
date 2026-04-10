"""End-to-end training pipeline for DREAMT sleep stage classification."""

from __future__ import annotations

import gc
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm.auto import tqdm

from src.config import DataConfig, ProjectConfig
from src.data.data_loading import (
    ParticipantData,
    build_split_manifest_payload,
    discover_participant_files,
    load_participant_files,
    split_participant_files,
)
from src.data.preprocessing import (
    NormalizationStats,
    apply_normalization,
    clean_participant_frame,
    compute_normalization_stats,
)
from src.data.windowing import WindowedSleepDataset
from src.models import build_model
from src.training.evaluate import evaluate_model, save_evaluation_artifacts
from src.utils import (
    build_logger,
    ensure_directory,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_json,
    seed_worker,
    set_seed,
)


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if alpha.ndim != 1:
                raise ValueError("alpha must be a 1D tensor with one value per class.")
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets = targets.long()
        target_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        loss = -torch.pow(1.0 - target_probs, self.gamma) * target_log_probs
        if self.alpha is not None:
            loss = loss * self.alpha.gather(dim=0, index=targets)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def run_training_pipeline(config: ProjectConfig) -> dict[str, Any]:
    """Run data preparation, training, validation, and final test evaluation."""

    _prepare_output_directories(config)
    logger = build_logger(config.paths.log_path)
    set_seed(config.training.seed)
    device = resolve_device()

    logger.info("Using device: %s", device)
    logger.info("Dataset directory: %s", config.data.dataset_dir)
    logger.info(
        (
            "Configured model=%s | loss=%s | weighted_sampler=%s | "
            "class_weighting=%s | label_smoothing=%.3f"
        ),
        config.model.model_name,
        config.training.loss_name,
        config.training.use_weighted_sampler,
        config.training.class_weighting_mode,
        config.training.label_smoothing,
    )
    _log_sequence_configuration(logger, config.data)

    participant_files = discover_participant_files(config.data.dataset_dir)
    participant_split = split_participant_files(
        file_paths=participant_files,
        data_config=config.data,
        manifest_path=config.paths.split_manifest_path,
        dataset_root=config.data.dataset_dir,
        seed=config.training.seed,
    )
    logger.info(
        "Split participants into %d train / %d validation / %d test files.",
        len(participant_split.train_files),
        len(participant_split.val_files),
        len(participant_split.test_files),
    )

    train_participants = _load_and_clean_participants(
        participant_split.train_files,
        config,
        split_name="train",
        logger=logger,
    )

    normalization_stats = compute_normalization_stats(
        train_participants,
        config.data.feature_columns,
    )
    save_json(normalization_stats.to_dict(), config.paths.normalization_stats_path)
    logger.info("Normalization statistics computed from TRAIN participants only.")

    train_participants = apply_normalization(
        train_participants,
        normalization_stats,
        config.data.feature_columns,
    )
    train_dataset = _build_windowed_dataset(
        train_participants,
        data_config=config.data,
        label_to_index=config.label_to_index,
        split_name="train",
        logger=logger,
    )
    train_participants.clear()
    gc.collect()

    val_dataset = _load_normalize_and_window_split(
        file_paths=participant_split.val_files,
        config=config,
        split_name="validation",
        logger=logger,
        normalization_stats=normalization_stats,
    )
    test_dataset = _load_normalize_and_window_split(
        file_paths=participant_split.test_files,
        config=config,
        split_name="test",
        logger=logger,
        normalization_stats=normalization_stats,
    )
    _validate_dataset_sizes(train_dataset, val_dataset, test_dataset)

    dataset_summary = {
        "windowing_config": _build_windowing_config_summary(config.data),
        "train": _build_split_dataset_summary(
            dataset=train_dataset,
            split_files=participant_split.train_files,
            dataset_root=config.data.dataset_dir,
        ),
        "validation": _build_split_dataset_summary(
            dataset=val_dataset,
            split_files=participant_split.val_files,
            dataset_root=config.data.dataset_dir,
        ),
        "test": _build_split_dataset_summary(
            dataset=test_dataset,
            split_files=participant_split.test_files,
            dataset_root=config.data.dataset_dir,
        ),
    }
    save_json(dataset_summary, config.paths.dataset_summary_path)

    save_json(
        build_split_manifest_payload(
            split=participant_split,
            dataset_root=config.data.dataset_dir,
            all_file_paths=participant_files,
            seed=config.training.seed,
            ratios={
                "train": config.data.train_ratio,
                "val": config.data.val_ratio,
                "test": config.data.test_ratio,
            },
            window_counts_by_split={
                "train": len(train_dataset),
                "validation": len(val_dataset),
                "test": len(test_dataset),
            },
            class_counts_by_split={
                "train": train_dataset.label_counts(),
                "validation": val_dataset.label_counts(),
                "test": test_dataset.label_counts(),
            },
        ),
        config.paths.split_manifest_path,
    )
    logger.info(
        (
            "Split manifest saved | train_files=%d validation_files=%d test_files=%d | "
            "train_windows=%d validation_windows=%d test_windows=%d"
        ),
        len(participant_split.train_files),
        len(participant_split.val_files),
        len(participant_split.test_files),
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    _log_window_summary(logger, "Train", train_dataset)
    _log_window_summary(logger, "Validation", val_dataset)
    _log_window_summary(logger, "Test", test_dataset)

    class_counts, class_weights = _compute_class_weights(
        labels=train_dataset.labels,
        num_classes=config.model.num_classes,
        label_names=config.data.label_names,
        class_weighting_mode=config.training.class_weighting_mode,
        output_path=config.paths.class_weights_path,
    )
    logger.info(
        "Class weights computed from TRAIN target windows only | mode=%s | %s",
        config.training.class_weighting_mode,
        _format_named_values(
            {
                label_name: float(class_weights[index].item())
                for index, label_name in enumerate(config.data.label_names)
            },
            digits=4,
        ),
    )

    train_sampler = (
        _build_weighted_sampler(
            labels=train_dataset.labels,
            class_weights=class_weights,
            seed=config.training.seed,
        )
        if config.training.use_weighted_sampler
        else None
    )

    train_loader = _build_dataloader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=train_sampler is None,
        seed=config.training.seed,
        pin_memory=device.type == "cuda",
        sampler=train_sampler,
    )
    val_loader = _build_dataloader(
        dataset=val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        seed=config.training.seed,
        pin_memory=device.type == "cuda",
    )
    test_loader = _build_dataloader(
        dataset=test_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False,
        seed=config.training.seed,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config.model).to(device)
    train_criterion, loss_description = _build_training_criterion(
        config=config,
        class_weights=class_weights,
        device=device,
    )
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    logger.info("Training criterion: %s", loss_description)

    history: list[dict[str, Any]] = []
    best_val_macro_f1 = float("-inf")
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=train_criterion,
            device=device,
            epoch=epoch,
            max_epochs=config.training.max_epochs,
            gradient_clip_norm=config.training.gradient_clip_norm,
        )
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=eval_criterion,
            device=device,
            label_names=config.data.label_names,
            split_name="validation",
        )
        save_evaluation_artifacts(
            metrics=val_metrics,
            label_names=config.data.label_names,
            output_dir=config.paths.validation_epochs_dir,
            artifact_prefix=f"validation_epoch_{epoch:03d}",
            title=f"Validation Confusion Matrix (Epoch {epoch})",
        )
        save_evaluation_artifacts(
            metrics=val_metrics,
            label_names=config.data.label_names,
            output_dir=config.paths.validation_dir,
            artifact_prefix="validation_latest",
            title=f"Validation Confusion Matrix (Epoch {epoch})",
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": {
                key: val_metrics[key]
                for key in (
                    "loss",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_precision",
                    "weighted_recall",
                    "weighted_f1",
                    "per_class",
                    "target_distribution",
                    "prediction_distribution",
                    "distribution_shift",
                    "confusion_matrix",
                    "num_examples",
                    "split",
                )
            },
        }
        history.append(epoch_record)
        save_json(history, config.paths.training_history_path)

        logger.info(
            (
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | "
                "val_bal_acc=%.4f | val_macro_f1=%.4f | val_weighted_f1=%.4f"
            ),
            epoch,
            config.training.max_epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["balanced_accuracy"],
            val_metrics["macro_f1"],
            val_metrics["weighted_f1"],
        )
        logger.info(
            "Validation per-class F1 | %s",
            _format_per_class_metric(val_metrics["per_class"], metric_name="f1"),
        )
        logger.info(
            "Validation prediction distribution | %s",
            _format_distribution(val_metrics["prediction_distribution"]),
        )
        logger.info(
            "Validation prediction minus target | %s",
            _format_distribution_shift(val_metrics["distribution_shift"]),
        )

        if val_metrics["macro_f1"] > (best_val_macro_f1 + config.training.min_delta):
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            logger.info("New best checkpoint identified at epoch %d.", epoch)
        else:
            epochs_without_improvement += 1
            logger.info(
                "Validation macro F1 did not improve for %d epoch(s).",
                epochs_without_improvement,
            )

        checkpoint_state = {
            "epoch": epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label_names": list(config.data.label_names),
            "feature_columns": list(config.data.feature_columns),
            "normalization_stats": normalization_stats.to_dict(),
            "config": _checkpoint_config_payload(config),
            "train_class_counts": {
                label_name: int(class_counts[index].item())
                for index, label_name in enumerate(config.data.label_names)
            },
            "train_class_weights": {
                label_name: float(class_weights[index].item())
                for index, label_name in enumerate(config.data.label_names)
            },
        }
        save_checkpoint(checkpoint_state, config.paths.latest_checkpoint_path)

        if best_epoch == epoch:
            save_checkpoint(checkpoint_state, config.paths.best_checkpoint_path)
            save_evaluation_artifacts(
                metrics=val_metrics,
                label_names=config.data.label_names,
                output_dir=config.paths.validation_dir,
                artifact_prefix="validation_best",
                title="Validation Confusion Matrix (Best Checkpoint)",
            )

        if epochs_without_improvement >= config.training.patience:
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    load_checkpoint(config.paths.best_checkpoint_path, model=model, device=device)
    logger.info(
        "Loaded best checkpoint from epoch %d with validation macro F1 %.4f.",
        best_epoch,
        best_val_macro_f1,
    )

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=eval_criterion,
        device=device,
        label_names=config.data.label_names,
        split_name="test",
    )
    save_evaluation_artifacts(
        metrics=test_metrics,
        label_names=config.data.label_names,
        output_dir=config.paths.test_dir,
        artifact_prefix="test",
        title="Test Confusion Matrix",
    )
    logger.info(
        (
            "Test results | loss=%.4f | accuracy=%.4f | balanced_accuracy=%.4f | "
            "macro_f1=%.4f | weighted_f1=%.4f"
        ),
        test_metrics["loss"],
        test_metrics["accuracy"],
        test_metrics["balanced_accuracy"],
        test_metrics["macro_f1"],
        test_metrics["weighted_f1"],
    )
    logger.info(
        "Test per-class F1 | %s",
        _format_per_class_metric(test_metrics["per_class"], metric_name="f1"),
    )
    logger.info(
        "Test prediction distribution | %s",
        _format_distribution(test_metrics["prediction_distribution"]),
    )
    logger.info(
        "Test prediction minus target | %s",
        _format_distribution_shift(test_metrics["distribution_shift"]),
    )

    summary = {
        "device": str(device),
        "dataset_dir": str(config.data.dataset_dir),
        "sequence_definition": _build_windowing_config_summary(config.data),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "train_class_counts": {
            label_name: int(class_counts[index].item())
            for index, label_name in enumerate(config.data.label_names)
        },
        "train_class_weights": {
            label_name: float(class_weights[index].item())
            for index, label_name in enumerate(config.data.label_names)
        },
        "test_metrics": test_metrics,
    }
    save_json(summary, config.paths.run_summary_path)
    return summary


def _load_and_clean_participants(
    file_paths: Sequence[Path],
    config: ProjectConfig,
    split_name: str,
    logger: logging.Logger,
) -> list[ParticipantData]:
    """Load raw CSV files and apply participant-level cleaning."""

    logger.info("Loading %s participants...", split_name)
    raw_participants = load_participant_files(file_paths, data_config=config.data)
    cleaned_participants = [
        clean_participant_frame(participant, config.data) for participant in raw_participants
    ]
    logger.info("Loaded and cleaned %d %s participants.", len(cleaned_participants), split_name)
    return cleaned_participants


def _load_normalize_and_window_split(
    file_paths: Sequence[Path],
    config: ProjectConfig,
    split_name: str,
    logger: logging.Logger,
    normalization_stats: NormalizationStats,
) -> WindowedSleepDataset:
    """Load one split, normalize it with train stats, and build its window dataset."""

    participants = _load_and_clean_participants(
        file_paths=file_paths,
        config=config,
        split_name=split_name,
        logger=logger,
    )
    participants = apply_normalization(
        participants,
        normalization_stats,
        config.data.feature_columns,
    )
    dataset = _build_windowed_dataset(
        participants,
        data_config=config.data,
        label_to_index=config.label_to_index,
        split_name=split_name,
        logger=logger,
    )
    participants.clear()
    gc.collect()
    return dataset


def _build_windowed_dataset(
    participants: Sequence[ParticipantData],
    data_config: DataConfig,
    label_to_index: dict[str, int],
    split_name: str,
    logger: logging.Logger,
) -> WindowedSleepDataset:
    """Build a compact windowed dataset and log its retained sample count."""

    dataset = WindowedSleepDataset(participants, data_config, label_to_index)
    logger.info(
        "Materialized %d retained %s windows from compact participant arrays.",
        len(dataset),
        split_name,
    )
    return dataset


def _prepare_output_directories(config: ProjectConfig) -> None:
    """Create all output directories used by the pipeline."""

    ensure_directory(config.paths.output_dir)
    ensure_directory(config.paths.validation_dir)
    ensure_directory(config.paths.validation_epochs_dir)
    ensure_directory(config.paths.test_dir)
    ensure_directory(config.paths.latest_checkpoint_path.parent)
    ensure_directory(config.paths.best_checkpoint_path.parent)


def _build_dataloader(
    dataset: WindowedSleepDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    """Construct a DataLoader with deterministic worker seeding."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def _compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
    label_names: Sequence[str],
    class_weighting_mode: str,
    output_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute train-only class weights according to the configured weighting mode."""

    if len(labels) == 0:
        raise ValueError("Train dataset produced no labels for class weight computation.")

    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes)
    if torch.any(counts == 0):
        zero_classes = [
            label_names[index]
            for index, count in enumerate(counts.tolist())
            if count == 0
        ]
        raise ValueError(
            "At least one supervised class has zero train windows: "
            + ", ".join(zero_classes)
        )

    if class_weighting_mode == "inverse_frequency":
        total = counts.sum().float()
        weights = total / (counts.float() * float(num_classes))
        weights = weights / weights.mean()
    elif class_weighting_mode == "sqrt_inverse_frequency":
        total = counts.sum().float()
        weights = torch.sqrt(total / (counts.float() * float(num_classes)))
        weights = weights / weights.mean()
    else:
        weights = torch.ones(num_classes, dtype=torch.float32)

    save_json(
        {
            "computed_from": "train_target_windows_only",
            "class_weighting_mode": class_weighting_mode,
            "counts": {
                label_names[index]: int(counts[index].item())
                for index in range(num_classes)
            },
            "weights": {
                label_names[index]: float(weights[index].item())
                for index in range(num_classes)
            },
        },
        output_path,
    )
    return counts, weights


def _build_weighted_sampler(
    labels: Sequence[int],
    class_weights: torch.Tensor,
    seed: int,
) -> WeightedRandomSampler:
    """Create a train-only weighted sampler using train-window class weights."""

    label_tensor = torch.tensor(labels, dtype=torch.long)
    sample_weights = class_weights[label_tensor].double()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def _build_training_criterion(
    config: ProjectConfig,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[nn.Module, str]:
    """Create the configured training loss."""

    if config.training.loss_name == "cross_entropy":
        return (
            nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing),
            f"cross_entropy(label_smoothing={config.training.label_smoothing:.3f})",
        )

    if config.training.loss_name == "weighted_cross_entropy":
        return (
            nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=config.training.label_smoothing,
            ),
            (
                "weighted_cross_entropy("
                f"label_smoothing={config.training.label_smoothing:.3f}, "
                f"class_weights=train_window_{config.training.class_weighting_mode})"
            ),
        )

    focal_alpha: torch.Tensor | None = None
    focal_alpha_description = "none"
    if config.training.focal_alpha is not None:
        focal_alpha = torch.tensor(config.training.focal_alpha, dtype=torch.float32, device=device)
        focal_alpha_description = "config_focal_alpha"
    elif config.training.loss_name == "weighted_focal_loss":
        focal_alpha = class_weights.to(device)
        focal_alpha_description = f"train_window_{config.training.class_weighting_mode}"

    return (
        FocalLoss(
            gamma=config.training.focal_gamma,
            alpha=focal_alpha,
            reduction=config.training.focal_reduction,
        ),
        (
            f"{config.training.loss_name}("
            f"gamma={config.training.focal_gamma:.3f}, "
            f"alpha={focal_alpha_description}, "
            f"reduction={config.training.focal_reduction}, "
            f"label_smoothing_applied={False})"
        ),
    )


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    gradient_clip_norm: float,
) -> float:
    """Run one supervised training epoch."""

    model.train()
    total_loss = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)
    for batch in progress:
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        target_start_indices = batch["target_start_idx"].to(device, non_blocking=True)
        target_end_indices = batch["target_end_idx"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            inputs,
            target_start_indices=target_start_indices,
            target_end_indices=target_end_indices,
        )
        raw_loss = criterion(logits, targets)
        if raw_loss.ndim > 0:
            loss = raw_loss.mean()
        else:
            loss = raw_loss
        loss.backward()

        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

        optimizer.step()

        batch_size = targets.size(0)
        if raw_loss.ndim > 0:
            display_loss = raw_loss.mean()
        elif getattr(criterion, "reduction", None) == "sum":
            display_loss = raw_loss / max(batch_size, 1)
        else:
            display_loss = raw_loss

        total_loss += float(display_loss.item()) * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{display_loss.item():.4f}")

    if total_examples == 0:
        raise ValueError("No training samples were available for this epoch.")
    return total_loss / total_examples


def _build_split_dataset_summary(
    dataset: WindowedSleepDataset,
    split_files: Sequence[Path],
    dataset_root: Path,
) -> dict[str, Any]:
    """Collect window diagnostics for one split."""

    return {
        "num_files": len(split_files),
        "source_files": [path.relative_to(dataset_root).as_posix() for path in split_files],
        "num_windows": len(dataset),
        "class_counts": dataset.label_counts(),
        "windowing": dataset.summary,
    }


def _log_window_summary(
    logger: logging.Logger,
    split_name: str,
    dataset: WindowedSleepDataset,
) -> None:
    """Log concise split-level window generation diagnostics."""

    sequence_definition = dataset.summary["sequence_definition"]
    if sequence_definition["use_context_windows"]:
        logger.info(
            (
                "%s sequence definition | target_length=%d left_context=%d "
                "right_context=%d total_input_length=%d"
            ),
            split_name,
            sequence_definition["target_window_length"],
            sequence_definition["left_context"],
            sequence_definition["right_context"],
            sequence_definition["total_input_length"],
        )

    logger.info(
        (
            "%s windows | candidates=%d kept=%d discarded=%d "
            "participants_with_no_kept_windows=%d ambiguous_windows_kept=%d"
        ),
        split_name,
        dataset.summary["candidate_windows"],
        dataset.summary["kept_windows"],
        dataset.summary["discarded_windows"],
        dataset.summary["participants_with_no_kept_windows"],
        dataset.summary["ambiguous_windows_kept"],
    )
    logger.info(
        "%s class counts | %s",
        split_name,
        _format_named_values(dataset.summary["kept_class_counts"], digits=0),
    )
    if dataset.summary["discard_reasons"]:
        logger.info(
            "%s discard reasons | %s",
            split_name,
            _format_named_values(dataset.summary["discard_reasons"], digits=0),
        )


def _log_sequence_configuration(
    logger: logging.Logger,
    data_config: DataConfig,
) -> None:
    """Log how each supervised sequence sample is constructed."""

    logger.info(
        (
            "Sequence windowing | target_length=%d | left_context=%d | "
            "right_context=%d | total_input_length=%d | step=%d | use_context_windows=%s"
        ),
        data_config.target_window_length,
        data_config.effective_left_context,
        data_config.effective_right_context,
        data_config.input_window_length,
        data_config.step,
        data_config.use_context_windows,
    )


def _build_windowing_config_summary(data_config: DataConfig) -> dict[str, int | float | bool]:
    """Serialize the sequence windowing configuration saved with artifacts."""

    return {
        "target_window_length": data_config.target_window_length,
        "step": data_config.step,
        "use_context_windows": data_config.use_context_windows,
        "configured_left_context": data_config.left_context,
        "configured_right_context": data_config.right_context,
        "left_context": data_config.effective_left_context,
        "right_context": data_config.effective_right_context,
        "total_input_length": data_config.input_window_length,
        "label_purity_threshold": data_config.label_purity_threshold,
        "min_valid_fraction": data_config.min_valid_fraction,
        "drop_ambiguous_windows": data_config.drop_ambiguous_windows,
        "require_min_valid_labels": data_config.require_min_valid_labels,
        "continuity_gap_factor": data_config.continuity_gap_factor,
    }


def _format_per_class_metric(
    per_class_metrics: dict[str, dict[str, float]],
    metric_name: str,
) -> str:
    """Render compact per-class metric logging."""

    parts = [
        f"{label}={metrics[metric_name]:.4f}"
        for label, metrics in per_class_metrics.items()
    ]
    return ", ".join(parts)


def _format_distribution(distribution: dict[str, dict[str, float]]) -> str:
    """Render label distributions using count values."""

    parts = [
        f"{label}={stats['count']}"
        for label, stats in distribution.items()
    ]
    return ", ".join(parts)


def _format_distribution_shift(distribution_shift: dict[str, dict[str, float]]) -> str:
    """Render prediction-minus-target count deltas in a compact log string."""

    parts = []
    for label, stats in distribution_shift.items():
        parts.append(
            f"{label}={stats['count_delta']:+d} ({stats['fraction_delta']:+.3f})"
        )
    return ", ".join(parts)


def _format_named_values(
    values: dict[str, int | float],
    digits: int,
) -> str:
    """Render scalar dictionaries in a stable order."""

    parts = []
    for label, value in values.items():
        if digits == 0:
            parts.append(f"{label}={int(value)}")
        else:
            parts.append(f"{label}={float(value):.{digits}f}")
    return ", ".join(parts)


def _checkpoint_config_payload(config: ProjectConfig) -> dict[str, Any]:
    """Serialize the configuration stored with checkpoints."""

    payload = asdict(config)
    payload["data"]["dataset_dir"] = str(config.data.dataset_dir)
    payload["paths"] = {
        key: str(value)
        for key, value in asdict(config.paths).items()
    }
    return payload


def _validate_dataset_sizes(
    train_dataset: WindowedSleepDataset,
    val_dataset: WindowedSleepDataset,
    test_dataset: WindowedSleepDataset,
) -> None:
    """Fail early when any split produced zero retained windows."""

    if len(train_dataset) == 0:
        raise ValueError("Train split produced zero valid windows.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split produced zero valid windows.")
    if len(test_dataset) == 0:
        raise ValueError("Test split produced zero valid windows.")
