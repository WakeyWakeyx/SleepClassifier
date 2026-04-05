"""End-to-end training pipeline for DREAMT sleep stage classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import ProjectConfig
from src.data.data_loading import (
    ParticipantData,
    discover_participant_files,
    load_participant_files,
    split_participant_files,
)
from src.data.preprocessing import (
    apply_normalization,
    clean_participant_frame,
    compute_normalization_stats,
)
from src.data.windowing import WindowedSleepDataset
from src.models import SleepStageCNN
from src.training.evaluate import evaluate_model
from src.utils import (
    build_logger,
    ensure_directory,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_confusion_matrix_figure,
    save_json,
    seed_worker,
    set_seed,
)


def run_training_pipeline(config: ProjectConfig) -> dict[str, Any]:
    """Run data preparation, training, validation, and final test evaluation."""

    _prepare_output_directories(config)
    logger = build_logger(config.paths.log_path)
    set_seed(config.training.seed)
    device = resolve_device()

    logger.info("Using device: %s", device)
    logger.info("Dataset directory: %s", config.data.dataset_dir)

    participant_files = discover_participant_files(config.data.dataset_dir)
    participant_split = split_participant_files(
        file_paths=participant_files,
        data_config=config.data,
        manifest_path=config.paths.split_manifest_path,
        dataset_root=config.data.dataset_dir,
        seed=config.training.seed,
    )
    logger.info(
        "Split participants into %d train / %d val / %d test files.",
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
    val_participants = _load_and_clean_participants(
        participant_split.val_files,
        config,
        split_name="validation",
        logger=logger,
    )
    test_participants = _load_and_clean_participants(
        participant_split.test_files,
        config,
        split_name="test",
        logger=logger,
    )

    normalization_stats = compute_normalization_stats(
        train_participants,
        config.data.feature_columns,
    )
    save_json(normalization_stats.to_dict(), config.paths.normalization_stats_path)

    train_participants = apply_normalization(
        train_participants,
        normalization_stats,
        config.data.feature_columns,
    )
    val_participants = apply_normalization(
        val_participants,
        normalization_stats,
        config.data.feature_columns,
    )
    test_participants = apply_normalization(
        test_participants,
        normalization_stats,
        config.data.feature_columns,
    )

    train_dataset = WindowedSleepDataset(
        train_participants,
        config.data,
        config.label_to_index,
    )
    val_dataset = WindowedSleepDataset(
        val_participants,
        config.data,
        config.label_to_index,
    )
    test_dataset = WindowedSleepDataset(
        test_participants,
        config.data,
        config.label_to_index,
    )
    _validate_dataset_sizes(train_dataset, val_dataset, test_dataset)

    dataset_summary = {
        "train": {
            "num_windows": len(train_dataset),
            "class_counts": train_dataset.label_counts(config.data.label_names),
            "summary": dict(train_dataset.summary),
        },
        "validation": {
            "num_windows": len(val_dataset),
            "class_counts": val_dataset.label_counts(config.data.label_names),
            "summary": dict(val_dataset.summary),
        },
        "test": {
            "num_windows": len(test_dataset),
            "class_counts": test_dataset.label_counts(config.data.label_names),
            "summary": dict(test_dataset.summary),
        },
    }
    save_json(dataset_summary, config.paths.output_dir / "dataset_summary.json")
    logger.info(
        "Window counts | train=%d validation=%d test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_loader = _build_dataloader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
        seed=config.training.seed,
        pin_memory=device.type == "cuda",
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

    class_weights = _compute_class_weights(
        labels=train_dataset.labels,
        num_classes=config.model.num_classes,
        label_names=config.data.label_names,
        output_path=config.paths.class_weights_path,
    )

    model = SleepStageCNN(
        input_channels=config.model.input_channels,
        num_classes=config.model.num_classes,
        conv_channels=config.model.conv_channels,
        kernel_sizes=config.model.kernel_sizes,
        dropout=config.model.dropout,
        classifier_hidden_dim=config.model.classifier_hidden_dim,
    ).to(device)

    train_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

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
            output_dir=config.paths.validation_dir,
            artifact_prefix="validation_latest",
            split_name="validation",
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class": val_metrics["per_class"],
        }
        history.append(epoch_record)
        save_json(history, config.paths.training_history_path)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_macro_f1=%.4f",
            epoch,
            config.training.max_epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )
        logger.info(
            "Validation per-class F1 | %s",
            _format_per_class_metric(val_metrics["per_class"], metric_name="f1"),
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
            "config": {
                "data": {
                    "window_length": config.data.window_length,
                    "step": config.data.step,
                    "label_agreement_threshold": config.data.label_agreement_threshold,
                    "min_valid_fraction": config.data.min_valid_fraction,
                },
                "model": {
                    "input_channels": config.model.input_channels,
                    "num_classes": config.model.num_classes,
                    "conv_channels": list(config.model.conv_channels),
                    "kernel_sizes": list(config.model.kernel_sizes),
                    "dropout": config.model.dropout,
                    "classifier_hidden_dim": config.model.classifier_hidden_dim,
                },
            },
        }
        save_checkpoint(checkpoint_state, config.paths.latest_checkpoint_path)

        if best_epoch == epoch:
            save_checkpoint(checkpoint_state, config.paths.best_checkpoint_path)
            save_json(
                val_metrics,
                config.paths.validation_dir / "validation_best_metrics.json",
            )
            save_confusion_matrix_figure(
                matrix=val_metrics["confusion_matrix"],
                label_names=config.data.label_names,
                output_path=(
                    config.paths.validation_dir / "validation_best_confusion_matrix.png"
                ),
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
        output_dir=config.paths.test_dir,
        artifact_prefix="test",
        split_name="test",
    )
    logger.info(
        "Test results | loss=%.4f | accuracy=%.4f | macro_precision=%.4f | macro_recall=%.4f | macro_f1=%.4f",
        test_metrics["loss"],
        test_metrics["accuracy"],
        test_metrics["macro_precision"],
        test_metrics["macro_recall"],
        test_metrics["macro_f1"],
    )
    logger.info(
        "Test per-class F1 | %s",
        _format_per_class_metric(test_metrics["per_class"], metric_name="f1"),
    )

    summary = {
        "device": str(device),
        "dataset_dir": str(config.data.dataset_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
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
    raw_participants = load_participant_files(file_paths)
    cleaned_participants = [
        clean_participant_frame(participant, config.data) for participant in raw_participants
    ]
    logger.info("Loaded and cleaned %d %s participants.", len(cleaned_participants), split_name)
    return cleaned_participants


def _prepare_output_directories(config: ProjectConfig) -> None:
    """Create all output directories used by the pipeline."""

    ensure_directory(config.paths.output_dir)
    ensure_directory(config.paths.validation_dir)
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
) -> DataLoader:
    """Construct a DataLoader with deterministic worker seeding."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def _compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
    label_names: Sequence[str],
    output_path: Path,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from train windows only."""

    if not labels:
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

    total = counts.sum().float()
    weights = total / (counts.float() * float(num_classes))

    save_json(
        {
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
    return weights


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
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()

        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_examples == 0:
        raise ValueError("No training samples were available for this epoch.")
    return total_loss / total_examples


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
