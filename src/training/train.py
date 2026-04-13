"""End-to-end training pipeline for DREAMT sleep stage classification."""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
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
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

        self.gamma = gamma
        self.label_smoothing = label_smoothing
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

        nll_loss = -target_log_probs
        if self.label_smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=1)
            ce_loss = (
                (1.0 - self.label_smoothing) * nll_loss
                + self.label_smoothing * smooth_loss
            )
        else:
            ce_loss = nll_loss

        loss = torch.pow(1.0 - target_probs, self.gamma) * ce_loss
        if self.alpha is not None:
            loss = loss * self.alpha.gather(dim=0, index=targets)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class BalancedSoftmaxLoss(nn.Module):
    """Balanced softmax loss that incorporates train-window class priors."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

        self.label_smoothing = label_smoothing
        self.reduction = reduction
        log_class_counts = class_counts.float().clamp_min(1.0).log()
        self.register_buffer("log_class_counts", log_class_counts)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.log_class_counts.unsqueeze(0)
        return F.cross_entropy(
            adjusted_logits,
            targets.long(),
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class SamplerPlan(dict):
    """Lightweight typed container for sampler metadata and reporting."""


def _extract_logits(
    model_outputs: torch.Tensor | Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize model outputs so the training loop always consumes logits."""

    if isinstance(model_outputs, torch.Tensor):
        return model_outputs, {}
    if "logits" not in model_outputs:
        raise ValueError("Structured model outputs must include a 'logits' tensor.")
    return model_outputs["logits"], {
        key: value
        for key, value in model_outputs.items()
        if key != "logits"
    }


def _reduce_loss(raw_loss: torch.Tensor, criterion: nn.Module, batch_size: int) -> torch.Tensor:
    """Reduce arbitrary criterion outputs into a mean-scaled scalar tensor."""

    if raw_loss.ndim > 0:
        return raw_loss.mean()
    if getattr(criterion, "reduction", None) == "sum":
        return raw_loss / max(batch_size, 1)
    return raw_loss


def run_training_pipeline(config: ProjectConfig) -> dict[str, Any]:
    """Run data preparation, training, validation, and final test evaluation."""

    _prepare_output_directories(config)
    logger = build_logger(config.paths.log_path)
    save_json(_checkpoint_config_payload(config), config.paths.config_snapshot_path)
    set_seed(
        config.training.seed,
        deterministic=config.training.deterministic,
        cudnn_benchmark=config.training.cudnn_benchmark,
        allow_tf32=config.training.allow_tf32,
    )
    device = resolve_device()
    amp_enabled = bool(config.training.use_amp and device.type == "cuda")
    amp_dtype = _resolve_amp_dtype(config.training.amp_dtype)
    evaluation_batch_size = config.training.eval_batch_size or config.training.batch_size
    active_num_workers = config.training.num_workers
    active_persistent_workers = (
        config.training.persistent_workers and active_num_workers > 0
    )

    logger.info(
        "Experiment | preset=%s | name=%s | group=%s | run_dir=%s",
        config.experiment.preset_name,
        config.experiment.name,
        config.experiment.group_name,
        config.paths.run_dir,
    )
    logger.info("Using device: %s", device)
    logger.info("Dataset directory: %s", config.data.dataset_dir)
    logger.info(
        (
            "Configured model=%s | loss=%s | sampler=%s | "
            "class_weighting=%s | label_smoothing=%.3f | early_stopping=%s"
        ),
        config.model.model_name,
        config.training.loss_name,
        config.training.resolved_sampler_strategy,
        config.training.class_weighting_mode,
        config.training.label_smoothing,
        config.training.early_stopping_metric,
    )
    logger.info(
        (
            "Runtime | batch_size=%d | eval_batch_size=%d | num_workers=%d | "
            "pin_memory=%s | persistent_workers=%s | prefetch_factor=%d | "
            "amp=%s (%s) | deterministic=%s | cudnn_benchmark=%s | allow_tf32=%s"
        ),
        config.training.batch_size,
        evaluation_batch_size,
        active_num_workers,
        device.type == "cuda",
        active_persistent_workers,
        config.training.prefetch_factor,
        amp_enabled,
        config.training.amp_dtype,
        config.training.deterministic,
        config.training.cudnn_benchmark,
        config.training.allow_tf32,
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

    class_counts = _compute_class_counts(
        labels=train_dataset.labels,
        num_classes=config.model.num_classes,
        label_names=config.data.label_names,
    )
    loss_class_weights = _compute_balancing_weights(
        counts=class_counts,
        mode=config.training.class_weighting_mode,
        beta=config.training.class_balance_beta,
        max_weight_ratio=config.training.loss_max_weight_ratio,
    )
    sampler_class_weights = _compute_balancing_weights(
        counts=class_counts,
        mode=config.training.resolved_sampler_strategy,
        beta=config.training.class_balance_beta,
        max_weight_ratio=config.training.sampler_max_weight_ratio,
    )
    _save_class_weight_report(
        counts=class_counts,
        label_names=config.data.label_names,
        loss_weight_mode=config.training.class_weighting_mode,
        loss_weights=loss_class_weights,
        sampler_weight_mode=config.training.resolved_sampler_strategy,
        sampler_weights=sampler_class_weights,
        beta=config.training.class_balance_beta,
        loss_max_weight_ratio=config.training.loss_max_weight_ratio,
        sampler_max_weight_ratio=config.training.sampler_max_weight_ratio,
        output_path=config.paths.class_weights_path,
    )
    logger.info(
        "Loss class weights | mode=%s | %s",
        config.training.class_weighting_mode,
        _format_named_values(
            _weights_to_named_dict(loss_class_weights, config.data.label_names),
            digits=4,
        ),
    )
    if config.training.loss_name == "balanced_softmax":
        logger.info(
            "Balanced softmax class counts | %s",
            _format_named_values(
                {
                    label_name: int(class_counts[index].item())
                    for index, label_name in enumerate(config.data.label_names)
                },
                digits=0,
            ),
        )
    sampler_plan = _build_weighted_sampler_plan(
        dataset=train_dataset,
        class_counts=class_counts,
        class_weights=sampler_class_weights,
        config=config,
    )
    save_json(
        {key: value for key, value in sampler_plan.items() if key != "sampler"},
        config.paths.sampler_report_path,
    )
    if config.training.sampler_enabled:
        logger.info(
            "Sampler weights | mode=%s | %s",
            config.training.resolved_sampler_strategy,
            _format_named_values(
                _weights_to_named_dict(sampler_class_weights, config.data.label_names),
                digits=4,
            ),
        )
        logger.info(
            "Sampler class probabilities | %s",
            _format_named_values(sampler_plan["class_probabilities"], digits=4),
        )
        logger.info(
            "Sampler expected class counts per epoch | %s | replacement=%s",
            _format_named_values(sampler_plan["expected_class_counts"], digits=1),
            sampler_plan["replacement"],
        )
        logger.info(
            "Sampler participants per class | %s",
            _format_named_values(sampler_plan["participants_per_class"], digits=0),
        )
        if sampler_plan.get("participant_balanced_sampling"):
            logger.info(
                "Sampler mode | participant-balanced within class to reduce subject concentration."
            )

    train_sampler = sampler_plan["sampler"]

    train_loader = _build_dataloader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        num_workers=active_num_workers,
        shuffle=train_sampler is None,
        seed=config.training.seed,
        pin_memory=device.type == "cuda",
        persistent_workers=active_persistent_workers,
        prefetch_factor=config.training.prefetch_factor,
        sampler=train_sampler,
    )
    val_loader = _build_dataloader(
        dataset=val_dataset,
        batch_size=evaluation_batch_size,
        num_workers=active_num_workers,
        shuffle=False,
        seed=config.training.seed,
        pin_memory=device.type == "cuda",
        persistent_workers=active_persistent_workers,
        prefetch_factor=config.training.prefetch_factor,
    )
    test_loader = _build_dataloader(
        dataset=test_dataset,
        batch_size=evaluation_batch_size,
        num_workers=active_num_workers,
        shuffle=False,
        seed=config.training.seed,
        pin_memory=device.type == "cuda",
        persistent_workers=active_persistent_workers,
        prefetch_factor=config.training.prefetch_factor,
    )

    model = build_model(config.model).to(device)
    logger.info(
        "Model parameters | trainable=%d",
        _count_trainable_parameters(model),
    )
    train_criterion, loss_description = _build_training_criterion(
        config=config,
        class_counts=class_counts,
        class_weights=loss_class_weights,
        device=device,
    )
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = _build_scheduler(optimizer=optimizer, config=config)
    scaler = _build_grad_scaler(amp_enabled=amp_enabled)
    logger.info("Training criterion: %s", loss_description)
    if config.training.auxiliary_target_loss_weight > 0.0:
        logger.info(
            "Auxiliary target loss | weight=%.3f",
            config.training.auxiliary_target_loss_weight,
        )
    if config.training.loss_name == "balanced_softmax" and config.training.sampler_enabled:
        logger.warning(
            "Balanced softmax is paired with a sampler; prefer bounded sampling and watch stability."
        )
    if scheduler is not None:
        logger.info("LR scheduler: %s", _describe_scheduler(config))

    history: list[dict[str, Any]] = []
    best_metric_trackers = _initialize_best_metric_trackers()
    monitored_metric_name = config.training.early_stopping_metric
    best_monitored_metric = float("-inf")
    epochs_without_improvement = 0
    best_epoch = 0
    best_epoch_metrics: dict[str, Any] = {}
    collapse_state = _initialize_collapse_state(config.data.label_names)

    for epoch in range(1, config.training.max_epochs + 1):
        train_metrics = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=train_criterion,
            device=device,
            epoch=epoch,
            max_epochs=config.training.max_epochs,
            gradient_clip_norm=config.training.gradient_clip_norm,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            auxiliary_target_loss_weight=config.training.auxiliary_target_loss_weight,
        )
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=eval_criterion,
            device=device,
            label_names=config.data.label_names,
            minority_labels=config.training.minority_labels,
            split_name="validation",
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        collapse_state, collapse_alerts = _update_collapse_state(
            collapse_state=collapse_state,
            val_metrics=val_metrics,
            config=config,
        )
        _update_best_metric_trackers(
            best_metric_trackers=best_metric_trackers,
            epoch=epoch,
            metrics=val_metrics,
        )
        if epoch % config.training.validation_artifact_frequency == 0:
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
        previous_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            _step_scheduler(
                scheduler=scheduler,
                config=config,
                monitored_metric=float(val_metrics[monitored_metric_name]),
            )
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "learning_rate": float(current_lr),
            "train_loss": train_metrics["loss"],
            "train": train_metrics,
            "collapse_alerts": collapse_alerts,
            "validation": {
                key: val_metrics[key]
                for key in (
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
                    "per_class",
                    "target_distribution",
                    "prediction_distribution",
                    "distribution_shift",
                    "confusion_matrix",
                    "num_examples",
                    "split",
                )
            }
            | (
                {"model_diagnostics": val_metrics["model_diagnostics"]}
                if "model_diagnostics" in val_metrics
                else {}
            ),
        }
        history.append(epoch_record)
        save_json(history, config.paths.training_history_path)

        monitored_metric_value = float(val_metrics[monitored_metric_name])

        logger.info(
            (
                "Epoch %d/%d | lr=%.2e | train_loss=%.4f | val_loss=%.4f | "
                "val_acc=%.4f | val_bal_acc=%.4f | val_macro_f1=%.4f | "
                "val_minority_f1=%.4f | val_weighted_f1=%.4f | epoch_sec=%.1f | "
                "train_examples_per_sec=%.1f"
            ),
            epoch,
            config.training.max_epochs,
            current_lr,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["balanced_accuracy"],
            val_metrics["macro_f1"],
            val_metrics["minority_macro_f1"],
            val_metrics["weighted_f1"],
            train_metrics["duration_seconds"],
            train_metrics["examples_per_second"],
        )
        logger.info(
            "Train runtime | data_wait_sec=%.2f | compute_sec=%.2f | workers=%d",
            train_metrics["data_wait_seconds"],
            train_metrics["compute_seconds"],
            train_metrics["num_workers"],
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
        if val_metrics.get("model_diagnostics"):
            diagnostics = val_metrics["model_diagnostics"]
            if "context_gate_mean" in diagnostics:
                logger.info(
                    "Validation context gate | mean=%.4f | std=%.4f",
                    diagnostics["context_gate_mean"],
                    diagnostics["context_gate_std"],
                )
        if device.type == "cuda" and train_metrics["max_cuda_memory_mb"] is not None:
            logger.info(
                "Training GPU peak memory | %.1f MiB (%.1f%% of device total)",
                train_metrics["max_cuda_memory_mb"],
                train_metrics["max_cuda_memory_fraction"] * 100.0,
            )
        if current_lr != previous_lr:
            logger.info(
                "Scheduler adjusted learning rate from %.2e to %.2e.",
                previous_lr,
                current_lr,
            )
        for alert in collapse_alerts:
            logger.warning("Stage collapse alert | %s", alert)

        if monitored_metric_value > (best_monitored_metric + config.training.min_delta):
            best_monitored_metric = monitored_metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_epoch_metrics = dict(val_metrics)
            logger.info(
                "New best checkpoint identified at epoch %d using %s=%.4f.",
                epoch,
                monitored_metric_name,
                monitored_metric_value,
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "Validation %s did not improve for %d epoch(s).",
                monitored_metric_name,
                epochs_without_improvement,
            )

        checkpoint_state = {
            "epoch": epoch,
            "best_monitored_metric": best_monitored_metric,
            "best_epoch_by_early_stopping_metric": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
            "label_names": list(config.data.label_names),
            "feature_columns": list(config.data.feature_columns),
            "normalization_stats": normalization_stats.to_dict(),
            "config": _checkpoint_config_payload(config),
            "train_class_counts": {
                label_name: int(class_counts[index].item())
                for index, label_name in enumerate(config.data.label_names)
            },
            "train_class_weights": {
                label_name: float(loss_class_weights[index].item())
                for index, label_name in enumerate(config.data.label_names)
            },
            "train_sampler_weights": {
                label_name: float(sampler_class_weights[index].item())
                for index, label_name in enumerate(config.data.label_names)
            },
            "best_metric_trackers": best_metric_trackers,
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
        "Loaded best checkpoint from epoch %d with validation %s %.4f.",
        best_epoch,
        monitored_metric_name,
        best_monitored_metric,
    )

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=eval_criterion,
        device=device,
        label_names=config.data.label_names,
        minority_labels=config.training.minority_labels,
        split_name="test",
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
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
            "macro_f1=%.4f | minority_macro_f1=%.4f | weighted_f1=%.4f"
        ),
        test_metrics["loss"],
        test_metrics["accuracy"],
        test_metrics["balanced_accuracy"],
        test_metrics["macro_f1"],
        test_metrics["minority_macro_f1"],
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
    if test_metrics.get("model_diagnostics"):
        diagnostics = test_metrics["model_diagnostics"]
        if "context_gate_mean" in diagnostics:
            logger.info(
                "Test context gate | mean=%.4f | std=%.4f",
                diagnostics["context_gate_mean"],
                diagnostics["context_gate_std"],
            )

    summary = {
        "experiment": {
            "preset_name": config.experiment.preset_name,
            "name": config.experiment.name,
            "group_name": config.experiment.group_name,
            "run_id": config.experiment.run_id,
            "notes": config.experiment.notes,
            "output_dir": str(config.paths.run_dir),
        },
        "device": str(device),
        "dataset_dir": str(config.data.dataset_dir),
        "model": {
            "name": config.model.model_name,
            "conv_dilations": list(config.model.conv_dilations),
            "use_target_indicator_channel": config.model.use_target_indicator_channel,
            "use_relative_position_channel": config.model.use_relative_position_channel,
            "pool_context_region": config.model.pool_context_region,
            "separate_context_regions": config.model.separate_context_regions,
        },
        "training_runtime": {
            "batch_size": config.training.batch_size,
            "eval_batch_size": evaluation_batch_size,
            "num_workers": active_num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": active_persistent_workers,
            "prefetch_factor": config.training.prefetch_factor if active_num_workers > 0 else None,
            "use_amp": amp_enabled,
            "amp_dtype": config.training.amp_dtype if amp_enabled else None,
            "scheduler_name": config.training.scheduler_name,
            "warmup_epochs": config.training.warmup_epochs,
            "deterministic": config.training.deterministic,
            "cudnn_benchmark": config.training.cudnn_benchmark,
            "allow_tf32": config.training.allow_tf32,
            "loss_name": config.training.loss_name,
            "class_weighting_mode": config.training.class_weighting_mode,
            "sampler_strategy": config.training.resolved_sampler_strategy,
            "use_weighted_sampler": config.training.sampler_enabled,
            "participant_balanced_sampling": config.training.participant_balanced_sampling,
            "auxiliary_target_loss_weight": config.training.auxiliary_target_loss_weight,
            "early_stopping_metric": config.training.early_stopping_metric,
        },
        "sequence_definition": _build_windowing_config_summary(config.data),
        "best_epoch": best_epoch,
        "best_monitored_metric_name": monitored_metric_name,
        "best_monitored_metric": best_monitored_metric,
        "best_epoch_validation_metrics": best_epoch_metrics,
        "best_epochs_by_metric": best_metric_trackers,
        "train_class_counts": {
            label_name: int(class_counts[index].item())
            for index, label_name in enumerate(config.data.label_names)
        },
        "train_class_weights": {
            label_name: float(loss_class_weights[index].item())
            for index, label_name in enumerate(config.data.label_names)
        },
        "train_sampler_weights": {
            label_name: float(sampler_class_weights[index].item())
            for index, label_name in enumerate(config.data.label_names)
        },
        "sampler_plan": {
            key: value
            for key, value in sampler_plan.items()
            if key != "sampler"
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
    load_start_time = perf_counter()
    raw_participants = load_participant_files(file_paths, data_config=config.data)
    load_duration_seconds = perf_counter() - load_start_time

    clean_start_time = perf_counter()
    clean_durations_seconds: list[float] = []
    cleaned_participants: list[ParticipantData] = []
    for participant in raw_participants:
        participant_clean_start_time = perf_counter()
        cleaned_participants.append(clean_participant_frame(participant, config.data))
        clean_durations_seconds.append(perf_counter() - participant_clean_start_time)

    clean_duration_seconds = perf_counter() - clean_start_time
    logger.info("Loaded and cleaned %d %s participants.", len(cleaned_participants), split_name)
    logger.info(
        (
            "%s data timings | load_participant_files_sec=%.2f | "
            "clean_participant_frame_total_sec=%.2f | clean_participant_frame_mean_sec=%.2f | "
            "clean_participant_frame_max_sec=%.2f"
        ),
        split_name.capitalize(),
        load_duration_seconds,
        clean_duration_seconds,
        (
            sum(clean_durations_seconds) / len(clean_durations_seconds)
            if clean_durations_seconds
            else 0.0
        ),
        max(clean_durations_seconds, default=0.0),
    )
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

    build_start_time = perf_counter()
    dataset = WindowedSleepDataset(participants, data_config, label_to_index)
    build_duration_seconds = perf_counter() - build_start_time

    label_count_start_time = perf_counter()
    label_counts = dataset.label_counts()
    label_count_duration_seconds = perf_counter() - label_count_start_time
    logger.info(
        "Materialized %d retained %s windows from compact participant arrays.",
        len(dataset),
        split_name,
    )
    logger.info(
        (
            "%s dataset timings | WindowedSleepDataset_sec=%.2f | "
            "label_counts_sec=%.4f"
        ),
        split_name.capitalize(),
        build_duration_seconds,
        label_count_duration_seconds,
    )
    logger.debug("%s label counts snapshot: %s", split_name.capitalize(), label_counts)
    return dataset


def _prepare_output_directories(config: ProjectConfig) -> None:
    """Create all output directories used by the pipeline."""

    ensure_directory(config.paths.output_dir)
    ensure_directory(config.paths.group_dir)
    ensure_directory(config.paths.validation_dir)
    ensure_directory(config.paths.validation_epochs_dir)
    ensure_directory(config.paths.test_dir)
    ensure_directory(config.paths.checkpoints_dir)
    ensure_directory(config.paths.latest_checkpoint_path.parent)
    ensure_directory(config.paths.best_checkpoint_path.parent)


def _build_dataloader(
    dataset: WindowedSleepDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    """Construct a DataLoader with deterministic worker seeding."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        **dataloader_kwargs,
    )


def _compute_class_counts(
    labels: Sequence[int],
    num_classes: int,
    label_names: Sequence[str],
) -> torch.Tensor:
    """Count retained train windows per class and fail fast on missing classes."""

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
    return counts


def _compute_balancing_weights(
    counts: torch.Tensor,
    mode: str,
    beta: float,
    max_weight_ratio: float | None = None,
) -> torch.Tensor:
    """Compute normalized class-balancing weights for loss or sampling."""

    counts = counts.float()
    if mode == "none":
        return torch.ones_like(counts, dtype=torch.float32)

    if mode == "inverse_frequency":
        weights = counts.sum() / (counts * float(counts.numel()))
    elif mode == "capped_inverse_frequency":
        weights = counts.sum() / (counts * float(counts.numel()))
    elif mode == "sqrt_inverse_frequency":
        weights = torch.sqrt(counts.sum() / (counts * float(counts.numel())))
    elif mode == "effective_number":
        effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / effective_num.clamp_min(1e-12)
    else:
        raise ValueError(f"Unsupported balancing mode: {mode}")

    if max_weight_ratio is not None:
        min_weight = weights.min().clamp_min(1e-12)
        weights = weights.clamp(max=min_weight * max_weight_ratio)
    return (weights / weights.mean()).to(dtype=torch.float32)


def _save_class_weight_report(
    counts: torch.Tensor,
    label_names: Sequence[str],
    loss_weight_mode: str,
    loss_weights: torch.Tensor,
    sampler_weight_mode: str,
    sampler_weights: torch.Tensor,
    beta: float,
    loss_max_weight_ratio: float | None,
    sampler_max_weight_ratio: float | None,
    output_path: Path,
) -> None:
    """Persist the train-only class-count and weighting diagnostics."""

    save_json(
        {
            "computed_from": "train_target_windows_only",
            "class_balance_beta": beta,
            "class_weighting_mode": loss_weight_mode,
            "sampler_weighting_mode": sampler_weight_mode,
            "loss_max_weight_ratio": loss_max_weight_ratio,
            "sampler_max_weight_ratio": sampler_max_weight_ratio,
            "counts": {
                label_names[index]: int(counts[index].item())
                for index in range(len(label_names))
            },
            "weights": {
                label_names[index]: float(loss_weights[index].item())
                for index in range(len(label_names))
            },
            "sampler_weights": {
                label_names[index]: float(sampler_weights[index].item())
                for index in range(len(label_names))
            },
            "loss_weighting": {
                "mode": loss_weight_mode,
                "weights": {
                    label_names[index]: float(loss_weights[index].item())
                    for index in range(len(label_names))
                },
            },
            "sampler_weighting": {
                "mode": sampler_weight_mode,
                "weights": {
                    label_names[index]: float(sampler_weights[index].item())
                    for index in range(len(label_names))
                },
            },
        },
        output_path,
    )


def _build_weighted_sampler_plan(
    dataset: WindowedSleepDataset,
    class_counts: torch.Tensor,
    class_weights: torch.Tensor,
    config: ProjectConfig,
) -> SamplerPlan:
    """Build a weighted sampler plus a report of its effective class probabilities."""

    labels = dataset.labels
    if not config.training.sampler_enabled:
        return SamplerPlan(
            sampler=None,
            class_probabilities={
                label_name: (
                    float(class_counts[index].item()) / float(class_counts.sum().item())
                )
                for index, label_name in enumerate(config.data.label_names)
            },
            expected_class_counts={
                label_name: float(class_counts[index].item())
                for index, label_name in enumerate(config.data.label_names)
            },
            participants_per_class={
                label_name: int(
                    np.count_nonzero(dataset.participant_class_counts_matrix[:, index] > 0)
                )
                for index, label_name in enumerate(config.data.label_names)
            },
            participant_balanced_sampling=False,
            replacement=False,
            num_samples=len(labels),
        )

    class_counts_float = class_counts.float()
    class_mass = class_counts_float * class_weights.float()
    class_probabilities = class_mass / class_mass.sum().clamp_min(1e-12)
    class_probabilities = _apply_probability_constraints(
        probabilities=class_probabilities,
        label_names=config.data.label_names,
        floors=config.training.sampler_probability_floors,
        caps=config.training.sampler_probability_caps,
    )

    label_tensor = torch.tensor(labels, dtype=torch.long)
    if config.training.participant_balanced_sampling:
        participant_tensor = torch.tensor(dataset.participant_indices, dtype=torch.long)
        participant_class_counts = torch.tensor(
            dataset.participant_class_counts_matrix,
            dtype=torch.float32,
        )
        participants_per_class = torch.tensor(
            (dataset.participant_class_counts_matrix > 0).sum(axis=0),
            dtype=torch.float32,
        )
        per_sample_group_counts = participant_class_counts[participant_tensor, label_tensor]
        per_sample_weights = (
            class_probabilities[label_tensor]
            / participants_per_class[label_tensor].clamp_min(1.0)
            / per_sample_group_counts.clamp_min(1.0)
        )
    else:
        per_sample_weights = (
            class_probabilities[label_tensor] / class_counts_float[label_tensor].clamp_min(1.0)
        )
    per_sample_weights = (per_sample_weights / per_sample_weights.mean()).double()
    num_samples = max(
        1,
        int(math.ceil(len(labels) * config.training.sampler_num_samples_multiplier)),
    )
    if not config.training.sampler_replacement:
        num_samples = min(num_samples, len(labels))

    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    sampler = WeightedRandomSampler(
        weights=per_sample_weights,
        num_samples=num_samples,
        replacement=config.training.sampler_replacement,
        generator=generator,
    )
    return SamplerPlan(
        sampler=sampler,
        class_probabilities={
            label_name: float(class_probabilities[index].item())
            for index, label_name in enumerate(config.data.label_names)
        },
        expected_class_counts={
            label_name: float(class_probabilities[index].item() * num_samples)
            for index, label_name in enumerate(config.data.label_names)
        },
        participants_per_class={
            label_name: int(
                np.count_nonzero(dataset.participant_class_counts_matrix[:, index] > 0)
            )
            for index, label_name in enumerate(config.data.label_names)
        },
        participant_balanced_sampling=config.training.participant_balanced_sampling,
        replacement=config.training.sampler_replacement,
        num_samples=num_samples,
    )


def _apply_probability_constraints(
    probabilities: torch.Tensor,
    label_names: Sequence[str],
    floors: dict[str, float],
    caps: dict[str, float],
) -> torch.Tensor:
    """Apply optional per-class probability floors and caps while preserving sum=1."""

    adjusted = probabilities.clone().float()
    floor_values = torch.tensor(
        [floors.get(label_name, 0.0) for label_name in label_names],
        dtype=torch.float32,
    )
    cap_values = torch.tensor(
        [caps.get(label_name, 1.0) for label_name in label_names],
        dtype=torch.float32,
    )
    base = adjusted.clone()

    for _ in range(8):
        adjusted = adjusted / adjusted.sum().clamp_min(1e-12)
        below_mask = adjusted < floor_values
        above_mask = adjusted > cap_values
        fixed_mask = below_mask | above_mask
        if not torch.any(fixed_mask):
            return adjusted

        adjusted = adjusted.clone()
        adjusted = torch.where(below_mask, floor_values, adjusted)
        adjusted = torch.where(above_mask, cap_values, adjusted)
        remaining_mass = 1.0 - adjusted[fixed_mask].sum()
        if remaining_mass < -1e-6:
            raise ValueError("Sampler probability floors and caps are inconsistent.")

        free_mask = ~fixed_mask
        if torch.any(free_mask):
            free_base = base[free_mask]
            if float(free_base.sum().item()) <= 0.0:
                adjusted[free_mask] = remaining_mass / float(free_mask.sum().item())
            else:
                adjusted[free_mask] = remaining_mass * (
                    free_base / free_base.sum().clamp_min(1e-12)
                )
        elif remaining_mass > 1e-6:
            raise ValueError("Sampler probability caps leave unused probability mass.")

    return adjusted / adjusted.sum().clamp_min(1e-12)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
) -> ReduceLROnPlateau | CosineAnnealingLR | SequentialLR | None:
    """Construct the configured learning-rate scheduler."""

    if config.training.scheduler_name == "none":
        return None
    if config.training.scheduler_name == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.training.scheduler_t_max,
            eta_min=config.training.scheduler_eta_min,
        )
    if config.training.scheduler_name == "linear_warmup_cosine":
        warmup_epochs = min(config.training.warmup_epochs, max(config.training.max_epochs - 1, 0))
        if warmup_epochs <= 0:
            return CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max(config.training.max_epochs, 1),
                eta_min=config.training.scheduler_eta_min,
            )
        warmup_scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_epochs = max(config.training.max_epochs - warmup_epochs, 1)
        cosine_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cosine_epochs,
            eta_min=config.training.scheduler_eta_min,
        )
        return SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    return ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=config.training.scheduler_min_lr,
    )


def _describe_scheduler(config: ProjectConfig) -> str:
    """Render the configured scheduler in a compact log-friendly form."""

    if config.training.scheduler_name == "cosine_annealing":
        return (
            "cosine_annealing("
            f"T_max={config.training.scheduler_t_max}, "
            f"eta_min={config.training.scheduler_eta_min:.2e})"
        )
    if config.training.scheduler_name == "linear_warmup_cosine":
        return (
            "linear_warmup_cosine("
            f"warmup_epochs={config.training.warmup_epochs}, "
            f"eta_min={config.training.scheduler_eta_min:.2e})"
        )
    return (
        "reduce_on_plateau("
        f"factor={config.training.scheduler_factor:.3f}, "
        f"patience={config.training.scheduler_patience}, "
        f"min_lr={config.training.scheduler_min_lr:.2e})"
    )


def _step_scheduler(
    scheduler: ReduceLROnPlateau | CosineAnnealingLR | SequentialLR,
    config: ProjectConfig,
    monitored_metric: float,
) -> None:
    """Advance the configured scheduler with the appropriate step signature."""

    if config.training.scheduler_name == "reduce_on_plateau":
        scheduler.step(monitored_metric)
        return
    scheduler.step()


def _temper_loss_weights_for_sampler(
    weights: torch.Tensor,
    sampler_enabled: bool,
    power: float,
) -> torch.Tensor:
    """Soften loss-side reweighting when a sampler is already rebalancing batches."""

    if not sampler_enabled:
        return weights.float()
    tempered = torch.pow(weights.float().clamp_min(1e-12), power)
    return tempered / tempered.mean().clamp_min(1e-12)


def _initialize_best_metric_trackers() -> dict[str, dict[str, float | int]]:
    """Create running best-epoch trackers for the key validation metrics."""

    return {
        "macro_f1": {"epoch": 0, "value": float("-inf")},
        "balanced_accuracy": {"epoch": 0, "value": float("-inf")},
        "minority_macro_f1": {"epoch": 0, "value": float("-inf")},
    }


def _update_best_metric_trackers(
    best_metric_trackers: dict[str, dict[str, float | int]],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    """Update tracked best epochs for the main validation metrics."""

    for metric_name in ("macro_f1", "balanced_accuracy", "minority_macro_f1"):
        metric_value = float(metrics[metric_name])
        if metric_value > float(best_metric_trackers[metric_name]["value"]):
            best_metric_trackers[metric_name] = {
                "epoch": epoch,
                "value": metric_value,
            }


def _initialize_collapse_state(
    label_names: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Create per-class streak counters used for collapse alerts."""

    return {
        label_name: {
            "low_recall_streak": 0,
            "near_zero_prediction_streak": 0,
        }
        for label_name in label_names
    }


def _update_collapse_state(
    collapse_state: dict[str, dict[str, int]],
    val_metrics: dict[str, Any],
    config: ProjectConfig,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Track persistent recall collapse or prediction collapse across epochs."""

    alerts: list[str] = []
    for label_name, metrics in val_metrics["per_class"].items():
        low_recall = metrics["recall"] < config.training.collapse_recall_threshold
        near_zero_prediction = (
            metrics["predicted_count"] <= config.training.collapse_predicted_count_threshold
            or metrics["predicted_fraction"]
            <= config.training.collapse_predicted_fraction_threshold
        )
        collapse_state[label_name]["low_recall_streak"] = (
            collapse_state[label_name]["low_recall_streak"] + 1 if low_recall else 0
        )
        collapse_state[label_name]["near_zero_prediction_streak"] = (
            collapse_state[label_name]["near_zero_prediction_streak"] + 1
            if near_zero_prediction
            else 0
        )

        if (
            collapse_state[label_name]["low_recall_streak"]
            >= config.training.collapse_patience_epochs
        ):
            alerts.append(
                f"{label_name} recall stayed below {config.training.collapse_recall_threshold:.3f} "
                f"for {collapse_state[label_name]['low_recall_streak']} epochs"
            )
        if (
            collapse_state[label_name]["near_zero_prediction_streak"]
            >= config.training.collapse_patience_epochs
        ):
            alerts.append(
                f"{label_name} predicted count stayed near zero for "
                f"{collapse_state[label_name]['near_zero_prediction_streak']} epochs"
            )
    return collapse_state, alerts


def _build_grad_scaler(amp_enabled: bool) -> Any:
    """Create a GradScaler compatible with both newer and older PyTorch APIs."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    """Map the configured AMP dtype string to a torch dtype."""

    if amp_dtype == "float16":
        return torch.float16
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {amp_dtype}")


def _build_training_criterion(
    config: ProjectConfig,
    class_counts: torch.Tensor,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[nn.Module, str]:
    """Create the configured training loss."""

    effective_loss_weights = _temper_loss_weights_for_sampler(
        weights=class_weights,
        sampler_enabled=config.training.sampler_enabled,
        power=config.training.loss_weight_power_when_sampling,
    )

    if config.training.loss_name == "cross_entropy":
        return (
            nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing),
            f"cross_entropy(label_smoothing={config.training.label_smoothing:.3f})",
        )

    if config.training.loss_name == "weighted_cross_entropy":
        return (
            nn.CrossEntropyLoss(
                weight=effective_loss_weights.to(device),
                label_smoothing=config.training.label_smoothing,
            ),
            (
                "weighted_cross_entropy("
                f"label_smoothing={config.training.label_smoothing:.3f}, "
                f"class_weights=train_window_{config.training.class_weighting_mode}, "
                f"tempered_for_sampler={config.training.sampler_enabled})"
            ),
        )

    if config.training.loss_name == "balanced_softmax":
        return (
            BalancedSoftmaxLoss(
                class_counts=class_counts.to(device),
                label_smoothing=config.training.label_smoothing,
            ),
            (
                "balanced_softmax("
                f"label_smoothing={config.training.label_smoothing:.3f}, "
                "class_priors=train_window_counts)"
            ),
        )

    focal_alpha: torch.Tensor | None = None
    focal_alpha_description = "none"
    if config.training.focal_alpha is not None:
        focal_alpha = torch.tensor(config.training.focal_alpha, dtype=torch.float32, device=device)
        focal_alpha_description = "config_focal_alpha"
    elif config.training.loss_name == "weighted_focal_loss":
        focal_alpha = effective_loss_weights.to(device)
        focal_alpha_description = (
            f"train_window_{config.training.class_weighting_mode}_tempered"
        )
    elif config.training.loss_name == "class_balanced_focal_loss":
        focal_alpha = _temper_loss_weights_for_sampler(
            weights=_compute_balancing_weights(
                counts=class_counts,
                mode="effective_number",
                beta=config.training.class_balance_beta,
                max_weight_ratio=config.training.loss_max_weight_ratio,
            ),
            sampler_enabled=config.training.sampler_enabled,
            power=config.training.loss_weight_power_when_sampling,
        ).to(device)
        focal_alpha_description = "effective_number_tempered"

    return (
        FocalLoss(
            gamma=config.training.focal_gamma,
            alpha=focal_alpha,
            label_smoothing=config.training.label_smoothing,
            reduction=config.training.focal_reduction,
        ),
        (
            f"{config.training.loss_name}("
            f"gamma={config.training.focal_gamma:.3f}, "
            f"alpha={focal_alpha_description}, "
            f"reduction={config.training.focal_reduction}, "
            f"label_smoothing={config.training.label_smoothing:.3f})"
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
    scaler: Any,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    auxiliary_target_loss_weight: float,
) -> dict[str, float | None]:
    """Run one supervised training epoch."""

    model.train()
    total_loss = 0.0
    total_primary_loss = 0.0
    total_auxiliary_loss = 0.0
    total_examples = 0
    epoch_start_time = perf_counter()
    data_wait_seconds = 0.0
    compute_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)
    batch_fetch_start = perf_counter()
    for batch in progress:
        batch_ready_time = perf_counter()
        data_wait_seconds += batch_ready_time - batch_fetch_start
        compute_start_time = perf_counter()
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        target_start_indices = batch["target_start_idx"].to(device, non_blocking=True)
        target_end_indices = batch["target_end_idx"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            model_outputs = model(
                inputs,
                target_start_indices=target_start_indices,
                target_end_indices=target_end_indices,
            )
            logits, auxiliary_outputs = _extract_logits(model_outputs)
            raw_loss = criterion(logits, targets)
            primary_loss = _reduce_loss(raw_loss, criterion=criterion, batch_size=targets.size(0))
            auxiliary_loss = torch.zeros_like(primary_loss)
            if (
                auxiliary_target_loss_weight > 0.0
                and "auxiliary_logits" in auxiliary_outputs
            ):
                auxiliary_raw_loss = criterion(auxiliary_outputs["auxiliary_logits"], targets)
                auxiliary_loss = _reduce_loss(
                    auxiliary_raw_loss,
                    criterion=criterion,
                    batch_size=targets.size(0),
                )
            loss = primary_loss + (auxiliary_target_loss_weight * auxiliary_loss)

        if amp_enabled:
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

        batch_size = targets.size(0)
        display_loss = loss.detach()
        display_primary_loss = primary_loss.detach()
        display_auxiliary_loss = auxiliary_loss.detach()

        total_loss += float(display_loss.item()) * batch_size
        total_primary_loss += float(display_primary_loss.item()) * batch_size
        total_auxiliary_loss += float(display_auxiliary_loss.item()) * batch_size
        total_examples += batch_size
        compute_seconds += perf_counter() - compute_start_time
        progress.set_postfix(
            loss=f"{display_loss.item():.4f}",
            aux=f"{display_auxiliary_loss.item():.4f}",
        )
        batch_fetch_start = perf_counter()

    if total_examples == 0:
        raise ValueError("No training samples were available for this epoch.")
    duration_seconds = perf_counter() - epoch_start_time
    max_cuda_memory_mb = None
    max_cuda_memory_fraction = None
    if device.type == "cuda":
        max_cuda_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        total_cuda_memory_bytes = torch.cuda.get_device_properties(device).total_memory
        max_cuda_memory_fraction = (
            torch.cuda.max_memory_allocated(device) / max(total_cuda_memory_bytes, 1)
        )
    return {
        "loss": total_loss / total_examples,
        "primary_loss": total_primary_loss / total_examples,
        "auxiliary_loss": total_auxiliary_loss / total_examples,
        "num_examples": float(total_examples),
        "duration_seconds": duration_seconds,
        "examples_per_second": total_examples / max(duration_seconds, 1e-8),
        "data_wait_seconds": data_wait_seconds,
        "compute_seconds": compute_seconds,
        "num_workers": int(dataloader.num_workers),
        "max_cuda_memory_mb": max_cuda_memory_mb,
        "max_cuda_memory_fraction": max_cuda_memory_fraction,
    }


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
                "right_context=%d total_input_length=%d | label_strategy=%s"
            ),
            split_name,
            sequence_definition["target_window_length"],
            sequence_definition["left_context"],
            sequence_definition["right_context"],
            sequence_definition["total_input_length"],
            sequence_definition["target_label_strategy"],
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
    missing_class_participants = dataset.summary.get("participants_missing_each_class", {})
    if missing_class_participants:
        logger.info(
            "%s participants missing each class | %s",
            split_name,
            _format_named_values(missing_class_participants, digits=0),
        )
    if dataset.summary["discard_reasons"]:
        logger.info(
            "%s discard reasons | %s",
            split_name,
            _format_named_values(dataset.summary["discard_reasons"], digits=0),
        )
    if dataset.summary.get("actual_purity_discards_by_class"):
        logger.info(
            "%s purity discards by class | %s",
            split_name,
            _format_named_values(dataset.summary["actual_purity_discards_by_class"], digits=0),
        )
        logger.info(
            "%s purity failure rates by class | %s",
            split_name,
            _format_named_values(dataset.summary["purity_threshold_failure_rates_by_class"], digits=3),
        )
        purity_relative_loss = dataset.summary.get("minority_purity_relative_loss", {})
        if purity_relative_loss:
            logger.info(
                "%s purity pressure vs overall | N1=%.2fx | N3=%.2fx | R=%.2fx",
                split_name,
                purity_relative_loss.get("N1", {}).get("relative_to_overall", 0.0),
                purity_relative_loss.get("N3", {}).get("relative_to_overall", 0.0),
                purity_relative_loss.get("R", {}).get("relative_to_overall", 0.0),
            )
    participant_coverage = dataset.summary.get("participant_class_coverage", {})
    if participant_coverage:
        logger.info(
            (
                "%s participant coverage | "
                "N1_nonzero=%d top5=%.2f | N3_nonzero=%d top5=%.2f | R_nonzero=%d top5=%.2f"
            ),
            split_name,
            participant_coverage.get("N1", {}).get("participants_with_any", 0),
            participant_coverage.get("N1", {}).get("top5_share", 0.0),
            participant_coverage.get("N3", {}).get("participants_with_any", 0),
            participant_coverage.get("N3", {}).get("top5_share", 0.0),
            participant_coverage.get("R", {}).get("participants_with_any", 0),
            participant_coverage.get("R", {}).get("top5_share", 0.0),
        )
    transition_proximity = dataset.summary.get("transition_proximity_by_class", {})
    if transition_proximity:
        for threshold_key in ("256", "512"):
            if threshold_key in transition_proximity:
                logger.info(
                    "%s transition proximity <=%s | %s",
                    split_name,
                    threshold_key,
                    ", ".join(
                        (
                            f"{label}="
                            f"{transition_proximity[threshold_key][label]['fraction']:.3f}"
                        )
                        for label in dataset.label_names
                    ),
                )


def _log_sequence_configuration(
    logger: logging.Logger,
    data_config: DataConfig,
) -> None:
    """Log how each supervised sequence sample is constructed."""

    logger.info(
        (
            "Sequence windowing | target_length=%d | left_context=%d | "
            "right_context=%d | total_input_length=%d | step=%d | use_context_windows=%s | "
            "label_strategy=%s | purity_threshold=%.2f | center_label_span=%d"
        ),
        data_config.target_window_length,
        data_config.effective_left_context,
        data_config.effective_right_context,
        data_config.input_window_length,
        data_config.step,
        data_config.use_context_windows,
        data_config.target_label_strategy,
        data_config.label_purity_threshold,
        data_config.center_label_span,
    )


def _build_windowing_config_summary(
    data_config: DataConfig,
) -> dict[str, int | float | bool | str]:
    """Serialize the sequence windowing configuration saved with artifacts."""

    return {
        "target_window_length": data_config.target_window_length,
        "step": data_config.step,
        "use_context_windows": data_config.use_context_windows,
        "target_label_strategy": data_config.target_label_strategy,
        "center_label_span": data_config.center_label_span,
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
        "transition_distance_thresholds": list(data_config.transition_distance_thresholds),
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


def _weights_to_named_dict(
    weights: torch.Tensor,
    label_names: Sequence[str],
) -> dict[str, float]:
    """Convert a class-weight tensor into a labeled dictionary."""

    return {
        label_name: float(weights[index].item())
        for index, label_name in enumerate(label_names)
    }


def _count_trainable_parameters(model: nn.Module) -> int:
    """Count the trainable parameters of a model."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _checkpoint_config_payload(config: ProjectConfig) -> dict[str, Any]:
    """Serialize the configuration stored with checkpoints."""

    payload = asdict(config)
    payload["data"]["dataset_dir"] = str(config.data.dataset_dir)
    payload["experiment"]["output_root"] = str(config.experiment.output_root)
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
