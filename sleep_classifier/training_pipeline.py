"""Training pipeline for the context-aware sleep-stage classifier."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler

from sleep_classifier.config import ExperimentConfig, apply_common_overrides
from sleep_classifier.data import PreparedDataBundle, SleepWindowDataset, prepare_datasets
from sleep_classifier.evaluation import create_dataloader, evaluate_model
from sleep_classifier.label_mapping import get_class_names
from sleep_classifier.losses import build_loss
from sleep_classifier.models import build_model
from sleep_classifier.utils import (
    build_logger,
    configure_torch_runtime,
    elapsed_seconds,
    format_seconds,
    get_device,
    plot_training_history,
    save_json,
    set_global_seed,
    time_block,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI."""

    parser = argparse.ArgumentParser(description="Train a context-aware DREAMT sleep-stage classifier.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to the DREAMT data_64Hz directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoints and reports.")
    parser.add_argument("--window-seconds", type=int, default=None, help="Legacy alias for total context window length in seconds.")
    parser.add_argument("--context-window-seconds", type=int, default=None, help="Total context window length in seconds.")
    parser.add_argument("--center-epoch-seconds", type=int, default=None, help="Center epoch duration in seconds.")
    parser.add_argument("--stride-seconds", type=int, default=None, help="Legacy alias that sets all split strides.")
    parser.add_argument("--train-stride-seconds", type=int, default=None, help="Training stride in seconds.")
    parser.add_argument("--val-stride-seconds", type=int, default=None, help="Validation stride in seconds.")
    parser.add_argument("--test-stride-seconds", type=int, default=None, help="Test stride in seconds.")
    parser.add_argument("--min-window-complete-fraction", type=float, default=None, help="Minimum fraction of rows in a context window that originally had all wearable features present before fill.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to 0 for Windows safety.")
    parser.add_argument("--random-seed", type=int, default=None, help="Global random seed.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Early stopping patience on validation macro F1.")
    parser.add_argument("--gradient-clip-norm", type=float, default=None, help="Gradient clipping norm.")
    parser.add_argument("--dropout", type=float, default=None, help="Model dropout probability.")
    parser.add_argument("--normalization-mode", choices=["global_train", "participant"], default=None, help="Feature normalization mode.")
    parser.add_argument("--loss-name", choices=["weighted_cross_entropy", "focal"], default=None, help="Classification loss.")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Focal loss gamma.")
    parser.add_argument("--lr-scheduler-patience", type=int, default=None, help="ReduceLROnPlateau patience.")
    parser.add_argument("--lr-scheduler-factor", type=float, default=None, help="ReduceLROnPlateau factor.")
    parser.add_argument("--lr-scheduler-metric", choices=["macro_f1", "loss"], default=None, help="Validation metric monitored by the LR scheduler.")
    parser.add_argument("--limit-files", type=int, default=None, help="Use only the first N CSV files after deterministic sorting.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume training from a saved checkpoint.")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt", help="Filename for the best checkpoint.")

    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", help="Enable CUDA mixed precision.")
    mixed_precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false", help="Disable CUDA mixed precision.")

    sampler_group = parser.add_mutually_exclusive_group()
    sampler_group.add_argument("--use-weighted-sampler", dest="use_weighted_sampler", action="store_true", help="Enable weighted random sampling for the training split.")
    sampler_group.add_argument("--no-weighted-sampler", dest="use_weighted_sampler", action="store_false", help="Disable weighted random sampling.")

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use-cache", dest="use_cache", action="store_true", help="Reuse cached cleaned participants and window metadata.")
    cache_group.add_argument("--no-use-cache", dest="use_cache", action="store_false", help="Disable cache reuse for this run.")

    rebuild_group = parser.add_mutually_exclusive_group()
    rebuild_group.add_argument("--rebuild-cache", dest="rebuild_cache", action="store_true", help="Rebuild cached cleaned participants and window metadata.")
    rebuild_group.add_argument("--no-rebuild-cache", dest="rebuild_cache", action="store_false", help="Keep any existing cache if compatible.")

    derived_group = parser.add_mutually_exclusive_group()
    derived_group.add_argument("--use-derived-features", dest="use_derived_features", action="store_true", help="Enable derived wearable features.")
    derived_group.add_argument("--no-derived-features", dest="use_derived_features", action="store_false", help="Disable derived wearable features.")

    parser.set_defaults(
        mixed_precision=None,
        use_weighted_sampler=None,
        use_cache=None,
        rebuild_cache=None,
        use_derived_features=None,
    )
    return parser


def build_train_sampler(dataset: SleepWindowDataset, class_weights: list[float]) -> WeightedRandomSampler:
    """Create a weighted random sampler for the training split."""

    labels = dataset.labels.cpu().numpy()
    sample_weights = np.asarray(class_weights, dtype=np.float64)[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_mixed_precision: bool,
    gradient_clip_norm: float,
) -> float:
    """Run one epoch of optimization."""

    model.train()
    running_loss = 0.0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
            logits = model(inputs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size_value = labels.size(0)
        running_loss += float(loss.item()) * batch_size_value
        total_examples += int(batch_size_value)

    if total_examples == 0:
        raise ValueError("Training loader produced zero examples.")
    return running_loss / total_examples


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    config: ExperimentConfig,
    bundle: PreparedDataBundle,
    history: dict[str, list[float]],
    best_val_macro_f1: float,
) -> None:
    """Persist training state for resume and evaluation."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.to_dict(),
        "normalization_stats": bundle.normalization_stats,
        "class_weights": bundle.class_weights,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "class_names": get_class_names(),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, dict[str, list[float]]]:
    """Restore training state from a checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_val_macro_f1", -math.inf)),
        dict(checkpoint.get("history", {})),
    )


def initialize_config(args: argparse.Namespace, logger: Any) -> ExperimentConfig:
    """Load config defaults or checkpoint config, then apply CLI overrides."""

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        config = ExperimentConfig.from_dict(checkpoint["config"])
        logger.info("Loaded base config from checkpoint %s.", args.resume_from)
    else:
        config = ExperimentConfig()

    config = apply_common_overrides(config, args)
    config.validate()
    config.ensure_output_dirs()
    save_json(config.to_dict(), config.config_path)
    return config


def run_training(
    config: ExperimentConfig,
    resume_from: Path | None = None,
    best_checkpoint_filename: str = "best_model.pt",
) -> None:
    """Train the model and automatically evaluate the best checkpoint on test data."""

    logger = build_logger()
    set_global_seed(config.random_seed)
    device = get_device(logger)
    configure_torch_runtime(device, logger)

    if config.limit_files is not None:
        logger.warning(
            "File limit enabled: using only the first %d file(s) after deterministic sorting.",
            config.limit_files,
        )

    bundle = prepare_datasets(config=config, logger=logger, requested_splits=("train", "val", "test"))
    train_dataset = bundle.prepared_splits["train"].dataset
    train_sampler = (
        build_train_sampler(train_dataset, bundle.class_weights)
        if config.use_weighted_sampler
        else None
    )
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )

    class_names = get_class_names()
    model = build_model(config=config, num_classes=len(class_names)).to(device)
    class_weights_tensor = torch.tensor(bundle.class_weights, dtype=torch.float32, device=device)
    criterion = build_loss(config=config, class_weights=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max" if config.lr_scheduler_metric == "macro_f1" else "min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )
    use_mixed_precision = device.type == "cuda" and config.mixed_precision
    scaler = torch.amp.GradScaler(device.type, enabled=use_mixed_precision)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
        "epoch_seconds": [],
        "learning_rate": [],
    }
    start_epoch = 0
    best_val_macro_f1 = -math.inf
    epochs_without_improvement = 0

    best_checkpoint_path = config.checkpoints_dir / best_checkpoint_filename
    last_checkpoint_path = config.checkpoints_dir / "last_model.pt"

    if resume_from is not None:
        start_epoch, best_val_macro_f1, history = load_checkpoint_state(
            checkpoint_path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        logger.info("Resumed training from epoch %d.", start_epoch)

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time_block()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_mixed_precision=use_mixed_precision,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_result = evaluate_model(
            model=model,
            dataset=bundle.prepared_splits["val"].dataset,
            criterion=criterion,
            device=device,
            use_mixed_precision=use_mixed_precision,
            class_names=class_names,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            split_name="validation",
        )
        val_metrics = val_result.metrics
        scheduler_value = float(
            val_metrics["macro_f1"]
            if config.lr_scheduler_metric == "macro_f1"
            else val_metrics["loss"]
        )
        scheduler.step(scheduler_value)

        epoch_duration = elapsed_seconds(epoch_start)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_balanced_accuracy"].append(float(val_metrics["balanced_accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["epoch_seconds"].append(float(epoch_duration))
        history["learning_rate"].append(current_lr)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                bundle=bundle,
                history=history,
                best_val_macro_f1=best_val_macro_f1,
            )
            logger.info("Saved new best checkpoint to %s.", best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            bundle=bundle,
            history=history,
            best_val_macro_f1=best_val_macro_f1,
        )
        save_json(history, config.history_path)
        plot_training_history(history, config.plots_dir / "training_history.png")

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_bal_acc=%.4f | val_macro_f1=%.4f | lr=%.6f | time=%s",
            epoch + 1,
            config.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["balanced_accuracy"],
            val_metrics["macro_f1"],
            current_lr,
            format_seconds(epoch_duration),
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            logger.info(
                "Early stopping triggered after %d epoch(s) without macro F1 improvement.",
                epochs_without_improvement,
            )
            break

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    logger.info("Reloaded best checkpoint from %s for final test evaluation.", best_checkpoint_path)

    test_result = evaluate_model(
        model=model,
        dataset=bundle.prepared_splits["test"].dataset,
        criterion=criterion,
        device=device,
        use_mixed_precision=use_mixed_precision,
        class_names=class_names,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split_name="test",
        metrics_output_path=config.reports_dir / "test_metrics.json",
        report_output_path=config.reports_dir / "test_classification_report.json",
        report_text_output_path=config.reports_dir / "test_classification_report.txt",
        confusion_output_path=config.plots_dir / "test_confusion_matrix.png",
    )

    test_metrics = test_result.metrics
    logger.info("Training complete. Best validation macro F1: %.4f", best_val_macro_f1)
    logger.info("Test accuracy: %.4f", test_metrics["accuracy"])
    logger.info("Test balanced accuracy: %.4f", test_metrics["balanced_accuracy"])
    logger.info("Test macro F1: %.4f", test_metrics["macro_f1"])
    logger.info("Saved test metrics to %s.", config.reports_dir / "test_metrics.json")
    logger.info("Saved test classification report to %s.", config.reports_dir / "test_classification_report.json")
    logger.info("Saved test confusion matrix to %s.", config.plots_dir / "test_confusion_matrix.png")

    print("Training complete.")
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Artifacts saved to: {config.output_dir}")


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()
    logger = build_logger()
    config = initialize_config(args, logger)
    run_training(
        config=config,
        resume_from=args.resume_from,
        best_checkpoint_filename=args.checkpoint_name,
    )
