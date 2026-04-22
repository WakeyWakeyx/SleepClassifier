"""Training pipeline for the modality-aware epoch-sequence sleep classifier."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler

from sleep_classifier.data import (
    EpochSequenceDataset,
    PreparedDataBundle,
    PreparedSplitData,
    SleepWindowDataset,
    prepare_datasets,
)
from sleep_classifier.experiment_config import ExperimentConfig, apply_common_overrides
from sleep_classifier.label_mapping import get_class_names
from sleep_classifier.models import build_model
from sleep_classifier.multitask_losses import build_loss
from sleep_classifier.sequence_evaluation import create_dataloader, evaluate_model, move_batch_to_device
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to the DREAMT data_64Hz directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoints and reports.")
    parser.add_argument("--window-seconds", type=int, default=None, help="Legacy alias for total context window seconds.")
    parser.add_argument("--context-window-seconds", type=int, default=None, help="Optional total context window seconds.")
    parser.add_argument("--context-length-epochs", type=int, default=None, help="Total number of epochs in each context sequence.")
    parser.add_argument("--center-epoch-seconds", type=int, default=None, help="Epoch duration in seconds.")
    parser.add_argument("--stride-seconds", type=int, default=None, help="Legacy alias that sets all split strides.")
    parser.add_argument("--train-stride-seconds", type=int, default=None, help="Training stride in seconds.")
    parser.add_argument("--val-stride-seconds", type=int, default=None, help="Validation stride in seconds.")
    parser.add_argument("--test-stride-seconds", type=int, default=None, help="Test stride in seconds.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--random-seed", type=int, default=None, help="Global random seed.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--gradient-clip-norm", type=float, default=None, help="Gradient clipping norm.")
    parser.add_argument("--dropout", type=float, default=None, help="Model dropout probability.")
    parser.add_argument("--normalization-mode", choices=["global_train", "participant"], default=None)
    parser.add_argument("--loss-name", choices=["weighted_cross_entropy", "focal", "class_balanced_focal"], default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--class-balanced-beta", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--final-loss-weight", type=float, default=None)
    parser.add_argument("--raw-loss-weight", type=float, default=None)
    parser.add_argument("--lr-scheduler-patience", type=int, default=None)
    parser.add_argument("--lr-scheduler-factor", type=float, default=None)
    parser.add_argument("--lr-scheduler-metric", choices=["macro_f1", "loss"], default=None)
    parser.add_argument("--sequence-model-name", choices=["bilstm", "transformer", "tcn"], default=None)
    parser.add_argument("--sequence-hidden-size", type=int, default=None)
    parser.add_argument("--sequence-layers", type=int, default=None)
    parser.add_argument("--transformer-heads", type=int, default=None)
    parser.add_argument("--epoch-embedding-dim", type=int, default=None)
    parser.add_argument("--branch-embedding-dim", type=int, default=None)
    parser.add_argument("--model-base-channels", type=int, default=None)
    parser.add_argument("--smoothing-mode", choices=["none", "median", "viterbi"], default=None)
    parser.add_argument("--median-filter-size", type=int, default=None)
    parser.add_argument("--time-mask-prob", type=float, default=None)
    parser.add_argument("--time-mask-max-fraction", type=float, default=None)
    parser.add_argument("--channel-dropout-prob", type=float, default=None)
    parser.add_argument("--min-window-complete-fraction", type=float, default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume training from a saved checkpoint.")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt", help="Filename for the best checkpoint.")

    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    mixed_precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")

    sampler_group = parser.add_mutually_exclusive_group()
    sampler_group.add_argument("--use-weighted-sampler", dest="use_weighted_sampler", action="store_true")
    sampler_group.add_argument("--no-weighted-sampler", dest="use_weighted_sampler", action="store_false")

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use-cache", dest="use_cache", action="store_true")
    cache_group.add_argument("--no-use-cache", dest="use_cache", action="store_false")

    rebuild_group = parser.add_mutually_exclusive_group()
    rebuild_group.add_argument("--rebuild-cache", dest="rebuild_cache", action="store_true")
    rebuild_group.add_argument("--no-rebuild-cache", dest="rebuild_cache", action="store_false")

    derived_group = parser.add_mutually_exclusive_group()
    derived_group.add_argument("--use-derived-features", dest="use_derived_features", action="store_true")
    derived_group.add_argument("--no-derived-features", dest="use_derived_features", action="store_false")

    branch_group = parser.add_mutually_exclusive_group()
    branch_group.add_argument("--use-multi-branch", dest="use_multi_branch", action="store_true")
    branch_group.add_argument("--single-branch", dest="use_multi_branch", action="store_false")

    multitask_group = parser.add_mutually_exclusive_group()
    multitask_group.add_argument("--use-multitask-heads", dest="use_multitask_heads", action="store_true")
    multitask_group.add_argument("--three-class-only", dest="use_multitask_heads", action="store_false")

    parser.set_defaults(
        mixed_precision=None,
        use_weighted_sampler=None,
        use_cache=None,
        rebuild_cache=None,
        use_derived_features=None,
        use_multi_branch=None,
        use_multitask_heads=None,
    )
    return parser


def build_train_sampler(
    dataset: EpochSequenceDataset | SleepWindowDataset,
    class_weights: list[float],
) -> WeightedRandomSampler:
    """Create a weighted random sampler for the training split."""

    labels = dataset.labels.cpu().numpy()
    sample_weights = np.asarray(class_weights, dtype=np.float64)[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def apply_channel_dropout(inputs: torch.Tensor, drop_probability: float) -> torch.Tensor:
    """Randomly zero whole feature channels across all epochs and times for a sample."""

    if drop_probability <= 0.0:
        return inputs
    outputs = inputs.clone()
    _, _, num_channels, _ = outputs.shape
    for batch_index in range(outputs.size(0)):
        channel_mask = torch.rand(num_channels, device=outputs.device) < drop_probability
        if bool(channel_mask.all()):
            keep_index = int(torch.randint(0, num_channels, (1,), device=outputs.device).item())
            channel_mask[keep_index] = False
        outputs[batch_index, :, channel_mask, :] = 0.0
    return outputs


def apply_time_masking(
    inputs: torch.Tensor,
    mask_probability: float,
    max_mask_fraction: float,
) -> torch.Tensor:
    """Randomly zero contiguous temporal spans inside individual epochs."""

    if mask_probability <= 0.0 or max_mask_fraction <= 0.0:
        return inputs
    outputs = inputs.clone()
    epoch_length = outputs.size(-1)
    max_mask_length = max(1, int(epoch_length * max_mask_fraction))
    for batch_index in range(outputs.size(0)):
        for epoch_index in range(outputs.size(1)):
            if random.random() >= mask_probability:
                continue
            mask_length = random.randint(1, max_mask_length)
            start_index = random.randint(0, max(0, epoch_length - mask_length))
            outputs[batch_index, epoch_index, :, start_index : start_index + mask_length] = 0.0
    return outputs


def train_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_mixed_precision: bool,
    gradient_clip_norm: float,
    config: ExperimentConfig,
) -> dict[str, float]:
    """Run one epoch of optimization."""

    model.train()
    running_total_loss = 0.0
    running_final_loss = 0.0
    running_raw_loss = 0.0
    total_examples = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        batch["features"] = apply_channel_dropout(batch["features"], config.channel_dropout_prob)
        batch["features"] = apply_time_masking(
            batch["features"],
            mask_probability=config.time_mask_prob,
            max_mask_fraction=config.time_mask_max_fraction,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
            outputs = model(batch["features"])
            loss, loss_components = criterion(outputs, batch)

        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size_value = batch["final_target"].size(0)
        running_total_loss += float(loss_components["total_loss"]) * batch_size_value
        running_final_loss += float(loss_components["final_loss"]) * batch_size_value
        running_raw_loss += float(loss_components["raw_loss"]) * batch_size_value
        total_examples += int(batch_size_value)

    if total_examples == 0:
        raise ValueError("Training loader produced zero examples.")
    return {
        "loss": running_total_loss / total_examples,
        "final_loss": running_final_loss / total_examples,
        "raw_loss": running_raw_loss / total_examples,
    }


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
        "final_class_counts": bundle.final_class_counts,
        "raw_class_counts": bundle.raw_class_counts,
        "transition_matrix": bundle.transition_matrix,
        "class_priors": bundle.class_priors,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "class_names": get_class_names(level="final"),
        "raw_class_names": get_class_names(level="raw"),
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


def _log_metric_bundle(logger: Any, label: str, metrics: dict[str, Any]) -> None:
    logger.info("%s accuracy: %.4f", label, metrics["accuracy"])
    logger.info("%s balanced accuracy: %.4f", label, metrics["balanced_accuracy"])
    logger.info("%s macro F1: %.4f", label, metrics["macro_f1"])
    for class_name, class_metrics in metrics["per_class"].items():
        logger.info(
            "%s %s | precision=%.4f recall=%.4f f1=%.4f support=%d",
            label,
            class_name,
            class_metrics["precision"],
            class_metrics["recall"],
            class_metrics["f1"],
            class_metrics["support"],
        )
    logger.info("%s confusion matrix:\n%s", label, np.asarray(metrics["confusion_matrix"], dtype=np.int64))


def run_training(
    config: ExperimentConfig,
    resume_from: Path | None = None,
    best_checkpoint_filename: str = "best_model.pt",
) -> None:
    """Train the model and automatically evaluate the best checkpoint on the test data."""

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

    class_names = get_class_names(level="final")
    model = build_model(
        config=config,
        final_num_classes=len(class_names),
        raw_num_classes=len(get_class_names(level="raw")),
    ).to(device)
    criterion = build_loss(
        config=config,
        final_class_counts=bundle.final_class_counts,
        raw_class_counts=bundle.raw_class_counts,
        device=device,
    )
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
        "train_final_loss": [],
        "train_raw_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
        "val_smoothed_macro_f1": [],
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
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_mixed_precision=use_mixed_precision,
            gradient_clip_norm=config.gradient_clip_norm,
            config=config,
        )
        val_result = evaluate_model(
            model=model,
            prepared_split=bundle.prepared_splits["val"],
            criterion=criterion,
            config=config,
            device=device,
            use_mixed_precision=use_mixed_precision,
            class_names=class_names,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            transition_matrix=bundle.transition_matrix,
            class_priors=bundle.class_priors,
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
        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_final_loss"].append(float(train_metrics["final_loss"]))
        history["train_raw_loss"].append(float(train_metrics["raw_loss"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_balanced_accuracy"].append(float(val_metrics["balanced_accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["val_smoothed_macro_f1"].append(
            float(val_result.smoothed_metrics["macro_f1"]) if val_result.smoothed_metrics is not None else float("nan")
        )
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
            "Epoch %d/%d | train_loss=%.4f | train_final=%.4f | train_raw=%.4f | val_loss=%.4f | val_acc=%.4f | val_macro_f1=%.4f | val_smoothed_f1=%s | lr=%.6f | time=%s",
            epoch + 1,
            config.epochs,
            train_metrics["loss"],
            train_metrics["final_loss"],
            train_metrics["raw_loss"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            (
                f"{val_result.smoothed_metrics['macro_f1']:.4f}"
                if val_result.smoothed_metrics is not None
                else "n/a"
            ),
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
        prepared_split=bundle.prepared_splits["test"],
        criterion=criterion,
        config=config,
        device=device,
        use_mixed_precision=use_mixed_precision,
        class_names=class_names,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        transition_matrix=bundle.transition_matrix,
        class_priors=bundle.class_priors,
        split_name="test",
        metrics_output_path=config.reports_dir / "test_metrics.json",
        report_output_path=config.reports_dir / "test_classification_report.json",
        report_text_output_path=config.reports_dir / "test_classification_report.txt",
        confusion_output_path=config.plots_dir / "test_confusion_matrix.png",
    )

    test_metrics = test_result.metrics
    logger.info("Training complete. Best validation macro F1: %.4f", best_val_macro_f1)
    _log_metric_bundle(logger, "Raw test", test_metrics)
    if test_result.smoothed_metrics is not None:
        _log_metric_bundle(logger, "Smoothed test", test_result.smoothed_metrics)
    logger.info("Saved test metrics to %s.", config.reports_dir / "test_metrics.json")

    print("Training complete.")
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
    print(f"Raw test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Raw test macro F1: {test_metrics['macro_f1']:.4f}")
    if test_result.smoothed_metrics is not None:
        print(f"Smoothed test accuracy: {test_result.smoothed_metrics['accuracy']:.4f}")
        print(f"Smoothed test macro F1: {test_result.smoothed_metrics['macro_f1']:.4f}")
    else:
        print("Smoothed test accuracy: n/a")
        print("Smoothed test macro F1: n/a")
    print("Per-class precision/recall/F1:")
    for class_name, class_metrics in test_metrics["per_class"].items():
        print(
            f"  {class_name}: precision={class_metrics['precision']:.4f} "
            f"recall={class_metrics['recall']:.4f} f1={class_metrics['f1']:.4f}"
        )
    print("Confusion matrix:")
    print(np.asarray(test_metrics["confusion_matrix"], dtype=np.int64))
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
