"""Train a ternary DREAMT sleep-stage classifier from 64 Hz wearable data only."""

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
from torch.utils.data import DataLoader

from sleep_classifier.config import ExperimentConfig, apply_common_overrides
from sleep_classifier.data import PreparedDataBundle, prepare_datasets
from sleep_classifier.label_mapping import get_class_names
from sleep_classifier.models import SleepStageCNN1D
from sleep_classifier.utils import (
    build_logger,
    compute_classification_metrics,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to the DREAMT data_64Hz directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoints and reports.")
    parser.add_argument("--window-seconds", type=int, default=None, help="Window length in seconds.")
    parser.add_argument("--stride-seconds", type=int, default=None, help="Stride length in seconds.")
    parser.add_argument("--min-window-complete-fraction", type=float, default=None, help="Minimum fraction of rows in a window that originally had all features present before fill.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to 0 for Windows safety.")
    parser.add_argument("--random-seed", type=int, default=None, help="Global random seed.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Early stopping patience on validation macro F1.")
    parser.add_argument("--gradient-clip-norm", type=float, default=None, help="Gradient clipping norm.")
    parser.add_argument("--dropout", type=float, default=None, help="Model dropout probability.")
    parser.add_argument("--limit-files", type=int, default=None, help="Use only the first N CSV files after deterministic sorting.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume training from a saved checkpoint.")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt", help="Filename for the best checkpoint.")
    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", help="Enable CUDA mixed precision.")
    mixed_precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false", help="Disable CUDA mixed precision.")
    parser.set_defaults(mixed_precision=None)
    return parser


def create_dataloader(
    dataset: torch.utils.data.Dataset[Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def evaluate_loader(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_mixed_precision: bool,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    total_examples = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                logits = model(inputs)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            total_examples += int(batch_size)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())

    if total_examples == 0:
        raise ValueError("Evaluation loader produced zero examples.")

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)
    metrics["loss"] = running_loss / total_examples
    return metrics


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_mixed_precision: bool,
    gradient_clip_norm: float,
) -> float:
    model.train()
    running_loss = 0.0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            logits = model(inputs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        total_examples += int(batch_size)

    if total_examples == 0:
        raise ValueError("Training loader produced zero examples.")
    return running_loss / total_examples


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: ExperimentConfig,
    bundle: PreparedDataBundle,
    history: dict[str, list[float]],
    best_val_macro_f1: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, dict[str, list[float]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_val_macro_f1", -math.inf)),
        dict(checkpoint.get("history", {})),
    )


def initialize_config(args: argparse.Namespace, logger: Any) -> ExperimentConfig:
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
    logger = build_logger()
    set_global_seed(config.random_seed)
    device = get_device(logger)
    configure_torch_runtime(device, logger)

    if config.limit_files is not None:
        logger.warning(
            "File limit enabled: using only the first %d file(s) after deterministic sorting.",
            config.limit_files,
        )

    bundle = prepare_datasets(config=config, logger=logger, requested_splits=("train", "val"))
    train_loader = create_dataloader(
        dataset=bundle.prepared_splits["train"].dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        device=device,
    )
    val_loader = create_dataloader(
        dataset=bundle.prepared_splits["val"].dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device,
    )

    model = SleepStageCNN1D(
        input_channels=len(config.feature_columns),
        num_classes=len(get_class_names()),
        dropout=config.dropout,
    ).to(device)

    class_weights_tensor = torch.tensor(bundle.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    use_mixed_precision = device.type == "cuda" and config.mixed_precision
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

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
            scaler=scaler,
            device=device,
        )
        logger.info("Resumed training from epoch %d.", start_epoch)

    class_names = get_class_names()
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time_block()
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
        val_metrics = evaluate_loader(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            use_mixed_precision=use_mixed_precision,
            class_names=class_names,
        )

        epoch_duration = elapsed_seconds(epoch_start)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_balanced_accuracy"].append(float(val_metrics["balanced_accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["epoch_seconds"].append(float(epoch_duration))
        history["learning_rate"].append(float(current_lr))

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

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            bundle=bundle,
            history=history,
            best_val_macro_f1=best_val_macro_f1,
        )
        save_json(history, config.history_path)
        plot_training_history(history, config.plots_dir / "training_history.png")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                bundle=bundle,
                history=history,
                best_val_macro_f1=best_val_macro_f1,
            )
            logger.info("Saved new best checkpoint to %s.", best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            logger.info(
                "Early stopping triggered after %d epoch(s) without macro F1 improvement.",
                epochs_without_improvement,
            )
            break

    logger.info("Training complete. Best validation macro F1: %.4f", best_val_macro_f1)


from sleep_classifier.epoch_training_pipeline import main as pipeline_main, run_training as pipeline_run_training

main = pipeline_main
run_training = pipeline_run_training


if __name__ == "__main__":
    pipeline_main()
