"""Shared utilities for logging, reproducibility, metrics, and plotting."""

from __future__ import annotations

import json
import logging
import pickle
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


class PathEncoder(json.JSONEncoder):
    """JSON encoder that understands Path and numpy scalar types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_logger(name: str = "sleep_classifier", log_level: int = logging.INFO) -> logging.Logger:
    """Create a console logger with a stable format."""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(device: torch.device, logger: logging.Logger) -> None:
    """Enable safe performance optimizations for NVIDIA GPUs."""

    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    logger.info("Enabled CUDA optimizations: cuDNN benchmark + TF32.")


def get_device(logger: logging.Logger) -> torch.device:
    """Resolve the best available torch device and log the choice."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info("Using device: %s (%s)", device, device_name)
        return device
    device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


def ensure_parent_dir(path: Path) -> None:
    """Create a file's parent directory if it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(payload: Any, path: Path) -> None:
    """Save JSON with deterministic formatting."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, cls=PathEncoder, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    """Load JSON from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_torch_checkpoint(path: Path, map_location: Any = "cpu") -> Any:
    """Load a checkpoint, preferring weights-only deserialization when supported."""

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=map_location)


def time_block() -> float:
    """Return a perf counter value for elapsed timing."""

    return perf_counter()


def elapsed_seconds(start_time: float) -> float:
    """Compute elapsed seconds from a perf counter start time."""

    return perf_counter() - start_time


def format_seconds(seconds: float) -> str:
    """Format elapsed seconds for logs."""

    minutes, secs = divmod(seconds, 60.0)
    if minutes >= 1:
        return f"{int(minutes)}m {secs:04.1f}s"
    return f"{secs:0.1f}s"


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Iterable[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion matrix figure."""

    ensure_parent_dir(output_path)
    class_names = list(class_names)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = confusion_matrix.max() / 2.0 if confusion_matrix.size else 0.0
    for row_index in range(confusion_matrix.shape[0]):
        for col_index in range(confusion_matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                int(confusion_matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if confusion_matrix[row_index, col_index] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Save a compact learning-curve plot."""

    ensure_parent_dir(output_path)
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        return

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history.get("train_loss", []), label="Train Loss")
    axes[0].plot(epochs, history.get("val_loss", []), label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.get("val_accuracy", []), label="Val Accuracy")
    axes[1].plot(epochs, history.get("val_macro_f1", []), label="Val Macro F1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute aggregate and per-class classification metrics."""

    labels = list(range(len(class_names)))
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    per_class = {
        class_names[idx]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_score[idx]),
            "support": int(support[idx]),
        }
        for idx in labels
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
