"""Shared utility functions for reproducibility, logging, and structured I/O."""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    """Convert common non-JSON objects into serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_json(payload: Any, path: Path) -> None:
    """Write structured JSON to disk with stable formatting."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def save_csv_rows(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    fieldnames: Sequence[str],
) -> None:
    """Write a list of dictionaries to CSV using a fixed column order."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> Any:
    """Load JSON content from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(
    seed: int,
    *,
    deterministic: bool = False,
    cudnn_benchmark: bool = True,
    allow_tf32: bool = True,
) -> None:
    """Set random seeds and configure PyTorch runtime behavior."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark if not deterministic else False


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader worker processes deterministically."""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device() -> torch.device:
    """Pick a CUDA device when available, otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_logger(log_path: Path) -> logging.Logger:
    """Create a console and file logger for the training run."""

    ensure_directory(log_path.parent)
    logger = logging.getLogger("sleep_classifier")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
