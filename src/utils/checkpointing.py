"""Checkpoint save/load helpers for PyTorch models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .utilities import ensure_directory


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    """Persist a checkpoint dictionary to disk."""

    ensure_directory(path.parent)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into the provided model and optional optimizer."""

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
