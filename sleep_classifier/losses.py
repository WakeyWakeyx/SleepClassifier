"""Loss helpers for sleep-stage classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sleep_classifier.config import ExperimentConfig


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight.gather(0, targets)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_loss(config: ExperimentConfig, class_weights: torch.Tensor) -> nn.Module:
    """Create the configured classification loss."""

    if config.loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if config.loss_name == "focal":
        return FocalLoss(gamma=config.focal_gamma, weight=class_weights)
    raise ValueError(f"Unsupported loss '{config.loss_name}'.")
