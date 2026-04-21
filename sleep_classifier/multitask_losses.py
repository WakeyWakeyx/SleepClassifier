"""Multitask loss helpers for sleep-stage classification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sleep_classifier.experiment_config import ExperimentConfig


def _inverse_frequency_weights(class_counts: list[int]) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts.numel() * counts)
    return weights


def _class_balanced_weights(class_counts: list[int], beta: float) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    effective_num = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
    weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-8)
    weights = weights / weights.sum() * counts.numel()
    return weights


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        num_classes = logits.size(1)
        if self.label_smoothing > 0.0 and num_classes > 1:
            off_value = self.label_smoothing / (num_classes - 1)
            target_distribution = torch.full_like(log_probs, off_value)
            target_distribution.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            target_distribution = torch.zeros_like(log_probs)
            target_distribution.scatter_(1, targets.unsqueeze(1), 1.0)

        ce_loss = -(target_distribution * log_probs).sum(dim=1)
        pt = torch.clamp((target_distribution * probs).sum(dim=1), min=1e-6, max=1.0)
        loss = torch.pow(1.0 - pt, self.gamma) * ce_loss
        if self.weight is not None:
            loss = loss * self.weight.gather(0, targets)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """Weighted combination of final-target and raw-target losses."""

    def __init__(
        self,
        final_criterion: nn.Module,
        raw_criterion: nn.Module | None,
        final_weight: float,
        raw_weight: float,
    ) -> None:
        super().__init__()
        self.final_criterion = final_criterion
        self.raw_criterion = raw_criterion
        self.final_weight = final_weight
        self.raw_weight = raw_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        final_logits = outputs["final_logits"]
        if final_logits is None:
            raise ValueError("Model outputs are missing final_logits.")

        final_loss = self.final_criterion(final_logits, batch["final_target"])
        total_loss = self.final_weight * final_loss
        raw_loss_value = torch.zeros_like(final_loss)

        raw_logits = outputs.get("raw_logits")
        if self.raw_criterion is not None and raw_logits is not None:
            raw_loss_value = self.raw_criterion(raw_logits, batch["raw_target"])
            total_loss = total_loss + self.raw_weight * raw_loss_value

        return total_loss, {
            "total_loss": float(total_loss.detach().item()),
            "final_loss": float(final_loss.detach().item()),
            "raw_loss": float(raw_loss_value.detach().item()),
        }


def build_loss(
    config: ExperimentConfig,
    final_class_counts: list[int],
    raw_class_counts: list[int],
    device: torch.device,
) -> MultiTaskLoss:
    """Create the configured multitask loss."""

    final_inverse_weights = _inverse_frequency_weights(final_class_counts).to(device)
    raw_inverse_weights = _inverse_frequency_weights(raw_class_counts).to(device)

    if config.loss_name == "weighted_cross_entropy":
        final_criterion: nn.Module = nn.CrossEntropyLoss(
            weight=final_inverse_weights,
            label_smoothing=config.label_smoothing,
        )
    elif config.loss_name == "focal":
        final_criterion = FocalLoss(
            gamma=config.focal_gamma,
            weight=final_inverse_weights,
            label_smoothing=config.label_smoothing,
        )
    elif config.loss_name == "class_balanced_focal":
        final_balanced_weights = _class_balanced_weights(
            final_class_counts,
            beta=config.class_balanced_beta,
        ).to(device)
        final_criterion = FocalLoss(
            gamma=config.focal_gamma,
            weight=final_balanced_weights,
            label_smoothing=config.label_smoothing,
        )
    else:  # pragma: no cover - validated earlier
        raise ValueError(f"Unsupported loss '{config.loss_name}'.")

    raw_criterion: nn.Module | None = None
    if config.use_multitask_heads:
        raw_criterion = nn.CrossEntropyLoss(
            weight=raw_inverse_weights,
            label_smoothing=config.label_smoothing,
        )

    return MultiTaskLoss(
        final_criterion=final_criterion,
        raw_criterion=raw_criterion,
        final_weight=config.final_loss_weight,
        raw_weight=config.raw_loss_weight,
    )
