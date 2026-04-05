"""Baseline 1D CNN model for windowed sleep stage classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    """A simple Conv1d -> BatchNorm -> ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SleepStageCNN(nn.Module):
    """A production-style CNN baseline for multichannel time-series windows."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        dropout: float,
        classifier_hidden_dim: int,
    ) -> None:
        super().__init__()

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length.")
        if not conv_channels:
            raise ValueError("At least one convolutional block is required.")

        layers: list[nn.Module] = []
        current_channels = input_channels
        for block_index, (out_channels, kernel_size) in enumerate(
            zip(conv_channels, kernel_sizes, strict=True)
        ):
            layers.append(ConvBlock(current_channels, out_channels, kernel_size))
            if block_index < len(conv_channels) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                layers.append(nn.Dropout(p=dropout))
            current_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(current_channels, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        return self.classifier(features)
