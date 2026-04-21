"""1D CNN baseline for ternary sleep-stage classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """A compact convolutional block with normalization and pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        pool_kernel_size: int = 2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SleepStageCNN1D(nn.Module):
    """A readable CNN baseline for 64 Hz wearable windows."""

    def __init__(
        self,
        input_channels: int = 8,
        num_classes: int = 3,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(input_channels, 32, kernel_size=9, dropout=dropout),
            ConvBlock(32, 64, kernel_size=7, dropout=dropout),
            ConvBlock(64, 128, kernel_size=5, dropout=dropout),
            ConvBlock(128, 256, kernel_size=3, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        return self.head(outputs)
