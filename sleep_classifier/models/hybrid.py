"""Context-aware wearable time-series models for sleep-stage classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sleep_classifier.config import ExperimentConfig


class ResidualConvBlock(nn.Module):
    """Residual convolutional block with normalization, dropout, and optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = F.gelu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.dropout(outputs)
        return F.gelu(outputs + residual)


class MultiScaleStem(nn.Module):
    """Parallel convolutional branches to capture multiple temporal scales."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 11, 17),
    ) -> None:
        super().__init__()
        branch_width = max(output_channels // len(kernel_sizes), 16)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        branch_width,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_width),
                    nn.GELU(),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.projection = nn.Sequential(
            nn.Conv1d(branch_width * len(kernel_sizes), output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = [branch(inputs) for branch in self.branches]
        return self.projection(torch.cat(outputs, dim=1))


class AttentionPool1d(nn.Module):
    """Attention pooling over temporal features."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attention_scores = self.score(inputs).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return torch.sum(inputs * attention_weights.unsqueeze(-1), dim=1)


class MultiScaleResidualBiLSTMClassifier(nn.Module):
    """Multi-scale residual CNN encoder with BiLSTM and attention pooling."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        dropout: float = 0.30,
        base_channels: int = 64,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
        attention_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.stem = MultiScaleStem(input_channels=input_channels, output_channels=base_channels)
        self.encoder = nn.Sequential(
            ResidualConvBlock(base_channels, base_channels, kernel_size=7, stride=2, dropout=dropout),
            ResidualConvBlock(base_channels, base_channels * 2, kernel_size=5, stride=2, dropout=dropout),
            ResidualConvBlock(base_channels * 2, base_channels * 2, kernel_size=5, dropout=dropout),
            ResidualConvBlock(base_channels * 2, base_channels * 3, kernel_size=3, stride=2, dropout=dropout),
        )
        conv_output_dim = base_channels * 3
        self.temporal_norm = nn.BatchNorm1d(conv_output_dim)
        self.temporal_model = nn.LSTM(
            input_size=conv_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        temporal_dim = lstm_hidden_size * 2
        self.attention_pool = AttentionPool1d(temporal_dim, attention_hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.encoder(outputs)
        outputs = self.temporal_norm(outputs)
        outputs = outputs.transpose(1, 2)
        outputs, _ = self.temporal_model(outputs)
        attention_context = self.attention_pool(outputs)
        mean_context = outputs.mean(dim=1)
        pooled = torch.cat([attention_context, mean_context], dim=1)
        return self.classifier(pooled)


SleepStageContextModel = MultiScaleResidualBiLSTMClassifier


def build_model(config: ExperimentConfig, num_classes: int) -> nn.Module:
    """Construct the configured model."""

    if config.model_name != "multiscale_bilstm":
        raise ValueError(f"Unsupported model '{config.model_name}'.")
    return MultiScaleResidualBiLSTMClassifier(
        input_channels=len(config.input_feature_columns),
        num_classes=num_classes,
        dropout=config.dropout,
        base_channels=config.model_base_channels,
        lstm_hidden_size=config.lstm_hidden_size,
        lstm_layers=config.lstm_layers,
        attention_hidden_size=config.attention_hidden_size,
    )
