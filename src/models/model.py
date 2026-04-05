"""Configurable CNN-based models for windowed sleep stage classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.config import ModelConfig


class ResidualTemporalBlock(nn.Module):
    """Residual 1D convolution block with dropout for regularized feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

        if in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = outputs + residual
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class TemporalConvEncoder(nn.Module):
    """Shared temporal CNN stem that preserves a sequence axis for downstream heads."""

    def __init__(
        self,
        input_channels: int,
        conv_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length.")
        if not conv_channels:
            raise ValueError("At least one convolutional block is required.")

        self.blocks = nn.ModuleList()
        current_channels = input_channels
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes, strict=True):
            self.blocks.append(
                ResidualTemporalBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            current_channels = out_channels

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.output_channels = current_channels
        self.num_pool_layers = max(len(self.blocks) - 1, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for block_index, block in enumerate(self.blocks):
            outputs = block(outputs)
            if block_index < len(self.blocks) - 1:
                outputs = self.pool(outputs)
        return outputs

    def downsample_target_bounds(
        self,
        target_start_indices: torch.Tensor | None,
        target_end_indices: torch.Tensor | None,
        encoded_length: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Project input-space target bounds onto the encoder's reduced time axis."""

        if target_start_indices is None or target_end_indices is None:
            return None, None
        if encoded_length <= 0:
            raise ValueError("encoded_length must be positive.")

        reduced_starts = target_start_indices.long()
        reduced_ends = target_end_indices.long()

        for _ in range(self.num_pool_layers):
            reduced_starts = torch.div(reduced_starts, 2, rounding_mode="floor")
            reduced_ends = torch.div(reduced_ends + 1, 2, rounding_mode="floor")

        reduced_starts = reduced_starts.clamp(min=0, max=encoded_length - 1)
        reduced_ends = reduced_ends.clamp(min=1, max=encoded_length)
        reduced_ends = torch.maximum(reduced_ends, reduced_starts + 1)
        reduced_ends = reduced_ends.clamp(max=encoded_length)
        return reduced_starts, reduced_ends


class MLPClassifier(nn.Module):
    """Compact classifier head shared across baseline variants."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SleepStageCNNBaseline(nn.Module):
    """Residual-style CNN baseline for multichannel time-series sequence windows."""

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
        self.encoder = TemporalConvEncoder(
            input_channels=input_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        self.classifier = MLPClassifier(
            input_dim=self.encoder.output_channels * 2,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        target_start_indices: torch.Tensor | None = None,
        target_end_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # inputs: [batch, channels, time]
        features = self.encoder(inputs)
        # sequence_features: [batch, reduced_time, feature_dim]
        sequence_features = features.transpose(1, 2).contiguous()
        reduced_starts, reduced_ends = self.encoder.downsample_target_bounds(
            target_start_indices=target_start_indices,
            target_end_indices=target_end_indices,
            encoded_length=sequence_features.size(1),
        )
        pooled_features = _pool_sequence_region(
            sequence_features=sequence_features,
            target_start_indices=reduced_starts,
            target_end_indices=reduced_ends,
        )
        return self.classifier(pooled_features)


class SleepStageCNNBiLSTM(nn.Module):
    """CNN encoder followed by a BiLSTM over temporal feature steps."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        dropout: float,
        classifier_hidden_dim: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lstm_dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = TemporalConvEncoder(
            input_channels=input_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        self.sequence_dropout = nn.Dropout(p=dropout)
        self.sequence_model = nn.LSTM(
            input_size=self.encoder.output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = MLPClassifier(
            input_dim=lstm_hidden_size * 4,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        target_start_indices: torch.Tensor | None = None,
        target_end_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # inputs: [batch, channels, time]
        features = self.encoder(inputs)
        # sequence_features: [batch, reduced_time, feature_dim]
        sequence_features = features.transpose(1, 2).contiguous()
        sequence_features = self.sequence_dropout(sequence_features)

        lstm_outputs, _ = self.sequence_model(sequence_features)
        reduced_starts, reduced_ends = self.encoder.downsample_target_bounds(
            target_start_indices=target_start_indices,
            target_end_indices=target_end_indices,
            encoded_length=lstm_outputs.size(1),
        )
        pooled_features = _pool_sequence_region(
            sequence_features=lstm_outputs,
            target_start_indices=reduced_starts,
            target_end_indices=reduced_ends,
        )
        return self.classifier(pooled_features)


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate the configured model variant."""

    common_kwargs = {
        "input_channels": config.input_channels,
        "num_classes": config.num_classes,
        "conv_channels": config.conv_channels,
        "kernel_sizes": config.kernel_sizes,
        "dropout": config.dropout,
        "classifier_hidden_dim": config.classifier_hidden_dim,
    }

    if config.model_name == "cnn_baseline":
        return SleepStageCNNBaseline(**common_kwargs)
    if config.model_name == "cnn_bilstm":
        return SleepStageCNNBiLSTM(
            **common_kwargs,
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            lstm_dropout=config.lstm_dropout,
        )
    raise ValueError(f"Unsupported model_name: {config.model_name}")


def _pool_sequence_region(
    sequence_features: torch.Tensor,
    target_start_indices: torch.Tensor | None,
    target_end_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Pool only the target region so context informs features without owning the label."""

    batch_size, sequence_length, _ = sequence_features.shape
    if target_start_indices is None or target_end_indices is None:
        target_start_indices = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=sequence_features.device,
        )
        target_end_indices = torch.full(
            (batch_size,),
            fill_value=sequence_length,
            dtype=torch.long,
            device=sequence_features.device,
        )
    else:
        target_start_indices = target_start_indices.to(sequence_features.device).long()
        target_end_indices = target_end_indices.to(sequence_features.device).long()

    time_indices = torch.arange(sequence_length, device=sequence_features.device).unsqueeze(0)
    target_mask = (time_indices >= target_start_indices.unsqueeze(1)) & (
        time_indices < target_end_indices.unsqueeze(1)
    )
    target_mask = target_mask.unsqueeze(-1)

    masked_sum = sequence_features.masked_fill(~target_mask, 0.0).sum(dim=1)
    target_lengths = target_mask.sum(dim=1).clamp_min(1).to(sequence_features.dtype)
    mean_features = masked_sum / target_lengths

    mask_fill_value = torch.finfo(sequence_features.dtype).min
    masked_max = sequence_features.masked_fill(~target_mask, mask_fill_value)
    max_features = masked_max.amax(dim=1)

    return torch.cat([mean_features, max_features], dim=1)


SleepStageCNN = SleepStageCNNBaseline
