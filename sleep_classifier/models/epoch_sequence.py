"""Modality-aware epoch-sequence models for sleep-stage classification."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from sleep_classifier.experiment_config import ExperimentConfig


def _unique_present_features(feature_names: Iterable[str], feature_to_index: dict[str, int]) -> list[str]:
    ordered: list[str] = []
    for feature_name in feature_names:
        if feature_name in feature_to_index and feature_name not in ordered:
            ordered.append(feature_name)
    return ordered


class ResidualConvBlock(nn.Module):
    """Compact residual 1D block used inside epoch encoders."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.AvgPool1d(kernel_size=stride, stride=stride)
            if stride > 1
            else nn.Identity()
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


class WaveformEpochEncoder(nn.Module):
    """Epoch encoder for raw or downsampled waveform channels."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        embedding_dim: int,
        dropout: float,
        downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        self.downsample = (
            nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
            if downsample_factor > 1
            else nn.Identity()
        )
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualConvBlock(base_channels, kernel_size=7, stride=2, dropout=dropout),
            ResidualConvBlock(base_channels, kernel_size=5, stride=2, dropout=dropout),
            ResidualConvBlock(base_channels, kernel_size=3, dilation=2, dropout=dropout),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(base_channels * 2),
            nn.Linear(base_channels * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.downsample(inputs)
        outputs = self.stem(outputs)
        outputs = self.blocks(outputs)
        avg_pool = outputs.mean(dim=-1)
        max_pool = outputs.amax(dim=-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.projection(pooled)


class SummaryEpochEncoder(nn.Module):
    """Epoch encoder for low-rate summary channels such as HR and IBI."""

    def __init__(
        self,
        input_channels: int,
        embedding_dim: int,
        dropout: float,
        chunk_count: int = 6,
    ) -> None:
        super().__init__()
        self.chunk_count = chunk_count
        feature_dim = input_channels * (6 + chunk_count)
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1)
        std = inputs.std(dim=-1, unbiased=False)
        minimum = inputs.amin(dim=-1)
        maximum = inputs.amax(dim=-1)
        last = inputs[:, :, -1]
        delta = inputs[:, :, -1] - inputs[:, :, 0]
        chunk_means = F.adaptive_avg_pool1d(inputs, self.chunk_count).flatten(start_dim=1)
        summary = torch.cat([mean, std, minimum, maximum, last, delta, chunk_means], dim=1)
        return self.network(summary)


class PositionalSequenceEncoder(nn.Module):
    """Transformer encoder over epoch embeddings."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.context_length_epochs, config.epoch_embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.epoch_embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.epoch_embedding_dim * config.transformer_ff_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.sequence_layers)
        self.output_norm = nn.LayerNorm(config.epoch_embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs + self.position_embeddings[:, : inputs.size(1)]
        outputs = self.encoder(outputs)
        return self.output_norm(outputs)


class BiLSTMSequenceEncoder(nn.Module):
    """Bidirectional LSTM over epoch embeddings."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.epoch_embedding_dim,
            hidden_size=config.sequence_hidden_size,
            num_layers=config.sequence_layers,
            dropout=config.dropout if config.sequence_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Linear(config.sequence_hidden_size * 2, config.epoch_embedding_dim)
        self.output_norm = nn.LayerNorm(config.epoch_embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        outputs = self.projection(outputs)
        return self.output_norm(outputs)


class TemporalConvBlock(nn.Module):
    """Residual temporal convolution block over epoch embeddings."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.block(inputs)
        outputs = outputs + inputs
        return self.norm(outputs.transpose(1, 2)).transpose(1, 2)


class TemporalConvSequenceEncoder(nn.Module):
    """Temporal convolutional network over epoch embeddings."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TemporalConvBlock(
                    channels=config.epoch_embedding_dim,
                    kernel_size=config.tcn_kernel_size,
                    dilation=2**layer_index,
                    dropout=config.dropout,
                )
                for layer_index in range(config.sequence_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.epoch_embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2)
        for block in self.blocks:
            outputs = block(outputs)
        outputs = outputs.transpose(1, 2)
        return self.output_norm(outputs)


class ModalityAwareEpochSequenceModel(nn.Module):
    """Modality-aware epoch encoder followed by a sequence model over epochs."""

    def __init__(
        self,
        config: ExperimentConfig,
        final_num_classes: int,
        raw_num_classes: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.final_num_classes = final_num_classes
        self.raw_num_classes = raw_num_classes
        self.feature_columns = list(config.input_feature_columns)
        self.feature_to_index = {name: idx for idx, name in enumerate(self.feature_columns)}
        self.use_multi_branch = config.use_multi_branch
        self.use_multitask_heads = config.use_multitask_heads

        if self.use_multi_branch:
            bvp_features = _unique_present_features(("BVP", "BVP_DELTA"), self.feature_to_index)
            acc_features = _unique_present_features(("ACC_X", "ACC_Y", "ACC_Z", "ACC_MAG"), self.feature_to_index)
            autonomic_features = _unique_present_features(
                ("EDA", "TEMP", "EDA_SLOPE", "TEMP_SLOPE", "EDA_DELTA", "TEMP_DELTA"),
                self.feature_to_index,
            )
            cardio_features = _unique_present_features(("HR", "IBI", "IBI_ROLLING_STD"), self.feature_to_index)

            self.branch_feature_names = {
                "bvp": bvp_features,
                "acc": acc_features,
                "autonomic": autonomic_features,
                "cardio": cardio_features,
            }
            self.bvp_encoder = WaveformEpochEncoder(
                input_channels=len(bvp_features),
                base_channels=config.model_base_channels,
                embedding_dim=config.branch_embedding_dim,
                dropout=config.dropout,
                downsample_factor=1,
            )
            self.acc_encoder = WaveformEpochEncoder(
                input_channels=len(acc_features),
                base_channels=config.model_base_channels,
                embedding_dim=config.branch_embedding_dim,
                dropout=config.dropout,
                downsample_factor=2,
            )
            self.autonomic_encoder = WaveformEpochEncoder(
                input_channels=len(autonomic_features),
                base_channels=max(config.model_base_channels // 2, 16),
                embedding_dim=config.branch_embedding_dim,
                dropout=config.dropout,
                downsample_factor=32,
            )
            self.cardio_encoder = SummaryEpochEncoder(
                input_channels=len(cardio_features),
                embedding_dim=config.branch_embedding_dim,
                dropout=config.dropout,
            )
            fusion_input_dim = config.branch_embedding_dim * 4
            self.epoch_fusion = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, config.epoch_embedding_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.epoch_embedding_dim, config.epoch_embedding_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.single_encoder = WaveformEpochEncoder(
                input_channels=len(self.feature_columns),
                base_channels=config.model_base_channels,
                embedding_dim=config.epoch_embedding_dim,
                dropout=config.dropout,
                downsample_factor=4,
            )

        if config.sequence_model_name == "bilstm":
            self.sequence_encoder = BiLSTMSequenceEncoder(config)
        elif config.sequence_model_name == "transformer":
            self.sequence_encoder = PositionalSequenceEncoder(config)
        elif config.sequence_model_name == "tcn":
            self.sequence_encoder = TemporalConvSequenceEncoder(config)
        else:  # pragma: no cover - validated earlier
            raise ValueError(f"Unsupported sequence model '{config.sequence_model_name}'.")

        head_input_dim = config.epoch_embedding_dim * 3
        self.shared_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, config.epoch_embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.final_head = nn.Linear(config.epoch_embedding_dim, final_num_classes)
        self.raw_head = nn.Linear(config.epoch_embedding_dim, raw_num_classes)

    def _select_features(self, inputs: torch.Tensor, feature_names: list[str]) -> torch.Tensor:
        indices = [self.feature_to_index[name] for name in feature_names]
        return inputs[:, indices, :]

    def _encode_epochs(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_epochs, num_channels, epoch_length = inputs.shape
        flattened = inputs.reshape(batch_size * num_epochs, num_channels, epoch_length)

        if self.use_multi_branch:
            bvp_embedding = self.bvp_encoder(self._select_features(flattened, self.branch_feature_names["bvp"]))
            acc_embedding = self.acc_encoder(self._select_features(flattened, self.branch_feature_names["acc"]))
            autonomic_embedding = self.autonomic_encoder(
                self._select_features(flattened, self.branch_feature_names["autonomic"])
            )
            cardio_embedding = self.cardio_encoder(
                self._select_features(flattened, self.branch_feature_names["cardio"])
            )
            fused = torch.cat(
                [bvp_embedding, acc_embedding, autonomic_embedding, cardio_embedding],
                dim=1,
            )
            epoch_embeddings = self.epoch_fusion(fused)
        else:
            epoch_embeddings = self.single_encoder(flattened)

        return epoch_embeddings.view(batch_size, num_epochs, -1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if inputs.ndim != 4:
            raise ValueError(
                "Expected model inputs with shape [batch, num_epochs, channels, epoch_length], "
                f"got {inputs.shape}"
            )

        epoch_embeddings = self._encode_epochs(inputs)
        sequence_outputs = self.sequence_encoder(epoch_embeddings)
        center_index = inputs.size(1) // 2
        center_context = sequence_outputs[:, center_index]
        global_context = sequence_outputs.mean(dim=1)
        center_epoch_embedding = epoch_embeddings[:, center_index]
        head_inputs = torch.cat([center_context, global_context, center_epoch_embedding], dim=1)
        shared = self.shared_head(head_inputs)
        return {
            "final_logits": self.final_head(shared),
            "raw_logits": self.raw_head(shared) if self.use_multitask_heads else None,
            "epoch_embeddings": epoch_embeddings,
            "sequence_outputs": sequence_outputs,
        }


SleepStageContextModel = ModalityAwareEpochSequenceModel
SleepStageCNN1D = SleepStageContextModel


def build_model(
    config: ExperimentConfig,
    final_num_classes: int,
    raw_num_classes: int,
) -> nn.Module:
    """Construct the configured model."""

    if config.model_name != "epoch_sequence":
        raise ValueError(f"Unsupported model '{config.model_name}'.")
    return ModalityAwareEpochSequenceModel(
        config=config,
        final_num_classes=final_num_classes,
        raw_num_classes=raw_num_classes,
    )
