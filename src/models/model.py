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
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
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
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length.")
        if len(conv_channels) != len(dilations):
            raise ValueError("conv_channels and dilations must have the same length.")
        if not conv_channels:
            raise ValueError("At least one convolutional block is required.")

        self.blocks = nn.ModuleList()
        current_channels = input_channels
        for out_channels, kernel_size, dilation in zip(
            conv_channels,
            kernel_sizes,
            dilations,
            strict=True,
        ):
            self.blocks.append(
                ResidualTemporalBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
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
        """Project input-space target bounds onto the encoder's reduced time axis.

        Convolution blocks preserve sequence length because they use symmetric
        padding. Only the intermediate max-pooling stages reduce the time axis.
        The target region is mapped with floor/ceil style bounds so that any
        encoded timestep whose pooled receptive field overlaps the target span is
        retained for target-aware pooling.
        """

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


class MaskedRegionAttentionPooling(nn.Module):
    """Pool one masked temporal region using mean, max, and attention statistics."""

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        attention_hidden_dim = max(feature_dim // 2, 32)
        self.attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(feature_dim * 3)
        self.output_dim = feature_dim * 3

    def forward(self, sequence_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded_mask = mask.unsqueeze(-1)
        valid_counts = mask.sum(dim=1, keepdim=True)
        has_valid = valid_counts > 0

        masked_sum = sequence_features.masked_fill(~expanded_mask, 0.0).sum(dim=1)
        mean_features = masked_sum / valid_counts.clamp_min(1).to(sequence_features.dtype)

        mask_fill_value = torch.finfo(sequence_features.dtype).min
        masked_max = sequence_features.masked_fill(~expanded_mask, mask_fill_value)
        max_features = masked_max.amax(dim=1)
        max_features = torch.where(has_valid, max_features, torch.zeros_like(max_features))

        attention_logits = self.attention(sequence_features).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, mask_fill_value)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = attention_weights.masked_fill(~mask, 0.0)
        attention_weights = attention_weights / attention_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        attention_features = torch.bmm(
            attention_weights.unsqueeze(1),
            sequence_features,
        ).squeeze(1)
        attention_features = torch.where(
            has_valid,
            attention_features,
            torch.zeros_like(attention_features),
        )

        pooled_features = torch.cat(
            [mean_features, max_features, attention_features],
            dim=1,
        )
        return self.output_norm(pooled_features)


class TargetContextPooling(nn.Module):
    """Fuse explicit target summaries with pooled surrounding context."""

    def __init__(
        self,
        feature_dim: int,
        dropout: float,
        include_context_region: bool,
        separate_context_regions: bool,
    ) -> None:
        super().__init__()
        self.include_context_region = include_context_region
        self.separate_context_regions = separate_context_regions and include_context_region
        self.region_pool = MaskedRegionAttentionPooling(feature_dim=feature_dim, dropout=dropout)

        if self.separate_context_regions:
            self.output_dim = self.region_pool.output_dim * 5
            self.output_norm = nn.LayerNorm(self.output_dim)
        elif include_context_region:
            self.output_dim = self.region_pool.output_dim * 3
            self.output_norm = nn.LayerNorm(self.output_dim)
        else:
            self.output_dim = self.region_pool.output_dim
            self.output_norm = nn.Identity()

    def forward(
        self,
        sequence_features: torch.Tensor,
        target_start_indices: torch.Tensor | None,
        target_end_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        target_mask, left_context_mask, right_context_mask = _build_region_masks(
            sequence_length=sequence_features.size(1),
            batch_size=sequence_features.size(0),
            device=sequence_features.device,
            target_start_indices=target_start_indices,
            target_end_indices=target_end_indices,
        )
        target_features = self.region_pool(sequence_features=sequence_features, mask=target_mask)

        if not self.include_context_region:
            return target_features

        if self.separate_context_regions:
            left_context_features = self.region_pool(
                sequence_features=sequence_features,
                mask=left_context_mask,
            )
            right_context_features = self.region_pool(
                sequence_features=sequence_features,
                mask=right_context_mask,
            )
            fused_features = torch.cat(
                [
                    target_features,
                    left_context_features,
                    right_context_features,
                    target_features - left_context_features,
                    target_features - right_context_features,
                ],
                dim=1,
            )
            return self.output_norm(fused_features)

        context_features = self.region_pool(
            sequence_features=sequence_features,
            mask=left_context_mask | right_context_mask,
        )
        fused_features = torch.cat(
            [
                target_features,
                context_features,
                target_features - context_features,
            ],
            dim=1,
        )
        return self.output_norm(fused_features)


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


class InputRegularization(nn.Module):
    """Apply optional per-feature and per-channel dropout to raw inputs."""

    def __init__(self, feature_dropout: float, channel_dropout: float) -> None:
        super().__init__()
        self.feature_dropout = (
            nn.Dropout(p=feature_dropout) if feature_dropout > 0.0 else nn.Identity()
        )
        self.channel_dropout = (
            nn.Dropout1d(p=channel_dropout) if channel_dropout > 0.0 else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.channel_dropout(inputs)
        outputs = self.feature_dropout(outputs)
        return outputs


class SequenceFeatureMixin:
    """Add positional cues and the supervised target indicator to the inputs."""

    use_target_indicator_channel: bool
    use_relative_position_channel: bool

    @property
    def extra_input_channels(self) -> int:
        return int(self.use_target_indicator_channel) + int(self.use_relative_position_channel)

    def _augment_inputs(
        self,
        inputs: torch.Tensor,
        target_start_indices: torch.Tensor | None,
        target_end_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        auxiliary_channels: list[torch.Tensor] = []
        batch_size = inputs.size(0)
        sequence_length = inputs.size(2)
        device = inputs.device
        dtype = inputs.dtype

        if self.use_relative_position_channel:
            relative_position = torch.linspace(
                -1.0,
                1.0,
                steps=sequence_length,
                device=device,
                dtype=dtype,
            ).view(1, 1, sequence_length)
            auxiliary_channels.append(relative_position.expand(batch_size, -1, -1))

        if self.use_target_indicator_channel:
            target_mask = _build_target_mask(
                sequence_length=sequence_length,
                batch_size=batch_size,
                device=device,
                target_start_indices=target_start_indices,
                target_end_indices=target_end_indices,
            )
            auxiliary_channels.append(target_mask.unsqueeze(1).to(dtype))

        if not auxiliary_channels:
            return inputs
        return torch.cat([inputs, *auxiliary_channels], dim=1)


class SleepStageCNNBaseline(SequenceFeatureMixin, nn.Module):
    """Residual-style CNN baseline with explicit target/context-aware pooling."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        conv_dilations: Sequence[int],
        dropout: float,
        feature_dropout: float,
        channel_dropout: float,
        classifier_hidden_dim: int,
        use_target_indicator_channel: bool,
        use_relative_position_channel: bool,
        pool_context_region: bool,
        separate_context_regions: bool,
    ) -> None:
        super().__init__()
        self.use_target_indicator_channel = use_target_indicator_channel
        self.use_relative_position_channel = use_relative_position_channel
        self.input_regularization = InputRegularization(
            feature_dropout=feature_dropout,
            channel_dropout=channel_dropout,
        )
        self.encoder = TemporalConvEncoder(
            input_channels=input_channels + self.extra_input_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dilations=conv_dilations,
            dropout=dropout,
        )
        self.target_pool = TargetContextPooling(
            feature_dim=self.encoder.output_channels,
            dropout=dropout,
            include_context_region=pool_context_region,
            separate_context_regions=separate_context_regions,
        )
        self.classifier = MLPClassifier(
            input_dim=self.target_pool.output_dim,
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
        inputs = self.input_regularization(inputs)
        augmented_inputs = self._augment_inputs(
            inputs=inputs,
            target_start_indices=target_start_indices,
            target_end_indices=target_end_indices,
        )
        features = self.encoder(augmented_inputs)
        sequence_features = features.transpose(1, 2).contiguous()
        reduced_starts, reduced_ends = self.encoder.downsample_target_bounds(
            target_start_indices=target_start_indices,
            target_end_indices=target_end_indices,
            encoded_length=sequence_features.size(1),
        )
        pooled_features = self.target_pool(
            sequence_features=sequence_features,
            target_start_indices=reduced_starts,
            target_end_indices=reduced_ends,
        )
        return self.classifier(pooled_features)


class SleepStageCNNBiLSTMTargetPool(SequenceFeatureMixin, nn.Module):
    """CNN encoder + BiLSTM with explicit target/context pooling."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        conv_dilations: Sequence[int],
        dropout: float,
        feature_dropout: float,
        channel_dropout: float,
        classifier_hidden_dim: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lstm_dropout: float,
        use_target_indicator_channel: bool,
        use_relative_position_channel: bool,
        pool_context_region: bool,
        separate_context_regions: bool,
    ) -> None:
        super().__init__()
        self.use_target_indicator_channel = use_target_indicator_channel
        self.use_relative_position_channel = use_relative_position_channel
        self.input_regularization = InputRegularization(
            feature_dropout=feature_dropout,
            channel_dropout=channel_dropout,
        )
        self.encoder = TemporalConvEncoder(
            input_channels=input_channels + self.extra_input_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dilations=conv_dilations,
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
        self.target_pool = TargetContextPooling(
            feature_dim=lstm_hidden_size * 2,
            dropout=dropout,
            include_context_region=pool_context_region,
            separate_context_regions=separate_context_regions,
        )
        self.classifier = MLPClassifier(
            input_dim=self.target_pool.output_dim,
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
        inputs = self.input_regularization(inputs)
        augmented_inputs = self._augment_inputs(
            inputs=inputs,
            target_start_indices=target_start_indices,
            target_end_indices=target_end_indices,
        )
        features = self.encoder(augmented_inputs)
        sequence_features = features.transpose(1, 2).contiguous()
        sequence_features = self.sequence_dropout(sequence_features)

        lstm_outputs, _ = self.sequence_model(sequence_features)
        reduced_starts, reduced_ends = self.encoder.downsample_target_bounds(
            target_start_indices=target_start_indices,
            target_end_indices=target_end_indices,
            encoded_length=lstm_outputs.size(1),
        )
        pooled_features = self.target_pool(
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
        "conv_dilations": config.conv_dilations,
        "dropout": config.dropout,
        "feature_dropout": config.feature_dropout,
        "channel_dropout": config.channel_dropout,
        "classifier_hidden_dim": config.classifier_hidden_dim,
        "use_target_indicator_channel": config.use_target_indicator_channel,
        "use_relative_position_channel": config.use_relative_position_channel,
        "pool_context_region": config.pool_context_region,
        "separate_context_regions": config.separate_context_regions,
    }

    if config.model_name == "cnn_baseline":
        return SleepStageCNNBaseline(**common_kwargs)
    if config.model_name in {"cnn_bilstm", "cnn_bilstm_target_pool"}:
        return SleepStageCNNBiLSTMTargetPool(
            **common_kwargs,
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            lstm_dropout=config.lstm_dropout,
        )
    raise ValueError(f"Unsupported model_name: {config.model_name}")


def _build_target_mask(
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    target_start_indices: torch.Tensor | None,
    target_end_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Create a boolean mask for the target region on the current time axis."""

    if target_start_indices is None or target_end_indices is None:
        target_start_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        target_end_indices = torch.full(
            (batch_size,),
            fill_value=sequence_length,
            dtype=torch.long,
            device=device,
        )
    else:
        target_start_indices = target_start_indices.to(device).long()
        target_end_indices = target_end_indices.to(device).long()

    time_indices = torch.arange(sequence_length, device=device).unsqueeze(0)
    return (time_indices >= target_start_indices.unsqueeze(1)) & (
        time_indices < target_end_indices.unsqueeze(1)
    )


def _build_region_masks(
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    target_start_indices: torch.Tensor | None,
    target_end_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create target, left-context, and right-context masks on the current time axis."""

    target_mask = _build_target_mask(
        sequence_length=sequence_length,
        batch_size=batch_size,
        device=device,
        target_start_indices=target_start_indices,
        target_end_indices=target_end_indices,
    )
    time_indices = torch.arange(sequence_length, device=device).unsqueeze(0)

    if target_start_indices is None or target_end_indices is None:
        left_context_mask = torch.zeros_like(target_mask)
        right_context_mask = torch.zeros_like(target_mask)
    else:
        target_start_indices = target_start_indices.to(device).long()
        target_end_indices = target_end_indices.to(device).long()
        left_context_mask = time_indices < target_start_indices.unsqueeze(1)
        right_context_mask = time_indices >= target_end_indices.unsqueeze(1)

    return target_mask, left_context_mask, right_context_mask


SleepStageCNN = SleepStageCNNBaseline
SleepStageCNNBiLSTM = SleepStageCNNBiLSTMTargetPool
