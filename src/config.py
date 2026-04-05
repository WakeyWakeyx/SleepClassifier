"""Central configuration for the sleep stage classification project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS: tuple[str, ...] = (
    "BVP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "TEMP",
    "EDA",
    "HR",
    "IBI",
)
LABEL_NAMES: tuple[str, ...] = ("W", "N1", "N2", "N3", "R")
EXCLUDED_LABELS: tuple[str, ...] = ("P", "Missing")


def _default_dataset_dir() -> Path:
    """Resolve the dataset directory with an environment override."""

    env_value = os.getenv("DREAMT_DATASET_DIR")
    if env_value:
        return Path(env_value)
    return PROJECT_ROOT / "dataset" / "data_64Hz"


@dataclass(slots=True, frozen=True)
class DataConfig:
    """Configuration for dataset discovery, cleaning, splitting, and windowing."""

    dataset_dir: Path = field(default_factory=_default_dataset_dir)
    timestamp_column: str = "TIMESTAMP"
    target_column: str = "Sleep_Stage"
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    label_names: tuple[str, ...] = LABEL_NAMES
    excluded_labels: tuple[str, ...] = EXCLUDED_LABELS
    window_length: int = 256
    step: int = 128
    label_purity_threshold: float = 0.80
    min_valid_fraction: float = 0.80
    drop_ambiguous_windows: bool = True
    require_min_valid_labels: bool = True
    continuity_gap_factor: float = 2.5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    @property
    def label_agreement_threshold(self) -> float:
        """Backward-compatible alias used by the existing pipeline."""

        return self.label_purity_threshold


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Configuration for the configurable CNN-based baselines."""

    model_type: str = "cnn_bilstm"
    input_channels: int = len(FEATURE_COLUMNS)
    num_classes: int = len(LABEL_NAMES)
    conv_channels: tuple[int, ...] = (64, 128, 192)
    kernel_sizes: tuple[int, ...] = (7, 5, 5)
    dropout: float = 0.40
    classifier_hidden_dim: int = 128
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.20


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    """Configuration for optimization, logging, and runtime behavior."""

    seed: int = 42
    batch_size: int = 256
    num_workers: int = 0
    max_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    patience: int = 8
    min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0
    use_weighted_sampler: bool = True
    loss_name: str = "cross_entropy"
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0
    focal_use_class_weights: bool = True
    focal_alpha: tuple[float, ...] | None = None


@dataclass(slots=True, frozen=True)
class PathConfig:
    """Filesystem locations used throughout the training pipeline."""

    project_root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "outputs"
    latest_checkpoint_path: Path = PROJECT_ROOT / "src" / "latest.pt"
    best_checkpoint_path: Path = PROJECT_ROOT / "src" / "best.pt"
    normalization_stats_path: Path = PROJECT_ROOT / "outputs" / "normalization_stats.json"
    split_manifest_path: Path = PROJECT_ROOT / "outputs" / "split_manifest.json"
    dataset_summary_path: Path = PROJECT_ROOT / "outputs" / "dataset_summary.json"
    training_history_path: Path = PROJECT_ROOT / "outputs" / "training_history.json"
    run_summary_path: Path = PROJECT_ROOT / "outputs" / "run_summary.json"
    class_weights_path: Path = PROJECT_ROOT / "outputs" / "class_weights.json"
    log_path: Path = PROJECT_ROOT / "outputs" / "train.log"
    validation_dir: Path = PROJECT_ROOT / "outputs" / "validation"
    validation_epochs_dir: Path = PROJECT_ROOT / "outputs" / "validation" / "epochs"
    test_dir: Path = PROJECT_ROOT / "outputs" / "test"


@dataclass(slots=True, frozen=True)
class ProjectConfig:
    """Top-level configuration bundle."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.data.label_names)}

    @property
    def index_to_label(self) -> dict[int, str]:
        return {index: label for index, label in enumerate(self.data.label_names)}


def build_config() -> ProjectConfig:
    """Build and validate the project configuration."""

    config = ProjectConfig()

    if config.model.input_channels != len(config.data.feature_columns):
        raise ValueError(
            "Model input_channels must match the number of configured feature columns."
        )
    if config.model.num_classes != len(config.data.label_names):
        raise ValueError(
            "Model num_classes must match the number of configured label names."
        )
    if config.model.model_type not in {"cnn_baseline", "cnn_bilstm"}:
        raise ValueError("model_type must be 'cnn_baseline' or 'cnn_bilstm'.")
    if len(config.model.conv_channels) != len(config.model.kernel_sizes):
        raise ValueError("conv_channels and kernel_sizes must have the same length.")
    if not config.model.conv_channels:
        raise ValueError("At least one convolutional stage is required.")
    if config.model.lstm_num_layers < 1:
        raise ValueError("lstm_num_layers must be at least 1.")
    if any(kernel_size < 3 or kernel_size % 2 == 0 for kernel_size in config.model.kernel_sizes):
        raise ValueError("kernel_sizes must be odd integers greater than or equal to 3.")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model dropout must be in the interval [0, 1).")
    if not 0.0 <= config.model.lstm_dropout < 1.0:
        raise ValueError("lstm_dropout must be in the interval [0, 1).")

    if abs(
        config.data.train_ratio + config.data.val_ratio + config.data.test_ratio - 1.0
    ) > 1e-6:
        raise ValueError("Train/validation/test split ratios must sum to 1.0.")
    if config.data.window_length <= 0 or config.data.step <= 0:
        raise ValueError("window_length and step must be positive integers.")
    if not 0.0 < config.data.label_purity_threshold <= 1.0:
        raise ValueError("label_purity_threshold must be in the interval (0, 1].")
    if not 0.0 < config.data.min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in the interval (0, 1].")
    if config.data.continuity_gap_factor <= 0.0:
        raise ValueError("continuity_gap_factor must be positive.")

    if config.training.loss_name not in {
        "cross_entropy",
        "weighted_cross_entropy",
        "focal_loss",
    }:
        raise ValueError(
            "loss_name must be one of: cross_entropy, weighted_cross_entropy, focal_loss."
        )
    if not 0.0 <= config.training.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in the interval [0, 1).")
    if config.training.focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative.")
    if (
        config.training.focal_alpha is not None
        and len(config.training.focal_alpha) != config.model.num_classes
    ):
        raise ValueError("focal_alpha must provide one value per supervised class.")

    return config
