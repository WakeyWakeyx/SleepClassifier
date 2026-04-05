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


@dataclass(slots=True, frozen=True)
class DataConfig:
    """Configuration for dataset discovery, cleaning, splitting, and windowing."""

    dataset_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("DREAMT_DATASET_DIR", r"D:\Dreamt\data_64Hz")
        )
    )
    timestamp_column: str = "TIMESTAMP"
    target_column: str = "Sleep_Stage"
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    label_names: tuple[str, ...] = LABEL_NAMES
    excluded_labels: tuple[str, ...] = EXCLUDED_LABELS
    window_length: int = 256
    step: int = 128
    label_agreement_threshold: float = 0.80
    min_valid_fraction: float = 0.80
    continuity_gap_factor: float = 2.5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Configuration for the baseline 1D CNN classifier."""

    input_channels: int = len(FEATURE_COLUMNS)
    num_classes: int = len(LABEL_NAMES)
    conv_channels: tuple[int, ...] = (64, 128, 256, 256)
    kernel_sizes: tuple[int, ...] = (7, 5, 5, 3)
    dropout: float = 0.30
    classifier_hidden_dim: int = 128


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    """Configuration for optimization, logging, and runtime behavior."""

    seed: int = 42
    batch_size: int = 256
    num_workers: int = 0
    max_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0


@dataclass(slots=True, frozen=True)
class PathConfig:
    """Filesystem locations used throughout the training pipeline."""

    project_root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "outputs"
    latest_checkpoint_path: Path = PROJECT_ROOT / "src" / "latest.pt"
    best_checkpoint_path: Path = PROJECT_ROOT / "src" / "best.pt"
    normalization_stats_path: Path = PROJECT_ROOT / "outputs" / "normalization_stats.json"
    split_manifest_path: Path = PROJECT_ROOT / "outputs" / "split_manifest.json"
    training_history_path: Path = PROJECT_ROOT / "outputs" / "training_history.json"
    run_summary_path: Path = PROJECT_ROOT / "outputs" / "run_summary.json"
    class_weights_path: Path = PROJECT_ROOT / "outputs" / "class_weights.json"
    log_path: Path = PROJECT_ROOT / "outputs" / "train.log"
    validation_dir: Path = PROJECT_ROOT / "outputs" / "validation"
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
            "Model num_classes must match the number of supervised label names."
        )
    if abs(
        config.data.train_ratio + config.data.val_ratio + config.data.test_ratio - 1.0
    ) > 1e-6:
        raise ValueError("Train/validation/test split ratios must sum to 1.0.")
    if not 0.0 < config.data.label_agreement_threshold <= 1.0:
        raise ValueError("label_agreement_threshold must be in the interval (0, 1].")
    if not 0.0 < config.data.min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in the interval (0, 1].")

    return config
