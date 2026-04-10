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

# Core sequence-window defaults. Each supervised example is one contiguous
# multichannel time series, and the label is assigned from the target segment only.
TARGET_WINDOW_LENGTH = 256
STEP = 128
USE_CONTEXT_WINDOWS = True
LEFT_CONTEXT = 256
RIGHT_CONTEXT = 256
LABEL_PURITY_THRESHOLD = 0.80

# Frequently tuned training/model defaults exposed as clear top-level knobs.
MODEL_NAME = "cnn_bilstm_target_pool"
DROPOUT = 0.40
LOSS_NAME = "weighted_cross_entropy"
LABEL_SMOOTHING = 0.0
USE_WEIGHTED_SAMPLER = False
CLASS_WEIGHTING = "sqrt_inverse_frequency"
SAMPLER_WEIGHTING = "inverse_frequency"
CLASS_BALANCE_BETA = 0.9999
FOCAL_REDUCTION = "mean"
USE_TARGET_INDICATOR_CHANNEL = True
USE_RELATIVE_POSITION_CHANNEL = True
POOL_CONTEXT_REGION = True
EVAL_BATCH_SIZE = 256
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
USE_AMP = True
AMP_DTYPE = "float16"
DETERMINISTIC = False
CUDNN_BENCHMARK = True
ALLOW_TF32 = True
SCHEDULER_NAME = "reduce_on_plateau"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 1e-5
VALIDATION_ARTIFACT_FREQUENCY = 1


def _default_dataset_dir() -> Path:
    """Resolve the dataset directory with an environment override."""

    env_value = os.getenv("DREAMT_DATASET_DIR")
    if env_value:
        return Path(env_value)

    project_local_dataset = PROJECT_ROOT / "dataset" / "data_64Hz"
    if project_local_dataset.exists():
        return project_local_dataset

    return Path(r"D:\Dreamt\data_64Hz")


@dataclass(slots=True, frozen=True)
class DataConfig:
    """Configuration for dataset discovery, cleaning, splitting, and sequence windowing."""

    dataset_dir: Path = field(default_factory=_default_dataset_dir)
    timestamp_column: str = "TIMESTAMP"
    target_column: str = "Sleep_Stage"
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    label_names: tuple[str, ...] = LABEL_NAMES
    excluded_labels: tuple[str, ...] = EXCLUDED_LABELS
    target_window_length: int = TARGET_WINDOW_LENGTH
    step: int = STEP
    use_context_windows: bool = USE_CONTEXT_WINDOWS
    left_context: int = LEFT_CONTEXT
    right_context: int = RIGHT_CONTEXT
    label_purity_threshold: float = LABEL_PURITY_THRESHOLD
    min_valid_fraction: float = 0.80
    drop_ambiguous_windows: bool = True
    require_min_valid_labels: bool = True
    continuity_gap_factor: float = 2.5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    @property
    def effective_left_context(self) -> int:
        """Context actually prepended to the target segment for model input."""

        return self.left_context if self.use_context_windows else 0

    @property
    def effective_right_context(self) -> int:
        """Context actually appended to the target segment for model input."""

        return self.right_context if self.use_context_windows else 0

    @property
    def input_window_length(self) -> int:
        """Total number of timesteps consumed by the model for one sample."""

        return (
            self.effective_left_context
            + self.target_window_length
            + self.effective_right_context
        )

    @property
    def target_start_offset(self) -> int:
        """Start index of the supervised target segment within the model input."""

        return self.effective_left_context

    @property
    def target_end_offset(self) -> int:
        """Exclusive end index of the supervised target segment within the input."""

        return self.target_start_offset + self.target_window_length

    @property
    def window_length(self) -> int:
        """Backward-compatible alias for the supervised target segment length."""

        return self.target_window_length

    @property
    def label_agreement_threshold(self) -> float:
        """Backward-compatible alias used by the existing pipeline."""

        return self.label_purity_threshold


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Configuration for the configurable CNN-based sequence models."""

    model_name: str = MODEL_NAME
    input_channels: int = len(FEATURE_COLUMNS)
    num_classes: int = len(LABEL_NAMES)
    conv_channels: tuple[int, ...] = (64, 128, 192)
    kernel_sizes: tuple[int, ...] = (7, 5, 5)
    dropout: float = DROPOUT
    classifier_hidden_dim: int = 128
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.20
    use_target_indicator_channel: bool = USE_TARGET_INDICATOR_CHANNEL
    use_relative_position_channel: bool = USE_RELATIVE_POSITION_CHANNEL
    pool_context_region: bool = POOL_CONTEXT_REGION

    @property
    def model_type(self) -> str:
        """Backward-compatible alias for the selected model architecture."""

        return self.model_name


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    """Configuration for optimization, logging, and runtime behavior."""

    seed: int = 42
    batch_size: int = 128
    eval_batch_size: int | None = EVAL_BATCH_SIZE
    num_workers: int = 0
    persistent_workers: bool = PERSISTENT_WORKERS
    prefetch_factor: int = PREFETCH_FACTOR
    max_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    patience: int = 8
    min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0
    use_weighted_sampler: bool = USE_WEIGHTED_SAMPLER
    loss_name: str = LOSS_NAME
    label_smoothing: float = LABEL_SMOOTHING
    class_weighting_mode: str = CLASS_WEIGHTING
    sampler_weighting_mode: str = SAMPLER_WEIGHTING
    class_balance_beta: float = CLASS_BALANCE_BETA
    focal_gamma: float = 2.0
    focal_reduction: str = FOCAL_REDUCTION
    focal_alpha: tuple[float, ...] | None = None
    use_amp: bool = USE_AMP
    amp_dtype: str = AMP_DTYPE
    deterministic: bool = DETERMINISTIC
    cudnn_benchmark: bool = CUDNN_BENCHMARK
    allow_tf32: bool = ALLOW_TF32
    scheduler_name: str = SCHEDULER_NAME
    scheduler_factor: float = SCHEDULER_FACTOR
    scheduler_patience: int = SCHEDULER_PATIENCE
    scheduler_min_lr: float = SCHEDULER_MIN_LR
    validation_artifact_frequency: int = VALIDATION_ARTIFACT_FREQUENCY


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
    if config.model.model_name not in {
        "cnn_baseline",
        "cnn_bilstm",
        "cnn_bilstm_target_pool",
    }:
        raise ValueError(
            "model_name must be 'cnn_baseline', 'cnn_bilstm', or 'cnn_bilstm_target_pool'."
        )
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
    if config.data.target_window_length <= 0 or config.data.step <= 0:
        raise ValueError("target_window_length and step must be positive integers.")
    if config.data.left_context < 0 or config.data.right_context < 0:
        raise ValueError("left_context and right_context must be non-negative integers.")
    if config.data.input_window_length <= 0:
        raise ValueError("input_window_length must be positive.")
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
        "weighted_focal_loss",
    }:
        raise ValueError(
            "loss_name must be one of: cross_entropy, weighted_cross_entropy, "
            "focal_loss, weighted_focal_loss."
        )
    if config.training.class_weighting_mode not in {
        "none",
        "inverse_frequency",
        "sqrt_inverse_frequency",
        "effective_number",
    }:
        raise ValueError(
            "class_weighting_mode must be 'none', 'inverse_frequency', "
            "'sqrt_inverse_frequency', or 'effective_number'."
        )
    if config.training.sampler_weighting_mode not in {
        "none",
        "inverse_frequency",
        "sqrt_inverse_frequency",
        "effective_number",
    }:
        raise ValueError(
            "sampler_weighting_mode must be 'none', 'inverse_frequency', "
            "'sqrt_inverse_frequency', or 'effective_number'."
        )
    if not 0.0 <= config.training.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in the interval [0, 1).")
    if config.training.focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative.")
    if config.training.focal_reduction not in {"mean", "sum", "none"}:
        raise ValueError("focal_reduction must be 'mean', 'sum', or 'none'.")
    if config.training.eval_batch_size is not None and config.training.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive when provided.")
    if config.training.num_workers < 0:
        raise ValueError("num_workers must be greater than or equal to 0.")
    if config.training.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive.")
    if config.training.amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be 'float16' or 'bfloat16'.")
    if config.training.scheduler_name not in {"none", "reduce_on_plateau"}:
        raise ValueError("scheduler_name must be 'none' or 'reduce_on_plateau'.")
    if not 0.0 < config.training.scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must be in the interval (0, 1).")
    if config.training.scheduler_patience < 0:
        raise ValueError("scheduler_patience must be non-negative.")
    if config.training.scheduler_min_lr < 0.0:
        raise ValueError("scheduler_min_lr must be non-negative.")
    if not 0.0 <= config.training.class_balance_beta < 1.0:
        raise ValueError("class_balance_beta must be in the interval [0, 1).")
    if config.training.validation_artifact_frequency <= 0:
        raise ValueError("validation_artifact_frequency must be positive.")
    if (
        config.training.focal_alpha is not None
        and len(config.training.focal_alpha) != config.model.num_classes
    ):
        raise ValueError("focal_alpha must provide one value per supervised class.")

    return config
