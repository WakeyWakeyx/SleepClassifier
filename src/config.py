"""Central configuration and experiment preset system for the project."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

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
MINORITY_LABELS: tuple[str, ...] = ("N1", "N3", "R")
TRANSITION_DISTANCE_THRESHOLDS: tuple[int, ...] = (128, 256, 512, 1024)
LABEL_SCHEMA_NAME = "five_class"

# Core sequence-window defaults. Each supervised example is one contiguous
# multichannel time series, and the label is assigned from the target segment only.
TARGET_WINDOW_LENGTH = 256
STEP = 128
USE_CONTEXT_WINDOWS = True
LEFT_CONTEXT = 256
RIGHT_CONTEXT = 256
LABEL_PURITY_THRESHOLD = 0.80
TARGET_LABEL_STRATEGY = "purity_constrained_majority"
CENTER_LABEL_SPAN = 1

# Frequently tuned training/model defaults exposed as clear top-level knobs.
MODEL_NAME = "cnn_bilstm_target_pool"
DROPOUT = 0.40
FEATURE_DROPOUT = 0.0
CHANNEL_DROPOUT = 0.0
LOSS_NAME = "balanced_softmax"
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
SCHEDULER_T_MAX = 12
SCHEDULER_ETA_MIN = 1e-5
VALIDATION_ARTIFACT_FREQUENCY = 1
EARLY_STOPPING_METRIC = "macro_f1"
SAMPLER_REPLACEMENT = True
SAMPLER_NUM_SAMPLES_MULTIPLIER = 1.0
SAMPLER_MAX_WEIGHT_RATIO = 4.0
LOSS_MAX_WEIGHT_RATIO = 4.0
LOSS_WEIGHT_POWER_WHEN_SAMPLING = 0.5
COLLAPSE_RECALL_THRESHOLD = 0.02
COLLAPSE_PREDICTED_COUNT_THRESHOLD = 2
COLLAPSE_PREDICTED_FRACTION_THRESHOLD = 0.001
COLLAPSE_PATIENCE_EPOCHS = 2
PARTICIPANT_BALANCED_SAMPLING = False
AUXILIARY_TARGET_LOSS_WEIGHT = 0.0
WARMUP_EPOCHS = 0


def _default_label_mapping() -> dict[str, str]:
    """Return the default identity mapping from source stages to supervised stages."""

    return {label_name: label_name for label_name in LABEL_NAMES}


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    """Preserve first-seen order while removing duplicates."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _derive_label_names(
    source_label_names: Sequence[str],
    label_mapping: Mapping[str, str],
) -> tuple[str, ...]:
    """Build the supervised label order from the configured source-stage mapping."""

    return _ordered_unique(label_mapping[source_label] for source_label in source_label_names)


def _remap_focus_labels(
    focus_labels: Sequence[str],
    label_mapping: Mapping[str, str],
) -> tuple[str, ...]:
    """Remap raw or already-remapped focus labels into supervised label space."""

    return _ordered_unique(label_mapping.get(label_name, label_name) for label_name in focus_labels)


def _default_dataset_dir() -> Path:
    """Resolve the dataset directory with an environment override."""

    env_value = os.getenv("DREAMT_DATASET_DIR")
    if env_value:
        return Path(env_value)

    project_local_dataset = PROJECT_ROOT / "dataset" / "data_64Hz"
    if project_local_dataset.exists():
        return project_local_dataset

    return Path(r"D:\Dreamt\data_64Hz")


def _default_output_root() -> Path:
    """Resolve the base output directory with an environment override."""

    env_value = os.getenv("DREAMT_OUTPUT_ROOT")
    return Path(env_value) if env_value else PROJECT_ROOT / "outputs"


def _default_num_workers() -> int:
    """Choose a conservative worker count that still improves throughput."""

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return min(4, max(2, cpu_count // 2))


def _default_experiment_group() -> str:
    """Build a stable group name for one invocation of the pipeline."""

    env_value = os.getenv("DREAMT_EXPERIMENT_GROUP")
    if env_value:
        return env_value
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _default_run_id() -> str:
    """Create a unique per-experiment run identifier."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _requested_preset_names() -> tuple[str, ...]:
    """Resolve the single preset or preset grid requested by the environment."""

    grid_value = os.getenv("DREAMT_EXPERIMENT_GRID")
    if grid_value:
        names = tuple(name.strip() for name in grid_value.split(",") if name.strip())
        if names:
            return names

    preset_name = os.getenv("DREAMT_EXPERIMENT", "baseline_current").strip()
    return (preset_name,) if preset_name else ("baseline_current",)


def _load_json_overrides(env_var_name: str) -> dict[str, Any]:
    """Parse JSON overrides from an environment variable when provided."""

    raw_value = os.getenv(env_var_name)
    if not raw_value:
        return {}
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{env_var_name} must contain valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{env_var_name} must decode to a JSON object.")
    return payload


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dictionaries without mutating inputs."""

    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True, frozen=True)
class DataConfig:
    """Configuration for dataset discovery, cleaning, splitting, and sequence windowing."""

    dataset_dir: Path = field(default_factory=_default_dataset_dir)
    timestamp_column: str = "TIMESTAMP"
    target_column: str = "Sleep_Stage"
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    label_schema_name: str = LABEL_SCHEMA_NAME
    source_label_names: tuple[str, ...] = LABEL_NAMES
    label_names: tuple[str, ...] = LABEL_NAMES
    label_mapping: dict[str, str] = field(default_factory=_default_label_mapping)
    excluded_labels: tuple[str, ...] = EXCLUDED_LABELS
    target_window_length: int = TARGET_WINDOW_LENGTH
    step: int = STEP
    use_context_windows: bool = USE_CONTEXT_WINDOWS
    left_context: int = LEFT_CONTEXT
    right_context: int = RIGHT_CONTEXT
    label_purity_threshold: float = LABEL_PURITY_THRESHOLD
    target_label_strategy: str = TARGET_LABEL_STRATEGY
    center_label_span: int = CENTER_LABEL_SPAN
    min_valid_fraction: float = 0.80
    drop_ambiguous_windows: bool = True
    require_min_valid_labels: bool = True
    continuity_gap_factor: float = 2.5
    transition_distance_thresholds: tuple[int, ...] = TRANSITION_DISTANCE_THRESHOLDS
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

    @property
    def purity_threshold(self) -> float:
        """Alias for concise references in diagnostics and summaries."""

        return self.label_purity_threshold

    @property
    def source_label_to_index(self) -> dict[str, int]:
        """Map raw dataset sleep-stage labels to stable integer ids."""

        return {
            label_name: index
            for index, label_name in enumerate(self.source_label_names)
        }

    @property
    def source_index_to_label(self) -> dict[int, str]:
        """Inverse mapping for raw dataset sleep-stage labels."""

        return {
            index: label_name
            for index, label_name in enumerate(self.source_label_names)
        }

    @property
    def output_label_to_index(self) -> dict[str, int]:
        """Map supervised output labels to stable integer ids."""

        return {
            label_name: index
            for index, label_name in enumerate(self.label_names)
        }

    @property
    def source_label_remap_indices(self) -> tuple[int, ...]:
        """Map each raw-label index onto the configured supervised-label index."""

        output_label_to_index = self.output_label_to_index
        return tuple(
            output_label_to_index[self.label_mapping[label_name]]
            for label_name in self.source_label_names
        )


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Configuration for the configurable CNN-based sequence models."""

    model_name: str = MODEL_NAME
    input_channels: int = len(FEATURE_COLUMNS)
    num_classes: int = len(LABEL_NAMES)
    conv_channels: tuple[int, ...] = (64, 128, 192)
    kernel_sizes: tuple[int, ...] = (7, 5, 5)
    conv_dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = DROPOUT
    feature_dropout: float = FEATURE_DROPOUT
    channel_dropout: float = CHANNEL_DROPOUT
    classifier_hidden_dim: int = 128
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.20
    use_target_indicator_channel: bool = USE_TARGET_INDICATOR_CHANNEL
    use_relative_position_channel: bool = USE_RELATIVE_POSITION_CHANNEL
    pool_context_region: bool = POOL_CONTEXT_REGION
    separate_context_regions: bool = True

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
    num_workers: int = field(default_factory=_default_num_workers)
    persistent_workers: bool = PERSISTENT_WORKERS
    prefetch_factor: int = PREFETCH_FACTOR
    max_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    patience: int = 8
    min_delta: float = 1e-4
    gradient_clip_norm: float = 1.0
    use_weighted_sampler: bool = USE_WEIGHTED_SAMPLER
    sampler_strategy: str | None = None
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
    scheduler_t_max: int = SCHEDULER_T_MAX
    scheduler_eta_min: float = SCHEDULER_ETA_MIN
    validation_artifact_frequency: int = VALIDATION_ARTIFACT_FREQUENCY
    early_stopping_metric: str = EARLY_STOPPING_METRIC
    minority_labels: tuple[str, ...] = MINORITY_LABELS
    sampler_replacement: bool = SAMPLER_REPLACEMENT
    sampler_num_samples_multiplier: float = SAMPLER_NUM_SAMPLES_MULTIPLIER
    sampler_probability_caps: dict[str, float] = field(default_factory=dict)
    sampler_probability_floors: dict[str, float] = field(default_factory=dict)
    sampler_max_weight_ratio: float | None = SAMPLER_MAX_WEIGHT_RATIO
    loss_max_weight_ratio: float | None = LOSS_MAX_WEIGHT_RATIO
    loss_weight_power_when_sampling: float = LOSS_WEIGHT_POWER_WHEN_SAMPLING
    collapse_recall_threshold: float = COLLAPSE_RECALL_THRESHOLD
    collapse_predicted_count_threshold: int = COLLAPSE_PREDICTED_COUNT_THRESHOLD
    collapse_predicted_fraction_threshold: float = COLLAPSE_PREDICTED_FRACTION_THRESHOLD
    collapse_patience_epochs: int = COLLAPSE_PATIENCE_EPOCHS
    participant_balanced_sampling: bool = PARTICIPANT_BALANCED_SAMPLING
    auxiliary_target_loss_weight: float = AUXILIARY_TARGET_LOSS_WEIGHT
    warmup_epochs: int = WARMUP_EPOCHS

    @property
    def resolved_sampler_strategy(self) -> str:
        """Normalize legacy sampler knobs to one strategy string."""

        if self.sampler_strategy is not None:
            return self.sampler_strategy
        return self.sampler_weighting_mode if self.use_weighted_sampler else "none"

    @property
    def sampler_enabled(self) -> bool:
        """Return whether the training loop should construct a sampler."""

        return self.resolved_sampler_strategy != "none"


@dataclass(slots=True, frozen=True)
class ExperimentConfig:
    """Metadata describing one named experiment or one member of an experiment grid."""

    preset_name: str = "baseline_current"
    name: str = "baseline_current"
    group_name: str = field(default_factory=_default_experiment_group)
    run_id: str = field(default_factory=_default_run_id)
    output_root: Path = field(default_factory=_default_output_root)
    notes: str | None = None


@dataclass(slots=True, frozen=True)
class PathConfig:
    """Filesystem locations used throughout the training pipeline."""

    project_root: Path
    output_root: Path
    output_dir: Path
    group_dir: Path
    run_dir: Path
    checkpoints_dir: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    normalization_stats_path: Path
    split_manifest_path: Path
    dataset_summary_path: Path
    training_history_path: Path
    run_summary_path: Path
    class_weights_path: Path
    sampler_report_path: Path
    config_snapshot_path: Path
    log_path: Path
    validation_dir: Path
    validation_epochs_dir: Path
    test_dir: Path


@dataclass(slots=True, frozen=True)
class ProjectConfig:
    """Top-level configuration bundle."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathConfig = field(
        default_factory=lambda: _build_paths(ExperimentConfig())
    )

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.data.label_names)}

    @property
    def index_to_label(self) -> dict[int, str]:
        return {index: label for index, label in enumerate(self.data.label_names)}

    @property
    def source_label_to_index(self) -> dict[str, int]:
        return self.data.source_label_to_index

    @property
    def source_index_to_label(self) -> dict[int, str]:
        return self.data.source_index_to_label


EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {
    "baseline_current": {
        "experiment": {
            "notes": "Current baseline reproduced in config form for clean comparisons.",
        },
        "data": {
            "target_window_length": 256,
            "left_context": 256,
            "right_context": 256,
            "step": 128,
            "use_context_windows": True,
            "target_label_strategy": "purity_constrained_majority",
            "label_purity_threshold": 0.80,
        },
        "model": {
            "model_name": "cnn_bilstm_target_pool",
            "dropout": 0.40,
        },
        "training": {
            "batch_size": 128,
            "eval_batch_size": 256,
            "num_workers": 0,
            "loss_name": "balanced_softmax",
            "use_weighted_sampler": False,
            "sampler_strategy": "none",
            "class_weighting_mode": "sqrt_inverse_frequency",
            "scheduler_name": "reduce_on_plateau",
            "early_stopping_metric": "macro_f1",
        },
    },
    "baseline_current_ternary": {
        "_extends": "baseline_current",
        "experiment": {
            "notes": (
                "Current baseline with the existing split/window/context pipeline preserved, "
                "but remapped into Awake/Light/Deep through the config-driven label schema."
            ),
        },
        "data": {
            "label_schema_name": "ternary_sleep_depth",
            "label_mapping": {
                "W": "Awake",
                "N1": "Light",
                "N2": "Light",
                "N3": "Deep",
                "R": "Deep",
            },
        },
        "training": {
            "minority_labels": ("Light", "Deep"),
        },
    },
    "target_only_cnn_centered": {
        "experiment": {
            "notes": "Serious target-only baseline: simpler CNN, centered labels, no context, and participant-aware sampling.",
        },
        "data": {
            "target_label_strategy": "center_label",
            "center_label_span": 1,
            "use_context_windows": False,
            "left_context": 0,
            "right_context": 0,
        },
        "model": {
            "model_name": "cnn_baseline",
            "dropout": 0.25,
            "pool_context_region": False,
            "separate_context_regions": False,
            "use_target_indicator_channel": False,
        },
        "training": {
            "batch_size": 128,
            "eval_batch_size": 256,
            "num_workers": 0,
            "max_epochs": 24,
            "patience": 6,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "scheduler_name": "linear_warmup_cosine",
            "warmup_epochs": 2,
            "loss_name": "cross_entropy",
            "use_weighted_sampler": True,
            "sampler_strategy": "sqrt_inverse_frequency",
            "participant_balanced_sampling": True,
            "class_weighting_mode": "none",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "current_arch_centered_stabilized": {
        "experiment": {
            "notes": "Current CNN+BiLSTM architecture with centered labels and stabilized participant-aware training.",
        },
        "data": {
            "target_label_strategy": "center_label",
            "center_label_span": 1,
        },
        "model": {
            "model_name": "cnn_bilstm_target_pool",
            "dropout": 0.30,
            "pool_context_region": False,
            "separate_context_regions": False,
            "use_target_indicator_channel": False,
        },
        "training": {
            "batch_size": 128,
            "eval_batch_size": 256,
            "num_workers": 0,
            "max_epochs": 24,
            "patience": 6,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "scheduler_name": "linear_warmup_cosine",
            "warmup_epochs": 2,
            "loss_name": "cross_entropy",
            "use_weighted_sampler": True,
            "sampler_strategy": "sqrt_inverse_frequency",
            "participant_balanced_sampling": True,
            "class_weighting_mode": "none",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "target_only_bilstm_centered": {
        "experiment": {
            "notes": "Target-only BiLSTM ablation to test whether context itself is causing collapse.",
        },
        "data": {
            "target_label_strategy": "center_label",
            "center_label_span": 1,
            "use_context_windows": False,
            "left_context": 0,
            "right_context": 0,
        },
        "model": {
            "model_name": "cnn_bilstm_target_pool",
            "dropout": 0.30,
        },
        "training": {
            "batch_size": 128,
            "eval_batch_size": 256,
            "num_workers": 0,
            "max_epochs": 24,
            "patience": 6,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "scheduler_name": "linear_warmup_cosine",
            "warmup_epochs": 2,
            "loss_name": "cross_entropy",
            "use_weighted_sampler": True,
            "sampler_strategy": "sqrt_inverse_frequency",
            "participant_balanced_sampling": True,
            "class_weighting_mode": "none",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "context_gated_centered": {
        "experiment": {
            "notes": "Promising architecture: centered supervision plus target-first gated context fusion and participant-aware sampling.",
        },
        "data": {
            "target_label_strategy": "center_label",
            "center_label_span": 1,
            "use_context_windows": True,
            "left_context": 256,
            "right_context": 256,
        },
        "model": {
            "model_name": "cnn_bilstm_context_gated",
            "dropout": 0.30,
            "channel_dropout": 0.05,
        },
        "training": {
            "batch_size": 128,
            "eval_batch_size": 256,
            "num_workers": 0,
            "max_epochs": 24,
            "patience": 6,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "scheduler_name": "linear_warmup_cosine",
            "warmup_epochs": 2,
            "loss_name": "cross_entropy",
            "use_weighted_sampler": True,
            "sampler_strategy": "sqrt_inverse_frequency",
            "participant_balanced_sampling": True,
            "class_weighting_mode": "none",
            "auxiliary_target_loss_weight": 0.35,
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "sampler_plus_focal": {
        "experiment": {
            "notes": "Recommended first pass: bounded minority oversampling with tempered focal loss.",
        },
        "model": {
            "dropout": 0.30,
            "channel_dropout": 0.05,
        },
        "training": {
            "batch_size": 192,
            "eval_batch_size": 384,
            "num_workers": _default_num_workers(),
            "use_weighted_sampler": True,
            "sampler_strategy": "sqrt_inverse_frequency",
            "loss_name": "weighted_focal_loss",
            "class_weighting_mode": "sqrt_inverse_frequency",
            "loss_weight_power_when_sampling": 0.5,
            "sampler_max_weight_ratio": 4.0,
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "stricter_purity_center_label": {
        "data": {
            "target_label_strategy": "center_label",
            "center_label_span": 9,
            "label_purity_threshold": 0.90,
            "drop_ambiguous_windows": True,
        },
        "training": {
            "use_weighted_sampler": True,
            "sampler_strategy": "sqrt_inverse_frequency",
            "loss_name": "weighted_focal_loss",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "shorter_target_no_context": {
        "data": {
            "target_window_length": 128,
            "step": 64,
            "use_context_windows": False,
            "left_context": 0,
            "right_context": 0,
        },
        "training": {
            "use_weighted_sampler": True,
            "sampler_strategy": "sqrt_inverse_frequency",
            "loss_name": "weighted_focal_loss",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "shorter_target_with_context": {
        "data": {
            "target_window_length": 128,
            "step": 64,
            "use_context_windows": True,
            "left_context": 128,
            "right_context": 128,
        },
        "training": {
            "use_weighted_sampler": True,
            "sampler_strategy": "sqrt_inverse_frequency",
            "loss_name": "weighted_focal_loss",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "lower_dropout": {
        "model": {
            "dropout": 0.25,
            "channel_dropout": 0.05,
        },
    },
    "higher_weight_decay": {
        "training": {
            "weight_decay": 1e-3,
        },
    },
    "imbalance_a1_weighted_sampler_sqrt_focal": {
        "training": {
            "use_weighted_sampler": True,
            "sampler_strategy": "sqrt_inverse_frequency",
            "loss_name": "focal_loss",
            "class_weighting_mode": "none",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "imbalance_a2_weighted_sampler_capped_ce": {
        "training": {
            "use_weighted_sampler": True,
            "sampler_strategy": "capped_inverse_frequency",
            "sampler_max_weight_ratio": 4.0,
            "loss_name": "cross_entropy",
            "class_weighting_mode": "none",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "imbalance_a3_no_sampler_cb_focal": {
        "training": {
            "use_weighted_sampler": False,
            "sampler_strategy": "none",
            "loss_name": "class_balanced_focal_loss",
            "class_weighting_mode": "effective_number",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "imbalance_a4_no_sampler_weighted_focal": {
        "training": {
            "use_weighted_sampler": False,
            "sampler_strategy": "none",
            "loss_name": "weighted_focal_loss",
            "class_weighting_mode": "sqrt_inverse_frequency",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "imbalance_a5_weighted_sampler_balanced_softmax": {
        "training": {
            "use_weighted_sampler": True,
            "sampler_strategy": "capped_inverse_frequency",
            "sampler_max_weight_ratio": 2.5,
            "loss_name": "balanced_softmax",
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "label_b6_center_label_current": {
        "data": {
            "target_label_strategy": "center_label",
            "center_label_span": 9,
        },
    },
    "label_b7_purity_090": {
        "data": {
            "target_label_strategy": "purity_constrained_majority",
            "label_purity_threshold": 0.90,
        },
    },
    "label_b8_shorter_target_192": {
        "data": {
            "target_window_length": 192,
            "step": 96,
        },
    },
    "label_b9_step_64": {
        "data": {
            "step": 64,
        },
    },
    "context_c10_no_context": {
        "data": {
            "use_context_windows": False,
            "left_context": 0,
            "right_context": 0,
        },
    },
    "context_c11_reduced_context": {
        "data": {
            "use_context_windows": True,
            "left_context": 128,
            "right_context": 128,
        },
    },
    "context_c12_current_context": {
        "data": {
            "use_context_windows": True,
            "left_context": 256,
            "right_context": 256,
        },
    },
    "train_d13_lower_dropout": {
        "model": {
            "dropout": 0.25,
        },
    },
    "train_d14_higher_weight_decay": {
        "training": {
            "weight_decay": 1e-3,
        },
    },
    "train_d15_early_stop_minority": {
        "training": {
            "early_stopping_metric": "minority_macro_f1",
        },
    },
    "train_d16_cosine_scheduler": {
        "training": {
            "scheduler_name": "cosine_annealing",
            "scheduler_t_max": 12,
            "scheduler_eta_min": 1e-5,
        },
    },
}


def _default_config_payload() -> dict[str, Any]:
    """Return the default configuration as a nested dictionary."""

    experiment_payload = asdict(ExperimentConfig())
    experiment_payload["preset_name"] = ""
    experiment_payload["name"] = ""
    return {
        "data": asdict(DataConfig()),
        "model": asdict(ModelConfig()),
        "training": asdict(TrainingConfig()),
        "experiment": experiment_payload,
    }


def _resolve_preset_payload(
    preset_name: str,
    ancestry: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resolve one preset, optionally inheriting from another preset."""

    if preset_name in ancestry:
        cycle = " -> ".join([*ancestry, preset_name])
        raise ValueError(f"Preset inheritance cycle detected: {cycle}")
    if preset_name not in EXPERIMENT_PRESETS:
        known_presets = ", ".join(sorted(EXPERIMENT_PRESETS))
        raise ValueError(
            f"Unknown experiment preset '{preset_name}'. Known presets: {known_presets}"
        )

    preset_payload = dict(EXPERIMENT_PRESETS[preset_name])
    parent_preset_name = preset_payload.pop("_extends", None)
    if parent_preset_name is None:
        return preset_payload

    inherited_payload = _resolve_preset_payload(
        parent_preset_name,
        ancestry=(*ancestry, preset_name),
    )
    return _deep_merge(inherited_payload, preset_payload)


def _build_paths(experiment: ExperimentConfig) -> PathConfig:
    """Derive run-specific output locations for one experiment."""

    group_dir = experiment.output_root / "experiments" / experiment.group_name
    run_dir = group_dir / f"{experiment.name}_{experiment.run_id}"
    checkpoints_dir = run_dir / "checkpoints"
    validation_dir = run_dir / "validation"
    test_dir = run_dir / "test"

    return PathConfig(
        project_root=PROJECT_ROOT,
        output_root=experiment.output_root,
        output_dir=run_dir,
        group_dir=group_dir,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        latest_checkpoint_path=checkpoints_dir / "latest.pt",
        best_checkpoint_path=checkpoints_dir / "best.pt",
        normalization_stats_path=run_dir / "normalization_stats.json",
        split_manifest_path=run_dir / "split_manifest.json",
        dataset_summary_path=run_dir / "dataset_summary.json",
        training_history_path=run_dir / "training_history.json",
        run_summary_path=run_dir / "run_summary.json",
        class_weights_path=run_dir / "class_weights.json",
        sampler_report_path=run_dir / "sampler_report.json",
        config_snapshot_path=run_dir / "config_snapshot.json",
        log_path=run_dir / "train.log",
        validation_dir=validation_dir,
        validation_epochs_dir=validation_dir / "epochs",
        test_dir=test_dir,
    )


def _coerce_paths(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert selected string fields back to Path objects before dataclass construction."""

    payload["data"]["dataset_dir"] = Path(payload["data"]["dataset_dir"])
    payload["experiment"]["output_root"] = Path(payload["experiment"]["output_root"])
    return payload


def _normalize_payload(payload: dict[str, Any], preset_name: str) -> dict[str, Any]:
    """Apply compatibility and naming rules after preset and env overrides merge."""

    training_payload = payload["training"]
    data_payload = payload["data"]
    model_payload = payload["model"]
    experiment_payload = payload["experiment"]

    data_payload["source_label_names"] = tuple(data_payload["source_label_names"])
    data_payload["label_mapping"] = dict(data_payload["label_mapping"])
    data_payload["label_names"] = _derive_label_names(
        source_label_names=data_payload["source_label_names"],
        label_mapping=data_payload["label_mapping"],
    )
    model_payload["num_classes"] = len(data_payload["label_names"])
    training_payload["minority_labels"] = _remap_focus_labels(
        focus_labels=tuple(training_payload["minority_labels"]),
        label_mapping=data_payload["label_mapping"],
    )

    sampler_strategy = training_payload.get("sampler_strategy")
    if sampler_strategy is None:
        training_payload["sampler_strategy"] = (
            training_payload["sampler_weighting_mode"]
            if training_payload["use_weighted_sampler"]
            else "none"
        )
    training_payload["use_weighted_sampler"] = (
        training_payload["sampler_strategy"] != "none"
    )

    if data_payload["use_context_windows"] is False:
        data_payload["left_context"] = 0
        data_payload["right_context"] = 0

    experiment_payload.setdefault("preset_name", preset_name)
    experiment_payload.setdefault("name", preset_name)
    experiment_payload["name"] = experiment_payload["name"] or preset_name
    if not experiment_payload.get("run_id"):
        experiment_payload["run_id"] = _default_run_id()

    return payload


def _instantiate_config(payload: dict[str, Any]) -> ProjectConfig:
    """Build and validate a ProjectConfig from a merged configuration payload."""

    payload = _coerce_paths(payload)
    data = DataConfig(**payload["data"])
    model = ModelConfig(**payload["model"])
    training = TrainingConfig(**payload["training"])
    experiment = ExperimentConfig(**payload["experiment"])
    paths = _build_paths(experiment)
    config = ProjectConfig(
        data=data,
        model=model,
        training=training,
        experiment=experiment,
        paths=paths,
    )
    _validate_config(config)
    return config


def build_configs() -> list[ProjectConfig]:
    """Build one or more experiment configs from presets and optional JSON overrides."""

    requested_presets = _requested_preset_names()
    overrides = _load_json_overrides("DREAMT_CONFIG_OVERRIDES")
    group_name = os.getenv("DREAMT_EXPERIMENT_GROUP") or _default_experiment_group()
    built_configs: list[ProjectConfig] = []

    for preset_name in requested_presets:
        if preset_name not in EXPERIMENT_PRESETS:
            known_presets = ", ".join(sorted(EXPERIMENT_PRESETS))
            raise ValueError(
                f"Unknown experiment preset '{preset_name}'. Known presets: {known_presets}"
            )

        payload = _default_config_payload()
        payload = _deep_merge(payload, _resolve_preset_payload(preset_name))
        payload = _deep_merge(payload, overrides)
        payload["experiment"]["preset_name"] = preset_name
        payload["experiment"].setdefault("name", preset_name)
        payload["experiment"]["group_name"] = group_name
        payload = _normalize_payload(payload, preset_name=preset_name)
        built_configs.append(_instantiate_config(payload))

    return built_configs


def build_config() -> ProjectConfig:
    """Build the first project configuration for backward-compatible callers."""

    return build_configs()[0]


def _validate_probability_constraints(
    values: Mapping[str, float],
    label_names: tuple[str, ...],
    field_name: str,
) -> None:
    """Validate optional class-specific sampling floors and caps."""

    unknown_labels = sorted(set(values) - set(label_names))
    if unknown_labels:
        raise ValueError(
            f"{field_name} contains unknown labels: {', '.join(unknown_labels)}."
        )
    for label_name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name}[{label_name}] must be in the interval [0, 1].")


def _validate_config(config: ProjectConfig) -> None:
    """Validate one fully materialized project configuration."""

    if config.model.input_channels != len(config.data.feature_columns):
        raise ValueError(
            "Model input_channels must match the number of configured feature columns."
        )
    if config.model.num_classes != len(config.data.label_names):
        raise ValueError(
            "Model num_classes must match the number of configured label names."
        )
    if not config.data.label_schema_name:
        raise ValueError("label_schema_name must be a non-empty string.")
    if not config.data.source_label_names:
        raise ValueError("source_label_names must contain at least one dataset label.")
    if not config.data.label_names:
        raise ValueError("label_names must contain at least one supervised label.")
    if len(set(config.data.source_label_names)) != len(config.data.source_label_names):
        raise ValueError("source_label_names must be unique.")
    if len(set(config.data.label_names)) != len(config.data.label_names):
        raise ValueError("label_names must be unique.")
    if set(config.data.excluded_labels) & set(config.data.source_label_names):
        raise ValueError("excluded_labels cannot overlap source_label_names.")
    missing_mapping_keys = sorted(
        set(config.data.source_label_names) - set(config.data.label_mapping)
    )
    if missing_mapping_keys:
        raise ValueError(
            "label_mapping must define every source label. Missing: "
            + ", ".join(missing_mapping_keys)
        )
    unknown_mapping_keys = sorted(
        set(config.data.label_mapping) - set(config.data.source_label_names)
    )
    if unknown_mapping_keys:
        raise ValueError(
            "label_mapping contains unknown source labels: "
            + ", ".join(unknown_mapping_keys)
        )
    unknown_supervised_labels = sorted(
        set(config.data.label_mapping.values()) - set(config.data.label_names)
    )
    if unknown_supervised_labels:
        raise ValueError(
            "label_mapping contains unknown supervised labels: "
            + ", ".join(unknown_supervised_labels)
        )
    if set(config.data.label_names) != set(config.data.label_mapping.values()):
        raise ValueError(
            "label_names must exactly match the supervised labels produced by label_mapping."
        )
    if config.model.model_name not in {
        "cnn_baseline",
        "cnn_bilstm",
        "cnn_bilstm_target_pool",
        "cnn_bilstm_context_gated",
    }:
        raise ValueError(
            "model_name must be 'cnn_baseline', 'cnn_bilstm', "
            "'cnn_bilstm_target_pool', or 'cnn_bilstm_context_gated'."
        )
    if len(config.model.conv_channels) != len(config.model.kernel_sizes):
        raise ValueError("conv_channels and kernel_sizes must have the same length.")
    if len(config.model.conv_channels) != len(config.model.conv_dilations):
        raise ValueError("conv_channels and conv_dilations must have the same length.")
    if not config.model.conv_channels:
        raise ValueError("At least one convolutional stage is required.")
    if config.model.lstm_num_layers < 1:
        raise ValueError("lstm_num_layers must be at least 1.")
    if any(
        kernel_size < 3 or kernel_size % 2 == 0
        for kernel_size in config.model.kernel_sizes
    ):
        raise ValueError("kernel_sizes must be odd integers greater than or equal to 3.")
    if any(dilation < 1 for dilation in config.model.conv_dilations):
        raise ValueError("conv_dilations must be positive integers.")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model dropout must be in the interval [0, 1).")
    if not 0.0 <= config.model.feature_dropout < 1.0:
        raise ValueError("feature_dropout must be in the interval [0, 1).")
    if not 0.0 <= config.model.channel_dropout < 1.0:
        raise ValueError("channel_dropout must be in the interval [0, 1).")
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
    if config.data.center_label_span <= 0 or config.data.center_label_span % 2 == 0:
        raise ValueError("center_label_span must be a positive odd integer.")
    if config.data.center_label_span > config.data.target_window_length:
        raise ValueError("center_label_span cannot exceed target_window_length.")
    if config.data.target_label_strategy not in {
        "majority_vote",
        "center_label",
        "purity_constrained_majority",
    }:
        raise ValueError(
            "target_label_strategy must be one of: majority_vote, center_label, "
            "purity_constrained_majority."
        )
    if not 0.0 < config.data.min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in the interval (0, 1].")
    if config.data.continuity_gap_factor <= 0.0:
        raise ValueError("continuity_gap_factor must be positive.")
    if any(threshold <= 0 for threshold in config.data.transition_distance_thresholds):
        raise ValueError("transition_distance_thresholds must contain only positive integers.")

    if config.training.loss_name not in {
        "cross_entropy",
        "weighted_cross_entropy",
        "balanced_softmax",
        "focal_loss",
        "weighted_focal_loss",
        "class_balanced_focal_loss",
    }:
        raise ValueError(
            "loss_name must be one of: cross_entropy, weighted_cross_entropy, "
            "balanced_softmax, focal_loss, weighted_focal_loss, class_balanced_focal_loss."
        )
    if config.training.class_weighting_mode not in {
        "none",
        "inverse_frequency",
        "sqrt_inverse_frequency",
        "effective_number",
        "capped_inverse_frequency",
    }:
        raise ValueError(
            "class_weighting_mode must be one of: none, inverse_frequency, "
            "sqrt_inverse_frequency, effective_number, capped_inverse_frequency."
        )
    if config.training.resolved_sampler_strategy not in {
        "none",
        "inverse_frequency",
        "sqrt_inverse_frequency",
        "effective_number",
        "capped_inverse_frequency",
    }:
        raise ValueError(
            "sampler_strategy must be one of: none, inverse_frequency, "
            "sqrt_inverse_frequency, effective_number, capped_inverse_frequency."
        )
    if config.training.early_stopping_metric not in {
        "macro_f1",
        "balanced_accuracy",
        "minority_macro_f1",
    }:
        raise ValueError(
            "early_stopping_metric must be macro_f1, balanced_accuracy, or minority_macro_f1."
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
    if config.training.scheduler_name not in {
        "none",
        "reduce_on_plateau",
        "cosine_annealing",
        "linear_warmup_cosine",
    }:
        raise ValueError(
            "scheduler_name must be 'none', 'reduce_on_plateau', "
            "'cosine_annealing', or 'linear_warmup_cosine'."
        )
    if not 0.0 < config.training.scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must be in the interval (0, 1).")
    if config.training.scheduler_patience < 0:
        raise ValueError("scheduler_patience must be non-negative.")
    if config.training.scheduler_min_lr < 0.0:
        raise ValueError("scheduler_min_lr must be non-negative.")
    if config.training.scheduler_t_max <= 0:
        raise ValueError("scheduler_t_max must be positive.")
    if config.training.scheduler_eta_min < 0.0:
        raise ValueError("scheduler_eta_min must be non-negative.")
    if not 0.0 <= config.training.class_balance_beta < 1.0:
        raise ValueError("class_balance_beta must be in the interval [0, 1).")
    if config.training.validation_artifact_frequency <= 0:
        raise ValueError("validation_artifact_frequency must be positive.")
    if config.training.sampler_num_samples_multiplier <= 0.0:
        raise ValueError("sampler_num_samples_multiplier must be positive.")
    if config.training.sampler_max_weight_ratio is not None:
        if config.training.sampler_max_weight_ratio <= 0.0:
            raise ValueError("sampler_max_weight_ratio must be positive when provided.")
    if config.training.loss_max_weight_ratio is not None:
        if config.training.loss_max_weight_ratio <= 0.0:
            raise ValueError("loss_max_weight_ratio must be positive when provided.")
    if not 0.0 <= config.training.loss_weight_power_when_sampling <= 1.0:
        raise ValueError("loss_weight_power_when_sampling must be in the interval [0, 1].")
    if not 0.0 <= config.training.collapse_recall_threshold <= 1.0:
        raise ValueError("collapse_recall_threshold must be in the interval [0, 1].")
    if config.training.collapse_predicted_count_threshold < 0:
        raise ValueError("collapse_predicted_count_threshold must be non-negative.")
    if not 0.0 <= config.training.collapse_predicted_fraction_threshold <= 1.0:
        raise ValueError(
            "collapse_predicted_fraction_threshold must be in the interval [0, 1]."
        )
    if config.training.collapse_patience_epochs <= 0:
        raise ValueError("collapse_patience_epochs must be positive.")
    if config.training.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if config.training.warmup_epochs >= config.training.max_epochs:
        raise ValueError("warmup_epochs must be smaller than max_epochs.")
    if config.training.auxiliary_target_loss_weight < 0.0:
        raise ValueError("auxiliary_target_loss_weight must be non-negative.")
    if (
        config.training.focal_alpha is not None
        and len(config.training.focal_alpha) != config.model.num_classes
    ):
        raise ValueError("focal_alpha must provide one value per supervised class.")
    if set(config.training.minority_labels) - set(config.data.label_names):
        raise ValueError("minority_labels must be a subset of the configured label names.")

    _validate_probability_constraints(
        config.training.sampler_probability_caps,
        config.data.label_names,
        field_name="sampler_probability_caps",
    )
    _validate_probability_constraints(
        config.training.sampler_probability_floors,
        config.data.label_names,
        field_name="sampler_probability_floors",
    )
    if sum(config.training.sampler_probability_floors.values()) > 1.0 + 1e-6:
        raise ValueError("sampler_probability_floors cannot sum to more than 1.0.")
    for label_name, floor_value in config.training.sampler_probability_floors.items():
        cap_value = config.training.sampler_probability_caps.get(label_name)
        if cap_value is not None and floor_value > cap_value:
            raise ValueError(
                f"sampler_probability_floors[{label_name}] cannot exceed its cap."
            )
