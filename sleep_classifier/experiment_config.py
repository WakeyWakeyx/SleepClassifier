"""Configuration objects and CLI helpers for the sleep classifier."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_FEATURE_COLUMNS: tuple[str, ...] = (
    "BVP",
    "IBI",
    "EDA",
    "TEMP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "HR",
)
FILTERED_FEATURE_COLUMNS: tuple[str, ...] = (
    "TEMP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "HR",
)
SENSOR_TO_FEATURE_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("BVP", ("BVP",)),
    ("IBI", ("IBI",)),
    ("EDA", ("EDA",)),
    ("TEMP", ("TEMP",)),
    ("ACC", ("ACC_X", "ACC_Y", "ACC_Z")),
    ("HR", ("HR",)),
)
PARTICIPANT_CACHE_DIRNAME = "participants"
LEGACY_PARTICIPANT_CACHE_DIRNAME = "cleaned_participants"
DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = FILTERED_FEATURE_COLUMNS
DEFAULT_DERIVED_FEATURE_COLUMNS: tuple[str, ...] = ()
DEFAULT_LABEL_COLUMN = "Sleep_Stage"
DEFAULT_TIMESTAMP_COLUMN = "TIMESTAMP"
DEFAULT_OUTPUT_DIR = Path("artifacts")
DEFAULT_DATASET_ROOT = Path("data_64Hz")
SUPPORTED_NORMALIZATION_MODES = {"global_train", "participant"}
SUPPORTED_LOSS_NAMES = {"weighted_cross_entropy", "focal", "class_balanced_focal"}
SUPPORTED_SCHEDULER_METRICS = {"macro_f1", "loss"}
SUPPORTED_SEQUENCE_MODELS = {"bilstm", "transformer", "tcn"}
SUPPORTED_SMOOTHING_MODES = {"none", "median", "viterbi"}
SUPPORTED_MODEL_NAMES = {"epoch_sequence"}


@dataclass(slots=True)
class ExperimentConfig:
    """Central experiment configuration."""

    dataset_root: Path = DEFAULT_DATASET_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    cache_version: str = "epoch_sequence_v2"
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    derived_feature_columns: tuple[str, ...] = DEFAULT_DERIVED_FEATURE_COLUMNS
    use_derived_features: bool = False
    label_column: str = DEFAULT_LABEL_COLUMN
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    sample_rate_hz: int = 64
    center_epoch_seconds: int = 30
    context_length_epochs: int = 11
    train_stride_epochs: int = 1
    val_stride_epochs: int = 1
    test_stride_epochs: int = 1
    min_window_complete_fraction: float = 0.90
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    batch_size: int = 64
    learning_rate: float = 8e-4
    weight_decay: float = 5e-4
    epochs: int = 30
    num_workers: int = 0
    mixed_precision: bool = True
    early_stopping_patience: int = 8
    gradient_clip_norm: float = 1.0
    dropout: float = 0.40
    normalization_mode: str = "global_train"
    loss_name: str = "focal"
    focal_gamma: float = 2.0
    class_balanced_beta: float = 0.999
    label_smoothing: float = 0.05
    final_loss_weight: float = 1.0
    raw_loss_weight: float = 0.35
    use_weighted_sampler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    lr_scheduler_metric: str = "macro_f1"
    limit_files: int | None = None
    use_cache: bool = True
    rebuild_cache: bool = False
    model_name: str = "epoch_sequence"
    use_multi_branch: bool = True
    use_multitask_heads: bool = True
    epoch_embedding_dim: int = 192
    branch_embedding_dim: int = 64
    model_base_channels: int = 48
    sequence_model_name: str = "bilstm"
    sequence_hidden_size: int = 160
    sequence_layers: int = 1
    transformer_heads: int = 4
    transformer_ff_multiplier: int = 4
    tcn_kernel_size: int = 3
    smoothing_mode: str = "viterbi"
    median_filter_size: int = 5
    time_mask_prob: float = 0.30
    time_mask_max_fraction: float = 0.15
    channel_dropout_prob: float = 0.12
    checkpoint_dirname: str = "checkpoints"
    report_dirname: str = "reports"
    plot_dirname: str = "plots"
    cache_dirname: str = "cache"
    cleaned_participant_dirname: str = PARTICIPANT_CACHE_DIRNAME
    history_filename: str = "training_history.json"
    config_filename: str = "config_snapshot.json"
    split_filename: str = "participant_splits.json"
    participant_summary_filename: str = "participant_summaries.json"
    window_manifest_filename: str = "window_manifest.csv"
    normalization_filename: str = "normalization_stats.json"
    class_weights_filename: str = "class_weights.json"
    transition_matrix_filename: str = "transition_stats.json"
    dataset_summary_filename: str = "dataset_summary.json"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.timestamp_column,
            *self.feature_columns,
            self.label_column,
        )

    @property
    def input_feature_columns(self) -> tuple[str, ...]:
        if not self.use_derived_features:
            return self.feature_columns
        return self.feature_columns + self.derived_feature_columns

    @property
    def canonical_feature_order(self) -> tuple[str, ...]:
        active_features = set(self.feature_columns)
        return tuple(feature_name for feature_name in SUPPORTED_FEATURE_COLUMNS if feature_name in active_features)

    @property
    def filtered_sensor_names(self) -> tuple[str, ...]:
        active_features = set(self.feature_columns)
        return tuple(
            sensor_name
            for sensor_name, sensor_features in SENSOR_TO_FEATURE_COLUMNS
            if any(feature_name in active_features for feature_name in sensor_features)
        )

    @property
    def center_window_size(self) -> int:
        return self.center_epoch_seconds * self.sample_rate_hz

    @property
    def context_window_size(self) -> int:
        return self.context_length_epochs * self.center_window_size

    @property
    def context_window_seconds(self) -> int:
        return self.context_length_epochs * self.center_epoch_seconds

    @property
    def left_context_epochs(self) -> int:
        return self.context_length_epochs // 2

    @property
    def right_context_epochs(self) -> int:
        return self.left_context_epochs

    @property
    def left_context_seconds(self) -> int:
        return self.left_context_epochs * self.center_epoch_seconds

    @property
    def right_context_seconds(self) -> int:
        return self.right_context_epochs * self.center_epoch_seconds

    @property
    def left_context_size(self) -> int:
        return self.left_context_seconds * self.sample_rate_hz

    @property
    def right_context_size(self) -> int:
        return self.right_context_seconds * self.sample_rate_hz

    @property
    def summary_stride_epochs(self) -> int:
        return 1

    @property
    def train_stride_seconds(self) -> int:
        return self.train_stride_epochs * self.center_epoch_seconds

    @property
    def val_stride_seconds(self) -> int:
        return self.val_stride_epochs * self.center_epoch_seconds

    @property
    def test_stride_seconds(self) -> int:
        return self.test_stride_epochs * self.center_epoch_seconds

    @property
    def summary_stride_seconds(self) -> int:
        return self.summary_stride_epochs * self.center_epoch_seconds

    def stride_epochs_for_split(self, split_name: str) -> int:
        mapping = {
            "train": self.train_stride_epochs,
            "val": self.val_stride_epochs,
            "test": self.test_stride_epochs,
            "summary": self.summary_stride_epochs,
            "inference": self.test_stride_epochs,
        }
        try:
            return mapping[split_name]
        except KeyError as exc:
            raise KeyError(f"Unknown split name: {split_name}") from exc

    def stride_seconds_for_split(self, split_name: str) -> int:
        return self.stride_epochs_for_split(split_name) * self.center_epoch_seconds

    def stride_size_for_split(self, split_name: str) -> int:
        return self.stride_seconds_for_split(split_name) * self.sample_rate_hz

    @property
    def cache_signature(self) -> str:
        payload = {
            "cache_version": self.cache_version,
            "dataset_root": str(self.dataset_root),
            "feature_columns": list(self.feature_columns),
            "derived_feature_columns": list(self.derived_feature_columns),
            "use_derived_features": self.use_derived_features,
            "sample_rate_hz": self.sample_rate_hz,
            "center_epoch_seconds": self.center_epoch_seconds,
            "context_length_epochs": self.context_length_epochs,
            "train_stride_epochs": self.train_stride_epochs,
            "val_stride_epochs": self.val_stride_epochs,
            "test_stride_epochs": self.test_stride_epochs,
            "min_window_complete_fraction": self.min_window_complete_fraction,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "limit_files": self.limit_files,
            "normalization_mode": self.normalization_mode,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:12]

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / self.checkpoint_dirname

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / self.report_dirname

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / self.plot_dirname

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / self.cache_dirname / self.cache_signature

    @property
    def cleaned_participants_dir(self) -> Path:
        return self.cache_dir / self.cleaned_participant_dirname

    @property
    def participant_cache_dir(self) -> Path:
        return self.cleaned_participants_dir

    @property
    def legacy_cleaned_participants_dir(self) -> Path:
        return self.cache_dir / LEGACY_PARTICIPANT_CACHE_DIRNAME

    @property
    def history_path(self) -> Path:
        return self.output_dir / self.history_filename

    @property
    def config_path(self) -> Path:
        return self.output_dir / self.config_filename

    @property
    def split_path(self) -> Path:
        return self.cache_dir / self.split_filename

    @property
    def participant_summary_path(self) -> Path:
        return self.cache_dir / self.participant_summary_filename

    @property
    def window_manifest_path(self) -> Path:
        return self.cache_dir / self.window_manifest_filename

    @property
    def normalization_path(self) -> Path:
        return self.cache_dir / self.normalization_filename

    @property
    def class_weights_path(self) -> Path:
        return self.cache_dir / self.class_weights_filename

    @property
    def transition_matrix_path(self) -> Path:
        return self.cache_dir / self.transition_matrix_filename

    @property
    def dataset_summary_path(self) -> Path:
        return self.cache_dir / self.dataset_summary_filename

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.participant_cache_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        ratios_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratios_sum - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, received {ratios_sum:.6f}.")
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive.")
        if self.center_epoch_seconds <= 0:
            raise ValueError("center_epoch_seconds must be positive.")
        if self.context_length_epochs <= 0 or self.context_length_epochs % 2 == 0:
            raise ValueError("context_length_epochs must be a positive odd integer.")
        if min(self.train_stride_epochs, self.val_stride_epochs, self.test_stride_epochs) <= 0:
            raise ValueError("All split strides must be positive.")
        if not 0.0 <= self.min_window_complete_fraction <= 1.0:
            raise ValueError("min_window_complete_fraction must be in [0, 1].")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if not self.feature_columns:
            raise ValueError("At least one input feature column must be configured.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must not contain duplicates.")
        unsupported_feature_columns = [
            feature_name for feature_name in self.feature_columns if feature_name not in SUPPORTED_FEATURE_COLUMNS
        ]
        if unsupported_feature_columns:
            raise ValueError(
                "Epoch-sequence training only supports wearable features from "
                f"{list(SUPPORTED_FEATURE_COLUMNS)}; received unsupported features "
                f"{unsupported_feature_columns}."
            )
        if tuple(self.feature_columns) != self.canonical_feature_order:
            raise ValueError(
                "feature_columns must be a non-empty ordered subset of "
                f"{list(SUPPORTED_FEATURE_COLUMNS)}; received {list(self.feature_columns)}."
            )
        if self.use_derived_features:
            raise ValueError(
                "Derived features are disabled for the epoch-sequence pipeline; "
                f"configure only base wearable columns from {list(SUPPORTED_FEATURE_COLUMNS)}."
            )
        if self.normalization_mode not in SUPPORTED_NORMALIZATION_MODES:
            raise ValueError(
                f"Unsupported normalization mode '{self.normalization_mode}'. "
                f"Expected one of {sorted(SUPPORTED_NORMALIZATION_MODES)}."
            )
        if self.loss_name not in SUPPORTED_LOSS_NAMES:
            raise ValueError(
                f"Unsupported loss '{self.loss_name}'. Expected one of {sorted(SUPPORTED_LOSS_NAMES)}."
            )
        if self.lr_scheduler_metric not in SUPPORTED_SCHEDULER_METRICS:
            raise ValueError(
                f"Unsupported scheduler metric '{self.lr_scheduler_metric}'. "
                f"Expected one of {sorted(SUPPORTED_SCHEDULER_METRICS)}."
            )
        if self.lr_scheduler_patience < 0:
            raise ValueError("lr_scheduler_patience cannot be negative.")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("lr_scheduler_factor must be in (0, 1).")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.sequence_model_name not in SUPPORTED_SEQUENCE_MODELS:
            raise ValueError(
                f"Unsupported sequence model '{self.sequence_model_name}'. "
                f"Expected one of {sorted(SUPPORTED_SEQUENCE_MODELS)}."
            )
        if self.smoothing_mode not in SUPPORTED_SMOOTHING_MODES:
            raise ValueError(
                f"Unsupported smoothing mode '{self.smoothing_mode}'. "
                f"Expected one of {sorted(SUPPORTED_SMOOTHING_MODES)}."
            )
        if self.model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Unsupported model '{self.model_name}'. Expected one of {sorted(SUPPORTED_MODEL_NAMES)}."
            )
        if self.sequence_layers <= 0:
            raise ValueError("sequence_layers must be positive.")
        if self.transformer_heads <= 0:
            raise ValueError("transformer_heads must be positive.")
        if self.epoch_embedding_dim <= 0 or self.branch_embedding_dim <= 0 or self.model_base_channels <= 0:
            raise ValueError("Embedding and channel sizes must be positive.")
        if not 0.0 <= self.time_mask_prob <= 1.0:
            raise ValueError("time_mask_prob must be in [0, 1].")
        if not 0.0 <= self.channel_dropout_prob <= 1.0:
            raise ValueError("channel_dropout_prob must be in [0, 1].")
        if not 0.0 <= self.time_mask_max_fraction < 1.0:
            raise ValueError("time_mask_max_fraction must be in [0, 1).")
        if self.median_filter_size <= 0 or self.median_filter_size % 2 == 0:
            raise ValueError("median_filter_size must be a positive odd integer.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_root"] = str(self.dataset_root)
        payload["output_dir"] = str(self.output_dir)
        payload["feature_columns"] = list(self.feature_columns)
        payload["derived_feature_columns"] = list(self.derived_feature_columns)
        payload["context_window_seconds"] = self.context_window_seconds
        payload["train_stride_seconds"] = self.train_stride_seconds
        payload["val_stride_seconds"] = self.val_stride_seconds
        payload["test_stride_seconds"] = self.test_stride_seconds
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        data = dict(payload)
        legacy_window_seconds = data.pop("window_seconds", None)
        legacy_stride_seconds = data.pop("stride_seconds", None)
        context_window_seconds = data.pop("context_window_seconds", None)
        train_stride_seconds = data.pop("train_stride_seconds", None)
        val_stride_seconds = data.pop("val_stride_seconds", None)
        test_stride_seconds = data.pop("test_stride_seconds", None)
        if "dataset_root" in data:
            data["dataset_root"] = Path(data["dataset_root"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        if "feature_columns" in data:
            data["feature_columns"] = tuple(data["feature_columns"])
        if "derived_feature_columns" in data:
            data["derived_feature_columns"] = tuple(data["derived_feature_columns"])
        center_epoch_seconds = int(data.get("center_epoch_seconds", 30))

        legacy_context_seconds = context_window_seconds if context_window_seconds is not None else legacy_window_seconds
        if legacy_context_seconds is not None and "context_length_epochs" not in data:
            data["context_length_epochs"] = max(1, int(legacy_context_seconds) // center_epoch_seconds)

        if legacy_stride_seconds is not None:
            stride_epochs = max(1, int(legacy_stride_seconds) // center_epoch_seconds)
            data.setdefault("train_stride_epochs", stride_epochs)
            data.setdefault("val_stride_epochs", stride_epochs)
            data.setdefault("test_stride_epochs", stride_epochs)

        if train_stride_seconds is not None:
            data["train_stride_epochs"] = max(1, int(train_stride_seconds) // center_epoch_seconds)
        if val_stride_seconds is not None:
            data["val_stride_epochs"] = max(1, int(val_stride_seconds) // center_epoch_seconds)
        if test_stride_seconds is not None:
            data["test_stride_epochs"] = max(1, int(test_stride_seconds) // center_epoch_seconds)

        if "lstm_hidden_size" in data and "sequence_hidden_size" not in data:
            data["sequence_hidden_size"] = int(data.pop("lstm_hidden_size"))
        else:
            data.pop("lstm_hidden_size", None)
        if "lstm_layers" in data and "sequence_layers" not in data:
            data["sequence_layers"] = int(data.pop("lstm_layers"))
        else:
            data.pop("lstm_layers", None)
        data.pop("attention_hidden_size", None)

        return cls(**data)


def _apply_stride_seconds_override(config: ExperimentConfig, attribute_prefix: str, stride_seconds: int) -> None:
    stride_epochs = max(1, int(stride_seconds) // config.center_epoch_seconds)
    setattr(config, f"{attribute_prefix}_stride_epochs", stride_epochs)


def apply_common_overrides(config: ExperimentConfig, args: Any) -> ExperimentConfig:
    """Update a config object from parsed CLI arguments."""

    if getattr(args, "dataset_root", None) is not None:
        config.dataset_root = Path(args.dataset_root)
    if getattr(args, "output_dir", None) is not None:
        config.output_dir = Path(args.output_dir)
    if getattr(args, "center_epoch_seconds", None) is not None:
        config.center_epoch_seconds = int(args.center_epoch_seconds)
    if getattr(args, "context_length_epochs", None) is not None:
        config.context_length_epochs = int(args.context_length_epochs)
    if getattr(args, "window_seconds", None) is not None:
        config.context_length_epochs = max(1, int(args.window_seconds) // config.center_epoch_seconds)
    if getattr(args, "context_window_seconds", None) is not None:
        config.context_length_epochs = max(1, int(args.context_window_seconds) // config.center_epoch_seconds)
    if getattr(args, "stride_seconds", None) is not None:
        _apply_stride_seconds_override(config, "train", int(args.stride_seconds))
        _apply_stride_seconds_override(config, "val", int(args.stride_seconds))
        _apply_stride_seconds_override(config, "test", int(args.stride_seconds))
    if getattr(args, "train_stride_seconds", None) is not None:
        _apply_stride_seconds_override(config, "train", int(args.train_stride_seconds))
    if getattr(args, "val_stride_seconds", None) is not None:
        _apply_stride_seconds_override(config, "val", int(args.val_stride_seconds))
    if getattr(args, "test_stride_seconds", None) is not None:
        _apply_stride_seconds_override(config, "test", int(args.test_stride_seconds))
    if getattr(args, "train_stride_epochs", None) is not None:
        config.train_stride_epochs = int(args.train_stride_epochs)
    if getattr(args, "val_stride_epochs", None) is not None:
        config.val_stride_epochs = int(args.val_stride_epochs)
    if getattr(args, "test_stride_epochs", None) is not None:
        config.test_stride_epochs = int(args.test_stride_epochs)
    if getattr(args, "batch_size", None) is not None:
        config.batch_size = int(args.batch_size)
    if getattr(args, "learning_rate", None) is not None:
        config.learning_rate = float(args.learning_rate)
    if getattr(args, "weight_decay", None) is not None:
        config.weight_decay = float(args.weight_decay)
    if getattr(args, "epochs", None) is not None:
        config.epochs = int(args.epochs)
    if getattr(args, "num_workers", None) is not None:
        config.num_workers = int(args.num_workers)
    if getattr(args, "limit_files", None) is not None:
        config.limit_files = int(args.limit_files)
    if getattr(args, "random_seed", None) is not None:
        config.random_seed = int(args.random_seed)
    if getattr(args, "early_stopping_patience", None) is not None:
        config.early_stopping_patience = int(args.early_stopping_patience)
    if getattr(args, "gradient_clip_norm", None) is not None:
        config.gradient_clip_norm = float(args.gradient_clip_norm)
    if getattr(args, "min_window_complete_fraction", None) is not None:
        config.min_window_complete_fraction = float(args.min_window_complete_fraction)
    if getattr(args, "dropout", None) is not None:
        config.dropout = float(args.dropout)
    if getattr(args, "mixed_precision", None) is not None:
        config.mixed_precision = bool(args.mixed_precision)
    if getattr(args, "normalization_mode", None) is not None:
        config.normalization_mode = str(args.normalization_mode)
    if getattr(args, "loss_name", None) is not None:
        config.loss_name = str(args.loss_name)
    if getattr(args, "focal_gamma", None) is not None:
        config.focal_gamma = float(args.focal_gamma)
    if getattr(args, "class_balanced_beta", None) is not None:
        config.class_balanced_beta = float(args.class_balanced_beta)
    if getattr(args, "label_smoothing", None) is not None:
        config.label_smoothing = float(args.label_smoothing)
    if getattr(args, "final_loss_weight", None) is not None:
        config.final_loss_weight = float(args.final_loss_weight)
    if getattr(args, "raw_loss_weight", None) is not None:
        config.raw_loss_weight = float(args.raw_loss_weight)
    if getattr(args, "use_weighted_sampler", None) is not None:
        config.use_weighted_sampler = bool(args.use_weighted_sampler)
    if getattr(args, "use_cache", None) is not None:
        config.use_cache = bool(args.use_cache)
    if getattr(args, "rebuild_cache", None) is not None:
        config.rebuild_cache = bool(args.rebuild_cache)
    if getattr(args, "use_derived_features", None) is not None:
        config.use_derived_features = bool(args.use_derived_features)
    if getattr(args, "use_multi_branch", None) is not None:
        config.use_multi_branch = bool(args.use_multi_branch)
    if getattr(args, "use_multitask_heads", None) is not None:
        config.use_multitask_heads = bool(args.use_multitask_heads)
    if getattr(args, "lr_scheduler_patience", None) is not None:
        config.lr_scheduler_patience = int(args.lr_scheduler_patience)
    if getattr(args, "lr_scheduler_factor", None) is not None:
        config.lr_scheduler_factor = float(args.lr_scheduler_factor)
    if getattr(args, "lr_scheduler_metric", None) is not None:
        config.lr_scheduler_metric = str(args.lr_scheduler_metric)
    if getattr(args, "sequence_model_name", None) is not None:
        config.sequence_model_name = str(args.sequence_model_name)
    if getattr(args, "sequence_hidden_size", None) is not None:
        config.sequence_hidden_size = int(args.sequence_hidden_size)
    if getattr(args, "sequence_layers", None) is not None:
        config.sequence_layers = int(args.sequence_layers)
    if getattr(args, "transformer_heads", None) is not None:
        config.transformer_heads = int(args.transformer_heads)
    if getattr(args, "epoch_embedding_dim", None) is not None:
        config.epoch_embedding_dim = int(args.epoch_embedding_dim)
    if getattr(args, "branch_embedding_dim", None) is not None:
        config.branch_embedding_dim = int(args.branch_embedding_dim)
    if getattr(args, "model_base_channels", None) is not None:
        config.model_base_channels = int(args.model_base_channels)
    if getattr(args, "smoothing_mode", None) is not None:
        config.smoothing_mode = str(args.smoothing_mode)
    if getattr(args, "median_filter_size", None) is not None:
        config.median_filter_size = int(args.median_filter_size)
    if getattr(args, "time_mask_prob", None) is not None:
        config.time_mask_prob = float(args.time_mask_prob)
    if getattr(args, "time_mask_max_fraction", None) is not None:
        config.time_mask_max_fraction = float(args.time_mask_max_fraction)
    if getattr(args, "channel_dropout_prob", None) is not None:
        config.channel_dropout_prob = float(args.channel_dropout_prob)
    return config
