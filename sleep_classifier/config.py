"""Configuration objects and CLI helpers for the sleep classifier."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "BVP",
    "IBI",
    "EDA",
    "TEMP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "HR",
)
DEFAULT_DERIVED_FEATURE_COLUMNS: tuple[str, ...] = (
    "ACC_MAG",
    "BVP_DELTA",
    "IBI_ROLLING_STD",
    "EDA_DELTA",
    "TEMP_DELTA",
)
DEFAULT_LABEL_COLUMN = "Sleep_Stage"
DEFAULT_TIMESTAMP_COLUMN = "TIMESTAMP"
DEFAULT_OUTPUT_DIR = Path("artifacts")
DEFAULT_DATASET_ROOT = Path("data_64Hz")
SUPPORTED_NORMALIZATION_MODES = {"global_train", "participant"}
SUPPORTED_LOSS_NAMES = {"weighted_cross_entropy", "focal"}
SUPPORTED_SCHEDULER_METRICS = {"macro_f1", "loss"}


@dataclass(slots=True)
class ExperimentConfig:
    """Central experiment configuration."""

    dataset_root: Path = DEFAULT_DATASET_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    derived_feature_columns: tuple[str, ...] = DEFAULT_DERIVED_FEATURE_COLUMNS
    use_derived_features: bool = True
    label_column: str = DEFAULT_LABEL_COLUMN
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    sample_rate_hz: int = 64
    center_epoch_seconds: int = 30
    context_window_seconds: int = 90
    train_stride_seconds: int = 15
    val_stride_seconds: int = 30
    test_stride_seconds: int = 30
    min_window_complete_fraction: float = 0.90
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 25
    num_workers: int = 0
    mixed_precision: bool = True
    early_stopping_patience: int = 7
    gradient_clip_norm: float = 1.0
    dropout: float = 0.30
    normalization_mode: str = "global_train"
    loss_name: str = "weighted_cross_entropy"
    focal_gamma: float = 2.0
    use_weighted_sampler: bool = False
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    lr_scheduler_metric: str = "macro_f1"
    limit_files: int | None = None
    use_cache: bool = True
    rebuild_cache: bool = False
    model_name: str = "multiscale_bilstm"
    model_base_channels: int = 64
    lstm_hidden_size: int = 128
    lstm_layers: int = 1
    attention_hidden_size: int = 128
    checkpoint_dirname: str = "checkpoints"
    report_dirname: str = "reports"
    plot_dirname: str = "plots"
    cache_dirname: str = "cache"
    cleaned_participant_dirname: str = "cleaned_participants"
    history_filename: str = "training_history.json"
    config_filename: str = "config_snapshot.json"
    split_filename: str = "participant_splits.json"
    participant_summary_filename: str = "participant_summaries.json"
    window_manifest_filename: str = "window_manifest.csv"
    normalization_filename: str = "normalization_stats.json"
    class_weights_filename: str = "class_weights.json"
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
    def center_window_size(self) -> int:
        return self.center_epoch_seconds * self.sample_rate_hz

    @property
    def context_window_size(self) -> int:
        return self.context_window_seconds * self.sample_rate_hz

    @property
    def left_context_seconds(self) -> int:
        return (self.context_window_seconds - self.center_epoch_seconds) // 2

    @property
    def right_context_seconds(self) -> int:
        return self.left_context_seconds

    @property
    def left_context_size(self) -> int:
        return self.left_context_seconds * self.sample_rate_hz

    @property
    def right_context_size(self) -> int:
        return self.right_context_seconds * self.sample_rate_hz

    @property
    def summary_stride_seconds(self) -> int:
        return self.center_epoch_seconds

    def stride_seconds_for_split(self, split_name: str) -> int:
        mapping = {
            "train": self.train_stride_seconds,
            "val": self.val_stride_seconds,
            "test": self.test_stride_seconds,
            "summary": self.summary_stride_seconds,
            "inference": self.test_stride_seconds,
        }
        try:
            return mapping[split_name]
        except KeyError as exc:
            raise KeyError(f"Unknown split name: {split_name}") from exc

    def stride_size_for_split(self, split_name: str) -> int:
        return self.stride_seconds_for_split(split_name) * self.sample_rate_hz

    @property
    def cache_signature(self) -> str:
        payload = {
            "dataset_root": str(self.dataset_root),
            "feature_columns": list(self.feature_columns),
            "derived_feature_columns": list(self.derived_feature_columns),
            "use_derived_features": self.use_derived_features,
            "sample_rate_hz": self.sample_rate_hz,
            "center_epoch_seconds": self.center_epoch_seconds,
            "context_window_seconds": self.context_window_seconds,
            "train_stride_seconds": self.train_stride_seconds,
            "val_stride_seconds": self.val_stride_seconds,
            "test_stride_seconds": self.test_stride_seconds,
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
    def dataset_summary_path(self) -> Path:
        return self.cache_dir / self.dataset_summary_filename

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_participants_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        ratios_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratios_sum - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, received {ratios_sum:.6f}.")
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive.")
        if self.center_epoch_seconds <= 0 or self.context_window_seconds <= 0:
            raise ValueError("Center epoch and context window must be positive.")
        if self.context_window_seconds < self.center_epoch_seconds:
            raise ValueError("Context window must be at least as long as the center epoch.")
        if (self.context_window_seconds - self.center_epoch_seconds) % 2 != 0:
            raise ValueError("Context window must leave equal left and right context in whole seconds.")
        if min(self.train_stride_seconds, self.val_stride_seconds, self.test_stride_seconds) <= 0:
            raise ValueError("All split strides must be positive.")
        if not 0.0 <= self.min_window_complete_fraction <= 1.0:
            raise ValueError("min_window_complete_fraction must be in [0, 1].")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
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
        if self.lr_scheduler_factor <= 0.0 or self.lr_scheduler_factor >= 1.0:
            raise ValueError("lr_scheduler_factor must be in (0, 1).")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative.")
        if self.lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_root"] = str(self.dataset_root)
        payload["output_dir"] = str(self.output_dir)
        payload["feature_columns"] = list(self.feature_columns)
        payload["derived_feature_columns"] = list(self.derived_feature_columns)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        data = dict(payload)
        legacy_window_seconds = data.pop("window_seconds", None)
        legacy_stride_seconds = data.pop("stride_seconds", None)
        if "dataset_root" in data:
            data["dataset_root"] = Path(data["dataset_root"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        if "feature_columns" in data:
            data["feature_columns"] = tuple(data["feature_columns"])
        if "derived_feature_columns" in data:
            data["derived_feature_columns"] = tuple(data["derived_feature_columns"])
        if legacy_window_seconds is not None and "context_window_seconds" not in data:
            data["context_window_seconds"] = int(legacy_window_seconds)
        if legacy_stride_seconds is not None:
            data.setdefault("train_stride_seconds", int(legacy_stride_seconds))
            data.setdefault("val_stride_seconds", int(legacy_stride_seconds))
            data.setdefault("test_stride_seconds", int(legacy_stride_seconds))
        return cls(**data)


def apply_common_overrides(config: ExperimentConfig, args: Any) -> ExperimentConfig:
    """Update a config object from parsed CLI arguments."""

    if getattr(args, "dataset_root", None) is not None:
        config.dataset_root = Path(args.dataset_root)
    if getattr(args, "output_dir", None) is not None:
        config.output_dir = Path(args.output_dir)
    if getattr(args, "window_seconds", None) is not None:
        config.context_window_seconds = int(args.window_seconds)
    if getattr(args, "context_window_seconds", None) is not None:
        config.context_window_seconds = int(args.context_window_seconds)
    if getattr(args, "center_epoch_seconds", None) is not None:
        config.center_epoch_seconds = int(args.center_epoch_seconds)
    if getattr(args, "stride_seconds", None) is not None:
        stride_seconds = int(args.stride_seconds)
        config.train_stride_seconds = stride_seconds
        config.val_stride_seconds = stride_seconds
        config.test_stride_seconds = stride_seconds
    if getattr(args, "train_stride_seconds", None) is not None:
        config.train_stride_seconds = int(args.train_stride_seconds)
    if getattr(args, "val_stride_seconds", None) is not None:
        config.val_stride_seconds = int(args.val_stride_seconds)
    if getattr(args, "test_stride_seconds", None) is not None:
        config.test_stride_seconds = int(args.test_stride_seconds)
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
    if getattr(args, "use_weighted_sampler", None) is not None:
        config.use_weighted_sampler = bool(args.use_weighted_sampler)
    if getattr(args, "use_cache", None) is not None:
        config.use_cache = bool(args.use_cache)
    if getattr(args, "rebuild_cache", None) is not None:
        config.rebuild_cache = bool(args.rebuild_cache)
    if getattr(args, "use_derived_features", None) is not None:
        config.use_derived_features = bool(args.use_derived_features)
    if getattr(args, "lr_scheduler_patience", None) is not None:
        config.lr_scheduler_patience = int(args.lr_scheduler_patience)
    if getattr(args, "lr_scheduler_factor", None) is not None:
        config.lr_scheduler_factor = float(args.lr_scheduler_factor)
    if getattr(args, "lr_scheduler_metric", None) is not None:
        config.lr_scheduler_metric = str(args.lr_scheduler_metric)
    return config
