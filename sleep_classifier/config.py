"""Configuration objects and CLI helpers for the sleep classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

DEFAULT_LABEL_COLUMN = "Sleep_Stage"
DEFAULT_TIMESTAMP_COLUMN = "TIMESTAMP"
DEFAULT_OUTPUT_DIR = Path("artifacts")
DEFAULT_DATASET_ROOT = Path("data_64Hz")


@dataclass(slots=True)
class ExperimentConfig:
    """Central experiment configuration."""

    dataset_root: Path = DEFAULT_DATASET_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    label_column: str = DEFAULT_LABEL_COLUMN
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    sample_rate_hz: int = 64
    window_seconds: int = 30
    stride_seconds: int = 30
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
    limit_files: int | None = None
    checkpoint_dirname: str = "checkpoints"
    report_dirname: str = "reports"
    plot_dirname: str = "plots"
    history_filename: str = "training_history.json"
    config_filename: str = "config_snapshot.json"
    split_filename: str = "participant_splits.json"
    manifest_filename: str = "dataset_manifest.json"
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
    def window_size(self) -> int:
        return self.window_seconds * self.sample_rate_hz

    @property
    def stride_size(self) -> int:
        return self.stride_seconds * self.sample_rate_hz

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
    def history_path(self) -> Path:
        return self.output_dir / self.history_filename

    @property
    def config_path(self) -> Path:
        return self.output_dir / self.config_filename

    @property
    def split_path(self) -> Path:
        return self.output_dir / self.split_filename

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / self.manifest_filename

    @property
    def normalization_path(self) -> Path:
        return self.output_dir / self.normalization_filename

    @property
    def class_weights_path(self) -> Path:
        return self.output_dir / self.class_weights_filename

    @property
    def dataset_summary_path(self) -> Path:
        return self.output_dir / self.dataset_summary_filename

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        ratios_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratios_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, received {ratios_sum:.6f}."
            )
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("Window and stride seconds must be positive.")
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive.")
        if not 0.0 <= self.min_window_complete_fraction <= 1.0:
            raise ValueError("min_window_complete_fraction must be in [0, 1].")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_root"] = str(self.dataset_root)
        payload["output_dir"] = str(self.output_dir)
        payload["feature_columns"] = list(self.feature_columns)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        data = dict(payload)
        if "dataset_root" in data:
            data["dataset_root"] = Path(data["dataset_root"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        if "feature_columns" in data:
            data["feature_columns"] = tuple(data["feature_columns"])
        return cls(**data)


def apply_common_overrides(config: ExperimentConfig, args: Any) -> ExperimentConfig:
    """Update a config object from parsed CLI arguments."""

    if getattr(args, "dataset_root", None) is not None:
        config.dataset_root = Path(args.dataset_root)
    if getattr(args, "output_dir", None) is not None:
        config.output_dir = Path(args.output_dir)
    if getattr(args, "window_seconds", None) is not None:
        config.window_seconds = int(args.window_seconds)
    if getattr(args, "stride_seconds", None) is not None:
        config.stride_seconds = int(args.stride_seconds)
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
    return config
