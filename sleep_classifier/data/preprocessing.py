"""CSV cleaning, caching, and context-window utilities for DREAMT wearable data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleep_classifier.config import ExperimentConfig
from sleep_classifier.label_mapping import ID_TO_NAME, map_sleep_stage


@dataclass(slots=True)
class CachedParticipantData:
    """Cleaned participant arrays and row-level accounting."""

    participant_id: str
    file_path: Path
    feature_columns: tuple[str, ...]
    timestamps: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    row_complete_before_fill: np.ndarray
    raw_row_count: int
    clean_row_count: int
    dropped_invalid_label_rows: int
    dropped_invalid_timestamp_rows: int
    dropped_all_feature_missing_rows: int
    dropped_duplicate_timestamp_rows: int
    dropped_remaining_nan_rows: int


@dataclass(slots=True)
class ParticipantSummary:
    """Participant-level summary for split generation and reporting."""

    participant_id: str
    file_path: Path
    raw_row_count: int
    clean_row_count: int
    window_count: int
    dominant_class_id: int | None
    dominant_class_name: str | None
    window_class_counts: dict[str, int]
    dropped_invalid_label_rows: int
    dropped_invalid_timestamp_rows: int
    dropped_all_feature_missing_rows: int
    dropped_duplicate_timestamp_rows: int
    dropped_remaining_nan_rows: int


@dataclass(slots=True)
class WindowedParticipantData:
    """Compatibility container for direct participant window extraction."""

    features: np.ndarray | None
    labels: np.ndarray
    start_timestamps: np.ndarray
    end_timestamps: np.ndarray
    skipped_low_quality_windows: int
    skipped_short_windows: int


def _coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _build_derived_feature_map(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, pd.Series]:
    rolling_window = max(config.sample_rate_hz * 5, 5)
    acc_mag = np.sqrt(
        np.square(frame["ACC_X"].to_numpy(dtype=np.float32))
        + np.square(frame["ACC_Y"].to_numpy(dtype=np.float32))
        + np.square(frame["ACC_Z"].to_numpy(dtype=np.float32))
    )
    return {
        "ACC_MAG": pd.Series(acc_mag, index=frame.index, dtype="float32"),
        "BVP_DELTA": frame["BVP"].diff().fillna(0.0).astype("float32"),
        "IBI_ROLLING_STD": frame["IBI"]
        .rolling(window=rolling_window, min_periods=1)
        .std(ddof=0)
        .fillna(0.0)
        .astype("float32"),
        "EDA_DELTA": frame["EDA"]
        .diff()
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .fillna(0.0)
        .astype("float32"),
        "TEMP_DELTA": frame["TEMP"]
        .diff()
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .fillna(0.0)
        .astype("float32"),
    }


def load_and_clean_participant_csv(
    file_path: Path,
    participant_id: str,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> CachedParticipantData | None:
    """Load, validate, clean, and featurize a participant CSV."""

    try:
        frame = pd.read_csv(file_path, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive runtime protection
        logger.warning("Skipping %s because it could not be read: %s", file_path.name, exc)
        return None

    missing_columns = [column for column in config.required_columns if column not in frame.columns]
    if missing_columns:
        logger.warning(
            "Skipping %s because required columns are missing: %s",
            file_path.name,
            ", ".join(missing_columns),
        )
        return None

    raw_row_count = len(frame)
    base_feature_columns = list(config.feature_columns)
    numeric_columns = [config.timestamp_column, *base_feature_columns]
    _coerce_numeric_columns(frame, numeric_columns)
    frame[config.label_column] = frame[config.label_column].astype("string").str.strip()

    mapped_labels = frame[config.label_column].apply(map_sleep_stage)
    valid_label_mask = mapped_labels.notna()
    dropped_invalid_label_rows = int((~valid_label_mask).sum())
    frame = frame.loc[valid_label_mask].copy()
    mapped_labels = mapped_labels.loc[valid_label_mask]

    valid_timestamp_mask = frame[config.timestamp_column].notna()
    dropped_invalid_timestamp_rows = int((~valid_timestamp_mask).sum())
    frame = frame.loc[valid_timestamp_mask].copy()
    mapped_labels = mapped_labels.loc[valid_timestamp_mask]

    all_features_missing_mask = frame[base_feature_columns].isna().all(axis=1)
    dropped_all_feature_missing_rows = int(all_features_missing_mask.sum())
    frame = frame.loc[~all_features_missing_mask].copy()
    mapped_labels = mapped_labels.loc[~all_features_missing_mask]

    raw_complete_mask = frame[base_feature_columns].notna().all(axis=1).astype(np.float32)
    frame["label_id"] = mapped_labels.astype(np.int64)
    frame["row_complete_before_fill"] = raw_complete_mask.to_numpy(dtype=np.float32)

    frame = frame.sort_values(config.timestamp_column, kind="mergesort")
    before_duplicates = len(frame)
    frame = frame.drop_duplicates(subset=[config.timestamp_column], keep="first").copy()
    dropped_duplicate_timestamp_rows = before_duplicates - len(frame)

    frame[base_feature_columns] = frame[base_feature_columns].ffill().bfill()
    remaining_nan_mask = frame[base_feature_columns].isna().any(axis=1)
    dropped_remaining_nan_rows = int(remaining_nan_mask.sum())
    frame = frame.loc[~remaining_nan_mask].copy()
    frame = frame.reset_index(drop=True)

    if config.use_derived_features:
        derived_feature_map = _build_derived_feature_map(frame, config)
        for feature_name in config.derived_feature_columns:
            if feature_name not in derived_feature_map:
                raise ValueError(f"Unsupported derived feature requested: {feature_name}")
            frame[feature_name] = derived_feature_map[feature_name]

    clean_row_count = len(frame)
    feature_columns = list(config.input_feature_columns)
    logger.info(
        "Participant %s: rows %d -> %d after cleaning (labels=%d, timestamps=%d, all-features-missing=%d, duplicates=%d, remaining-nan=%d).",
        participant_id,
        raw_row_count,
        clean_row_count,
        dropped_invalid_label_rows,
        dropped_invalid_timestamp_rows,
        dropped_all_feature_missing_rows,
        dropped_duplicate_timestamp_rows,
        dropped_remaining_nan_rows,
    )

    return CachedParticipantData(
        participant_id=participant_id,
        file_path=file_path,
        feature_columns=tuple(feature_columns),
        timestamps=frame[config.timestamp_column].to_numpy(dtype=np.float64, copy=True),
        features=frame[feature_columns].to_numpy(dtype=np.float32, copy=True),
        labels=frame["label_id"].to_numpy(dtype=np.int64, copy=True),
        row_complete_before_fill=frame["row_complete_before_fill"].to_numpy(dtype=np.float32, copy=True),
        raw_row_count=raw_row_count,
        clean_row_count=clean_row_count,
        dropped_invalid_label_rows=dropped_invalid_label_rows,
        dropped_invalid_timestamp_rows=dropped_invalid_timestamp_rows,
        dropped_all_feature_missing_rows=dropped_all_feature_missing_rows,
        dropped_duplicate_timestamp_rows=dropped_duplicate_timestamp_rows,
        dropped_remaining_nan_rows=dropped_remaining_nan_rows,
    )


def save_cached_participant(cleaned: CachedParticipantData, cache_path: Path) -> None:
    """Persist cleaned participant arrays for later reuse."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        participant_id=np.asarray([cleaned.participant_id]),
        file_path=np.asarray([str(cleaned.file_path)]),
        feature_columns=np.asarray(cleaned.feature_columns),
        timestamps=cleaned.timestamps,
        features=cleaned.features,
        labels=cleaned.labels,
        row_complete_before_fill=cleaned.row_complete_before_fill,
        raw_row_count=np.asarray([cleaned.raw_row_count], dtype=np.int64),
        clean_row_count=np.asarray([cleaned.clean_row_count], dtype=np.int64),
        dropped_invalid_label_rows=np.asarray([cleaned.dropped_invalid_label_rows], dtype=np.int64),
        dropped_invalid_timestamp_rows=np.asarray([cleaned.dropped_invalid_timestamp_rows], dtype=np.int64),
        dropped_all_feature_missing_rows=np.asarray([cleaned.dropped_all_feature_missing_rows], dtype=np.int64),
        dropped_duplicate_timestamp_rows=np.asarray([cleaned.dropped_duplicate_timestamp_rows], dtype=np.int64),
        dropped_remaining_nan_rows=np.asarray([cleaned.dropped_remaining_nan_rows], dtype=np.int64),
    )


def load_cached_participant(cache_path: Path) -> CachedParticipantData:
    """Load a cached participant from disk."""

    with np.load(cache_path, allow_pickle=False) as payload:
        return CachedParticipantData(
            participant_id=str(payload["participant_id"][0]),
            file_path=Path(str(payload["file_path"][0])),
            feature_columns=tuple(str(value) for value in payload["feature_columns"].tolist()),
            timestamps=payload["timestamps"].astype(np.float64, copy=False),
            features=payload["features"].astype(np.float32, copy=False),
            labels=payload["labels"].astype(np.int64, copy=False),
            row_complete_before_fill=payload["row_complete_before_fill"].astype(np.float32, copy=False),
            raw_row_count=int(payload["raw_row_count"][0]),
            clean_row_count=int(payload["clean_row_count"][0]),
            dropped_invalid_label_rows=int(payload["dropped_invalid_label_rows"][0]),
            dropped_invalid_timestamp_rows=int(payload["dropped_invalid_timestamp_rows"][0]),
            dropped_all_feature_missing_rows=int(payload["dropped_all_feature_missing_rows"][0]),
            dropped_duplicate_timestamp_rows=int(payload["dropped_duplicate_timestamp_rows"][0]),
            dropped_remaining_nan_rows=int(payload["dropped_remaining_nan_rows"][0]),
        )


def iter_window_bounds(num_rows: int, window_size: int, stride_size: int) -> list[tuple[int, int]]:
    """Generate deterministic sliding-window bounds."""

    if num_rows < window_size:
        return []
    return [(start, start + window_size) for start in range(0, num_rows - window_size + 1, stride_size)]


def resolve_window_label(label_ids: np.ndarray) -> int:
    """Choose a center-epoch label by majority vote, falling back to the last sample on ties."""

    values, counts = np.unique(label_ids, return_counts=True)
    max_count = counts.max()
    winners = values[counts == max_count]
    if len(winners) == 1:
        return int(winners[0])

    last_label = int(label_ids[-1])
    if last_label in winners:
        return last_label
    return int(winners[0])


def build_window_manifest_rows(
    cleaned: CachedParticipantData,
    config: ExperimentConfig,
    split_name: str,
    stride_seconds: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Create context-window metadata rows for a participant."""

    bounds = iter_window_bounds(
        num_rows=len(cleaned.labels),
        window_size=config.context_window_size,
        stride_size=stride_seconds * config.sample_rate_hz,
    )
    if not bounds:
        return [], 0, 1

    rows: list[dict[str, Any]] = []
    skipped_low_quality_windows = 0
    center_width = config.center_window_size
    left_context = config.left_context_size
    for candidate_index, (start_index, end_index) in enumerate(bounds):
        completion_fraction = float(cleaned.row_complete_before_fill[start_index:end_index].mean())
        if completion_fraction < config.min_window_complete_fraction:
            skipped_low_quality_windows += 1
            continue

        center_start_index = start_index + left_context
        center_end_index = center_start_index + center_width
        label_id = resolve_window_label(cleaned.labels[center_start_index:center_end_index])
        rows.append(
            {
                "split": split_name,
                "participant_id": cleaned.participant_id,
                "window_index": candidate_index,
                "input_start_index": start_index,
                "input_end_index": end_index,
                "center_start_index": center_start_index,
                "center_end_index": center_end_index,
                "window_start_timestamp": float(cleaned.timestamps[start_index]),
                "window_end_timestamp": float(cleaned.timestamps[end_index - 1]),
                "center_start_timestamp": float(cleaned.timestamps[center_start_index]),
                "center_end_timestamp": float(cleaned.timestamps[center_end_index - 1]),
                "label_id": int(label_id),
                "label_name": ID_TO_NAME[int(label_id)],
                "completion_fraction": completion_fraction,
                "context_window_seconds": config.context_window_seconds,
                "center_epoch_seconds": config.center_epoch_seconds,
                "stride_seconds": stride_seconds,
            }
        )

    return rows, skipped_low_quality_windows, 0


def build_windows_from_clean_dataframe(
    cleaned: CachedParticipantData,
    config: ExperimentConfig,
    include_features: bool,
) -> WindowedParticipantData:
    """Compatibility wrapper that materializes participant windows directly."""

    rows, skipped_low_quality_windows, skipped_short_windows = build_window_manifest_rows(
        cleaned=cleaned,
        config=config,
        split_name="inference",
        stride_seconds=config.test_stride_seconds,
    )
    manifest = pd.DataFrame(rows)
    features = None
    if include_features and not manifest.empty:
        features = extract_window_features(cleaned, manifest)
    elif include_features:
        features = np.empty((0, len(cleaned.feature_columns), config.context_window_size), dtype=np.float32)

    return WindowedParticipantData(
        features=features,
        labels=np.asarray([row["label_id"] for row in rows], dtype=np.int64),
        start_timestamps=np.asarray([row["center_start_timestamp"] for row in rows], dtype=np.float64),
        end_timestamps=np.asarray([row["center_end_timestamp"] for row in rows], dtype=np.float64),
        skipped_low_quality_windows=skipped_low_quality_windows,
        skipped_short_windows=skipped_short_windows,
    )


def extract_window_features(cleaned: CachedParticipantData, manifest_rows: pd.DataFrame) -> np.ndarray:
    """Slice feature tensors for a participant from manifest rows."""

    if manifest_rows.empty:
        return np.empty((0, len(cleaned.feature_columns), 0), dtype=np.float32)

    windows: list[np.ndarray] = []
    for row in manifest_rows.itertuples(index=False):
        window = cleaned.features[row.input_start_index:row.input_end_index]
        windows.append(window.T.astype(np.float32, copy=False))
    return np.stack(windows, axis=0)


def summarize_participant(cleaned: CachedParticipantData, config: ExperimentConfig) -> ParticipantSummary:
    """Build a participant summary without materializing all split tensors."""

    rows, _, skipped_short_windows = build_window_manifest_rows(
        cleaned=cleaned,
        config=config,
        split_name="summary",
        stride_seconds=config.summary_stride_seconds,
    )
    labels = np.asarray([row["label_id"] for row in rows], dtype=np.int64)
    counts = (
        np.bincount(labels, minlength=len(ID_TO_NAME))
        if len(labels)
        else np.zeros(len(ID_TO_NAME), dtype=np.int64)
    )
    dominant_class_id = int(np.argmax(counts)) if counts.sum() > 0 else None
    return ParticipantSummary(
        participant_id=cleaned.participant_id,
        file_path=cleaned.file_path,
        raw_row_count=cleaned.raw_row_count,
        clean_row_count=cleaned.clean_row_count,
        window_count=0 if skipped_short_windows else int(len(rows)),
        dominant_class_id=dominant_class_id,
        dominant_class_name=ID_TO_NAME.get(dominant_class_id) if dominant_class_id is not None else None,
        window_class_counts={ID_TO_NAME[idx]: int(count) for idx, count in enumerate(counts)},
        dropped_invalid_label_rows=cleaned.dropped_invalid_label_rows,
        dropped_invalid_timestamp_rows=cleaned.dropped_invalid_timestamp_rows,
        dropped_all_feature_missing_rows=cleaned.dropped_all_feature_missing_rows,
        dropped_duplicate_timestamp_rows=cleaned.dropped_duplicate_timestamp_rows,
        dropped_remaining_nan_rows=cleaned.dropped_remaining_nan_rows,
    )


def participant_summary_to_dict(summary: ParticipantSummary) -> dict[str, Any]:
    """Convert a participant summary to JSON-friendly output."""

    return {
        "participant_id": summary.participant_id,
        "file_path": str(summary.file_path),
        "raw_row_count": summary.raw_row_count,
        "clean_row_count": summary.clean_row_count,
        "window_count": summary.window_count,
        "dominant_class_id": summary.dominant_class_id,
        "dominant_class_name": summary.dominant_class_name,
        "window_class_counts": dict(summary.window_class_counts),
        "dropped_invalid_label_rows": summary.dropped_invalid_label_rows,
        "dropped_invalid_timestamp_rows": summary.dropped_invalid_timestamp_rows,
        "dropped_all_feature_missing_rows": summary.dropped_all_feature_missing_rows,
        "dropped_duplicate_timestamp_rows": summary.dropped_duplicate_timestamp_rows,
        "dropped_remaining_nan_rows": summary.dropped_remaining_nan_rows,
    }


def participant_summary_from_dict(payload: dict[str, Any]) -> ParticipantSummary:
    """Load a participant summary from JSON."""

    return ParticipantSummary(
        participant_id=str(payload["participant_id"]),
        file_path=Path(str(payload["file_path"])),
        raw_row_count=int(payload["raw_row_count"]),
        clean_row_count=int(payload["clean_row_count"]),
        window_count=int(payload["window_count"]),
        dominant_class_id=(
            int(payload["dominant_class_id"])
            if payload.get("dominant_class_id") is not None
            else None
        ),
        dominant_class_name=payload.get("dominant_class_name"),
        window_class_counts={str(key): int(value) for key, value in payload["window_class_counts"].items()},
        dropped_invalid_label_rows=int(payload["dropped_invalid_label_rows"]),
        dropped_invalid_timestamp_rows=int(payload["dropped_invalid_timestamp_rows"]),
        dropped_all_feature_missing_rows=int(payload["dropped_all_feature_missing_rows"]),
        dropped_duplicate_timestamp_rows=int(payload["dropped_duplicate_timestamp_rows"]),
        dropped_remaining_nan_rows=int(payload["dropped_remaining_nan_rows"]),
    )
