"""CSV cleaning and windowing logic for DREAMT wearable data."""

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
class CleanedParticipantData:
    """Cleaned participant dataframe and row-level accounting."""

    participant_id: str
    file_path: Path
    dataframe: pd.DataFrame
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
    """Window tensors and metadata for a single participant."""

    features: np.ndarray | None
    labels: np.ndarray
    start_timestamps: np.ndarray
    end_timestamps: np.ndarray
    skipped_low_quality_windows: int
    skipped_short_windows: int


def _coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def load_and_clean_participant_csv(
    file_path: Path,
    participant_id: str,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> CleanedParticipantData | None:
    """Load, validate, and clean a participant CSV."""

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
    numeric_columns = [config.timestamp_column, *config.feature_columns]
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

    all_features_missing_mask = frame[list(config.feature_columns)].isna().all(axis=1)
    dropped_all_feature_missing_rows = int(all_features_missing_mask.sum())
    frame = frame.loc[~all_features_missing_mask].copy()
    mapped_labels = mapped_labels.loc[~all_features_missing_mask]

    raw_complete_mask = frame[list(config.feature_columns)].notna().all(axis=1).astype(np.float32)

    frame["label_id"] = mapped_labels.astype(np.int64)
    frame["row_complete_before_fill"] = raw_complete_mask.to_numpy(dtype=np.float32)

    frame = frame.sort_values(config.timestamp_column, kind="mergesort")
    before_duplicates = len(frame)
    frame = frame.drop_duplicates(subset=[config.timestamp_column], keep="first").copy()
    dropped_duplicate_timestamp_rows = before_duplicates - len(frame)

    frame[list(config.feature_columns)] = frame[list(config.feature_columns)].ffill().bfill()
    remaining_nan_mask = frame[list(config.feature_columns)].isna().any(axis=1)
    dropped_remaining_nan_rows = int(remaining_nan_mask.sum())
    frame = frame.loc[~remaining_nan_mask].copy()
    frame = frame.reset_index(drop=True)

    clean_row_count = len(frame)
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

    return CleanedParticipantData(
        participant_id=participant_id,
        file_path=file_path,
        dataframe=frame,
        raw_row_count=raw_row_count,
        clean_row_count=clean_row_count,
        dropped_invalid_label_rows=dropped_invalid_label_rows,
        dropped_invalid_timestamp_rows=dropped_invalid_timestamp_rows,
        dropped_all_feature_missing_rows=dropped_all_feature_missing_rows,
        dropped_duplicate_timestamp_rows=dropped_duplicate_timestamp_rows,
        dropped_remaining_nan_rows=dropped_remaining_nan_rows,
    )


def iter_window_bounds(num_rows: int, window_size: int, stride_size: int) -> list[tuple[int, int]]:
    """Generate deterministic sliding-window bounds."""

    if num_rows < window_size:
        return []
    return [(start, start + window_size) for start in range(0, num_rows - window_size + 1, stride_size)]


def resolve_window_label(label_ids: np.ndarray) -> int:
    """Choose a window label by majority vote, falling back to the last sample on ties."""

    values, counts = np.unique(label_ids, return_counts=True)
    max_count = counts.max()
    winners = values[counts == max_count]
    if len(winners) == 1:
        return int(winners[0])

    last_label = int(label_ids[-1])
    if last_label in winners:
        return last_label
    return int(winners[0])


def build_windows_from_clean_dataframe(
    cleaned: CleanedParticipantData,
    config: ExperimentConfig,
    include_features: bool,
) -> WindowedParticipantData:
    """Create fixed windows from a cleaned participant dataframe."""

    frame = cleaned.dataframe
    bounds = iter_window_bounds(len(frame), config.window_size, config.stride_size)

    if not bounds:
        empty_features = (
            None
            if not include_features
            else np.empty((0, len(config.feature_columns), config.window_size), dtype=np.float32)
        )
        return WindowedParticipantData(
            features=empty_features,
            labels=np.empty((0,), dtype=np.int64),
            start_timestamps=np.empty((0,), dtype=np.float64),
            end_timestamps=np.empty((0,), dtype=np.float64),
            skipped_low_quality_windows=0,
            skipped_short_windows=1,
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    start_timestamps: list[float] = []
    end_timestamps: list[float] = []
    skipped_low_quality_windows = 0

    feature_columns = list(config.feature_columns)
    for start_index, end_index in bounds:
        window = frame.iloc[start_index:end_index]
        completion_fraction = float(window["row_complete_before_fill"].mean())
        if completion_fraction < config.min_window_complete_fraction:
            skipped_low_quality_windows += 1
            continue

        labels.append(resolve_window_label(window["label_id"].to_numpy(dtype=np.int64)))
        start_timestamps.append(float(window[config.timestamp_column].iloc[0]))
        end_timestamps.append(float(window[config.timestamp_column].iloc[-1]))
        if include_features:
            features.append(window[feature_columns].to_numpy(dtype=np.float32).T)

    feature_array = None
    if include_features:
        feature_array = (
            np.stack(features, axis=0)
            if features
            else np.empty((0, len(config.feature_columns), config.window_size), dtype=np.float32)
        )

    return WindowedParticipantData(
        features=feature_array,
        labels=np.asarray(labels, dtype=np.int64),
        start_timestamps=np.asarray(start_timestamps, dtype=np.float64),
        end_timestamps=np.asarray(end_timestamps, dtype=np.float64),
        skipped_low_quality_windows=skipped_low_quality_windows,
        skipped_short_windows=0,
    )


def summarize_participant(cleaned: CleanedParticipantData, config: ExperimentConfig) -> ParticipantSummary:
    """Build a participant summary without storing full feature windows."""

    windowed = build_windows_from_clean_dataframe(cleaned, config, include_features=False)
    counts = (
        np.bincount(windowed.labels, minlength=len(ID_TO_NAME))
        if len(windowed.labels)
        else np.zeros(len(ID_TO_NAME), dtype=np.int64)
    )
    dominant_class_id = int(np.argmax(counts)) if counts.sum() > 0 else None
    return ParticipantSummary(
        participant_id=cleaned.participant_id,
        file_path=cleaned.file_path,
        raw_row_count=cleaned.raw_row_count,
        clean_row_count=cleaned.clean_row_count,
        window_count=int(len(windowed.labels)),
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
