"""Participant-level preprocessing and normalization logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import DataConfig

from .data_loading import ParticipantData


@dataclass(slots=True, frozen=True)
class NormalizationStats:
    """Per-channel normalization statistics computed from train participants only."""

    means: dict[str, float]
    stds: dict[str, float]

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Convert the stats into a JSON-serializable dictionary."""

        return {"means": self.means, "stds": self.stds}


def clean_participant_frame(
    participant: ParticipantData,
    data_config: DataConfig,
) -> ParticipantData:
    """Sort, validate, coerce, and impute a participant dataframe."""

    required_columns = {
        data_config.timestamp_column,
        data_config.target_column,
        *data_config.feature_columns,
    }
    missing_columns = required_columns.difference(participant.frame.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{participant.source_path} is missing required columns: {missing_display}"
        )

    frame = participant.frame.loc[
        :,
        [
            data_config.timestamp_column,
            *data_config.feature_columns,
            data_config.target_column,
        ],
    ].copy()

    frame[data_config.timestamp_column] = pd.to_numeric(
        frame[data_config.timestamp_column],
        errors="coerce",
    )
    if frame[data_config.timestamp_column].isna().any():
        raise ValueError(
            f"{participant.source_path} contains non-numeric or missing TIMESTAMP values."
        )

    frame = frame.sort_values(data_config.timestamp_column, kind="stable").reset_index(drop=True)

    allowed_labels = set(data_config.source_label_names) | set(data_config.excluded_labels)
    labels = (
        frame[data_config.target_column]
        .astype("string")
        .fillna("Missing")
        .str.strip()
        .replace({"": "Missing", "<NA>": "Missing", "nan": "Missing", "NaN": "Missing"})
    )
    unexpected_labels = sorted(set(labels.dropna().tolist()).difference(allowed_labels))
    if unexpected_labels:
        raise ValueError(
            f"{participant.source_path} contains unexpected sleep stage labels: "
            f"{', '.join(unexpected_labels)}"
        )
    frame[data_config.target_column] = labels

    for column_name in data_config.feature_columns:
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")

    for column_name in data_config.feature_columns:
        if column_name == "IBI":
            frame[column_name] = _fill_ibi_channel(frame[column_name])
        else:
            frame[column_name] = _fill_aligned_feature_channel(frame[column_name])

    if frame.loc[:, data_config.feature_columns].isna().any().any():
        raise ValueError(
            f"{participant.source_path} still contains NaN values after preprocessing."
        )

    if not frame[data_config.timestamp_column].is_monotonic_increasing:
        raise ValueError(
            f"{participant.source_path} TIMESTAMP values are not monotonic after sorting."
        )

    return ParticipantData(
        participant_id=participant.participant_id,
        source_path=participant.source_path,
        frame=frame,
    )


def compute_normalization_stats(
    participants: Sequence[ParticipantData],
    feature_columns: Sequence[str],
) -> NormalizationStats:
    """Compute train-only per-channel means and standard deviations."""

    if not participants:
        raise ValueError("At least one train participant is required for normalization.")

    sums = np.zeros(len(feature_columns), dtype=np.float64)
    squared_sums = np.zeros(len(feature_columns), dtype=np.float64)
    total_rows = 0

    for participant in participants:
        values = participant.frame.loc[:, feature_columns].to_numpy(
            dtype=np.float32,
            copy=False,
        )
        if values.size == 0:
            continue
        sums += values.sum(axis=0, dtype=np.float64)
        squared_sums += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_rows += values.shape[0]

    if total_rows == 0:
        raise ValueError("No numeric rows were available to compute normalization statistics.")

    means = sums / total_rows
    variances = (squared_sums / total_rows) - np.square(means)
    variances = np.maximum(variances, 1e-12)
    stds = np.sqrt(variances)
    stds[stds < 1e-6] = 1.0

    return NormalizationStats(
        means={column: float(means[index]) for index, column in enumerate(feature_columns)},
        stds={column: float(stds[index]) for index, column in enumerate(feature_columns)},
    )


def apply_normalization(
    participants: Sequence[ParticipantData],
    stats: NormalizationStats,
    feature_columns: Sequence[str],
) -> list[ParticipantData]:
    """Apply precomputed normalization statistics in place."""

    normalized_participants = list(participants)
    means = np.asarray(
        [stats.means[column_name] for column_name in feature_columns],
        dtype=np.float32,
    )
    stds = np.asarray(
        [stats.stds[column_name] for column_name in feature_columns],
        dtype=np.float32,
    )

    for participant in normalized_participants:
        values = participant.frame.loc[:, feature_columns].to_numpy(
            dtype=np.float32,
            copy=False,
        )
        normalized_values = ((values - means) / stds).astype(np.float32, copy=False)
        participant.frame.loc[:, feature_columns] = normalized_values

    return normalized_participants


def _fill_aligned_feature_channel(series: pd.Series) -> pd.Series:
    """Fill missing values for aligned channels without assuming a native 64 Hz origin."""

    filled = (
        series.astype(np.float64)
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    return filled.astype(np.float32)


def _fill_ibi_channel(series: pd.Series) -> pd.Series:
    """Fill sparse IBI using forward fill, backward fill, then a neutral zero fallback."""

    filled = series.astype(np.float64).ffill().bfill().fillna(0.0)
    return filled.astype(np.float32)
