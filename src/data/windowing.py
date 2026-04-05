"""Window construction logic for sequence classification."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DataConfig

from .data_loading import ParticipantData


@dataclass(slots=True, frozen=True)
class WindowMetadata:
    """Metadata describing a single retained window."""

    participant_index: int
    participant_id: str
    source_path: str
    start_index: int
    end_index: int
    label_index: int


class WindowedSleepDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch dataset that slices normalized participant arrays into windows."""

    def __init__(
        self,
        participants: Sequence[ParticipantData],
        data_config: DataConfig,
        label_to_index: Mapping[str, int],
    ) -> None:
        self.feature_columns = data_config.feature_columns
        self.window_length = data_config.window_length
        self.samples: list[WindowMetadata] = []
        self.labels: list[int] = []
        self._signals: list[np.ndarray] = []
        self.summary: Counter[str] = Counter()

        for participant_index, participant in enumerate(participants):
            signal_array = participant.frame.loc[:, self.feature_columns].to_numpy(
                dtype=np.float32,
                copy=True,
            )
            timestamp_array = participant.frame.loc[:, data_config.timestamp_column].to_numpy(
                dtype=np.float64,
                copy=False,
            )
            encoded_labels = _encode_labels(
                participant.frame.loc[:, data_config.target_column].to_numpy(dtype=object),
                label_to_index=label_to_index,
                excluded_labels=set(data_config.excluded_labels),
                source_path=participant.source_path,
            )

            self._signals.append(signal_array)
            participant_windows, participant_summary = _generate_windows_for_participant(
                participant_index=participant_index,
                participant_id=participant.participant_id,
                source_path=participant.source_path,
                encoded_labels=encoded_labels,
                timestamps=timestamp_array,
                data_config=data_config,
                num_classes=len(label_to_index),
            )
            self.samples.extend(participant_windows)
            self.labels.extend(window.label_index for window in participant_windows)
            self.summary.update(participant_summary)

        self.summary["kept_windows"] = len(self.samples)
        self.summary["participants"] = len(participants)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        metadata = self.samples[index]
        window = self._signals[metadata.participant_index][
            metadata.start_index : metadata.end_index
        ]
        signal_tensor = torch.from_numpy(window.T.copy())
        return signal_tensor, metadata.label_index

    def label_counts(self, label_names: Sequence[str]) -> dict[str, int]:
        """Return the class distribution of retained windows."""

        counts = Counter(self.labels)
        return {
            label_name: int(counts.get(index, 0))
            for index, label_name in enumerate(label_names)
        }


def _encode_labels(
    labels: np.ndarray,
    label_to_index: Mapping[str, int],
    excluded_labels: set[str],
    source_path: Path,
) -> np.ndarray:
    """Map labels to integers, keeping excluded labels as -1."""

    encoded = np.full(shape=(labels.shape[0],), fill_value=-1, dtype=np.int16)
    unexpected: set[str] = set()

    for index, raw_label in enumerate(labels.tolist()):
        label = str(raw_label).strip()
        if label in label_to_index:
            encoded[index] = label_to_index[label]
        elif label in excluded_labels:
            continue
        else:
            unexpected.add(label)

    if unexpected:
        unexpected_display = ", ".join(sorted(unexpected))
        raise ValueError(f"{source_path} contains unsupported labels: {unexpected_display}")

    return encoded


def _generate_windows_for_participant(
    participant_index: int,
    participant_id: str,
    source_path: Path,
    encoded_labels: np.ndarray,
    timestamps: np.ndarray,
    data_config: DataConfig,
    num_classes: int,
) -> tuple[list[WindowMetadata], Counter[str]]:
    """Create window metadata for one participant using conservative label rules."""

    total_rows = encoded_labels.shape[0]
    summary: Counter[str] = Counter(total_rows=total_rows)
    windows: list[WindowMetadata] = []

    if total_rows < data_config.window_length:
        summary["too_short_participants"] += 1
        return windows, summary

    valid_prefix = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(encoded_labels >= 0, dtype=np.int64)]
    )
    class_prefixes = [
        np.concatenate(
            [
                np.array([0], dtype=np.int64),
                np.cumsum(encoded_labels == class_index, dtype=np.int64),
            ]
        )
        for class_index in range(num_classes)
    ]
    gap_prefix = _build_gap_prefix(
        timestamps=timestamps,
        continuity_gap_factor=data_config.continuity_gap_factor,
    )

    min_valid_count = math.ceil(data_config.min_valid_fraction * data_config.window_length)

    for start_index in range(0, total_rows - data_config.window_length + 1, data_config.step):
        summary["candidate_windows"] += 1
        end_index = start_index + data_config.window_length

        if _window_crosses_gap(gap_prefix, start_index, end_index):
            summary["dropped_temporal_gap"] += 1
            continue

        valid_count = int(valid_prefix[end_index] - valid_prefix[start_index])
        if valid_count < min_valid_count:
            summary["dropped_low_valid_fraction"] += 1
            continue

        class_counts = np.array(
            [
                class_prefix[end_index] - class_prefix[start_index]
                for class_prefix in class_prefixes
            ],
            dtype=np.int64,
        )
        majority_class = int(class_counts.argmax())
        majority_count = int(class_counts[majority_class])

        if valid_count == 0:
            summary["dropped_no_valid_labels"] += 1
            continue

        if (majority_count / valid_count) < data_config.label_agreement_threshold:
            summary["dropped_low_agreement"] += 1
            continue

        windows.append(
            WindowMetadata(
                participant_index=participant_index,
                participant_id=participant_id,
                source_path=str(source_path),
                start_index=start_index,
                end_index=end_index,
                label_index=majority_class,
            )
        )

    return windows, summary


def _build_gap_prefix(
    timestamps: np.ndarray,
    continuity_gap_factor: float,
) -> np.ndarray:
    """Build a prefix sum that marks boundaries with large timestamp gaps."""

    if timestamps.shape[0] < 2:
        return np.array([0], dtype=np.int64)

    diffs = np.diff(timestamps)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        bad_boundaries = np.zeros_like(diffs, dtype=np.int64)
    else:
        expected_step = float(np.median(positive_diffs))
        bad_boundaries = (
            (diffs < 0.0) | (diffs > (expected_step * continuity_gap_factor))
        ).astype(np.int64)

    return np.concatenate([np.array([0], dtype=np.int64), np.cumsum(bad_boundaries)])


def _window_crosses_gap(gap_prefix: np.ndarray, start_index: int, end_index: int) -> bool:
    """Return True when a window spans a detected timestamp discontinuity."""

    if end_index - start_index < 2:
        return False
    return bool(gap_prefix[end_index - 1] - gap_prefix[start_index])
