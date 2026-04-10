"""Window construction logic for sequence classification."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DataConfig

from .data_loading import ParticipantData


class WindowSample(TypedDict):
    """One supervised sample made from a contiguous multichannel time sequence."""

    inputs: torch.Tensor
    target: int
    target_start_idx: int
    target_end_idx: int


@dataclass(slots=True, frozen=True)
class WindowMetadata:
    """Metadata describing one retained supervised target segment and its input span."""

    participant_index: int
    input_start_index: int
    input_end_index: int
    target_start_index: int
    target_end_index: int
    label_index: int


@dataclass(slots=True)
class ParticipantWindowReport:
    """Structured per-participant windowing diagnostics."""

    participant_id: str
    source_path: str
    total_rows: int
    candidate_windows: int
    kept_windows: int
    discarded_windows: int
    ambiguous_windows_kept: int
    too_short_for_windowing: bool
    kept_class_counts: dict[str, int]
    discard_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into a JSON-friendly dictionary."""

        return {
            "participant_id": self.participant_id,
            "source_path": self.source_path,
            "total_rows": self.total_rows,
            "candidate_windows": self.candidate_windows,
            "kept_windows": self.kept_windows,
            "discarded_windows": self.discarded_windows,
            "ambiguous_windows_kept": self.ambiguous_windows_kept,
            "too_short_for_windowing": self.too_short_for_windowing,
            "kept_class_counts": dict(self.kept_class_counts),
            "discard_reasons": dict(self.discard_reasons),
        }


class WindowedSleepDataset(Dataset[WindowSample]):
    """Slice participant arrays into contiguous sequence windows for one target label."""

    def __init__(
        self,
        participants: Sequence[ParticipantData],
        data_config: DataConfig,
        label_to_index: Mapping[str, int],
    ) -> None:
        self.feature_columns = data_config.feature_columns
        self.target_window_length = data_config.target_window_length
        self.left_context = data_config.effective_left_context
        self.right_context = data_config.effective_right_context
        self.input_window_length = data_config.input_window_length
        self.label_names = tuple(
            label_name
            for label_name, _ in sorted(label_to_index.items(), key=lambda item: item[1])
        )
        self.samples: list[WindowMetadata] = []
        label_buffer: list[int] = []
        self._signals: list[np.ndarray] = []
        self.participant_summaries: list[dict[str, Any]] = []

        aggregate_reasons: Counter[str] = Counter()

        for participant_index, participant in enumerate(participants):
            signal_array = participant.frame.loc[:, self.feature_columns].to_numpy(
                dtype=np.float32,
                copy=False,
            )
            signal_array = np.ascontiguousarray(signal_array.T)
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
            participant.frame = participant.frame.iloc[0:0]
            participant_windows, participant_summary = _generate_windows_for_participant(
                participant_index=participant_index,
                participant_id=participant.participant_id,
                source_path=participant.source_path,
                encoded_labels=encoded_labels,
                timestamps=timestamp_array,
                data_config=data_config,
                label_names=self.label_names,
            )
            self.samples.extend(participant_windows)
            label_buffer.extend(window.label_index for window in participant_windows)
            self.participant_summaries.append(participant_summary.to_dict())
            aggregate_reasons.update(participant_summary.discard_reasons)

        self.labels = np.asarray(label_buffer, dtype=np.int64)
        label_counts = self.label_counts()
        total_candidate_windows = sum(
            report["candidate_windows"] for report in self.participant_summaries
        )
        total_ambiguous_windows_kept = sum(
            report["ambiguous_windows_kept"] for report in self.participant_summaries
        )
        participants_with_no_kept_windows = sum(
            int(report["kept_windows"] == 0) for report in self.participant_summaries
        )
        self.summary: dict[str, Any] = {
            "participants": len(participants),
            "candidate_windows": int(total_candidate_windows),
            "kept_windows": len(self.samples),
            "discarded_windows": int(total_candidate_windows - len(self.samples)),
            "ambiguous_windows_kept": int(total_ambiguous_windows_kept),
            "participants_with_no_kept_windows": int(participants_with_no_kept_windows),
            "kept_class_counts": label_counts,
            "participants_missing_each_class": {
                label_name: int(
                    sum(
                        report["kept_class_counts"].get(label_name, 0) == 0
                        for report in self.participant_summaries
                    )
                )
                for label_name in self.label_names
            },
            "discard_reasons": {
                reason: int(count)
                for reason, count in sorted(aggregate_reasons.items())
                if count > 0
            },
            "sequence_definition": _build_sequence_definition(data_config),
            "participant_summaries": self.participant_summaries,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> WindowSample:
        return self._materialize_sample(index)

    def __getitems__(self, indices: Sequence[int]) -> list[WindowSample]:
        """Materialize a batch of samples when the DataLoader supports batched fetching."""

        return [self._materialize_sample(index) for index in indices]

    def _materialize_sample(self, index: int) -> WindowSample:
        metadata = self.samples[index]
        window = self._signals[metadata.participant_index][
            :,
            metadata.input_start_index : metadata.input_end_index,
        ]
        signal_tensor = torch.from_numpy(window)
        target_start_idx = metadata.target_start_index - metadata.input_start_index
        target_end_idx = metadata.target_end_index - metadata.input_start_index

        return {
            "inputs": signal_tensor,
            "target": metadata.label_index,
            "target_start_idx": target_start_idx,
            "target_end_idx": target_end_idx,
        }

    def label_counts(self, label_names: Sequence[str] | None = None) -> dict[str, int]:
        """Return the class distribution of retained target windows."""

        names = tuple(label_names) if label_names is not None else self.label_names
        counts = np.bincount(self.labels, minlength=len(names)) if len(self.labels) else np.zeros(
            len(names),
            dtype=np.int64,
        )
        return {
            label_name: int(counts[index])
            for index, label_name in enumerate(names)
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
    label_names: Sequence[str],
) -> tuple[list[WindowMetadata], ParticipantWindowReport]:
    """Create sequence windows and target labels for one participant.

    Each retained sample is one contiguous input span:
    left context + target segment + right context.

    The supervised label is determined from the target segment only. Context rows
    can help the model disambiguate transitions, but they never influence label
    assignment or purity checks.
    """

    total_rows = int(encoded_labels.shape[0])
    windows: list[WindowMetadata] = []
    discard_reasons: Counter[str] = Counter()

    if total_rows < data_config.input_window_length:
        report = ParticipantWindowReport(
            participant_id=participant_id,
            source_path=str(source_path),
            total_rows=total_rows,
            candidate_windows=0,
            kept_windows=0,
            discarded_windows=0,
            ambiguous_windows_kept=0,
            too_short_for_windowing=True,
            kept_class_counts=_empty_label_counts(label_names),
            discard_reasons={"too_short_for_windowing": 1},
        )
        return windows, report

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
        for class_index in range(len(label_names))
    ]
    gap_prefix = _build_gap_prefix(
        timestamps=timestamps,
        continuity_gap_factor=data_config.continuity_gap_factor,
    )

    min_valid_count = math.ceil(
        data_config.min_valid_fraction * data_config.target_window_length
    )
    candidate_windows = 0
    ambiguous_windows_kept = 0

    for target_start_index in _candidate_target_start_indices(total_rows, data_config):
        candidate_windows += 1
        input_start_index = target_start_index - data_config.effective_left_context
        target_end_index = target_start_index + data_config.target_window_length
        input_end_index = target_end_index + data_config.effective_right_context

        if _window_crosses_gap(gap_prefix, input_start_index, input_end_index):
            discard_reasons["temporal_gap"] += 1
            continue

        valid_count = int(valid_prefix[target_end_index] - valid_prefix[target_start_index])
        if valid_count == 0:
            discard_reasons["no_valid_supervised_labels"] += 1
            continue

        if data_config.require_min_valid_labels and valid_count < min_valid_count:
            discard_reasons["excluded_label_contamination"] += 1
            continue

        class_counts = np.array(
            [
                class_prefix[target_end_index] - class_prefix[target_start_index]
                for class_prefix in class_prefixes
            ],
            dtype=np.int64,
        )
        majority_class = int(class_counts.argmax())
        majority_count = int(class_counts[majority_class])
        purity = majority_count / valid_count

        if purity < data_config.label_purity_threshold and data_config.drop_ambiguous_windows:
            discard_reasons["insufficient_purity"] += 1
            continue
        if purity < data_config.label_purity_threshold:
            ambiguous_windows_kept += 1

        windows.append(
            WindowMetadata(
                participant_index=participant_index,
                input_start_index=input_start_index,
                input_end_index=input_end_index,
                target_start_index=target_start_index,
                target_end_index=target_end_index,
                label_index=majority_class,
            )
        )

    kept_label_counts = Counter(window.label_index for window in windows)
    report = ParticipantWindowReport(
        participant_id=participant_id,
        source_path=str(source_path),
        total_rows=total_rows,
        candidate_windows=candidate_windows,
        kept_windows=len(windows),
        discarded_windows=candidate_windows - len(windows),
        ambiguous_windows_kept=ambiguous_windows_kept,
        too_short_for_windowing=False,
        kept_class_counts={
            label_name: int(kept_label_counts.get(index, 0))
            for index, label_name in enumerate(label_names)
        },
        discard_reasons={
            reason: int(count)
            for reason, count in sorted(discard_reasons.items())
            if count > 0
        },
    )
    return windows, report


def _candidate_target_start_indices(
    total_rows: int,
    data_config: DataConfig,
) -> range:
    """Yield target starts whose full input spans fit within participant boundaries."""

    first_target_start = data_config.effective_left_context
    last_target_start = (
        total_rows
        - data_config.target_window_length
        - data_config.effective_right_context
    )
    return range(first_target_start, last_target_start + 1, data_config.step)


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
    """Return True when a contiguous input span crosses a detected timestamp gap."""

    if end_index - start_index < 2:
        return False
    return bool(gap_prefix[end_index - 1] - gap_prefix[start_index])


def _build_sequence_definition(data_config: DataConfig) -> dict[str, int | bool]:
    """Serialize how each supervised sample is constructed."""

    return {
        "use_context_windows": data_config.use_context_windows,
        "target_window_length": data_config.target_window_length,
        "configured_left_context": data_config.left_context,
        "configured_right_context": data_config.right_context,
        "left_context": data_config.effective_left_context,
        "right_context": data_config.effective_right_context,
        "total_input_length": data_config.input_window_length,
        "step": data_config.step,
    }


def _empty_label_counts(label_names: Sequence[str]) -> dict[str, int]:
    """Create a zero-filled label-count dictionary."""

    return {label_name: 0 for label_name in label_names}
