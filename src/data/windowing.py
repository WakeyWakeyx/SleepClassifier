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


@dataclass(slots=True, frozen=True)
class WindowAssignment:
    """Resolved label assignment and ambiguity diagnostics for one target segment."""

    label_index: int | None
    candidate_label_index: int | None
    purity_reference_label_index: int | None
    purity: float
    is_ambiguous: bool
    discard_reason: str | None


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
    candidate_class_counts: dict[str, int]
    kept_class_counts: dict[str, int]
    actual_purity_discards_by_class: dict[str, int]
    purity_threshold_failures_by_class: dict[str, int]
    discarded_class_counts_by_reason: dict[str, dict[str, int]]
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
            "candidate_class_counts": dict(self.candidate_class_counts),
            "kept_class_counts": dict(self.kept_class_counts),
            "actual_purity_discards_by_class": dict(self.actual_purity_discards_by_class),
            "purity_threshold_failures_by_class": dict(self.purity_threshold_failures_by_class),
            "discarded_class_counts_by_reason": {
                reason: dict(counts)
                for reason, counts in self.discarded_class_counts_by_reason.items()
            },
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
        aggregate_candidate_class_counts: Counter[str] = Counter()
        aggregate_actual_purity_discards: Counter[str] = Counter()
        aggregate_purity_threshold_failures: Counter[str] = Counter()
        aggregate_discarded_class_counts_by_reason: dict[str, Counter[str]] = {}

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
            aggregate_candidate_class_counts.update(participant_summary.candidate_class_counts)
            aggregate_actual_purity_discards.update(
                participant_summary.actual_purity_discards_by_class
            )
            aggregate_purity_threshold_failures.update(
                participant_summary.purity_threshold_failures_by_class
            )
            for reason, class_counts in participant_summary.discarded_class_counts_by_reason.items():
                aggregate_discarded_class_counts_by_reason.setdefault(reason, Counter()).update(
                    class_counts
                )

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
        overall_purity_failure_total = int(sum(aggregate_purity_threshold_failures.values()))
        overall_candidate_total = int(sum(aggregate_candidate_class_counts.values()))
        overall_purity_failure_rate = (
            overall_purity_failure_total / overall_candidate_total
            if overall_candidate_total
            else 0.0
        )
        self.summary: dict[str, Any] = {
            "participants": len(participants),
            "candidate_windows": int(total_candidate_windows),
            "kept_windows": len(self.samples),
            "discarded_windows": int(total_candidate_windows - len(self.samples)),
            "ambiguous_windows_kept": int(total_ambiguous_windows_kept),
            "participants_with_no_kept_windows": int(participants_with_no_kept_windows),
            "candidate_class_counts": _counter_to_label_counts(
                aggregate_candidate_class_counts,
                self.label_names,
            ),
            "kept_class_counts": label_counts,
            "actual_purity_discards_by_class": _counter_to_label_counts(
                aggregate_actual_purity_discards,
                self.label_names,
            ),
            "purity_threshold_failures_by_class": _counter_to_label_counts(
                aggregate_purity_threshold_failures,
                self.label_names,
            ),
            "purity_threshold_failure_rates_by_class": _compute_classwise_rates(
                numerators=aggregate_purity_threshold_failures,
                denominators=aggregate_candidate_class_counts,
                label_names=self.label_names,
            ),
            "minority_purity_relative_loss": _build_relative_loss_summary(
                rates=_compute_classwise_rates(
                    numerators=aggregate_purity_threshold_failures,
                    denominators=aggregate_candidate_class_counts,
                    label_names=self.label_names,
                ),
                baseline_rate=overall_purity_failure_rate,
                label_names=self.label_names,
            ),
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
            "discarded_class_counts_by_reason": {
                reason: _counter_to_label_counts(class_counts, self.label_names)
                for reason, class_counts in sorted(aggregate_discarded_class_counts_by_reason.items())
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
            candidate_class_counts=_empty_label_counts(label_names),
            kept_class_counts=_empty_label_counts(label_names),
            actual_purity_discards_by_class=_empty_label_counts(label_names),
            purity_threshold_failures_by_class=_empty_label_counts(label_names),
            discarded_class_counts_by_reason={},
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
    candidate_label_counts: Counter[int] = Counter()
    actual_purity_discards_by_class: Counter[int] = Counter()
    purity_threshold_failures_by_class: Counter[int] = Counter()
    discarded_class_counts_by_reason: dict[str, Counter[int]] = {}

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
        assignment = _resolve_window_assignment(
            class_counts=class_counts,
            encoded_labels=encoded_labels,
            target_start_index=target_start_index,
            target_end_index=target_end_index,
            valid_count=valid_count,
            data_config=data_config,
        )
        if assignment.candidate_label_index is not None:
            candidate_label_counts[assignment.candidate_label_index] += 1
        if assignment.is_ambiguous and assignment.purity_reference_label_index is not None:
            purity_threshold_failures_by_class[assignment.purity_reference_label_index] += 1
        if assignment.discard_reason is not None:
            discard_reasons[assignment.discard_reason] += 1
            if assignment.discard_reason == "insufficient_purity":
                if assignment.purity_reference_label_index is not None:
                    actual_purity_discards_by_class[assignment.purity_reference_label_index] += 1
            if assignment.candidate_label_index is not None:
                discarded_class_counts_by_reason.setdefault(
                    assignment.discard_reason,
                    Counter(),
                )[assignment.candidate_label_index] += 1
            continue
        if assignment.is_ambiguous:
            ambiguous_windows_kept += 1

        windows.append(
            WindowMetadata(
                participant_index=participant_index,
                input_start_index=input_start_index,
                input_end_index=input_end_index,
                target_start_index=target_start_index,
                target_end_index=target_end_index,
                label_index=int(assignment.label_index),
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
        candidate_class_counts={
            label_name: int(candidate_label_counts.get(index, 0))
            for index, label_name in enumerate(label_names)
        },
        kept_class_counts={
            label_name: int(kept_label_counts.get(index, 0))
            for index, label_name in enumerate(label_names)
        },
        actual_purity_discards_by_class={
            label_name: int(actual_purity_discards_by_class.get(index, 0))
            for index, label_name in enumerate(label_names)
        },
        purity_threshold_failures_by_class={
            label_name: int(purity_threshold_failures_by_class.get(index, 0))
            for index, label_name in enumerate(label_names)
        },
        discarded_class_counts_by_reason={
            reason: {
                label_name: int(class_counts_by_reason.get(index, 0))
                for index, label_name in enumerate(label_names)
            }
            for reason, class_counts_by_reason in sorted(discarded_class_counts_by_reason.items())
        },
        discard_reasons={
            reason: int(count)
            for reason, count in sorted(discard_reasons.items())
            if count > 0
        },
    )
    return windows, report


def _resolve_window_assignment(
    class_counts: np.ndarray,
    encoded_labels: np.ndarray,
    target_start_index: int,
    target_end_index: int,
    valid_count: int,
    data_config: DataConfig,
) -> WindowAssignment:
    """Assign one window label according to the configured target-labeling strategy."""

    majority_class = int(class_counts.argmax())
    majority_count = int(class_counts[majority_class])
    purity = majority_count / valid_count
    is_ambiguous = purity < data_config.label_purity_threshold

    if data_config.target_label_strategy == "majority_vote":
        return WindowAssignment(
            label_index=majority_class,
            candidate_label_index=majority_class,
            purity_reference_label_index=majority_class,
            purity=purity,
            is_ambiguous=is_ambiguous,
            discard_reason=None,
        )

    if data_config.target_label_strategy == "center_label":
        center_start_index, center_end_index = _center_region_bounds(
            target_start_index=target_start_index,
            target_end_index=target_end_index,
            center_label_span=data_config.center_label_span,
        )
        center_labels = encoded_labels[center_start_index:center_end_index]
        valid_center_labels = center_labels[center_labels >= 0]
        if valid_center_labels.size == 0:
            return WindowAssignment(
                label_index=None,
                candidate_label_index=majority_class,
                purity_reference_label_index=majority_class,
                purity=purity,
                is_ambiguous=is_ambiguous,
                discard_reason="invalid_center_label",
            )

        center_class_counts = np.bincount(
            valid_center_labels.astype(np.int64),
            minlength=class_counts.shape[0],
        )
        center_class = int(center_class_counts.argmax())
        return WindowAssignment(
            label_index=center_class,
            candidate_label_index=center_class,
            purity_reference_label_index=majority_class,
            purity=purity,
            is_ambiguous=is_ambiguous,
            discard_reason=None,
        )

    discard_reason = None
    if is_ambiguous and data_config.drop_ambiguous_windows:
        discard_reason = "insufficient_purity"

    return WindowAssignment(
        label_index=majority_class if discard_reason is None else None,
        candidate_label_index=majority_class,
        purity_reference_label_index=majority_class,
        purity=purity,
        is_ambiguous=is_ambiguous,
        discard_reason=discard_reason,
    )


def _center_region_bounds(
    target_start_index: int,
    target_end_index: int,
    center_label_span: int,
) -> tuple[int, int]:
    """Return the centered sub-region used by the center-label strategy."""

    target_length = target_end_index - target_start_index
    center_index = target_start_index + (target_length // 2)
    half_span = center_label_span // 2
    center_start_index = max(target_start_index, center_index - half_span)
    center_end_index = min(target_end_index, center_index + half_span + 1)
    if (center_end_index - center_start_index) < center_label_span:
        center_start_index = max(target_start_index, center_end_index - center_label_span)
        center_end_index = min(target_end_index, center_start_index + center_label_span)
    return center_start_index, center_end_index


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


def _build_sequence_definition(
    data_config: DataConfig,
) -> dict[str, int | float | bool | str]:
    """Serialize how each supervised sample is constructed."""

    return {
        "use_context_windows": data_config.use_context_windows,
        "target_window_length": data_config.target_window_length,
        "target_label_strategy": data_config.target_label_strategy,
        "center_label_span": data_config.center_label_span,
        "label_purity_threshold": data_config.label_purity_threshold,
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


def _counter_to_label_counts(
    counter: Counter[str] | Counter[int],
    label_names: Sequence[str],
) -> dict[str, int]:
    """Render integer Counters into a stable label-keyed dictionary."""

    return {
        label_name: int(counter.get(index, counter.get(label_name, 0)))
        for index, label_name in enumerate(label_names)
    }


def _compute_classwise_rates(
    numerators: Counter[str] | Counter[int],
    denominators: Counter[str] | Counter[int],
    label_names: Sequence[str],
) -> dict[str, float]:
    """Compute per-class rates from aligned numerator and denominator counters."""

    rates: dict[str, float] = {}
    for index, label_name in enumerate(label_names):
        numerator = float(numerators.get(index, numerators.get(label_name, 0)))
        denominator = float(denominators.get(index, denominators.get(label_name, 0)))
        rates[label_name] = numerator / denominator if denominator > 0.0 else 0.0
    return rates


def _build_relative_loss_summary(
    rates: Mapping[str, float],
    baseline_rate: float,
    label_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compare each class purity-loss rate against the overall rate."""

    summary: dict[str, dict[str, float]] = {}
    for label_name in label_names:
        rate = float(rates.get(label_name, 0.0))
        summary[label_name] = {
            "purity_failure_rate": rate,
            "relative_to_overall": (
                rate / baseline_rate if baseline_rate > 0.0 else 0.0
            ),
        }
    return summary
