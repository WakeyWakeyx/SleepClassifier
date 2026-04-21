"""Dataset preparation and PyTorch dataset classes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from sleep_classifier.config import ExperimentConfig
from sleep_classifier.data.preprocessing import (
    ParticipantSummary,
    build_windows_from_clean_dataframe,
    load_and_clean_participant_csv,
    participant_summary_to_dict,
    summarize_participant,
)
from sleep_classifier.data.scan_dataset import ParticipantFile, scan_dataset_files
from sleep_classifier.data.splits import (
    ParticipantSplits,
    create_participant_splits,
    load_existing_splits,
    splits_to_dict,
)
from sleep_classifier.label_mapping import ID_TO_NAME
from sleep_classifier.utils import load_json, save_json


@dataclass(slots=True)
class PreparedSplitData:
    """Features, labels, metadata, and distribution for one split."""

    name: str
    dataset: "SleepWindowDataset"
    metadata: list[dict[str, Any]]
    label_counts: dict[str, int]
    num_windows: int


@dataclass(slots=True)
class PreparedDataBundle:
    """Container returned by dataset preparation."""

    config: ExperimentConfig
    participant_files: dict[str, ParticipantFile]
    summaries: list[ParticipantSummary]
    splits: ParticipantSplits
    normalization_stats: dict[str, list[float]]
    class_weights: list[float]
    prepared_splits: dict[str, PreparedSplitData]


class SleepWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple tensor dataset for windowed time-series classification."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.ndim != 3:
            raise ValueError(
                f"Expected features with shape [num_windows, channels, sequence_length], got {features.shape}"
            )
        if len(features) != len(labels):
            raise ValueError("Features and labels must contain the same number of windows.")
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def _collect_participant_summaries(
    participant_files: list[ParticipantFile],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[list[ParticipantSummary], dict[str, ParticipantFile]]:
    summaries: list[ParticipantSummary] = []
    file_lookup: dict[str, ParticipantFile] = {}
    skipped_files = 0

    for participant_file in participant_files:
        file_lookup[participant_file.participant_id] = participant_file
        cleaned = load_and_clean_participant_csv(
            file_path=participant_file.file_path,
            participant_id=participant_file.participant_id,
            config=config,
            logger=logger,
        )
        if cleaned is None:
            skipped_files += 1
            continue

        summary = summarize_participant(cleaned, config)
        if summary.window_count == 0:
            logger.warning(
                "Skipping %s because no usable windows were produced after cleaning.",
                participant_file.file_path.name,
            )
            skipped_files += 1
            continue
        summaries.append(summary)

    if not summaries:
        raise ValueError("All scanned files were invalid or produced zero usable windows.")

    logger.info(
        "Prepared participant summaries for %d file(s); skipped %d file(s).",
        len(summaries),
        skipped_files,
    )
    return summaries, file_lookup


def _compute_normalization_stats(
    participant_ids: list[str],
    participant_files: dict[str, ParticipantFile],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> dict[str, list[float]]:
    sums = np.zeros(len(config.feature_columns), dtype=np.float64)
    squared_sums = np.zeros(len(config.feature_columns), dtype=np.float64)
    total_values = 0

    for participant_id in participant_ids:
        participant_file = participant_files[participant_id]
        cleaned = load_and_clean_participant_csv(
            file_path=participant_file.file_path,
            participant_id=participant_id,
            config=config,
            logger=logger,
        )
        if cleaned is None:
            continue
        windowed = build_windows_from_clean_dataframe(cleaned, config, include_features=True)
        if windowed.features is None or len(windowed.features) == 0:
            continue
        features = windowed.features.astype(np.float64, copy=False)
        sums += features.sum(axis=(0, 2))
        squared_sums += np.square(features).sum(axis=(0, 2))
        total_values += int(features.shape[0] * features.shape[2])

    if total_values == 0:
        raise ValueError("Zero training windows available to compute normalization statistics.")

    means = sums / total_values
    variances = (squared_sums / total_values) - np.square(means)
    stds = np.sqrt(np.maximum(variances, 1e-8))
    return {
        "mean": means.tolist(),
        "std": stds.tolist(),
        "feature_columns": list(config.feature_columns),
    }


def _compute_class_weights_from_summaries(
    train_participant_ids: list[str],
    summary_lookup: dict[str, ParticipantSummary],
) -> tuple[list[float], dict[str, int]]:
    name_to_id = {name: idx for idx, name in ID_TO_NAME.items()}
    counts = np.zeros(len(ID_TO_NAME), dtype=np.int64)
    for participant_id in train_participant_ids:
        summary = summary_lookup[participant_id]
        for class_name, count in summary.window_class_counts.items():
            counts[name_to_id[class_name]] += int(count)

    if counts.sum() == 0:
        raise ValueError("Training split contains zero windows.")

    safe_counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(counts) * safe_counts.astype(np.float64))
    label_counts = {ID_TO_NAME[idx]: int(count) for idx, count in enumerate(counts)}
    return weights.tolist(), label_counts


def _normalize_features(features: np.ndarray, normalization_stats: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(normalization_stats["mean"], dtype=np.float32).reshape(1, -1, 1)
    std = np.asarray(normalization_stats["std"], dtype=np.float32).reshape(1, -1, 1)
    return (features - mean) / std


def _prepare_split_arrays(
    split_name: str,
    participant_ids: list[str],
    participant_files: dict[str, ParticipantFile],
    config: ExperimentConfig,
    normalization_stats: dict[str, list[float]],
    logger: logging.Logger,
) -> PreparedSplitData:
    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for participant_id in participant_ids:
        participant_file = participant_files[participant_id]
        cleaned = load_and_clean_participant_csv(
            file_path=participant_file.file_path,
            participant_id=participant_id,
            config=config,
            logger=logger,
        )
        if cleaned is None:
            continue
        windowed = build_windows_from_clean_dataframe(cleaned, config, include_features=True)
        if windowed.features is None or len(windowed.features) == 0:
            continue
        normalized_features = _normalize_features(windowed.features, normalization_stats)
        features_list.append(normalized_features.astype(np.float32, copy=False))
        labels_list.append(windowed.labels.astype(np.int64, copy=False))

        for index, (start_timestamp, end_timestamp, label_id) in enumerate(
            zip(windowed.start_timestamps, windowed.end_timestamps, windowed.labels, strict=True)
        ):
            metadata.append(
                {
                    "participant_id": participant_id,
                    "window_index": index,
                    "window_start_timestamp": float(start_timestamp),
                    "window_end_timestamp": float(end_timestamp),
                    "label_id": int(label_id),
                    "label_name": ID_TO_NAME[int(label_id)],
                }
            )

    if not features_list:
        raise ValueError(f"Split '{split_name}' produced zero usable windows.")

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    label_counts = np.bincount(labels, minlength=len(ID_TO_NAME))

    return PreparedSplitData(
        name=split_name,
        dataset=SleepWindowDataset(features=features, labels=labels),
        metadata=metadata,
        label_counts={ID_TO_NAME[idx]: int(count) for idx, count in enumerate(label_counts)},
        num_windows=int(len(labels)),
    )


def _load_or_create_splits(
    summaries: list[ParticipantSummary],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> ParticipantSplits:
    if config.split_path.exists():
        payload = load_json(config.split_path)
        available_ids = {summary.participant_id for summary in summaries}
        splits = load_existing_splits(payload, available_ids)
        logger.info("Loaded existing participant split file from %s.", config.split_path)
        return splits

    splits = create_participant_splits(summaries, config, logger)
    save_json(splits_to_dict(splits, config), config.split_path)
    logger.info("Saved participant split file to %s.", config.split_path)
    return splits


def _load_or_compute_normalization_stats(
    splits: ParticipantSplits,
    participant_files: dict[str, ParticipantFile],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> dict[str, list[float]]:
    if config.normalization_path.exists():
        payload = load_json(config.normalization_path)
        logger.info("Loaded normalization statistics from %s.", config.normalization_path)
        return payload

    stats = _compute_normalization_stats(
        participant_ids=splits.train,
        participant_files=participant_files,
        config=config,
        logger=logger,
    )
    save_json(stats, config.normalization_path)
    logger.info("Saved normalization statistics to %s.", config.normalization_path)
    return stats


def _load_or_compute_class_weights(
    splits: ParticipantSplits,
    summaries: list[ParticipantSummary],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> list[float]:
    if config.class_weights_path.exists():
        payload = load_json(config.class_weights_path)
        logger.info("Loaded class weights from %s.", config.class_weights_path)
        return list(payload["class_weights"])

    summary_lookup = {summary.participant_id: summary for summary in summaries}
    class_weights, label_counts = _compute_class_weights_from_summaries(splits.train, summary_lookup)
    save_json(
        {
            "class_weights": class_weights,
            "train_label_counts": label_counts,
        },
        config.class_weights_path,
    )
    logger.info("Saved class weights to %s.", config.class_weights_path)
    return class_weights


def prepare_datasets(
    config: ExperimentConfig,
    logger: logging.Logger,
    requested_splits: tuple[str, ...] = ("train", "val", "test"),
) -> PreparedDataBundle:
    """Run the full data preparation pipeline."""

    config.validate()
    config.ensure_output_dirs()

    participant_files = scan_dataset_files(
        dataset_root=config.dataset_root,
        logger=logger,
        limit_files=config.limit_files,
    )
    summaries, participant_lookup = _collect_participant_summaries(participant_files, config, logger)
    save_json(
        {
            "dataset_root": str(config.dataset_root),
            "limit_files": config.limit_files,
            "participants": [participant_summary_to_dict(summary) for summary in summaries],
        },
        config.manifest_path,
    )

    splits = _load_or_create_splits(summaries, config, logger)
    normalization_stats = _load_or_compute_normalization_stats(splits, participant_lookup, config, logger)
    class_weights = _load_or_compute_class_weights(splits, summaries, config, logger)

    split_to_ids = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }
    prepared_splits: dict[str, PreparedSplitData] = {}
    for split_name in requested_splits:
        prepared_splits[split_name] = _prepare_split_arrays(
            split_name=split_name,
            participant_ids=split_to_ids[split_name],
            participant_files=participant_lookup,
            config=config,
            normalization_stats=normalization_stats,
            logger=logger,
        )

    summary_lookup = {summary.participant_id: summary for summary in summaries}
    dataset_summary = {
        "requested_splits": list(requested_splits),
        "splits": {
            split_name: {
                "participants": split_to_ids[split_name],
                "participant_count": len(split_to_ids[split_name]),
                "num_windows": prepared_splits[split_name].num_windows if split_name in prepared_splits else None,
                "window_label_counts": prepared_splits[split_name].label_counts if split_name in prepared_splits else None,
                "source_window_counts_by_participant": {
                    participant_id: summary_lookup[participant_id].window_count
                    for participant_id in split_to_ids[split_name]
                },
            }
            for split_name in split_to_ids
        },
    }
    save_json(dataset_summary, config.dataset_summary_path)

    return PreparedDataBundle(
        config=config,
        participant_files=participant_lookup,
        summaries=summaries,
        splits=splits,
        normalization_stats=normalization_stats,
        class_weights=class_weights,
        prepared_splits=prepared_splits,
    )
