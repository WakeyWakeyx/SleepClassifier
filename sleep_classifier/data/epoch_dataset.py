"""Dataset preparation and PyTorch datasets for epoch-sequence modeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sleep_classifier.experiment_config import ExperimentConfig
from sleep_classifier.data.epoch_preprocessing import (
    CachedParticipantData,
    ParticipantSummary,
    build_window_manifest_rows,
    extract_window_features,
    load_and_clean_participant_csv,
    load_cached_participant,
    participant_summary_from_dict,
    participant_summary_to_dict,
    save_cached_participant,
    summarize_participant,
)
from sleep_classifier.data.scan_dataset import ParticipantFile, scan_dataset_files
from sleep_classifier.data.splits import (
    ParticipantSplits,
    create_participant_splits,
    load_existing_splits,
    splits_to_dict,
)
from sleep_classifier.label_mapping import FINAL_ID_TO_NAME, RAW_ID_TO_NAME
from sleep_classifier.utils import load_json, save_json


@dataclass(slots=True)
class PreparedSplitData:
    """Features, multitask labels, metadata, and distribution for one split."""

    name: str
    dataset: "SleepWindowDataset"
    metadata: list[dict[str, Any]]
    label_counts: dict[str, int]
    raw_label_counts: dict[str, int]
    num_windows: int


@dataclass(slots=True)
class PreparedDataBundle:
    """Container returned by dataset preparation."""

    config: ExperimentConfig
    participant_files: dict[str, ParticipantFile]
    summaries: list[ParticipantSummary]
    splits: ParticipantSplits
    normalization_stats: dict[str, Any]
    class_weights: list[float]
    raw_class_weights: list[float]
    final_class_counts: list[int]
    raw_class_counts: list[int]
    transition_matrix: list[list[float]]
    class_priors: list[float]
    prepared_splits: dict[str, PreparedSplitData]
    window_manifest: pd.DataFrame


class SleepWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Tensor dataset for epoch-sequence sleep-stage classification."""

    def __init__(self, features: np.ndarray, raw_labels: np.ndarray, final_labels: np.ndarray) -> None:
        if features.ndim != 4:
            raise ValueError(
                "Expected features with shape "
                f"[num_windows, num_epochs, channels, epoch_length], got {features.shape}"
            )
        if len(features) != len(raw_labels) or len(features) != len(final_labels):
            raise ValueError("Features and labels must contain the same number of windows.")
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.raw_labels = torch.from_numpy(raw_labels.astype(np.int64, copy=False))
        self.final_labels = torch.from_numpy(final_labels.astype(np.int64, copy=False))
        self.labels = self.final_labels

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "raw_target": self.raw_labels[index],
            "final_target": self.final_labels[index],
        }


class ParticipantStore:
    """Lazy participant cache backed by in-memory objects and saved npz files."""

    def __init__(
        self,
        cache_paths: dict[str, Path],
        initial_cache: dict[str, CachedParticipantData] | None = None,
    ) -> None:
        self.cache_paths = dict(cache_paths)
        self._loaded = dict(initial_cache or {})

    def load(self, participant_id: str) -> CachedParticipantData:
        if participant_id not in self._loaded:
            cache_path = self.cache_paths.get(participant_id)
            if cache_path is None:
                raise KeyError(f"No cached participant available for {participant_id}")
            self._loaded[participant_id] = load_cached_participant(cache_path)
        return self._loaded[participant_id]


def _normalize_features(features: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, -1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, -1, 1)
    return (features - mean) / std


def _participant_cache_path(config: ExperimentConfig, participant_id: str) -> Path:
    return config.cleaned_participants_dir / f"{participant_id}.npz"


def _load_existing_summary_lookup(config: ExperimentConfig) -> dict[str, ParticipantSummary]:
    if not config.use_cache or config.rebuild_cache or not config.participant_summary_path.exists():
        return {}
    payload = load_json(config.participant_summary_path)
    return {
        item["participant_id"]: participant_summary_from_dict(item)
        for item in payload.get("participants", [])
    }


def _collect_participant_summaries(
    participant_files: list[ParticipantFile],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[list[ParticipantSummary], dict[str, ParticipantFile], ParticipantStore]:
    summaries: list[ParticipantSummary] = []
    file_lookup: dict[str, ParticipantFile] = {}
    in_memory_cache: dict[str, CachedParticipantData] = {}
    cache_paths: dict[str, Path] = {}
    summary_lookup = _load_existing_summary_lookup(config)
    skipped_files = 0

    for participant_file in participant_files:
        participant_id = participant_file.participant_id
        file_lookup[participant_id] = participant_file
        cache_path = _participant_cache_path(config, participant_id)
        cache_paths[participant_id] = cache_path

        if (
            config.use_cache
            and not config.rebuild_cache
            and cache_path.exists()
            and participant_id in summary_lookup
        ):
            summaries.append(summary_lookup[participant_id])
            continue

        cleaned = load_and_clean_participant_csv(
            file_path=participant_file.file_path,
            participant_id=participant_id,
            config=config,
            logger=logger,
        )
        if cleaned is None:
            skipped_files += 1
            continue

        summary = summarize_participant(cleaned, config)
        if summary.window_count == 0:
            logger.warning(
                "Skipping %s because no usable epoch sequences were produced after cleaning.",
                participant_file.file_path.name,
            )
            skipped_files += 1
            continue

        summaries.append(summary)
        in_memory_cache[participant_id] = cleaned
        if config.use_cache:
            save_cached_participant(cleaned, cache_path)

    if not summaries:
        raise ValueError("All scanned files were invalid or produced zero usable windows.")

    usable_ids = {summary.participant_id for summary in summaries}
    filtered_lookup = {
        participant_id: participant_file
        for participant_id, participant_file in file_lookup.items()
        if participant_id in usable_ids
    }
    filtered_cache_paths = {
        participant_id: cache_path
        for participant_id, cache_path in cache_paths.items()
        if participant_id in usable_ids
    }
    if config.use_cache:
        save_json(
            {
                "dataset_root": str(config.dataset_root),
                "limit_files": config.limit_files,
                "input_feature_columns": list(config.input_feature_columns),
                "participants": [participant_summary_to_dict(summary) for summary in summaries],
            },
            config.participant_summary_path,
        )

    logger.info(
        "Prepared participant summaries for %d file(s); skipped %d file(s).",
        len(summaries),
        skipped_files,
    )
    return summaries, filtered_lookup, ParticipantStore(filtered_cache_paths, initial_cache=in_memory_cache)


def _load_or_create_splits(
    summaries: list[ParticipantSummary],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> ParticipantSplits:
    if config.use_cache and config.split_path.exists() and not config.rebuild_cache:
        payload = load_json(config.split_path)
        available_ids = {summary.participant_id for summary in summaries}
        splits = load_existing_splits(payload, available_ids)
        logger.info("Loaded existing participant split file from %s.", config.split_path)
        return splits

    splits = create_participant_splits(summaries, config, logger)
    save_json(splits_to_dict(splits, config), config.split_path)
    logger.info("Saved participant split file to %s.", config.split_path)
    return splits


def _build_window_manifest(
    participant_store: ParticipantStore,
    splits: ParticipantSplits,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_to_ids = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }
    for split_name, participant_ids in split_to_ids.items():
        stride_epochs = config.stride_epochs_for_split(split_name)
        for participant_id in participant_ids:
            cleaned = participant_store.load(participant_id)
            participant_rows, skipped_low_quality, skipped_short = build_window_manifest_rows(
                cleaned=cleaned,
                config=config,
                split_name=split_name,
                stride_epochs=stride_epochs,
            )
            if skipped_low_quality:
                logger.info(
                    "Skipped %d low-quality %s window(s) for participant %s.",
                    skipped_low_quality,
                    split_name,
                    participant_id,
                )
            if skipped_short:
                logger.info(
                    "Participant %s is too short for %s epoch sequences.",
                    participant_id,
                    split_name,
                )
            rows.extend(participant_rows)

    if not rows:
        raise ValueError("Window manifest generation produced zero windows across all splits.")

    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values(
        ["split", "participant_id", "window_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    manifest.to_csv(config.window_manifest_path, index=False)
    logger.info("Saved window manifest to %s.", config.window_manifest_path)
    return manifest


def _load_or_create_window_manifest(
    participant_store: ParticipantStore,
    splits: ParticipantSplits,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    if config.use_cache and config.window_manifest_path.exists() and not config.rebuild_cache:
        manifest = pd.read_csv(config.window_manifest_path)
        logger.info("Loaded window manifest from %s.", config.window_manifest_path)
        return manifest
    return _build_window_manifest(participant_store, splits, config, logger)


def _compute_feature_stats(feature_matrix: np.ndarray) -> dict[str, list[float]]:
    means = feature_matrix.mean(axis=0, dtype=np.float64)
    stds = np.maximum(feature_matrix.std(axis=0, dtype=np.float64), 1e-6)
    return {
        "mean": means.tolist(),
        "std": stds.tolist(),
    }


def _load_or_compute_normalization_stats(
    participant_store: ParticipantStore,
    manifest: pd.DataFrame,
    splits: ParticipantSplits,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    if config.use_cache and config.normalization_path.exists() and not config.rebuild_cache:
        payload = load_json(config.normalization_path)
        logger.info("Loaded normalization statistics from %s.", config.normalization_path)
        return payload

    if config.normalization_mode == "global_train":
        train_features: list[np.ndarray] = []
        for participant_id in splits.train:
            cleaned = participant_store.load(participant_id)
            train_features.append(cleaned.features.astype(np.float32, copy=False))
        if not train_features:
            raise ValueError("Zero training participants available to compute normalization statistics.")
        feature_matrix = np.concatenate(train_features, axis=0)
        stats = _compute_feature_stats(feature_matrix)
        payload: dict[str, Any] = {
            "mode": "global_train",
            "feature_columns": list(config.input_feature_columns),
            **stats,
        }
    else:
        participant_stats: dict[str, dict[str, Any]] = {}
        participant_ids = sorted(set(manifest["participant_id"].tolist()))
        for participant_id in participant_ids:
            cleaned = participant_store.load(participant_id)
            participant_stats[participant_id] = _compute_feature_stats(
                cleaned.features.astype(np.float32, copy=False)
            )
        payload = {
            "mode": "participant",
            "feature_columns": list(config.input_feature_columns),
            "participants": participant_stats,
        }

    save_json(payload, config.normalization_path)
    logger.info("Saved normalization statistics to %s.", config.normalization_path)
    return payload


def _compute_inverse_frequency_weights(counts: np.ndarray) -> list[float]:
    safe_counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(counts) * safe_counts.astype(np.float64))
    return weights.tolist()


def _load_or_compute_class_stats(
    manifest: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[list[float], list[float], list[int], list[int]]:
    if config.use_cache and config.class_weights_path.exists() and not config.rebuild_cache:
        payload = load_json(config.class_weights_path)
        logger.info("Loaded class weights from %s.", config.class_weights_path)
        return (
            list(payload["final_class_weights"]),
            list(payload["raw_class_weights"]),
            [int(value) for value in payload["final_class_counts"]],
            [int(value) for value in payload["raw_class_counts"]],
        )

    train_manifest = manifest.loc[manifest["split"] == "train"]
    final_counts = np.bincount(
        train_manifest["final_label_id"].to_numpy(dtype=np.int64),
        minlength=len(FINAL_ID_TO_NAME),
    )
    raw_counts = np.bincount(
        train_manifest["raw_label_id"].to_numpy(dtype=np.int64),
        minlength=len(RAW_ID_TO_NAME),
    )
    if final_counts.sum() == 0 or raw_counts.sum() == 0:
        raise ValueError("Training split contains zero windows.")

    final_weights = _compute_inverse_frequency_weights(final_counts)
    raw_weights = _compute_inverse_frequency_weights(raw_counts)
    payload = {
        "final_class_weights": final_weights,
        "raw_class_weights": raw_weights,
        "final_class_counts": final_counts.tolist(),
        "raw_class_counts": raw_counts.tolist(),
        "train_final_label_counts": {
            FINAL_ID_TO_NAME[idx]: int(count) for idx, count in enumerate(final_counts)
        },
        "train_raw_label_counts": {
            RAW_ID_TO_NAME[idx]: int(count) for idx, count in enumerate(raw_counts)
        },
    }
    save_json(payload, config.class_weights_path)
    logger.info("Saved class weights to %s.", config.class_weights_path)
    return final_weights, raw_weights, final_counts.tolist(), raw_counts.tolist()


def _load_or_compute_transition_stats(
    manifest: pd.DataFrame,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> tuple[list[list[float]], list[float]]:
    if config.use_cache and config.transition_matrix_path.exists() and not config.rebuild_cache:
        payload = load_json(config.transition_matrix_path)
        logger.info("Loaded transition statistics from %s.", config.transition_matrix_path)
        return (
            [list(map(float, row)) for row in payload["transition_matrix"]],
            [float(value) for value in payload["class_priors"]],
        )

    train_manifest = manifest.loc[manifest["split"] == "train"].copy()
    train_manifest = train_manifest.sort_values(
        ["participant_id", "center_epoch_index"],
        kind="mergesort",
    )
    transition_counts = np.ones((len(FINAL_ID_TO_NAME), len(FINAL_ID_TO_NAME)), dtype=np.float64)
    for _, participant_frame in train_manifest.groupby("participant_id", sort=False):
        labels = participant_frame["final_label_id"].to_numpy(dtype=np.int64)
        epochs = participant_frame["center_epoch_index"].to_numpy(dtype=np.int64)
        strides = participant_frame["stride_epochs"].to_numpy(dtype=np.int64)
        if len(labels) < 2:
            continue
        for previous_label, next_label, previous_epoch, next_epoch, previous_stride in zip(
            labels[:-1],
            labels[1:],
            epochs[:-1],
            epochs[1:],
            strides[:-1],
            strict=False,
        ):
            if int(next_epoch) - int(previous_epoch) != int(previous_stride):
                continue
            transition_counts[int(previous_label), int(next_label)] += 1.0

    priors = np.bincount(
        train_manifest["final_label_id"].to_numpy(dtype=np.int64),
        minlength=len(FINAL_ID_TO_NAME),
    ).astype(np.float64)
    priors = np.maximum(priors, 1.0)
    priors = priors / priors.sum()
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    payload = {
        "transition_matrix": transition_matrix.tolist(),
        "class_priors": priors.tolist(),
    }
    save_json(payload, config.transition_matrix_path)
    logger.info("Saved transition statistics to %s.", config.transition_matrix_path)
    return transition_matrix.tolist(), priors.tolist()


def _stats_for_participant(
    normalization_stats: dict[str, Any],
    participant_id: str,
) -> dict[str, Any]:
    if normalization_stats["mode"] == "global_train":
        return normalization_stats
    return normalization_stats["participants"][participant_id]


def _reshape_epoch_sequences(features: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    if features.shape[2] != config.context_window_size:
        raise ValueError(
            "Unexpected sequence length for epoch reshaping: "
            f"expected {config.context_window_size}, got {features.shape[2]}"
        )
    num_windows, num_channels, _ = features.shape
    reshaped = features.reshape(
        num_windows,
        num_channels,
        config.context_length_epochs,
        config.center_window_size,
    )
    return np.transpose(reshaped, (0, 2, 1, 3))


def _prepare_split_arrays(
    split_name: str,
    manifest: pd.DataFrame,
    participant_store: ParticipantStore,
    normalization_stats: dict[str, Any],
    config: ExperimentConfig,
) -> PreparedSplitData:
    split_manifest = manifest.loc[manifest["split"] == split_name].copy()
    if split_manifest.empty:
        raise ValueError(f"Split '{split_name}' produced zero usable windows.")

    feature_batches: list[np.ndarray] = []
    raw_label_batches: list[np.ndarray] = []
    final_label_batches: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for participant_id, participant_frame in split_manifest.groupby("participant_id", sort=False):
        participant_frame = participant_frame.sort_values("window_index", kind="mergesort")
        cleaned = participant_store.load(participant_id)
        window_features = extract_window_features(cleaned, participant_frame)
        stats = _stats_for_participant(normalization_stats, participant_id)
        normalized_features = _normalize_features(window_features, stats)
        feature_batches.append(_reshape_epoch_sequences(normalized_features.astype(np.float32, copy=False), config))
        raw_label_batches.append(participant_frame["raw_label_id"].to_numpy(dtype=np.int64))
        final_label_batches.append(participant_frame["final_label_id"].to_numpy(dtype=np.int64))
        metadata.extend(participant_frame.to_dict(orient="records"))

    features = np.concatenate(feature_batches, axis=0)
    raw_labels = np.concatenate(raw_label_batches, axis=0)
    final_labels = np.concatenate(final_label_batches, axis=0)
    raw_counts = np.bincount(raw_labels, minlength=len(RAW_ID_TO_NAME))
    final_counts = np.bincount(final_labels, minlength=len(FINAL_ID_TO_NAME))
    return PreparedSplitData(
        name=split_name,
        dataset=SleepWindowDataset(features=features, raw_labels=raw_labels, final_labels=final_labels),
        metadata=metadata,
        label_counts={FINAL_ID_TO_NAME[idx]: int(count) for idx, count in enumerate(final_counts)},
        raw_label_counts={RAW_ID_TO_NAME[idx]: int(count) for idx, count in enumerate(raw_counts)},
        num_windows=int(len(final_labels)),
    )


def prepare_datasets(
    config: ExperimentConfig,
    logger: logging.Logger,
    requested_splits: tuple[str, ...] = ("train", "val", "test"),
) -> PreparedDataBundle:
    """Run the cached data preparation pipeline."""

    config.validate()
    config.ensure_output_dirs()

    participant_files = scan_dataset_files(
        dataset_root=config.dataset_root,
        logger=logger,
        limit_files=config.limit_files,
    )
    summaries, participant_lookup, participant_store = _collect_participant_summaries(
        participant_files,
        config,
        logger,
    )
    splits = _load_or_create_splits(summaries, config, logger)
    window_manifest = _load_or_create_window_manifest(participant_store, splits, config, logger)
    normalization_stats = _load_or_compute_normalization_stats(
        participant_store=participant_store,
        manifest=window_manifest,
        splits=splits,
        config=config,
        logger=logger,
    )
    final_class_weights, raw_class_weights, final_class_counts, raw_class_counts = _load_or_compute_class_stats(
        window_manifest,
        config,
        logger,
    )
    transition_matrix, class_priors = _load_or_compute_transition_stats(window_manifest, config, logger)

    prepared_splits: dict[str, PreparedSplitData] = {}
    for split_name in requested_splits:
        prepared_splits[split_name] = _prepare_split_arrays(
            split_name=split_name,
            manifest=window_manifest,
            participant_store=participant_store,
            normalization_stats=normalization_stats,
            config=config,
        )

    split_to_ids = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }
    summary_lookup = {summary.participant_id: summary for summary in summaries}
    dataset_summary = {
        "requested_splits": list(requested_splits),
        "context_length_epochs": config.context_length_epochs,
        "context_window_seconds": config.context_window_seconds,
        "center_epoch_seconds": config.center_epoch_seconds,
        "train_stride_epochs": config.train_stride_epochs,
        "val_stride_epochs": config.val_stride_epochs,
        "test_stride_epochs": config.test_stride_epochs,
        "input_feature_columns": list(config.input_feature_columns),
        "normalization_mode": config.normalization_mode,
        "splits": {
            split_name: {
                "participants": split_to_ids[split_name],
                "participant_count": len(split_to_ids[split_name]),
                "num_windows": int((window_manifest["split"] == split_name).sum()),
                "final_window_label_counts": (
                    prepared_splits[split_name].label_counts
                    if split_name in prepared_splits
                    else None
                ),
                "raw_window_label_counts": (
                    prepared_splits[split_name].raw_label_counts
                    if split_name in prepared_splits
                    else None
                ),
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
        class_weights=final_class_weights,
        raw_class_weights=raw_class_weights,
        final_class_counts=final_class_counts,
        raw_class_counts=raw_class_counts,
        transition_matrix=transition_matrix,
        class_priors=class_priors,
        prepared_splits=prepared_splits,
        window_manifest=window_manifest,
    )
