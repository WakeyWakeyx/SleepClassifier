"""Participant-level dataset splitting helpers."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

from sklearn.model_selection import train_test_split

from sleep_classifier.data.epoch_preprocessing import ParticipantSummary
from sleep_classifier.experiment_config import ExperimentConfig


@dataclass(slots=True)
class ParticipantSplits:
    """Train, validation, and test participant ids."""

    train: list[str]
    val: list[str]
    test: list[str]
    stratified: bool


def _split_payload_to_dataclass(payload: dict[str, Any]) -> ParticipantSplits:
    return ParticipantSplits(
        train=list(payload["train"]),
        val=list(payload["val"]),
        test=list(payload["test"]),
        stratified=bool(payload.get("stratified", False)),
    )


def load_existing_splits(
    payload: dict[str, Any],
    available_participants: set[str],
) -> ParticipantSplits:
    """Load and validate a saved split file."""

    splits = _split_payload_to_dataclass(payload)
    split_sets = {
        "train": set(splits.train),
        "val": set(splits.val),
        "test": set(splits.test),
    }

    overlap = (split_sets["train"] & split_sets["val"]) | (split_sets["train"] & split_sets["test"]) | (split_sets["val"] & split_sets["test"])
    if overlap:
        raise ValueError(f"Saved splits contain overlapping participants: {sorted(overlap)}")

    all_split_participants = split_sets["train"] | split_sets["val"] | split_sets["test"]
    missing_from_scan = all_split_participants - available_participants
    if missing_from_scan:
        raise ValueError(
            "Saved split file references participants that are not available in the current scan: "
            + ", ".join(sorted(missing_from_scan))
        )
    return splits


def _try_stratified_split(
    participant_ids: list[str],
    dominant_labels: list[int],
    config: ExperimentConfig,
) -> ParticipantSplits:
    total_ratio = config.train_ratio + config.val_ratio
    val_ratio_of_remaining = config.val_ratio / total_ratio

    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        participant_ids,
        dominant_labels,
        test_size=config.test_ratio,
        shuffle=True,
        random_state=config.random_seed,
        stratify=dominant_labels,
    )
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        train_val_labels,
        test_size=val_ratio_of_remaining,
        shuffle=True,
        random_state=config.random_seed,
        stratify=train_val_labels,
    )
    return ParticipantSplits(
        train=sorted(train_ids),
        val=sorted(val_ids),
        test=sorted(test_ids),
        stratified=True,
    )


def _fallback_random_split(
    participant_ids: list[str],
    config: ExperimentConfig,
) -> ParticipantSplits:
    shuffled_ids = list(participant_ids)
    rng = random.Random(config.random_seed)
    rng.shuffle(shuffled_ids)

    total = len(shuffled_ids)
    train_count = max(1, int(round(total * config.train_ratio)))
    val_count = max(1, int(round(total * config.val_ratio)))
    test_count = total - train_count - val_count

    if test_count <= 0:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1

    if train_count + val_count + test_count != total:
        train_count = total - val_count - test_count

    train_ids = sorted(shuffled_ids[:train_count])
    val_ids = sorted(shuffled_ids[train_count : train_count + val_count])
    test_ids = sorted(shuffled_ids[train_count + val_count :])
    return ParticipantSplits(train=train_ids, val=val_ids, test=test_ids, stratified=False)


def create_participant_splits(
    summaries: list[ParticipantSummary],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> ParticipantSplits:
    """Create deterministic subject-level train/val/test splits."""

    usable = [summary for summary in summaries if summary.window_count > 0]
    if len(usable) < 3:
        raise ValueError(
            "Need at least 3 participants with usable windows to create train/val/test splits."
        )

    participant_ids = [summary.participant_id for summary in usable]
    dominant_labels = [
        summary.dominant_final_class_id if summary.dominant_final_class_id is not None else 0
        for summary in usable
    ]

    try:
        splits = _try_stratified_split(participant_ids, dominant_labels, config)
        logger.info("Created participant-level splits using dominant-class stratification.")
        return splits
    except ValueError as exc:
        logger.warning(
            "Stratified participant split was not possible (%s). Falling back to deterministic random splitting.",
            exc,
        )
        return _fallback_random_split(participant_ids, config)


def splits_to_dict(
    splits: ParticipantSplits,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Serialize split assignments for JSON output."""

    return {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
        "stratified": splits.stratified,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "random_seed": config.random_seed,
    }
