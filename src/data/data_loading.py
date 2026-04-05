"""Dataset discovery, CSV loading, and participant-level split management."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import DataConfig
from src.utils.utilities import load_json, save_json


@dataclass(slots=True)
class ParticipantData:
    """Container for a single participant/session dataframe."""

    participant_id: str
    source_path: Path
    frame: pd.DataFrame


@dataclass(slots=True, frozen=True)
class ParticipantSplit:
    """Grouped file lists for reproducible participant-wise splitting."""

    train_files: tuple[Path, ...]
    val_files: tuple[Path, ...]
    test_files: tuple[Path, ...]


def discover_participant_files(dataset_dir: Path) -> list[Path]:
    """Recursively find participant CSV files under the configured dataset root."""

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found: {dataset_dir}. "
            "Set DREAMT_DATASET_DIR or update src/config.py."
        )

    files = sorted(path for path in dataset_dir.rglob("*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files were found under {dataset_dir}.")
    return files


def load_participant_file(file_path: Path) -> ParticipantData:
    """Load a single participant CSV into a dataframe."""

    frame = pd.read_csv(file_path, low_memory=False)
    return ParticipantData(
        participant_id=file_path.stem,
        source_path=file_path,
        frame=frame,
    )


def load_participant_files(file_paths: Sequence[Path]) -> list[ParticipantData]:
    """Load multiple participant CSV files with progress reporting."""

    participants: list[ParticipantData] = []
    for file_path in tqdm(file_paths, desc="Loading participants", unit="file"):
        participants.append(load_participant_file(file_path))
    return participants


def split_participant_files(
    file_paths: Sequence[Path],
    data_config: DataConfig,
    manifest_path: Path,
    dataset_root: Path,
    seed: int,
) -> ParticipantSplit:
    """Create or reuse a deterministic participant-level split manifest."""

    file_paths = sorted(file_paths)
    discovered_rel_paths = [_to_relative_string(path, dataset_root) for path in file_paths]

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if sorted(manifest.get("all_files", [])) == sorted(discovered_rel_paths):
            return ParticipantSplit(
                train_files=tuple(dataset_root / rel_path for rel_path in manifest["train_files"]),
                val_files=tuple(dataset_root / rel_path for rel_path in manifest["val_files"]),
                test_files=tuple(dataset_root / rel_path for rel_path in manifest["test_files"]),
            )

    if len(file_paths) < 3:
        raise ValueError(
            "At least 3 participant files are required to create train/val/test splits."
        )

    counts = _compute_split_counts(
        total_count=len(file_paths),
        ratios=(
            data_config.train_ratio,
            data_config.val_ratio,
            data_config.test_ratio,
        ),
    )
    shuffled = list(file_paths)
    random.Random(seed).shuffle(shuffled)

    train_count, val_count, test_count = counts
    train_files = tuple(sorted(shuffled[:train_count]))
    val_files = tuple(sorted(shuffled[train_count : train_count + val_count]))
    test_files = tuple(sorted(shuffled[train_count + val_count : train_count + val_count + test_count]))

    split = ParticipantSplit(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
    )
    manifest = {
        "dataset_root": str(dataset_root),
        "seed": seed,
        "ratios": {
            "train": data_config.train_ratio,
            "val": data_config.val_ratio,
            "test": data_config.test_ratio,
        },
        "all_files": discovered_rel_paths,
        "train_files": [_to_relative_string(path, dataset_root) for path in train_files],
        "val_files": [_to_relative_string(path, dataset_root) for path in val_files],
        "test_files": [_to_relative_string(path, dataset_root) for path in test_files],
        "counts": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
    }
    save_json(manifest, manifest_path)
    return split


def _to_relative_string(file_path: Path, dataset_root: Path) -> str:
    """Serialize paths relative to the dataset root when possible."""

    return file_path.relative_to(dataset_root).as_posix()


def _compute_split_counts(
    total_count: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Convert split ratios into concrete counts while keeping every split non-empty."""

    raw_counts = np.array(ratios, dtype=np.float64) * total_count
    counts = np.floor(raw_counts).astype(int)
    remainder = int(total_count - counts.sum())

    fractional_parts = raw_counts - counts
    for index in np.argsort(-fractional_parts)[:remainder]:
        counts[index] += 1

    minimums = np.array([1, 1, 1], dtype=int)
    for index, minimum in enumerate(minimums):
        while counts[index] < minimum:
            donor_index = int(np.argmax(counts))
            if donor_index == index or counts[donor_index] <= minimums[donor_index]:
                raise ValueError("Unable to build non-empty train/val/test splits.")
            counts[donor_index] -= 1
            counts[index] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])
