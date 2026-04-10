"""Dataset discovery, CSV loading, and participant-level split management."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def load_participant_file(
    file_path: Path,
    data_config: DataConfig | None = None,
) -> ParticipantData:
    """Load a single participant CSV into a dataframe."""

    read_csv_kwargs: dict[str, Any] = {"low_memory": False}
    if data_config is not None:
        required_columns = set(_participant_csv_columns(data_config))
        read_csv_kwargs["usecols"] = lambda column_name: column_name in required_columns

    frame = pd.read_csv(file_path, **read_csv_kwargs)
    return ParticipantData(
        participant_id=file_path.stem,
        source_path=file_path,
        frame=frame,
    )


def load_participant_files(
    file_paths: Sequence[Path],
    data_config: DataConfig | None = None,
) -> list[ParticipantData]:
    """Load multiple participant CSV files with progress reporting."""

    participants: list[ParticipantData] = []
    for file_path in tqdm(file_paths, desc="Loading participants", unit="file"):
        participants.append(load_participant_file(file_path, data_config=data_config))
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
            reused_split = ParticipantSplit(
                train_files=tuple(dataset_root / rel_path for rel_path in manifest["train_files"]),
                val_files=tuple(dataset_root / rel_path for rel_path in manifest["val_files"]),
                test_files=tuple(dataset_root / rel_path for rel_path in manifest["test_files"]),
            )
            _validate_split_integrity(reused_split)
            return reused_split

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
    test_files = tuple(
        sorted(shuffled[train_count + val_count : train_count + val_count + test_count])
    )

    split = ParticipantSplit(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
    )
    _validate_split_integrity(split)
    save_json(
        build_split_manifest_payload(
            split=split,
            dataset_root=dataset_root,
            all_file_paths=file_paths,
            seed=seed,
            ratios={
                "train": data_config.train_ratio,
                "val": data_config.val_ratio,
                "test": data_config.test_ratio,
            },
        ),
        manifest_path,
    )
    return split


def build_split_manifest_payload(
    split: ParticipantSplit,
    dataset_root: Path,
    all_file_paths: Sequence[Path],
    seed: int,
    ratios: Mapping[str, float],
    window_counts_by_split: Mapping[str, int] | None = None,
    class_counts_by_split: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Build a split manifest with optional window-level diagnostics."""

    _validate_split_integrity(split)
    all_rel_paths = [_to_relative_string(path, dataset_root) for path in sorted(all_file_paths)]
    train_rel_paths = [_to_relative_string(path, dataset_root) for path in split.train_files]
    val_rel_paths = [_to_relative_string(path, dataset_root) for path in split.val_files]
    test_rel_paths = [_to_relative_string(path, dataset_root) for path in split.test_files]

    split_sets = {
        "train": set(train_rel_paths),
        "validation": set(val_rel_paths),
        "test": set(test_rel_paths),
    }
    assigned_files = split_sets["train"] | split_sets["validation"] | split_sets["test"]

    manifest: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "seed": seed,
        "ratios": dict(ratios),
        "all_files": all_rel_paths,
        "train_files": train_rel_paths,
        "val_files": val_rel_paths,
        "test_files": test_rel_paths,
        "counts": {
            "train_files": len(train_rel_paths),
            "validation_files": len(val_rel_paths),
            "test_files": len(test_rel_paths),
        },
        "integrity": {
            "no_file_overlap": True,
            "all_discovered_files_assigned_once": assigned_files == set(all_rel_paths),
        },
    }

    if window_counts_by_split is not None:
        manifest["window_counts"] = {
            split_name: int(window_counts_by_split.get(split_name, 0))
            for split_name in ("train", "validation", "test")
        }
    if class_counts_by_split is not None:
        manifest["window_class_counts"] = {
            split_name: {
                label_name: int(count)
                for label_name, count in class_counts_by_split.get(split_name, {}).items()
            }
            for split_name in ("train", "validation", "test")
        }

    return manifest


def _validate_split_integrity(split: ParticipantSplit) -> None:
    """Fail fast when any file is assigned to multiple splits."""

    train_set = set(split.train_files)
    val_set = set(split.val_files)
    test_set = set(split.test_files)

    if train_set & val_set:
        raise ValueError("Train and validation splits share participant files.")
    if train_set & test_set:
        raise ValueError("Train and test splits share participant files.")
    if val_set & test_set:
        raise ValueError("Validation and test splits share participant files.")


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


def _participant_csv_columns(data_config: DataConfig) -> list[str]:
    """Return the minimal set of CSV columns required by the pipeline."""

    return [
        data_config.timestamp_column,
        *data_config.feature_columns,
        data_config.target_column,
    ]
