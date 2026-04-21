"""Dataset discovery helpers for DREAMT 64 Hz CSV files."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ParticipantFile:
    """Reference to a participant CSV file."""

    participant_id: str
    file_path: Path


def extract_participant_id(file_path: Path) -> str:
    """Extract a stable participant id from a DREAMT CSV filename."""

    stem = file_path.stem
    match = re.match(r"(.+?)_whole_df$", stem, flags=re.IGNORECASE)
    participant_id = match.group(1) if match else stem
    participant_id = re.sub(r"[^A-Za-z0-9_-]+", "_", participant_id).strip("_")
    return participant_id or stem


def scan_dataset_files(
    dataset_root: Path,
    logger: logging.Logger,
    limit_files: int | None = None,
    recursive: bool = True,
) -> list[ParticipantFile]:
    """Scan the dataset directory for CSV files with deterministic ordering."""

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. Expected the DREAMT 64 Hz CSV folder."
        )
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    csv_iterable = dataset_root.rglob("*.csv") if recursive else dataset_root.glob("*.csv")
    csv_paths = sorted(csv_iterable, key=lambda path: path.as_posix().lower())
    if limit_files is not None:
        csv_paths = csv_paths[:limit_files]

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {dataset_root}")

    participant_files: list[ParticipantFile] = []
    seen_ids: dict[str, int] = {}
    for csv_path in csv_paths:
        base_id = extract_participant_id(csv_path)
        count = seen_ids.get(base_id, 0)
        seen_ids[base_id] = count + 1
        participant_id = base_id if count == 0 else f"{base_id}__{count + 1}"
        if participant_id != base_id:
            logger.warning(
                "Duplicate participant id '%s' detected. Using '%s' for %s.",
                base_id,
                participant_id,
                csv_path.name,
            )
        participant_files.append(ParticipantFile(participant_id=participant_id, file_path=csv_path))

    logger.info(
        "Discovered %d CSV file(s) under %s%s.",
        len(participant_files),
        dataset_root,
        f" (limited to first {limit_files})" if limit_files is not None else "",
    )
    return participant_files
