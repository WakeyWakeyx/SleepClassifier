"""Label mapping helpers for DREAMT sleep-stage classification."""

from __future__ import annotations

from typing import Any


CLASS_NAMES: tuple[str, ...] = ("Awake", "Light Sleep", "Deep Sleep")
ID_TO_NAME: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NAME_TO_ID: dict[str, int] = {name: idx for idx, name in ID_TO_NAME.items()}

RAW_TO_NAME: dict[str, str] = {
    "W": "Awake",
    "N1": "Light Sleep",
    "N2": "Light Sleep",
    "N3": "Deep Sleep",
    "R": "Deep Sleep",
}
RAW_TO_ID: dict[str, int] = {raw: NAME_TO_ID[name] for raw, name in RAW_TO_NAME.items()}

EXCLUDED_LABELS: set[str] = {"P", "Missing"}


def map_sleep_stage(label: Any) -> int | None:
    """Map a raw DREAMT sleep stage label to a ternary class id."""

    if label is None:
        return None
    text = str(label).strip()
    if not text or text in EXCLUDED_LABELS:
        return None
    return RAW_TO_ID.get(text)


def get_class_names() -> list[str]:
    """Return human-readable class names in id order."""

    return [ID_TO_NAME[idx] for idx in range(len(ID_TO_NAME))]
