"""Label mapping helpers for multitask DREAMT sleep-stage classification."""

from __future__ import annotations

from typing import Any, Literal


RAW_CLASS_NAMES: tuple[str, ...] = ("W", "N1", "N2", "N3", "R")
RAW_ID_TO_NAME: dict[int, str] = {idx: name for idx, name in enumerate(RAW_CLASS_NAMES)}
RAW_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in RAW_ID_TO_NAME.items()}

FINAL_CLASS_NAMES: tuple[str, ...] = ("Awake", "Light Sleep", "Deep Sleep")
FINAL_ID_TO_NAME: dict[int, str] = {idx: name for idx, name in enumerate(FINAL_CLASS_NAMES)}
FINAL_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in FINAL_ID_TO_NAME.items()}

RAW_TO_FINAL_NAME: dict[str, str] = {
    "W": "Awake",
    "N1": "Light Sleep",
    "N2": "Light Sleep",
    "N3": "Deep Sleep",
    "R": "Deep Sleep",
}
RAW_TO_FINAL_ID: dict[int, int] = {
    RAW_NAME_TO_ID[raw_name]: FINAL_NAME_TO_ID[final_name]
    for raw_name, final_name in RAW_TO_FINAL_NAME.items()
}

ID_TO_NAME = FINAL_ID_TO_NAME
NAME_TO_ID = FINAL_NAME_TO_ID
EXCLUDED_LABELS: set[str] = {"P", "Missing"}


def map_raw_sleep_stage(label: Any) -> int | None:
    """Map a raw DREAMT label to its original 5-class id."""

    if label is None:
        return None
    text = str(label).strip()
    if not text or text in EXCLUDED_LABELS:
        return None
    return RAW_NAME_TO_ID.get(text)


def collapse_raw_to_final(raw_label_id: int) -> int:
    """Collapse a 5-class sleep stage into the deployment 3-class target."""

    try:
        return RAW_TO_FINAL_ID[int(raw_label_id)]
    except KeyError as exc:
        raise KeyError(f"Unsupported raw label id: {raw_label_id}") from exc


def map_sleep_stage(label: Any) -> int | None:
    """Backward-compatible helper that maps directly to the final 3-class target."""

    raw_label_id = map_raw_sleep_stage(label)
    if raw_label_id is None:
        return None
    return collapse_raw_to_final(raw_label_id)


def get_class_names(level: Literal["final", "raw"] = "final") -> list[str]:
    """Return class names for either the final 3-class or raw 5-class label space."""

    if level == "raw":
        return [RAW_ID_TO_NAME[idx] for idx in range(len(RAW_ID_TO_NAME))]
    return [FINAL_ID_TO_NAME[idx] for idx in range(len(FINAL_ID_TO_NAME))]
