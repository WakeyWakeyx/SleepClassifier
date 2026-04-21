"""Run window-level inference for a single DREAMT participant CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sleep_classifier.config import ExperimentConfig, apply_common_overrides
from sleep_classifier.data.preprocessing import build_windows_from_clean_dataframe, load_and_clean_participant_csv
from sleep_classifier.data.scan_dataset import extract_participant_id
from sleep_classifier.label_mapping import ID_TO_NAME, get_class_names
from sleep_classifier.models import SleepStageCNN1D
from sleep_classifier.utils import build_logger, configure_torch_runtime, get_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to a single participant CSV.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path for prediction export.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional artifact directory override.")
    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    mixed_precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.set_defaults(mixed_precision=None)
    return parser


def initialize_config(checkpoint_path: Path, args: argparse.Namespace) -> tuple[ExperimentConfig, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    config = apply_common_overrides(config, args)
    config.ensure_output_dirs()
    return config, checkpoint


def normalize_features(features: np.ndarray, normalization_stats: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(normalization_stats["mean"], dtype=np.float32).reshape(1, -1, 1)
    std = np.asarray(normalization_stats["std"], dtype=np.float32).reshape(1, -1, 1)
    return (features - mean) / std


def main() -> None:
    args = build_parser().parse_args()
    logger = build_logger()
    config, checkpoint = initialize_config(args.checkpoint, args)
    device = get_device(logger)
    configure_torch_runtime(device, logger)

    participant_id = extract_participant_id(args.input_csv)
    cleaned = load_and_clean_participant_csv(
        file_path=args.input_csv,
        participant_id=participant_id,
        config=config,
        logger=logger,
    )
    if cleaned is None:
        raise ValueError(f"Input CSV could not be processed: {args.input_csv}")

    windowed = build_windows_from_clean_dataframe(cleaned, config, include_features=True)
    if windowed.features is None or len(windowed.features) == 0:
        raise ValueError("No usable windows were produced for inference.")

    normalization_stats = checkpoint.get("normalization_stats")
    if normalization_stats is None:
        raise ValueError("Checkpoint is missing normalization statistics required for inference.")

    normalized_features = normalize_features(windowed.features, normalization_stats)
    dataset = TensorDataset(torch.from_numpy(normalized_features.astype(np.float32, copy=False)))
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    model = SleepStageCNN1D(
        input_channels=len(config.feature_columns),
        num_classes=len(get_class_names()),
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probabilities: list[np.ndarray] = []
    use_mixed_precision = device.type == "cuda" and config.mixed_precision
    with torch.no_grad():
        for (inputs,) in data_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                logits = model(inputs)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    probability_array = np.concatenate(probabilities, axis=0)
    predicted_ids = probability_array.argmax(axis=1)
    confidences = probability_array.max(axis=1)

    prediction_frame = pd.DataFrame(
        {
            "participant_id": participant_id,
            "window_start_timestamp": windowed.start_timestamps,
            "window_end_timestamp": windowed.end_timestamps,
            "predicted_class_id": predicted_ids,
            "predicted_class_name": [ID_TO_NAME[int(label_id)] for label_id in predicted_ids],
            "confidence": confidences,
        }
    )

    output_csv = args.output_csv or (config.reports_dir / f"{participant_id}_predictions.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_csv, index=False)

    logger.info("Generated %d window-level predictions for %s.", len(prediction_frame), participant_id)
    logger.info("Saved inference output to %s.", output_csv)
    logger.info("Predicted class counts:\n%s", prediction_frame["predicted_class_name"].value_counts())


if __name__ == "__main__":
    main()
