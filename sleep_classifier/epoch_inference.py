"""Run epoch-sequence inference for a single DREAMT participant CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sleep_classifier.data.epoch_preprocessing import (
    build_window_manifest_rows,
    extract_window_features,
    load_and_clean_participant_csv,
)
from sleep_classifier.data.scan_dataset import extract_participant_id
from sleep_classifier.experiment_config import ExperimentConfig, apply_common_overrides
from sleep_classifier.label_mapping import FINAL_ID_TO_NAME, get_class_names
from sleep_classifier.models import build_model
from sleep_classifier.sequence_evaluation import smooth_predictions
from sleep_classifier.utils import build_logger, configure_torch_runtime, get_device, load_torch_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to a single participant CSV.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path for prediction export.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional artifact directory override.")
    parser.add_argument("--smoothing-mode", choices=["none", "median", "viterbi"], default=None)
    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    mixed_precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.set_defaults(mixed_precision=None)
    return parser


def initialize_config(checkpoint_path: Path, args: argparse.Namespace) -> tuple[ExperimentConfig, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    config = apply_common_overrides(config, args)
    config.validate()
    config.ensure_output_dirs()
    return config, checkpoint


def normalize_features(
    features: np.ndarray,
    normalization_stats: dict[str, object],
    participant_features: np.ndarray,
    participant_id: str,
) -> np.ndarray:
    if normalization_stats.get("mode") == "participant":
        participant_stats = normalization_stats.get("participants", {}).get(participant_id)
        if participant_stats is None:
            channel_count = participant_features.shape[1]
            sums = np.zeros(channel_count, dtype=np.float64)
            squared_sums = np.zeros(channel_count, dtype=np.float64)
            count = int(participant_features.shape[0] * participant_features.shape[2])
            for channel_index in range(channel_count):
                channel_values = np.asarray(participant_features[:, channel_index, :], dtype=np.float64)
                sums[channel_index] += channel_values.sum(dtype=np.float64)
                squared_sums[channel_index] += np.square(channel_values).sum(dtype=np.float64)
            means = sums / count
            variances = np.maximum((squared_sums / count) - np.square(means), 1e-12)
            participant_stats = {
                "mean": means.tolist(),
                "std": np.sqrt(variances).tolist(),
            }
        mean = np.asarray(participant_stats["mean"], dtype=np.float32).reshape(1, -1, 1)
        std = np.asarray(participant_stats["std"], dtype=np.float32).reshape(1, -1, 1)
    else:
        mean = np.asarray(normalization_stats["mean"], dtype=np.float32).reshape(1, -1, 1)
        std = np.asarray(normalization_stats["std"], dtype=np.float32).reshape(1, -1, 1)
    return (features - mean) / std


def reshape_epoch_sequences(features: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    num_windows, num_channels, _ = features.shape
    reshaped = features.reshape(
        num_windows,
        num_channels,
        config.context_length_epochs,
        config.center_window_size,
    )
    return np.transpose(reshaped, (0, 2, 1, 3))


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

    rows, _, skipped_short = build_window_manifest_rows(
        cleaned=cleaned,
        config=config,
        split_name="inference",
        stride_epochs=config.test_stride_epochs,
    )
    if skipped_short or not rows:
        raise ValueError("No usable epoch sequences were produced for inference.")
    manifest = pd.DataFrame(rows)
    features = extract_window_features(cleaned, manifest)

    normalization_stats = checkpoint.get("normalization_stats")
    if normalization_stats is None:
        raise ValueError("Checkpoint is missing normalization statistics required for inference.")

    normalized_features = normalize_features(
        features,
        normalization_stats,
        cleaned.features,
        participant_id,
    )
    epoch_features = reshape_epoch_sequences(normalized_features.astype(np.float32, copy=False), config)
    dataset = TensorDataset(torch.from_numpy(epoch_features.astype(np.float32, copy=False)))
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    model = build_model(
        config=config,
        final_num_classes=len(get_class_names(level="final")),
        raw_num_classes=len(get_class_names(level="raw")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probabilities: list[np.ndarray] = []
    use_mixed_precision = device.type == "cuda" and config.mixed_precision
    with torch.no_grad():
        for (inputs,) in data_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                outputs = model(inputs)
                logits = outputs["final_logits"]
                if logits is None:
                    raise ValueError("Model outputs are missing final_logits.")
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    probability_array = np.concatenate(probabilities, axis=0)
    raw_predicted_ids = probability_array.argmax(axis=1)
    raw_confidences = probability_array.max(axis=1)
    smoothed_predicted_ids = smooth_predictions(
        metadata=manifest.to_dict(orient="records"),
        raw_predictions=raw_predicted_ids,
        raw_probabilities=probability_array,
        config=config,
        transition_matrix=checkpoint.get("transition_matrix"),
        class_priors=checkpoint.get("class_priors"),
    )

    prediction_frame = pd.DataFrame(
        {
            "participant_id": participant_id,
            "window_start_timestamp": manifest["center_start_timestamp"].to_numpy(dtype=np.float64),
            "window_end_timestamp": manifest["center_end_timestamp"].to_numpy(dtype=np.float64),
            "raw_predicted_class_id": raw_predicted_ids,
            "raw_predicted_class_name": [FINAL_ID_TO_NAME[int(label_id)] for label_id in raw_predicted_ids],
            "raw_confidence": raw_confidences,
        }
    )
    if smoothed_predicted_ids is not None:
        prediction_frame["smoothed_predicted_class_id"] = smoothed_predicted_ids
        prediction_frame["smoothed_predicted_class_name"] = [
            FINAL_ID_TO_NAME[int(label_id)] for label_id in smoothed_predicted_ids
        ]
    for class_index, class_name in enumerate(get_class_names(level="final")):
        column_name = class_name.lower().replace(" ", "_")
        prediction_frame[f"prob_{column_name}"] = probability_array[:, class_index]

    output_csv = args.output_csv or (config.reports_dir / f"{participant_id}_predictions.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_csv, index=False)

    logger.info("Generated %d epoch-sequence predictions for %s.", len(prediction_frame), participant_id)
    logger.info("Saved inference output to %s.", output_csv)
    logger.info("Raw predicted class counts:\n%s", prediction_frame["raw_predicted_class_name"].value_counts())
    if smoothed_predicted_ids is not None:
        logger.info(
            "Smoothed predicted class counts:\n%s",
            prediction_frame["smoothed_predicted_class_name"].value_counts(),
        )
