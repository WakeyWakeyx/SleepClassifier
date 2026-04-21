"""Standalone checkpoint evaluation entry point for the epoch-sequence model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from sleep_classifier.data import prepare_datasets
from sleep_classifier.experiment_config import ExperimentConfig, apply_common_overrides
from sleep_classifier.label_mapping import get_class_names
from sleep_classifier.models import build_model
from sleep_classifier.multitask_losses import build_loss
from sleep_classifier.sequence_evaluation import evaluate_model
from sleep_classifier.utils import build_logger, configure_torch_runtime, get_device, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone evaluation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--context-window-seconds", type=int, default=None, help="Optional context window override.")
    parser.add_argument("--context-length-epochs", type=int, default=None, help="Optional context length override.")
    parser.add_argument("--center-epoch-seconds", type=int, default=None, help="Optional epoch duration override.")
    parser.add_argument("--test-stride-seconds", type=int, default=None, help="Optional test stride override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional file limit for debug evaluation.")
    parser.add_argument("--smoothing-mode", choices=["none", "median", "viterbi"], default=None)

    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    mixed_precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use-cache", dest="use_cache", action="store_true")
    cache_group.add_argument("--no-use-cache", dest="use_cache", action="store_false")

    rebuild_group = parser.add_mutually_exclusive_group()
    rebuild_group.add_argument("--rebuild-cache", dest="rebuild_cache", action="store_true")
    rebuild_group.add_argument("--no-rebuild-cache", dest="rebuild_cache", action="store_false")

    parser.set_defaults(mixed_precision=None, use_cache=None, rebuild_cache=None)
    return parser


def initialize_config(checkpoint_path: Path, args: argparse.Namespace, logger: Any) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Load config from checkpoint and apply any CLI overrides."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    config = apply_common_overrides(config, args)
    config.validate()
    config.ensure_output_dirs()
    logger.info("Loaded evaluation config from %s.", checkpoint_path)
    return config, checkpoint


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    logger = build_logger()
    config, checkpoint = initialize_config(args.checkpoint, args, logger)

    set_global_seed(config.random_seed)
    device = get_device(logger)
    configure_torch_runtime(device, logger)

    bundle = prepare_datasets(config=config, logger=logger, requested_splits=("test",))
    class_names = get_class_names(level="final")
    model = build_model(
        config=config,
        final_num_classes=len(class_names),
        raw_num_classes=len(get_class_names(level="raw")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = build_loss(
        config=config,
        final_class_counts=bundle.final_class_counts,
        raw_class_counts=bundle.raw_class_counts,
        device=device,
    )
    use_mixed_precision = device.type == "cuda" and config.mixed_precision
    result = evaluate_model(
        model=model,
        prepared_split=bundle.prepared_splits["test"],
        criterion=criterion,
        config=config,
        device=device,
        use_mixed_precision=use_mixed_precision,
        class_names=class_names,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        transition_matrix=checkpoint.get("transition_matrix", bundle.transition_matrix),
        class_priors=checkpoint.get("class_priors", bundle.class_priors),
        split_name="test",
        metrics_output_path=config.reports_dir / "test_metrics.json",
        report_output_path=config.reports_dir / "test_classification_report.json",
        report_text_output_path=config.reports_dir / "test_classification_report.txt",
        confusion_output_path=config.plots_dir / "test_confusion_matrix.png",
    )

    logger.info("Raw test accuracy: %.4f", result.metrics["accuracy"])
    logger.info("Raw test macro F1: %.4f", result.metrics["macro_f1"])
    if result.smoothed_metrics is not None:
        logger.info("Smoothed test accuracy: %.4f", result.smoothed_metrics["accuracy"])
        logger.info("Smoothed test macro F1: %.4f", result.smoothed_metrics["macro_f1"])
    logger.info("Saved evaluation artifacts to %s.", config.output_dir)
