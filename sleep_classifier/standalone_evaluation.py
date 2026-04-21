"""Standalone checkpoint evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from sleep_classifier.config import ExperimentConfig, apply_common_overrides
from sleep_classifier.data import prepare_datasets
from sleep_classifier.evaluation import evaluate_model
from sleep_classifier.label_mapping import get_class_names
from sleep_classifier.losses import build_loss
from sleep_classifier.models import build_model
from sleep_classifier.utils import build_logger, configure_torch_runtime, get_device, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone evaluation CLI."""

    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--context-window-seconds", type=int, default=None, help="Optional context window override.")
    parser.add_argument("--center-epoch-seconds", type=int, default=None, help="Optional center epoch override.")
    parser.add_argument("--test-stride-seconds", type=int, default=None, help="Optional test stride override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional file limit for debug evaluation.")

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
    class_names = get_class_names()
    model = build_model(config=config, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    class_weights = torch.tensor(bundle.class_weights, dtype=torch.float32, device=device)
    criterion = build_loss(config=config, class_weights=class_weights)
    use_mixed_precision = device.type == "cuda" and config.mixed_precision
    result = evaluate_model(
        model=model,
        dataset=bundle.prepared_splits["test"].dataset,
        criterion=criterion,
        device=device,
        use_mixed_precision=use_mixed_precision,
        class_names=class_names,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split_name="test",
        metrics_output_path=config.reports_dir / "test_metrics.json",
        report_output_path=config.reports_dir / "test_classification_report.json",
        report_text_output_path=config.reports_dir / "test_classification_report.txt",
        confusion_output_path=config.plots_dir / "test_confusion_matrix.png",
    )

    metrics = result.metrics
    logger.info("Test loss: %.4f", metrics["loss"])
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Balanced Accuracy: %.4f", metrics["balanced_accuracy"])
    logger.info("Macro Precision: %.4f", metrics["macro_precision"])
    logger.info("Macro Recall: %.4f", metrics["macro_recall"])
    logger.info("Macro F1: %.4f", metrics["macro_f1"])
    logger.info("Saved evaluation artifacts to %s.", config.output_dir)
