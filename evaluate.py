"""Evaluate a trained DREAMT sleep-stage classifier on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from sleep_classifier.config import ExperimentConfig, apply_common_overrides
from sleep_classifier.data import prepare_datasets
from sleep_classifier.label_mapping import get_class_names
from sleep_classifier.models import SleepStageCNN1D
from sleep_classifier.utils import (
    build_logger,
    compute_classification_metrics,
    configure_torch_runtime,
    get_device,
    plot_confusion_matrix,
    save_json,
    set_global_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the best checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional file limit for debug evaluation.")
    mixed_precision_group = parser.add_mutually_exclusive_group()
    mixed_precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    mixed_precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.set_defaults(mixed_precision=None)
    return parser


def create_dataloader(
    dataset: torch.utils.data.Dataset[Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def evaluate_loader(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_mixed_precision: bool,
    class_names: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    total_examples = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                logits = model(inputs)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            total_examples += int(batch_size)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())

    if total_examples == 0:
        raise ValueError("Test loader produced zero examples.")

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)
    metrics["loss"] = running_loss / total_examples
    return metrics, y_true, y_pred


def initialize_config(checkpoint_path: Path, args: argparse.Namespace, logger: Any) -> tuple[ExperimentConfig, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    config = apply_common_overrides(config, args)
    config.ensure_output_dirs()
    logger.info("Loaded evaluation config from %s.", checkpoint_path)
    return config, checkpoint


def main() -> None:
    args = build_parser().parse_args()
    logger = build_logger()
    config, checkpoint = initialize_config(args.checkpoint, args, logger)

    set_global_seed(config.random_seed)
    device = get_device(logger)
    configure_torch_runtime(device, logger)

    bundle = prepare_datasets(config=config, logger=logger, requested_splits=("test",))
    test_loader = create_dataloader(
        dataset=bundle.prepared_splits["test"].dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )

    model = SleepStageCNN1D(
        input_channels=len(config.feature_columns),
        num_classes=len(get_class_names()),
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    class_weights = torch.tensor(bundle.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_mixed_precision = device.type == "cuda" and config.mixed_precision
    class_names = get_class_names()
    metrics, y_true, y_pred = evaluate_loader(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        use_mixed_precision=use_mixed_precision,
        class_names=class_names,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    report_json_path = config.reports_dir / "test_classification_report.json"
    report_text_path = config.reports_dir / "test_classification_report.txt"
    confusion_path = config.plots_dir / "test_confusion_matrix.png"
    metrics_path = config.reports_dir / "test_metrics.json"

    save_json(report_dict, report_json_path)
    save_json(metrics, metrics_path)
    report_text_path.write_text(report_text, encoding="utf-8")
    plot_confusion_matrix(confusion_matrix=confusion, class_names=class_names, output_path=confusion_path, title="Test Confusion Matrix")

    logger.info("Test loss: %.4f", metrics["loss"])
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Balanced Accuracy: %.4f", metrics["balanced_accuracy"])
    logger.info("Macro Precision: %.4f", metrics["macro_precision"])
    logger.info("Macro Recall: %.4f", metrics["macro_recall"])
    logger.info("Macro F1: %.4f", metrics["macro_f1"])
    for class_name, class_metrics in metrics["per_class"].items():
        logger.info(
            "%s | precision=%.4f recall=%.4f f1=%.4f support=%d",
            class_name,
            class_metrics["precision"],
            class_metrics["recall"],
            class_metrics["f1"],
            class_metrics["support"],
        )
    logger.info("Confusion matrix:\n%s", confusion)
    logger.info("Saved evaluation report to %s and %s.", report_text_path, report_json_path)
    logger.info("Saved confusion matrix plot to %s.", confusion_path)


from sleep_classifier.standalone_evaluation import main as evaluation_main

main = evaluation_main


if __name__ == "__main__":
    evaluation_main()
