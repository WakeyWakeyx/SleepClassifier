"""Run a short end-to-end debug training job on only the first 10 CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from sleep_classifier.config import ExperimentConfig
from train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data_64Hz"), help="Path to the DREAMT data_64Hz folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "debug_small_run", help="Directory for debug artifacts.")
    parser.add_argument("--epochs", type=int, default=2, help="Short debug epoch count.")
    parser.add_argument("--batch-size", type=int, default=16, help="Debug batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Defaults to 0 for Windows.")
    parser.add_argument("--stride-seconds", type=int, default=30, help="Window stride in seconds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride_seconds=args.stride_seconds,
        limit_files=10,
        early_stopping_patience=2,
    )
    run_training(config=config, resume_from=None)


if __name__ == "__main__":
    main()
