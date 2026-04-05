"""CLI entry point for end-to-end sleep stage model training."""

from __future__ import annotations

from src.config import build_config
from src.training import run_training_pipeline


def main() -> None:
    """Run the full training pipeline with the configured settings."""

    config = build_config()
    run_training_pipeline(config)


if __name__ == "__main__":
    main()
