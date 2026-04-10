"""CLI entry point for end-to-end sleep stage model training."""

from __future__ import annotations

from src.config import build_configs
from src.training import run_training_pipeline
from src.utils import save_json


def main() -> None:
    """Run the full training pipeline with the configured settings."""

    configs = build_configs()
    summaries = [run_training_pipeline(config) for config in configs]

    if len(configs) > 1:
        comparison_payload = {
            "group_name": configs[0].experiment.group_name,
            "experiments": summaries,
        }
        comparison_path = configs[0].paths.group_dir / "experiment_comparison.json"
        save_json(comparison_payload, comparison_path)


if __name__ == "__main__":
    main()
