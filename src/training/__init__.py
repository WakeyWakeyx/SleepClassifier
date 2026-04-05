"""Training and evaluation entry points."""

from .evaluate import evaluate_model
from .train import run_training_pipeline

__all__ = ["evaluate_model", "run_training_pipeline"]
