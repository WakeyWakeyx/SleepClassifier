"""Backward-compatible import for the main context-aware sleep model."""

from .hybrid import MultiScaleResidualBiLSTMClassifier as SleepStageCNN1D

__all__ = ["SleepStageCNN1D"]
