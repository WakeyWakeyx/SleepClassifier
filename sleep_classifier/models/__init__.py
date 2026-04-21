"""Model package."""

from .hybrid import MultiScaleResidualBiLSTMClassifier, SleepStageContextModel, build_model

SleepStageCNN1D = SleepStageContextModel

__all__ = [
    "MultiScaleResidualBiLSTMClassifier",
    "SleepStageContextModel",
    "SleepStageCNN1D",
    "build_model",
]
