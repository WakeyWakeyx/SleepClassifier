"""Model package."""

from .epoch_sequence import ModalityAwareEpochSequenceModel, SleepStageCNN1D, SleepStageContextModel, build_model

MultiScaleResidualBiLSTMClassifier = ModalityAwareEpochSequenceModel

__all__ = [
    "ModalityAwareEpochSequenceModel",
    "MultiScaleResidualBiLSTMClassifier",
    "SleepStageContextModel",
    "SleepStageCNN1D",
    "build_model",
]
