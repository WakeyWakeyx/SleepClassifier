import unittest
from unittest.mock import patch

import torch

from sleep_classifier.experiment_config import ExperimentConfig
from sleep_classifier.models.epoch_sequence import ModalityAwareEpochSequenceModel


class ModalityAwareEpochSequenceModelTests(unittest.TestCase):
    def _make_config(self, feature_columns: tuple[str, ...]) -> ExperimentConfig:
        config = ExperimentConfig(
            feature_columns=feature_columns,
            sample_rate_hz=32,
            center_epoch_seconds=2,
            context_length_epochs=5,
            epoch_embedding_dim=24,
            branch_embedding_dim=8,
            model_base_channels=8,
            sequence_hidden_size=12,
            dropout=0.0,
            mixed_precision=False,
        )
        config.validate()
        return config

    def _make_inputs(self, config: ExperimentConfig) -> torch.Tensor:
        return torch.randn(
            2,
            config.context_length_epochs,
            len(config.input_feature_columns),
            config.center_window_size,
        )

    def _assert_output_shapes(
        self,
        outputs: dict[str, torch.Tensor | None],
        config: ExperimentConfig,
        final_num_classes: int,
        raw_num_classes: int,
    ) -> None:
        self.assertIsNotNone(outputs["final_logits"])
        self.assertIsNotNone(outputs["raw_logits"])
        self.assertEqual(outputs["final_logits"].shape, (2, final_num_classes))
        self.assertEqual(outputs["raw_logits"].shape, (2, raw_num_classes))
        self.assertEqual(
            outputs["epoch_embeddings"].shape,
            (2, config.context_length_epochs, config.epoch_embedding_dim),
        )
        self.assertEqual(
            outputs["sequence_outputs"].shape,
            (2, config.context_length_epochs, config.epoch_embedding_dim),
        )

    def test_multi_branch_forward_with_full_feature_set(self) -> None:
        config = self._make_config(("BVP", "IBI", "EDA", "TEMP", "ACC_X", "ACC_Y", "ACC_Z", "HR"))
        model = ModalityAwareEpochSequenceModel(config, final_num_classes=3, raw_num_classes=5)

        self.assertEqual(model.active_branch_names, ("bvp", "acc", "autonomic", "cardio"))
        self.assertEqual(model.epoch_fusion[1].in_features, config.branch_embedding_dim * 4)

        outputs = model(self._make_inputs(config))
        self._assert_output_shapes(outputs, config, final_num_classes=3, raw_num_classes=5)

    def test_filtered_feature_set_skips_empty_bvp_branch(self) -> None:
        config = self._make_config(("TEMP", "ACC_X", "ACC_Y", "ACC_Z", "HR"))
        model = ModalityAwareEpochSequenceModel(config, final_num_classes=3, raw_num_classes=5)

        self.assertEqual(model.branch_feature_names["bvp"], [])
        self.assertEqual(model.active_branch_names, ("acc", "autonomic", "cardio"))
        self.assertNotIn("bvp", model.branch_encoders)
        self.assertEqual(model.epoch_fusion[1].in_features, config.branch_embedding_dim * 3)

        with (
            patch.object(model.branch_encoders["acc"], "forward", wraps=model.branch_encoders["acc"].forward) as acc_forward,
            patch.object(
                model.branch_encoders["autonomic"],
                "forward",
                wraps=model.branch_encoders["autonomic"].forward,
            ) as autonomic_forward,
            patch.object(
                model.branch_encoders["cardio"],
                "forward",
                wraps=model.branch_encoders["cardio"].forward,
            ) as cardio_forward,
        ):
            outputs = model(self._make_inputs(config))

        self.assertEqual(acc_forward.call_count, 1)
        self.assertEqual(autonomic_forward.call_count, 1)
        self.assertEqual(cardio_forward.call_count, 1)
        self._assert_output_shapes(outputs, config, final_num_classes=3, raw_num_classes=5)

    def test_acc_only_feature_set_uses_single_active_branch(self) -> None:
        config = self._make_config(("ACC_X", "ACC_Y", "ACC_Z"))
        model = ModalityAwareEpochSequenceModel(config, final_num_classes=3, raw_num_classes=5)

        self.assertEqual(model.active_branch_names, ("acc",))
        self.assertEqual(model.branch_feature_names["autonomic"], [])
        self.assertEqual(model.branch_feature_names["cardio"], [])
        self.assertEqual(model.epoch_fusion[1].in_features, config.branch_embedding_dim)

        outputs = model(self._make_inputs(config))
        self._assert_output_shapes(outputs, config, final_num_classes=3, raw_num_classes=5)

    def test_config_rejects_invalid_feature_subsets(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one input feature column"):
            ExperimentConfig(feature_columns=()).validate()

        with self.assertRaisesRegex(ValueError, "ordered subset"):
            ExperimentConfig(feature_columns=("HR", "TEMP")).validate()


if __name__ == "__main__":
    unittest.main()
