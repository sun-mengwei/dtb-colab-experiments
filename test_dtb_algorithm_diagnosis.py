"""Runtime checks for the single-mode DTB diagnosis helpers."""

import math
import unittest

import torch

from DTB_elliptic_utils import MLP, flatten_parameters
from dtb_algorithm_diagnosis import (
    SingleModeConfig,
    layer_balanced_parameter_indices,
    run_single_mode_configuration,
    selection_counts,
)


class DTBAlgorithmDiagnosisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_dtype(torch.float64)

    def test_layer_balanced_selection_reaches_every_layer(self) -> None:
        model = MLP(10, width=16, depth=2)
        theta, spec = flatten_parameters(model)
        indices = layer_balanced_parameter_indices(spec, 32, seed=9)
        counts = selection_counts(indices, spec)
        self.assertEqual(indices.unique().numel(), 32)
        self.assertTrue(all(count > 0 for count in counts.values()))
        self.assertIn(theta.numel() - 1, indices.tolist())

    def test_small_outer_run_reports_basis_drift(self) -> None:
        evaluation_points = {
            "axis": torch.zeros(5, 2, dtype=torch.get_default_dtype())
        }
        result = run_single_mode_configuration(
            SingleModeConfig(
                dimension=2,
                train_count=32,
                validation_count=64,
                tangent_dimension=6,
                selection="layer_balanced",
                outer_steps=1,
                width=4,
                depth=1,
                model_seed=4,
                quadrature_seed=5,
                validation_seed=6,
                selection_seed=7,
                probe_count=16,
                probe_seed=8,
            ),
            evaluation_points=evaluation_points,
        )
        self.assertEqual(len(result["history"]), 2)
        for key in (
            "relative_l2",
            "relative_feature_change",
            "maximum_subspace_sine",
            "relative_prediction_change",
        ):
            self.assertTrue(math.isfinite(result["summary"][key]))
        self.assertEqual(len(result["evaluations"]["axis"]["predictions"]), 2)
        self.assertEqual(tuple(result["evaluations"]["axis"]["reference"].shape), (5,))


if __name__ == "__main__":
    unittest.main()
