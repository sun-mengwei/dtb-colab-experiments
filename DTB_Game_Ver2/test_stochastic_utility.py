import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")

from dtb import flat_params
from utility import (
    create_run_directory,
    diagnostics_markdown,
    euler_score_update,
    plot_dtb_em_snapshots,
    plot_tangent_diagnostics,
    sample_initial_with_score,
    tangent_velocity_spatial_terms,
)


class StochasticUtilityTests(unittest.TestCase):
    def test_gaussian_initial_score_is_exact(self):
        generator = torch.Generator().manual_seed(7)
        x, score, log_density = sample_initial_with_score(
            12,
            3,
            law="gaussian",
            device=torch.device("cpu"),
            dtype=torch.float64,
            generator=generator,
            gaussian_mean=0.4,
            gaussian_std=0.2,
        )
        expected = -(x - 0.4) / 0.2**2
        torch.testing.assert_close(score, expected)
        self.assertEqual(log_density.shape, (12,))

    def test_affine_tangent_derivatives_and_score_step(self):
        model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
        theta, structure, _ = flat_params(model)
        selected = torch.arange(theta.numel())
        alpha = torch.tensor([0.3, -0.2, 0.4, 0.1], dtype=torch.float64)
        x = torch.tensor([[0.2, -0.5], [0.7, 0.1]], dtype=torch.float64)

        velocity, jacobian, divergence, grad_divergence = tangent_velocity_spatial_terms(
            theta, selected, alpha, x, model, structure, chunk_size=1
        )
        direction_matrix = alpha.reshape(2, 2)
        torch.testing.assert_close(velocity, x @ direction_matrix.T)
        torch.testing.assert_close(
            jacobian, direction_matrix.expand(x.shape[0], -1, -1)
        )
        torch.testing.assert_close(
            divergence,
            torch.full((x.shape[0],), torch.trace(direction_matrix), dtype=x.dtype),
        )
        torch.testing.assert_close(grad_divergence, torch.zeros_like(x))

        score = torch.tensor([[1.0, 2.0], [-0.5, 0.25]], dtype=torch.float64)
        next_score, transported, source = euler_score_update(
            score, jacobian, grad_divergence, 0.05
        )
        expected_transport = score @ direction_matrix
        torch.testing.assert_close(transported, expected_transport)
        torch.testing.assert_close(source, torch.zeros_like(source))
        torch.testing.assert_close(next_score, score - 0.05 * expected_transport)

    def test_visualizations_and_math_table(self):
        rng = np.random.default_rng(2)
        dtb = rng.normal(size=(3, 10, 3))
        em = rng.normal(size=(3, 10, 3))
        records = [
            {
                "relative_projection_error": 0.1,
                "alpha_norm": 2.0,
                "score_rms": 1.0,
                "diffusion_rms": 0.01,
                "target_rms": 0.4,
                "tangent_rms": 0.39,
            },
            {
                "relative_projection_error": 0.08,
                "alpha_norm": 1.8,
                "score_rms": 1.1,
                "diffusion_rms": 0.011,
                "target_rms": 0.35,
                "tangent_rms": 0.34,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            figures = plot_dtb_em_snapshots(
                dtb,
                em,
                np.array([0.0, 0.5, 1.0]),
                [0, 2],
                coordinate_pairs=[(1, 2)],
                equilibria=np.zeros((1, 3)),
                stable_mask=[False],
                output_dir=folder,
            )
            plot_tangent_diagnostics(
                [0.0, 0.5],
                [0.1, 0.08],
                [2.0, 1.8],
                output_path=folder / "diagnostics.png",
            )
            self.assertIn((1, 2), figures)
            self.assertTrue((folder / "dtb_vs_em_x1_x2.png").exists())
            self.assertTrue((folder / "diagnostics.png").exists())
        table = diagnostics_markdown(records)
        self.assertIn("Relative projection error", table)
        self.assertIn(r"\|\alpha_k\|_2", table)

    def test_saved_run_folder_records_dimension_and_network_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = create_run_directory(
                temporary,
                dim=5,
                network="mlp",
                activation="tanh",
                width=32,
                depth=4,
                parameter_count=3525,
                basis_size=128,
                seed=2026,
            )
            self.assertTrue(folder.is_dir())
            self.assertIn("cournot_5d_stochastic_mlp-tanh_w32_d4", folder.name)
            self.assertIn("p3525_basis128_seed2026", folder.name)


if __name__ == "__main__":
    unittest.main()
