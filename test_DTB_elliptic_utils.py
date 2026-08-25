"""Small runtime checks for the shared Colab utility module."""

import math
import unittest

import torch

from DTB_elliptic_utils import (
    MLP,
    assemble_ritz_system,
    expand_direction,
    flatten_parameters,
    make_box_trial,
    manufactured_forcing,
    matrix_ritz_energy,
    predict_tangent,
    sample_box,
    sample_box_boundary,
    select_parameter_indices,
    selected_tangent_basis,
    solve_ritz_system,
    direct_ritz_energy,
)


class DTBEllipticUtilitiesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1)
        cls.model = MLP(2, width=5, depth=1)
        cls.theta, cls.spec = flatten_parameters(cls.model)
        # Keep the closure from becoming a bound method on instance access.
        cls.trial = staticmethod(make_box_trial(cls.model, cls.spec))
        cls.indices = select_parameter_indices(cls.theta.numel(), 6, seed=2)

    def test_selected_basis_and_boundary(self) -> None:
        points = sample_box(16, 2, seed=3)
        features, gradients = selected_tangent_basis(
            self.trial, self.theta, points, self.indices, chunk_size=8
        )
        self.assertEqual(features.shape, (16, 6))
        self.assertEqual(gradients.shape, (16, 2, 6))
        alpha = torch.arange(1, 7, dtype=torch.float64)
        direction = expand_direction(alpha, self.indices, self.theta.numel())
        boundary = sample_box_boundary(5, 2, seed=4)
        values = predict_tangent(self.trial, self.theta, direction, boundary)
        self.assertLess(float(values.abs().max()), 1.0e-12)

    def test_matrix_and_direct_energy_agree(self) -> None:
        points = sample_box(20, 2, seed=5)
        forcing = torch.sin(points[:, 0]) + points[:, 1]
        diffusion = torch.tensor([[1.0, 0.2], [0.2, 0.5]])
        features, gradients = selected_tangent_basis(
            self.trial, self.theta, points, self.indices, chunk_size=10
        )
        stiffness, load = assemble_ritz_system(
            features, gradients, forcing, 4.0, diffusion=diffusion
        )
        solution = solve_ritz_system(stiffness, load, ridge_relative=1.0e-8)
        direction = expand_direction(solution.alpha, self.indices, self.theta.numel())
        matrix_energy = matrix_ritz_energy(stiffness, load, solution.alpha)
        direct_energy = direct_ritz_energy(
            self.trial,
            self.theta,
            direction,
            points,
            forcing,
            4.0,
            diffusion=diffusion,
        )
        self.assertLess(float(torch.abs(matrix_energy - direct_energy)), 1.0e-10)

    def test_manufactured_poisson_forcing(self) -> None:
        def exact(point: torch.Tensor) -> torch.Tensor:
            return torch.cos(0.5 * math.pi * point[0]) * torch.cos(
                0.5 * math.pi * point[1]
            )

        points = sample_box(12, 2, seed=6)
        expected = 0.5 * math.pi**2 * torch.vmap(exact)(points)
        actual = manufactured_forcing(exact, points)
        torch.testing.assert_close(actual, expected, rtol=1.0e-11, atol=1.0e-11)


if __name__ == "__main__":
    unittest.main()
