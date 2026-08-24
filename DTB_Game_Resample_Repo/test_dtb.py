"""Small standard-library test suite for the reusable DTB machinery.

Run with:

    python -m unittest -v test_dtb.py
"""

from __future__ import annotations

import unittest

import torch

from dtb import (
    NeuralDTB,
    ParticleState,
    PeriodicReset,
    diffusion_score_term,
    dtb_basis_matrix,
    evaluate_map,
    fit_model_to_map,
    flat_params,
    form_velocity,
    projected_action_terms,
    select_parameter_subset,
    svd_projection,
    unflatten,
    write_flat_into_model,
)
from network import MLP, ResidualMLP, ResidualNetwork


class NetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_network_shapes_and_identity_residual(self) -> None:
        points = torch.randn(5, 2)
        self.assertEqual(MLP(2, 3, width=8, depth=2)(points).shape, (5, 3))
        self.assertEqual(ResidualNetwork(2, 3, width=8, blocks=2)(points).shape, (5, 3))
        identity = ResidualMLP(2, width=8, depth=2, last_layer_scale=0.0)
        torch.testing.assert_close(identity(points), points)


class DTBPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.generator = torch.Generator(device="cpu").manual_seed(11)
        self.model = ResidualMLP(
            2, width=8, depth=1, last_layer_scale=1e-3
        )
        self.theta, self.structure = flat_params(self.model)

    def test_flat_parameter_round_trip_and_functional_evaluation(self) -> None:
        rebuilt = unflatten(self.theta, self.structure)
        self.assertEqual(tuple(rebuilt), self.structure.names)
        points = torch.randn(6, 2)
        torch.testing.assert_close(
            evaluate_map(self.theta, points, self.model, self.structure),
            self.model(points),
        )

        shifted = self.theta + 0.01
        write_flat_into_model(self.model, shifted, self.structure)
        written, _ = flat_params(self.model)
        torch.testing.assert_close(written, shifted)

    def test_basis_matrix_and_score_velocity(self) -> None:
        points = torch.randn(7, 2)
        selected = select_parameter_subset(
            self.theta.numel(),
            12,
            device=self.theta.device,
            generator=self.generator,
        )
        basis = dtb_basis_matrix(
            self.theta,
            selected,
            points,
            self.model,
            self.structure,
            chunk_size=3,
        )
        self.assertEqual(basis.values.shape, (7, 2))
        self.assertEqual(basis.jacobian.shape, (7, 2, 12))
        self.assertEqual(basis.matrix.shape, (14, 12))

        # The projected field returned by the spatial-derivative routine must
        # be the same J alpha represented by the basis tensor.
        alpha = torch.randn(12)
        action = projected_action_terms(
            self.theta,
            selected,
            alpha,
            points,
            self.model,
            self.structure,
            chunk_size=3,
        )
        torch.testing.assert_close(
            action.velocity,
            torch.einsum("ndm,m->nd", basis.jacobian, alpha),
        )
        torch.testing.assert_close(
            action.divergence,
            torch.diagonal(action.jacobian, dim1=-2, dim2=-1).sum(dim=-1),
        )

        score = torch.randn_like(points)
        diffusion = torch.tensor([[0.04, 0.01], [0.01, 0.09]])
        expected_score_term = -0.5 * (score @ diffusion.T)
        torch.testing.assert_close(
            diffusion_score_term(score, diffusion), expected_score_term
        )
        torch.testing.assert_close(
            form_velocity(points, score, lambda x: -x, diffusion),
            -points + expected_score_term,
        )

    def test_selected_jacobian_matches_central_finite_difference(self) -> None:
        """Validate actual Jacobian values, not only the returned dimensions."""

        model = ResidualMLP(
            2,
            width=6,
            depth=1,
            last_layer_scale=0.2,
            dtype=torch.float64,
        )
        theta, structure = flat_params(model)
        point = torch.tensor([[0.15, -0.35]], dtype=torch.float64)
        selected = torch.tensor(
            [0, theta.numel() // 2, theta.numel() - 1], dtype=torch.long
        )
        basis = dtb_basis_matrix(
            theta, selected, point, model, structure, chunk_size=1
        )

        epsilon = 1e-6
        for local_column, parameter_index in enumerate(selected.tolist()):
            direction = torch.zeros_like(theta)
            direction[parameter_index] = epsilon
            plus = evaluate_map(theta + direction, point, model, structure)
            minus = evaluate_map(theta - direction, point, model, structure)
            finite_difference = ((plus - minus) / (2.0 * epsilon)).squeeze(0)
            torch.testing.assert_close(
                basis.jacobian[0, :, local_column],
                finite_difference,
                rtol=2e-6,
                atol=2e-7,
            )

    def test_svd_projection_recovers_a_known_solution(self) -> None:
        matrix = torch.randn(30, 6, dtype=torch.float64)
        alpha_true = torch.randn(6, dtype=torch.float64)
        result = svd_projection(matrix, matrix @ alpha_true, rtol=1e-12)
        torch.testing.assert_close(result.alpha, alpha_true, rtol=1e-10, atol=1e-10)
        self.assertEqual(result.rank, 6)
        self.assertLess(result.relative_residual, 1e-10)

    def test_map_refit_reduces_error(self) -> None:
        model = ResidualMLP(2, width=8, depth=1, last_layer_scale=0.0)
        points = torch.rand(64, 2, generator=self.generator) * 2.0 - 1.0
        target = points + torch.stack(
            (0.15 * points[:, 1], -0.10 * points[:, 0]), dim=1
        )
        before, after = fit_model_to_map(
            model,
            points,
            target,
            optimizer_steps=100,
            learning_rate=1e-2,
            batch_size=32,
            generator=self.generator,
        )
        self.assertLess(after, 0.1 * before)

    def test_stateful_driver_steps_and_resets(self) -> None:
        labels = 0.1 * torch.randn(12, 2, generator=self.generator)
        state = ParticleState(
            labels=labels,
            particles=labels.clone(),
            log_density=torch.zeros(12),
            score=-labels,
        )
        solver = NeuralDTB(
            self.model,
            lambda x: -x,
            0.01 * torch.eye(2),
            step_size=0.01,
            basis_size=16,
            jacobian_chunk_size=6,
            derivative_chunk_size=6,
            reset=PeriodicReset(
                interval=1,
                optimizer_steps=3,
                learning_rate=1e-3,
                batch_size=6,
            ),
            seed=5,
        )
        result = solver.step(state, completed_steps=1)
        result.state.validate()
        self.assertEqual(result.projection.alpha.shape, (16,))
        self.assertIsNotNone(result.reset)
        assert result.reset is not None
        self.assertEqual(result.reset.step, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
