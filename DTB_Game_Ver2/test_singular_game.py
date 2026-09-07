"""Acceptance tests execute the notebook's definitions, avoiding a second implementation.

Run: python -m unittest test_singular_game -v
Requires the same PyTorch, NumPy, pandas, Matplotlib and IPython as the notebook.
The full plotting cells and experiment matrix are intentionally not run here.
"""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
import torch


def load_notebook_definitions():
    path = Path(__file__).with_name('singular_game_2d_parameter_evolving_dtb.ipynb')
    notebook = json.loads(path.read_text())
    namespace = {'__name__': 'singular_game_notebook'}
    definitions = {'setup', 'config', 'benchmark', 'tangent_helpers', 'dtb_loop', 'baseline_loop'}
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        for cell in notebook['cells']:
            if cell['id'] in definitions:
                source = ''.join(cell['source'])
                # The standalone checks are called in a dedicated test below.
                if cell['id'] == 'baseline_loop':
                    source = source.rsplit('acceptance_checks()', 1)[0]
                exec(compile(source, f'{path.name}:{cell["id"]}', 'exec'), namespace)
    finally:
        os.chdir(previous)
    return namespace


class SingularGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = load_notebook_definitions()

    def small_config(self, **changes):
        return self.n['replace'](self.n['BASE'], **{
            'n_train': 32, 'n_test': 64, 'basis_size': 32,
            't_final': 0.003, **changes,
        })

    def test_analytic_benchmark_and_initialization(self):
        self.n['acceptance_checks']()

    def test_singular_reference_rejects_collision_and_invalid_time(self):
        x = torch.tensor([[2., 1.]], dtype=torch.float64)
        for t in (-1., .5, .6, float('nan')):
            with self.assertRaises(ValueError):
                self.n['exact_singular_flow'](x, t)
        with self.assertRaises(FloatingPointError):
            self.n['singular_game_drift'](torch.ones(1, 2))

    def test_ridge_solves_particle_normalized_objective(self):
        generator = torch.Generator().manual_seed(100)
        J = torch.randn(7, 2, 5, dtype=torch.float64, generator=generator)
        v = torch.randn(7, 2, dtype=torch.float64, generator=generator)
        eta = .03
        alpha, _ = self.n['ridge_tangent_solve'](J, v, eta)
        # Check the optimality equation independently, not the SVD formula.
        A = J.reshape(14, 5)
        gradient = A.T @ (A @ alpha - v.reshape(-1)) / 7 + eta * alpha
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
        # Repeating the empirical distribution must not change effective ridge.
        repeated, _ = self.n['ridge_tangent_solve'](J.repeat(3, 1, 1), v.repeat(3, 1), eta)
        torch.testing.assert_close(alpha, repeated)

    def test_test_jvp_matches_explicit_restricted_basis(self):
        cfg = self.small_config()
        model, theta, structure, selected, train, test = self.n['make_problem'](cfg)
        # Move away from identity, so hidden-layer derivatives are also exercised.
        theta = theta + .01 * torch.sin(torch.arange(theta.numel(), dtype=theta.dtype))
        alpha = torch.cos(torch.arange(cfg.basis_size, dtype=theta.dtype))
        _, J, _ = self.n['game_dtb_basis_matrices'](theta, selected, test, model, structure)
        actual = self.n['test_tangent_velocity'](theta, selected, alpha, test, model, structure)
        torch.testing.assert_close(actual, torch.einsum('ndm,m->nd', J, alpha))

    def test_official_history_evaluates_updated_map(self):
        cfg = self.small_config(basis_size=64)
        result = self.n['run_dtb'](cfg, verbose=False)
        model, _, structure, selected, _, test = self.n['make_problem'](cfg)
        mapped = self.n['map_at'](result['theta_final'], test, model, structure)
        torch.testing.assert_close(result['particles'][-1], mapped)
        frozen = torch.ones(result['parameter_count'], dtype=torch.bool)
        frozen[selected] = False
        self.assertTrue(torch.equal(result['theta_final'][frozen], result['theta_initial'][frozen]))
        self.assertGreater(float(result['steps'].map_step_defect.iloc[-1]), 0)
        self.assertAlmostEqual(result['metrics'].iloc[-1].time, cfg.t_final)

    def test_gap_controller_and_euler_use_matched_times(self):
        cfg = self.small_config(policy='gap-aware', gamma=1e-4, t_final=.001)
        result = self.n['run_dtb'](cfg, verbose=False)
        self.assertEqual(result['status'], 'completed')
        for step in result['steps'].itertuples():
            before = result['metrics'].loc[result['metrics'].time == step.time].iloc[0]
            self.assertLessEqual(step.step_size, cfg.gamma * before.min_gap_monitor**2 + 1e-14)
        euler = self.n['direct_euler_singular'](result['test_initial'], result['times'], cfg.h_max)
        self.assertEqual(len(euler['times']), len(result['times']))
        self.assertTrue(self.n['np'].allclose(euler['times'], result['times'], atol=1e-14, rtol=0))

    def test_collision_proposal_is_rejected_without_entering_history(self):
        # At identity only output weights are active, so amplifying the solved
        # direction yields a deliberately bad first proposal through the real map.
        original = self.n['ridge_tangent_solve']

        def oversized_direction(*args):
            alpha, diagnostics = original(*args)
            return 1e8 * alpha, diagnostics

        with patch.dict(self.n, {'ridge_tangent_solve': oversized_direction}):
            # Full coordinates include both output rows; a tiny random subset
            # could move only x2 and increase the gap despite a huge velocity.
            result = self.n['run_dtb'](self.small_config(basis_size=354), verbose=False)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('crosses', result['stop_reason'])
        self.assertEqual(result['times'].tolist(), [0.0])
        self.assertEqual(len(result['steps']), 0)
        self.assertTrue((self.n['gap'](result['particles']) > 0).all())
        torch.testing.assert_close(result['theta_initial'], result['theta_final'])

    def test_gap_stop_and_iteration_limit_are_explicit(self):
        stopped = self.n['run_dtb'](self.small_config(r_stop=3.0), verbose=False)
        self.assertEqual(stopped['status'], 'stopped')
        self.assertEqual(stopped['times'].tolist(), [0.0])
        limited = self.n['run_dtb'](self.small_config(max_steps=1), verbose=False)
        self.assertEqual(limited['status'], 'stopped')
        self.assertEqual(len(limited['steps']), 1)

    def test_failed_tangent_solve_retains_the_correct_last_state_and_time(self):
        original = self.n['ridge_tangent_solve']
        calls = 0

        def fail_second_solve(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise torch.linalg.LinAlgError('simulated failed SVD')
            return original(*args)

        with patch.dict(self.n, {'ridge_tangent_solve': fail_second_solve}):
            result = self.n['run_dtb'](self.small_config(), verbose=False)
        self.assertEqual(result['status'], 'failed')
        self.assertAlmostEqual(result['times'][-1], .001)
        self.assertAlmostEqual(result['snapshot_times'][-1], .001)
        truth = self.n['exact_singular_flow'](result['test_initial'], .001)
        self.assertAlmostEqual(result['metrics'].iloc[-1].state_rmse,
                               self.n['vector_rms'](result['particles'][-1] - truth))
        self.assertTrue(self.n['np'].isnan(result['metrics'].iloc[-1].alpha_norm))

    def test_baseline_rejects_bad_grids_and_collision_requests(self):
        x = torch.tensor([[2., 1.]], dtype=torch.float64)
        for times in ([], [.1], [0, .1, .1], [0, float('nan')], [0, .5]):
            with self.assertRaises(ValueError):
                self.n['direct_euler_singular'](x, times, .001)

    def test_regularized_reference_has_correct_time_derivative(self):
        x = torch.tensor([[2., 1.], [-2., -1.]], dtype=torch.float64)
        dt = 1e-6
        for eps in (.2, .1, .05):
            flow = self.n['exact_regularized_flow']
            derivative = (flow(x, .49 + dt, eps) - flow(x, .49 - dt, eps)) / (2 * dt)
            truth = self.n['regularized_game_drift'](flow(x, .49, eps), eps)
            torch.testing.assert_close(derivative, truth, atol=2e-7, rtol=2e-7)


if __name__ == '__main__':
    unittest.main()
