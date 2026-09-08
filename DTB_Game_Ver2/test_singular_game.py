"""Checks for the singular map update, shared functions, and fixed parameters."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch


def definitions():
    path = Path(__file__).with_name('singular_game_2d.ipynb')
    notebook = json.loads(path.read_text())
    namespace = {'__name__': 'singular_game_notebook'}
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        for cell in notebook['cells']:
            if cell['id'] in {'setup', 'drift', 'dtb_loop', 'euler_loop', 'plots'}:
                source = ''.join(cell['source']).split('\n# Run')[0]
                exec(compile(source, f'{path.name}:{cell["id"]}', 'exec'), namespace)
    finally:
        os.chdir(previous)
    return namespace


class SingularGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = definitions()

    def test_drift_and_singular_denominator(self):
        x = torch.tensor([[2., 1.], [-2., -1.]], dtype=torch.float64, requires_grad=True)
        r = x[:, 0]-x[:, 1]
        v1 = torch.autograd.grad((-2*torch.log(r.abs())).sum(), x, retain_graph=True)[0][:, 0]
        v2 = torch.autograd.grad(torch.log(r.abs()).sum(), x)[0][:, 1]
        torch.testing.assert_close(self.n['dynamic_drift'](x), torch.stack((v1, v2), -1))
        with self.assertRaises(FloatingPointError):
            self.n['dynamic_drift'](torch.ones(1, 2))

    def test_shared_imports_and_truncated_svd(self):
        for name, module in (
            ('ResidualMLPMap', 'run_game_dtb'), ('game_dtb_basis_matrices', 'run_game_dtb'),
            ('flat_params', 'dtb'), ('jform_solve', 'dtb'),
            ('count_trainable', 'network'), ('plot_tangent_diagnostics', 'utility'),
        ):
            self.assertEqual(self.n[name].__module__, module)
        A = torch.diag(torch.tensor([4., .01, 0.], dtype=torch.float64))
        b = torch.tensor([8., 1., 1.], dtype=torch.float64)
        actual = self.n['jform_solve'](A, b, rtol=.01, method=self.n['SVD_METHOD'])
        torch.testing.assert_close(actual, torch.tensor([2., 0., 0.], dtype=actual.dtype))

    def test_accumulated_map_uses_current_positions_and_frozen_theta(self):
        # A known position-dependent tangent field makes a cached-label Jacobian
        # or a substituted network-state update fail this regression test.
        basis_calls, solve_targets = [], []

        def known_basis(theta, selected, x, model, structure, **kwargs):
            basis_calls.append((theta.clone(), x.clone()))
            J = torch.zeros(len(x), 2, len(selected), dtype=x.dtype)
            J[:, :, 0] = 1+x.square()
            return torch.zeros_like(x), J, J.reshape(-1, len(selected))

        def known_coefficients(J, target, **kwargs):
            solve_targets.append(target.clone())
            alpha = torch.zeros(J.shape[1], dtype=J.dtype)
            alpha[0] = 1
            return alpha

        with patch.dict(self.n, {'game_dtb_basis_matrices': known_basis,
                                 'jform_solve': known_coefficients}):
            result = self.n['run_dtb'](n=16, n_snapshot=24, t_final=.002)
        train0, cloud0 = result['labels'], result['snapshot_labels']
        train1, cloud1 = train0+.001*(1+train0.square()), cloud0+.001*(1+cloud0.square())
        expected_train = train1+.001*(1+train1.square())
        expected_cloud = cloud1+.001*(1+cloud1.square())
        torch.testing.assert_close(result['train_particles'], expected_train)
        torch.testing.assert_close(result['snapshots'][-1], expected_cloud)
        torch.testing.assert_close(basis_calls[2][1], train1)
        torch.testing.assert_close(basis_calls[3][1], cloud1)
        torch.testing.assert_close(solve_targets[1], self.n['dynamic_drift'](train1).reshape(-1))
        for theta, _ in basis_calls:
            torch.testing.assert_close(theta, result['theta_0'], rtol=0, atol=0)
        actual_parameters = torch.cat([p.detach().reshape(-1) for p in result['model'].parameters()])
        torch.testing.assert_close(actual_parameters, result['theta_0'], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in result['model'].parameters()))

    def test_real_step_and_projection_diagnostic(self):
        result = self.n['run_dtb'](n=32, n_snapshot=64, t_final=.001)
        self.assertEqual(result['status'], 'completed')
        theta, selected = result['theta_0'], result['selected']
        model, structure = result['model'], result['structure']
        _, J, _ = self.n['game_dtb_basis_matrices'](theta, selected, result['labels'], model, structure)
        target = self.n['dynamic_drift'](result['labels'])
        projected = torch.einsum('ndm,m->nd', J, result['alpha'][0])
        torch.testing.assert_close(result['train_particles'], result['labels']+.001*projected)
        error = (projected-target).norm()/target.norm()
        self.assertAlmostEqual(result['projection_error'][0], float(error))
        self.assertEqual(result['diagnostic_times'].tolist(), [0.0])
        torch.manual_seed(0)
        initial_model = self.n['ResidualMLPMap'](dim=2, width=16, depth=2,
            activation='tanh', dtype=torch.float64, zero_init_output=False).net
        expected_theta, _, _ = self.n['flat_params'](initial_model)
        torch.testing.assert_close(result['theta_0'], expected_theta, rtol=0, atol=0)

    def test_rejected_step_never_enters_snapshots(self):
        original = self.n['jform_solve']
        with patch.dict(self.n, {'jform_solve': lambda *args, **kwargs: 1e8*original(*args, **kwargs)}):
            result = self.n['run_dtb'](n=32, n_snapshot=64, basis_size=354, t_final=.003)
        self.assertIn('rejected', result['status'])
        self.assertEqual(result['snapshot_times'].tolist(), [0.0])
        self.assertEqual(len(result['alpha']), 0)
        torch.testing.assert_close(result['snapshots'][0], result['snapshot_labels'])
        torch.testing.assert_close(result['train_particles'], result['labels'])

    def test_endpoints_and_minimal_notebook(self):
        result = self.n['run_dtb'](n=32, n_snapshot=64, t_final=.0025)
        self.assertEqual(result['snapshot_times'][-1], .0025)
        self.assertEqual(len(result['alpha']), 3)
        self.assertTrue(np.all(np.diff(result['snapshot_times']) > 0))
        for t in (0., float('nan')):
            with self.assertRaises(ValueError):
                self.n['run_dtb'](t_final=t)
        notebook = json.loads(Path(__file__).with_name('singular_game_2d.ipynb').read_text())
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        for removed in ('exact_singular_flow', 'RUN_FULL_STUDY',
                        'RUN_REGULARIZED', 'SAVE_RUN', 'proposed_theta', 'ridge_solve'):
            self.assertNotIn(removed, source)

    def test_euler_is_independent_and_matches_direct_recurrence(self):
        with patch.dict(self.n, {'ResidualMLPMap': lambda *a, **k: self.fail('Euler constructed a network')}):
            result = self.n['run_euler'](n_snapshot=32, h=.001, t_final=.002)
        initial = result['snapshot_labels']
        first = initial+.001*self.n['dynamic_drift'](initial)
        second = first+.001*self.n['dynamic_drift'](first)
        torch.testing.assert_close(result['snapshots'][1], first)
        torch.testing.assert_close(result['snapshots'][-1], second)
        # Euler preserves 2*x2-x1 for this drift; this independent identity checks signs.
        invariant = lambda q: 2*q[..., 1]-q[..., 0]
        torch.testing.assert_close(invariant(result['snapshots']), invariant(initial).expand(3, -1))
        dtb = self.n['run_dtb'](n=16, n_snapshot=32, h=.001, t_final=.002)
        torch.testing.assert_close(dtb['snapshot_labels'], result['snapshot_labels'], rtol=0, atol=0)
        np.testing.assert_array_equal(dtb['snapshot_times'], result['snapshot_times'])

    def test_euler_rejects_invalid_proposal_and_keeps_last_valid_state(self):
        def bad_drift(x):
            out = torch.zeros_like(x)
            out[:, 0] = -1e8
            return out
        with patch.dict(self.n, {'dynamic_drift': bad_drift}):
            result = self.n['run_euler'](n_snapshot=16, t_final=.003)
        self.assertIn('rejected', result['status'])
        self.assertEqual(result['snapshot_times'].tolist(), [0.0])
        self.assertEqual(len(result['requested_snapshot_times']), 4)
        torch.testing.assert_close(result['snapshots'][0], result['snapshot_labels'])

    def test_aligned_plots_share_times_axes_colors_and_mark_missing_states(self):
        euler = self.n['run_euler'](n_snapshot=16, t_final=.003)
        partial = {**euler, 'snapshots': euler['snapshots'][:1],
                   'snapshot_times': euler['snapshot_times'][:1], 'status': 'stopped'}
        figure = self.n['plot_aligned_snapshots'](partial, euler)
        axes = figure.axes[:-1]  # Shared colorbar is last.
        self.assertEqual(len(axes), 8)
        for ax in axes:
            np.testing.assert_allclose(ax.get_xlim(), axes[0].get_xlim())
            np.testing.assert_allclose(ax.get_ylim(), axes[0].get_ylim())
        for column in range(4):
            self.assertEqual(axes[column].get_title(), axes[column+4].get_title())
        for ax in axes[1:4]:
            self.assertIn('Unavailable', ax.texts[0].get_text())
            self.assertEqual(len(ax.collections), 0)
        colors = axes[0].collections[-1].get_array()
        for ax in axes[4:]:
            np.testing.assert_array_equal(ax.collections[-1].get_array(), colors)
            self.assertIs(ax.collections[-1].norm, axes[0].collections[-1].norm)
        self.n['plt'].close(figure)

    def test_switches_allow_euler_only_dtb_only_and_neither_without_stale_results(self):
        path = Path(__file__).with_name('singular_game_2d.ipynb')
        notebook = json.loads(path.read_text())
        for do_dtb, do_euler in ((False, True), (True, False), (False, False)):
            namespace = {'__name__': 'singular_switch_test'}
            previous = Path.cwd()
            try:
                os.chdir(path.parent)
                for cell in notebook['cells']:
                    if cell['cell_type'] != 'code':
                        continue
                    source = ''.join(cell['source'])
                    if cell['id'] == 'setup':
                        source = source.replace('RUN_DTB = True', f'RUN_DTB = {do_dtb}')
                        source = source.replace('RUN_EULER = True', f'RUN_EULER = {do_euler}')
                    exec(compile(source, f'{path.name}:{cell["id"]}', 'exec'), namespace)
                    if cell['id'] == 'setup':
                        namespace.update(N=16, N_SNAPSHOT=24, T=.002)
                        if not do_dtb:
                            self.assertNotIn('game_dtb_basis_matrices', namespace)
                self.assertEqual(namespace['dtb_result'] is not None, do_dtb)
                self.assertEqual(namespace['euler_result'] is not None, do_euler)
                self.assertEqual(namespace['fig_diagnostics'] is not None, do_dtb)
                if do_dtb or do_euler:
                    self.assertEqual(len(namespace['fig_snapshots'].axes), 4)  # Three times + colorbar.
                else:
                    self.assertIsNone(namespace['fig_snapshots'])
            finally:
                namespace.get('plt', self.n['plt']).close('all')
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
