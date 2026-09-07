"""Checks for the simplified notebook. Run: python -m unittest test_singular_game -v."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch


def definitions():
    path = Path(__file__).with_name('singular_game_2d_parameter_evolving_dtb.ipynb')
    notebook = json.loads(path.read_text())
    namespace = {'__name__': 'singular_game_notebook'}
    for cell in notebook['cells']:
        if cell['id'] in {'setup', 'drift', 'tangent_helpers', 'dtb_loop'}:
            source = ''.join(cell['source']).split('\nresult = run_dtb()')[0]
            exec(compile(source, f'{path.name}:{cell["id"]}', 'exec'), namespace)
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

    def test_normalized_ridge_optimality(self):
        generator = torch.Generator().manual_seed(123)
        J = torch.randn(9, 2, 5, generator=generator, dtype=torch.float64)
        v = torch.randn(9, 2, generator=generator, dtype=torch.float64)
        alpha = self.n['ridge_solve'](J, v, 0.03)
        A = J.reshape(18, 5)
        optimality = A.T @ (A @ alpha - v.reshape(-1))/9 + 0.03*alpha
        torch.testing.assert_close(optimality, torch.zeros_like(alpha), atol=1e-12, rtol=0)
        repeated = self.n['ridge_solve'](J.repeat(3, 1, 1), v.repeat(3, 1), 0.03)
        torch.testing.assert_close(alpha, repeated)

    def test_selected_update_uses_actual_map_and_projection_error(self):
        result = self.n['run_dtb'](n=32, n_snapshot=64, basis_size=128, t_final=0.001)
        self.assertEqual(result['status'], 'completed')
        initial = torch.cat([p.detach().reshape(-1) for p in result['model'].parameters()])
        expected_theta = initial.index_add(0, result['selected'], 0.001*result['alpha'][0])
        torch.testing.assert_close(result['theta'], expected_theta)
        mapped = self.n['map_at'](result['theta'], result['snapshot_labels'], result['model'])
        torch.testing.assert_close(result['snapshots'][-1], mapped)
        torch.testing.assert_close(result['snapshots'][0], result['snapshot_labels'])
        J = self.n['tangent_jacobian'](initial, result['selected'], result['labels'], result['model'])
        target = self.n['dynamic_drift'](result['labels'])
        residual = torch.einsum('ndm,m->nd', J, result['alpha'][0])-target
        self.assertAlmostEqual(result['projection_error'][0], float(residual.square().sum(-1).mean().sqrt()))
        self.assertEqual(result['diagnostic_times'].tolist(), [0.0])
        self.assertEqual(result['snapshot_times'].tolist(), [0.0, 0.001])

    def test_rejected_step_never_enters_snapshots(self):
        original = self.n['ridge_solve']
        with patch.dict(self.n, {'ridge_solve': lambda *args: 1e8*original(*args)}):
            result = self.n['run_dtb'](n=32, n_snapshot=64, basis_size=354, t_final=0.003)
        self.assertIn('rejected', result['status'])
        self.assertEqual(result['snapshot_times'].tolist(), [0.0])
        self.assertEqual(len(result['alpha']), 0)
        torch.testing.assert_close(result['snapshots'][0], result['snapshot_labels'])

    def test_endpoints_and_minimal_notebook(self):
        result = self.n['run_dtb'](n=32, n_snapshot=64, basis_size=128, t_final=0.0025)
        self.assertEqual(result['snapshot_times'][-1], 0.0025)
        self.assertEqual(len(result['alpha']), 3)
        self.assertTrue(np.all(np.diff(result['snapshot_times']) > 0))
        self.assertTrue(torch.isfinite(result['snapshots']).all())
        for t in (0.5, float('nan')):
            with self.assertRaises(ValueError):
                self.n['run_dtb'](t_final=t)
        notebook = json.loads(Path(__file__).with_name('singular_game_2d_parameter_evolving_dtb.ipynb').read_text())
        self.assertEqual(len(notebook['cells']), 8)
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        for removed in ('direct_euler', 'exact_singular_flow', 'RUN_FULL_STUDY', 'RUN_REGULARIZED', 'SAVE_RUN'):
            self.assertNotIn(removed, source)


if __name__ == '__main__':
    unittest.main()
