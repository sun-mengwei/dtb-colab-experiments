"""Run with: python -m unittest discover -s DTB_Game_Ver2 -p test_game_visualization.py."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from game_visualization import (
    format_dtb_metrics,
    plot_coordinate_snapshots,
    save_game_visualizations,
)


class GameVisualizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_coordinate_axes_are_fixed_across_time(self):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5) / 100
        fig = plot_coordinate_snapshots(data, [0, .1, .2], [0, 2])
        self.assertEqual(len(fig.axes), 4)
        for row in range(2):
            first, second = fig.axes[2 * row:2 * row + 2]
            np.testing.assert_allclose(first.get_xlim(), second.get_xlim())
            np.testing.assert_allclose(first.get_ylim(), second.get_ylim())
        self.assertEqual(fig.axes[0].get_xlabel(), "$x_{1}$")
        self.assertEqual(fig.axes[2].get_xlabel(), "$x_{3}$")
        with self.assertRaises(ValueError):
            plot_coordinate_snapshots(data, [0, .1, .2], [0, 2], coordinate_pairs=[(1, 6)])

    def test_metrics_use_recorded_values_and_distinguish_last_solve_time(self):
        states = [dict(time=t, game_drift_rms=.125, median_known_distance=.25,
                       p90_known_distance=.5, minimum_coordinate=-.002,
                       negative_coordinate_fraction=.01) for t in (0, 1)]
        projections = [dict(time=t, projection_residual=.075, projected_drift_rms=.1,
                            alpha_norm=2, basis_seconds=.01, solve_seconds=.02,
                            update_seconds=.003) for t in (0, .99)]
        text = format_dtb_metrics(states, projections)
        self.assertIn("Last: $t=1$", text)
        self.assertIn("Last: $t=0.99$", text)
        self.assertIn(r"v_k^b=\frac{\lVert\mathbf{g}_k\rVert_2}{\sqrt{N}}", text)
        self.assertIn(r"1.250\times 10^{-1}", text)
        self.assertIn(r"-2.000\times 10^{-3}", text)
        with tempfile.TemporaryDirectory() as directory:
            result = save_game_visualizations(
                np.zeros((2, 4, 3)), output_dir=directory, show=False,
                state_diagnostics=states, projection_diagnostics=projections)
            self.assertEqual(result["paths"]["dtb_metrics"].read_text(), text)
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {"coordinate_snapshots.png", "dtb_metrics.md"})

    def test_only_coordinate_figures_for_multiple_game_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            for dim in (3, 5, 6, 10):
                with self.subTest(dim=dim):
                    equilibria = np.stack([np.zeros(dim), np.full(dim, .1)])
                    data = equilibria[None] + np.array([.2, .1, 0])[:, None, None]
                    original = data.copy()
                    folder = Path(directory) / str(dim)
                    output = save_game_visualizations(
                        data, h=.2, equilibria=equilibria, output_dir=folder, show=False)
                    self.assertEqual(set(output["paths"]), {"coordinate_snapshots"})
                    self.assertEqual([p.name for p in folder.iterdir()], ["coordinate_snapshots.png"])
                    self.assertGreater(output["paths"]["coordinate_snapshots"].stat().st_size, 0)
                    np.testing.assert_array_equal(data, original)


if __name__ == "__main__":
    unittest.main()
