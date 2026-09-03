"""Run with: python -m unittest discover -s DTB_Game_Ver2 -p test_game_visualization.py."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from game_visualization import (
    compute_equilibrium_occupancy,
    plot_coordinate_snapshots,
    plot_occupancy_over_time,
    plot_state_heatmaps,
    save_game_visualizations,
)


class GameVisualizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_full_state_distance_and_feasibility_control_assignments(self):
        equilibria = np.array([[.5, .5, 0], [0, 0, 0]])
        points = np.array([[[.5, .5, 0], [.5, .5, .3], [.5, .5, -.01], [0, 0, 0]]])
        original = points.copy()
        occupancy = compute_equilibrium_occupancy(
            points, equilibria, radius=.05, feasible_mask=(points >= 0).all(axis=-1))
        # The second particle overlaps the first in (x1,x2), but is far away in x3.
        np.testing.assert_array_equal(occupancy.assignments, [[0, -1, -1, 1]])
        np.testing.assert_array_equal(occupancy.counts, [[1, 1, 2]])
        self.assertEqual(occupancy.pair_fractions[0, 0, 1], .25)
        np.testing.assert_allclose(sum(occupancy.families.values()), 1)
        np.testing.assert_array_equal(points, original)

    def test_overlapping_reference_neighborhoods_do_not_double_count(self):
        points = np.array([[[0., 0., 0.], [.01, 0., 0.]]])
        equilibria = np.array([[0., 0., 0.], [.02, 0., 0.]])
        occupancy = compute_equilibrium_occupancy(points, equilibria, radius=.1)
        np.testing.assert_array_equal(occupancy.assignments, [[0, 0]])
        self.assertEqual(occupancy.counts.sum(), 2)
        self.assertAlmostEqual(occupancy.fractions.sum(), 1)

    def test_unknown_equilibria_and_invalid_inputs(self):
        points = np.zeros((2, 3, 5))
        occupancy = compute_equilibrium_occupancy(points, np.empty((0, 5)))
        np.testing.assert_array_equal(occupancy.assignments, -1)
        np.testing.assert_array_equal(occupancy.fractions, 1)
        with self.assertRaises(ValueError):
            compute_equilibrium_occupancy(points, np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            compute_equilibrium_occupancy(points, np.zeros((1, 5)), radius=-1)
        with self.assertRaises(ValueError):
            compute_equilibrium_occupancy(points, np.zeros((1, 5)), feasible_mask=np.ones((2, 3)))

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

    def test_heatmaps_share_particle_order_and_color_scale(self):
        data = np.array([[[.1, -.1, 0], [.2, .3, 0], [.4, .2, .1]],
                         [[.9, 0, 0], [0, 0, .8], [.89, 0, .01]]])
        fig, order = plot_state_heatmaps(data, [0, 1], [0, 1])
        self.assertEqual(len(set(order)), 3)
        for step, axis in enumerate(fig.axes[:2]):
            np.testing.assert_array_equal(axis.images[0].get_array(), data[step, order])
        self.assertIs(fig.axes[0].images[0].norm, fig.axes[1].images[0].norm)

    def test_single_snapshot_has_visible_occupancy(self):
        points = np.full((1, 1, 3), .1)
        occupancy = compute_equilibrium_occupancy(points, points[0])
        fig = plot_occupancy_over_time(occupancy, np.array([0.]), radius=.05)
        self.assertAlmostEqual(sum(patch.get_height() for patch in fig.axes[0].patches), 1)

    def test_complete_outputs_for_multiple_game_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            for dim in (3, 5, 6, 10):
                with self.subTest(dim=dim):
                    pair = np.zeros(dim)
                    pair[:2] = .5
                    equilibria = np.stack([np.zeros(dim), np.full(dim, .1), pair])
                    endpoints = np.vstack([equilibria, np.ones(dim)])
                    data = endpoints[None] + np.array([.2, .1, 0])[:, None, None]
                    original = data.copy()
                    output = save_game_visualizations(
                        data, h=.2, equilibria=equilibria,
                        equilibrium_names=["origin", "symmetric", "pair(1,2)"],
                        output_dir=Path(directory) / str(dim), show=False)
                    for key in ("coordinate_snapshots", "state_heatmaps", "equilibrium_pair_occupancy",
                                "equilibrium_occupancy_over_time", "equilibrium_occupancy_csv",
                                "visualization_data"):
                        self.assertGreater(output["paths"][key].stat().st_size, 0)
                    np.testing.assert_array_equal(output["occupancy"].counts[-1], [1, 1, 1, 1])
                    np.testing.assert_allclose(output["occupancy"].fractions.sum(axis=1), 1)
                    self.assertEqual(output["occupancy"].pair_fractions.shape, (3, dim, dim))
                    with np.load(output["paths"]["visualization_data"], allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["heatmap_particle_ids"], output["heatmap_particle_ids"])
                    np.testing.assert_array_equal(data, original)


if __name__ == "__main__":
    unittest.main()
