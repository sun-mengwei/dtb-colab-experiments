"""Compare three-player Neural--DTB runs with different particle counts.

Each input directory must contain the ``history.npz`` written by
``replicate_three_player_game.py``.  The script keeps a shared camera and axis
range across panels so apparent concentration is comparable rather than a
plotting-scale artifact.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STABLE_EQUILIBRIA = np.asarray(
    [
        [3.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]
)
UNSTABLE_EQUILIBRIUM = np.zeros((1, 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/sample_study_stable")
    parser.add_argument("--counts", default="512,1024,5000")
    parser.add_argument(
        "--output", default="outputs/sample_study_stable/sample_count_comparison.png"
    )
    parser.add_argument(
        "--csv", default="outputs/sample_study_stable/sample_count_metrics.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    counts = [int(value) for value in args.counts.split(",")]
    histories = {n: np.load(root / f"n{n}" / "history.npz") for n in counts}

    rows: list[dict[str, float | int]] = []
    for n, history in histories.items():
        particles = history["final_particles"]
        distances = np.linalg.norm(
            particles[:, None, :] - STABLE_EQUILIBRIA[None, :, :], axis=2
        )
        nearest = distances.min(axis=1)
        assignments = np.bincount(distances.argmin(axis=1), minlength=4)
        rows.append(
            {
                "particles": n,
                "mean_x1": float(particles[:, 0].mean()),
                "mean_x2": float(particles[:, 1].mean()),
                "mean_x3": float(particles[:, 2].mean()),
                "final_projection_residual": float(history["projection_residuals"][-1]),
                "median_nearest_stable_distance": float(np.median(nearest)),
                "fraction_within_0.15": float(np.mean(nearest < 0.15)),
                "nearest_symmetric": int(assignments[0]),
                "nearest_110": int(assignments[1]),
                "nearest_101": int(assignments[2]),
                "nearest_011": int(assignments[3]),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_comparison(histories, rows, output_path)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {output_path}")
    print(f"saved {csv_path}")


def _plot_comparison(
    histories: dict[int, np.lib.npyio.NpzFile],
    rows: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    counts = list(histories)
    figure = plt.figure(figsize=(18, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, len(counts), height_ratios=(2.3, 1.0))

    for column, n in enumerate(counts):
        axis = figure.add_subplot(grid[0, column], projection="3d")
        particles = histories[n]["final_particles"]
        point_size = max(2.0, 12.0 * (128.0 / n) ** 0.35)
        axis.scatter(
            particles[:, 0],
            particles[:, 1],
            particles[:, 2],
            s=point_size,
            alpha=0.32,
            color="#1686c9",
            linewidths=0,
        )
        axis.scatter(
            STABLE_EQUILIBRIA[:, 0],
            STABLE_EQUILIBRIA[:, 1],
            STABLE_EQUILIBRIA[:, 2],
            s=34,
            color="#d62728",
            marker="o",
            depthshade=False,
            label="stable equilibria (4)" if column == 0 else "_nolegend_",
        )
        axis.scatter(
            UNSTABLE_EQUILIBRIUM[:, 0],
            UNSTABLE_EQUILIBRIUM[:, 1],
            UNSTABLE_EQUILIBRIUM[:, 2],
            s=64,
            color="#ffbf00",
            edgecolors="#222222",
            linewidths=0.7,
            marker="X",
            depthshade=False,
            label=(
                "unstable equilibrium (origin)" if column == 0 else "_nolegend_"
            ),
        )
        axis.set(xlim=(-0.4, 1.0), ylim=(-0.4, 1.0), zlim=(-0.4, 1.0))
        axis.set_xlabel(r"$x_1$")
        axis.set_ylabel(r"$x_2$")
        axis.set_zlabel(r"$x_3$")
        axis.set_title(f"N={n:,} at t=1")
        axis.view_init(elev=23, azim=-58)
        try:
            axis.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    residual_axis = figure.add_subplot(grid[1, :2])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(counts)))
    for color, n in zip(colors, counts):
        history = histories[n]
        times = history["times"][1:]
        residual_axis.plot(
            times,
            history["projection_residuals"],
            label=f"N={n:,}",
            color=color,
            linewidth=1.6,
        )
    residual_axis.set(
        xlabel="time",
        ylabel="relative tangent-projection residual",
        xlim=(0.0, 1.0),
    )
    residual_axis.grid(alpha=0.25)
    residual_axis.legend(ncol=2, frameon=False)

    table_axis = figure.add_subplot(grid[1, 2:])
    table_axis.axis("off")
    table_values = [
        [
            f"{int(row['particles']):,}",
            f"{float(row['final_projection_residual']):.3f}",
            f"{float(row['median_nearest_stable_distance']):.3f}",
            f"{100 * float(row['fraction_within_0.15']):.1f}%",
        ]
        for row in rows
    ]
    table = table_axis.table(
        cellText=table_values,
        colLabels=("N", "final residual", "median distance", "within 0.15"),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    figure.suptitle(
        "Three-player Neural--DTB sample-count study (fixed m=128)", fontsize=16
    )
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
