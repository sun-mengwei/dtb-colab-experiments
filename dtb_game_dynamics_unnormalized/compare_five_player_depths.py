"""Compare matched depth-2 and depth-4 five-player Neural--DTB runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def known_equilibria() -> np.ndarray:
    symmetric = np.full(5, 7.0 / 32.0)
    one_zero = []
    for zero_index in range(5):
        equilibrium = np.full(5, 5.0 / 18.0)
        equilibrium[zero_index] = 0.0
        one_zero.append(equilibrium)
    return np.vstack([np.zeros(5), symmetric, np.asarray(one_zero)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="outputs/five_player_depth_comparison"
    )
    parser.add_argument(
        "--output",
        default="outputs/five_player_depth_comparison/depth_comparison.png",
    )
    parser.add_argument(
        "--csv",
        default="outputs/five_player_depth_comparison/depth_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    histories = {
        2: np.load(root / "depth2" / "history.npz"),
        4: np.load(root / "depth4" / "history.npz"),
    }
    equilibria = known_equilibria()
    rows = [_metrics(depth, history, equilibria) for depth, history in histories.items()]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot(histories, rows, equilibria, output_path)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {output_path}")
    print(f"saved {csv_path}")


def _metrics(
    depth: int, history: np.lib.npyio.NpzFile, equilibria: np.ndarray
) -> dict[str, float | int]:
    particles = history["final_particles"]
    distances = np.linalg.norm(
        particles[:, None, :] - equilibria[None, :, :], axis=2
    )
    nearest = distances.min(axis=1)
    means = particles.mean(axis=0)
    stds = particles.std(axis=0)
    return {
        "depth": depth,
        "particles": len(particles),
        "final_projection_residual": float(history["projection_residuals"][-1]),
        "final_retained_rank": int(history["retained_ranks"][-1]),
        "median_nearest_known_equilibrium_distance": float(np.median(nearest)),
        "fraction_within_0.15_of_known_equilibrium": float(np.mean(nearest < 0.15)),
        "mean_coordinate": float(means.mean()),
        "mean_coordinate_spread": float(means.max() - means.min()),
        "mean_marginal_std": float(stds.mean()),
    }


def _plot(
    histories: dict[int, np.lib.npyio.NpzFile],
    rows: list[dict[str, float | int]],
    equilibria: np.ndarray,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(13.5, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(2.1, 1.0))
    projected_equilibria = np.column_stack(
        [equilibria[:, 0], equilibria[:, 1:].mean(axis=1)]
    )

    for column, depth in enumerate((2, 4)):
        axis = figure.add_subplot(grid[0, column])
        particles = histories[depth]["final_particles"]
        axis.scatter(
            particles[:, 0],
            particles[:, 1:].mean(axis=1),
            s=2.2,
            alpha=0.25,
            color="#1786c7",
            edgecolors="none",
            rasterized=True,
        )
        axis.scatter(
            projected_equilibria[:, 0],
            projected_equilibria[:, 1],
            s=70,
            color="#ffbf00",
            edgecolors="#222222",
            linewidths=0.7,
            marker="X",
            zorder=5,
            label="7 unstable equilibria (some overlap in projection)",
        )
        axis.set(
            title=f"depth {depth}, N=2,000 at t=1",
            xlabel=r"$x_1$",
            ylabel=r"mean$(x_2,\ldots,x_5)$",
            xlim=(-0.4, 1.0),
            ylim=(-0.4, 1.0),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="upper right")

    residual_axis = figure.add_subplot(grid[1, 0])
    for depth, color in zip((2, 4), ("#2a7ab0", "#d95f02")):
        history = histories[depth]
        residual_axis.plot(
            history["times"][1:],
            history["projection_residuals"],
            label=f"depth {depth}",
            linewidth=1.7,
            color=color,
        )
    residual_axis.set(
        xlabel="time",
        ylabel="relative tangent-projection residual",
        xlim=(0.0, 1.0),
    )
    residual_axis.grid(alpha=0.25)
    residual_axis.legend(frameon=False)

    table_axis = figure.add_subplot(grid[1, 1])
    table_axis.axis("off")
    table_values = [
        [
            str(int(row["depth"])),
            f"{float(row['final_projection_residual']):.3f}",
            f"{float(row['median_nearest_known_equilibrium_distance']):.3f}",
            f"{100 * float(row['fraction_within_0.15_of_known_equilibrium']):.1f}%",
        ]
        for row in rows
    ]
    table = table_axis.table(
        cellText=table_values,
        colLabels=("depth", "final residual", "median distance", "within 0.15"),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.65)

    figure.suptitle(
        "Five-player Neural--DTB depth comparison: matched random seed and basis",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
