"""Run and compare MLP and frozen-feature MMNN tangent dictionaries in 3D."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from game_dtb.runner import run_experiment
from replicate_three_player_game import experiment_args


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
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--basis-size", type=int, default=128)
    parser.add_argument("--svd-rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root", default="outputs/three_player_mlp_mmnn_n500"
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="reuse existing mlp/history.npz and mmnn/history.npz",
    )
    return parser.parse_args()


def _configuration(
    cli: argparse.Namespace, architecture: str, output_dir: Path
) -> SimpleNamespace:
    base_cli = SimpleNamespace(
        paper_scale=False,
        particles=cli.particles,
        depth=cli.depth,
        architecture=architecture,
        rank=cli.rank,
        width=cli.width,
        svd_rtol=cli.svd_rtol,
        device=cli.device,
        output_dir=str(output_dir),
        skip_sde_baseline=True,
        seed=cli.seed,
    )
    args = experiment_args(base_cli)
    args.basis_size = cli.basis_size
    args.run_sde_baseline = False
    return args


def main() -> None:
    cli = parse_args()
    if min(cli.particles, cli.width, cli.rank, cli.depth, cli.basis_size) < 1:
        raise ValueError("particle, width, rank, depth, and basis values must be positive")
    root = Path(cli.output_root)
    root.mkdir(parents=True, exist_ok=True)

    if not cli.skip_runs:
        for architecture in ("mlp", "mmnn"):
            print(f"=== Running {architecture.upper()} tangent dictionary ===")
            run_experiment(
                _configuration(cli, architecture, root / architecture)
            )

    histories = {
        name: np.load(root / name / "history.npz")
        for name in ("mlp", "mmnn")
    }
    if not np.array_equal(
        histories["mlp"]["initial_particles"],
        histories["mmnn"]["initial_particles"],
    ):
        raise RuntimeError("MLP and MMNN did not start from identical particles")

    configs = {
        name: json.loads((root / name / "config.json").read_text())
        for name in ("mlp", "mmnn")
    }
    rows = [
        _metrics(name, histories[name], configs[name])
        for name in ("mlp", "mmnn")
    ]

    figure_path = root / "architecture_comparison.png"
    csv_path = root / "architecture_metrics.csv"
    _plot(histories, rows, figure_path)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    particle_difference = np.mean(
        np.abs(
            histories["mlp"]["final_particles"]
            - histories["mmnn"]["final_particles"]
        )
    )
    print(f"mean absolute paired-particle difference: {particle_difference:.6f}")
    print(f"saved {figure_path}")
    print(f"saved {csv_path}")


def _metrics(
    architecture: str,
    history: np.lib.npyio.NpzFile,
    config: dict[str, object],
) -> dict[str, float | int | str]:
    particles = history["final_particles"]
    distances = np.linalg.norm(
        particles[:, None, :] - STABLE_EQUILIBRIA[None, :, :], axis=2
    )
    nearest = distances.min(axis=1)
    initial_diagnostics = config.get("tanh_diagnostics_initial", [])
    maximum_saturation = max(
        (
            float(layer["fraction_tanh_derivative_lt_0.05"])
            for layer in initial_diagnostics
        ),
        default=float("nan"),
    )
    return {
        "architecture": architecture.upper(),
        "particles": len(particles),
        "trainable_tangent_parameters": int(config["trainable_parameters_M"]),
        "selected_basis_size": int(config["actual_basis_size_m"]),
        "mean_retained_rank": float(history["retained_ranks"].mean()),
        "final_retained_rank": int(history["retained_ranks"][-1]),
        "mean_projection_residual": float(history["projection_residuals"].mean()),
        "final_projection_residual": float(history["projection_residuals"][-1]),
        "median_nearest_stable_distance": float(np.median(nearest)),
        "fraction_within_0.15_of_stable": float(np.mean(nearest < 0.15)),
        "maximum_initial_saturation_fraction": maximum_saturation,
    }


def _plot(
    histories: dict[str, np.lib.npyio.NpzFile],
    rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(13.5, 8.7), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(2.15, 1.0))

    for column, name in enumerate(("mlp", "mmnn")):
        axis = figure.add_subplot(grid[0, column], projection="3d")
        particles = histories[name]["final_particles"]
        axis.scatter(
            particles[:, 0], particles[:, 1], particles[:, 2],
            s=4.0, alpha=0.32, color="#1786c7", edgecolors="none",
            rasterized=True,
        )
        axis.scatter(
            STABLE_EQUILIBRIA[:, 0], STABLE_EQUILIBRIA[:, 1],
            STABLE_EQUILIBRIA[:, 2], s=38, color="#d62728", marker="o",
            depthshade=False, label="stable equilibria (4)",
        )
        axis.scatter(
            UNSTABLE_EQUILIBRIUM[:, 0], UNSTABLE_EQUILIBRIUM[:, 1],
            UNSTABLE_EQUILIBRIUM[:, 2], s=68, color="#ffbf00",
            edgecolors="#222222", linewidths=0.7, marker="X",
            depthshade=False, label="unstable origin",
        )
        axis.set(
            xlim=(-0.4, 1.0), ylim=(-0.4, 1.0), zlim=(-0.4, 1.0),
            xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$",
            title=f"{name.upper()}, N=500 at t=1",
        )
        axis.view_init(elev=23, azim=-58)
        axis.set_box_aspect((1, 1, 1))
        if column == 0:
            axis.legend(frameon=False, loc="upper left")

    residual_axis = figure.add_subplot(grid[1, 0])
    colors = {"mlp": "#2a7ab0", "mmnn": "#d95f02"}
    for name in ("mlp", "mmnn"):
        history = histories[name]
        residual_axis.plot(
            history["times"][1:], history["projection_residuals"],
            color=colors[name], linewidth=1.7, label=name.upper(),
        )
    residual_axis.set(
        xlabel="time", ylabel="relative tangent-projection residual",
        xlim=(0.0, 1.0),
    )
    residual_axis.grid(alpha=0.25)
    residual_axis.legend(frameon=False)

    table_axis = figure.add_subplot(grid[1, 1])
    table_axis.axis("off")
    values = [
        [
            str(row["architecture"]),
            str(int(row["trainable_tangent_parameters"])),
            str(int(row["final_retained_rank"])),
            f"{float(row['final_projection_residual']):.3f}",
            f"{float(row['median_nearest_stable_distance']):.3f}",
            f"{100 * float(row['fraction_within_0.15_of_stable']):.1f}%",
        ]
        for row in rows
    ]
    table = table_axis.table(
        cellText=values,
        colLabels=("model", "M", "rank", "residual", "median dist.", "within .15"),
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.65)

    figure.suptitle(
        "Three-player Neural--DTB: ordinary MLP versus frozen-feature MMNN",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
