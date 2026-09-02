"""Simple matched MLP-versus-NODE tangent-basis experiment in two dimensions."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from game_dtb.runner import run_experiment
from replicate_thesis_figures import experiment_args


STABLE_EQUILIBRIUM = np.asarray([0.5, 0.5])
UNSTABLE_EQUILIBRIUM = np.asarray([0.0, 0.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MLP and Neural-ODE DTB basis generators on the 2D game"
    )
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--basis-size", type=int, default=64)
    parser.add_argument("--svd-rtol", type=float, default=1e-5)
    parser.add_argument("--node-inner-steps", type=int, default=4)
    parser.add_argument("--node-integration-time", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root", default="outputs/two_player_mlp_node_n1000"
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="reuse existing mlp/history.npz and node/history.npz",
    )
    return parser.parse_args()


def _configuration(
    cli: argparse.Namespace, architecture: str, output_dir: Path
) -> SimpleNamespace:
    # BLOCK 1 -- Use the small Figure 4.2 preset: uniform [0,1]^2 particles,
    # 50 physical steps of size 0.02, float32, and no duplicated SDE baseline.
    base_cli = SimpleNamespace(
        paper_scale=False,
        particles=cli.particles,
        device=cli.device,
        skip_sde_baseline=True,
        seed=cli.seed,
    )
    args = experiment_args("uniform", output_dir, base_cli)
    args.architecture = architecture
    args.width = cli.width
    args.depth = cli.depth
    args.basis_size = cli.basis_size
    args.svd_rtol = cli.svd_rtol
    args.node_inner_steps = cli.node_inner_steps
    args.node_integration_time = cli.node_integration_time
    args.run_sde_baseline = False
    return args


def main() -> None:
    cli = parse_args()
    positive_integer_values = (
        cli.particles,
        cli.width,
        cli.depth,
        cli.basis_size,
        cli.node_inner_steps,
    )
    if min(positive_integer_values) < 1:
        raise ValueError("particle, width, depth, basis, and NODE-step values must be positive")
    if cli.node_integration_time <= 0:
        raise ValueError("NODE integration time must be positive")

    root = Path(cli.output_root)
    root.mkdir(parents=True, exist_ok=True)
    runtimes: dict[str, float] = {"mlp": float("nan"), "node": float("nan")}

    # BLOCK 2 -- Run both methods in one process.  The random seed is reset by
    # run_experiment, so they receive identical particles and matched network
    # initialization streams.  Only the interpretation of the network differs.
    if not cli.skip_runs:
        for architecture in ("mlp", "node"):
            print(f"=== Running {architecture.upper()} tangent dictionary ===")
            started = time.perf_counter()
            run_experiment(_configuration(cli, architecture, root / architecture))
            runtimes[architecture] = time.perf_counter() - started
            print(f"{architecture.upper()} elapsed seconds: {runtimes[architecture]:.3f}")

    histories = {
        name: np.load(root / name / "history.npz") for name in ("mlp", "node")
    }
    if not np.array_equal(
        histories["mlp"]["initial_particles"],
        histories["node"]["initial_particles"],
    ):
        raise RuntimeError("MLP and NODE did not start from identical particles")

    configs = {
        name: json.loads((root / name / "config.json").read_text())
        for name in ("mlp", "node")
    }
    rows = [
        _metrics(name, histories[name], configs[name], runtimes[name])
        for name in ("mlp", "node")
    ]

    figure_path = root / "architecture_comparison.png"
    csv_path = root / "architecture_metrics.csv"
    readme_path = root / "README.md"
    _plot(histories, rows, figure_path, cli.particles)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    readme_path.write_text(_readme(cli, rows), encoding="utf-8")

    particle_difference = float(
        np.mean(
            np.abs(
                histories["mlp"]["final_particles"]
                - histories["node"]["final_particles"]
            )
        )
    )
    print(f"mean absolute paired-particle difference: {particle_difference:.6f}")
    print(f"saved {figure_path}")
    print(f"saved {csv_path}")
    print(f"saved {readme_path}")


def _metrics(
    architecture: str,
    history: np.lib.npyio.NpzFile,
    config: dict[str, object],
    runtime_seconds: float,
) -> dict[str, float | int | str]:
    distances = np.linalg.norm(
        history["final_particles"] - STABLE_EQUILIBRIUM[None, :], axis=1
    )
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
        "particles": len(distances),
        "trainable_tangent_parameters": int(config["trainable_parameters_M"]),
        "selected_basis_size": int(config["actual_basis_size_m"]),
        "mean_retained_rank": float(history["retained_ranks"].mean()),
        "final_retained_rank": int(history["retained_ranks"][-1]),
        "mean_projection_residual": float(history["projection_residuals"].mean()),
        "final_projection_residual": float(history["projection_residuals"][-1]),
        "median_distance_to_stable": float(np.median(distances)),
        "fraction_within_0.15_of_stable": float(np.mean(distances < 0.15)),
        "maximum_initial_saturation_fraction": maximum_saturation,
        "runtime_seconds": runtime_seconds,
    }


def _plot(
    histories: dict[str, np.lib.npyio.NpzFile],
    rows: list[dict[str, float | int | str]],
    output_path: Path,
    particles: int,
) -> None:
    figure = plt.figure(figsize=(12.4, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.55, 1.0))
    colors = {"mlp": "#2a7ab0", "node": "#d95f02"}

    # BLOCK 3 -- The top row makes the physical outcome easy to compare.  The
    # unstable origin and stable Nash equilibrium use different markers.
    for column, name in enumerate(("mlp", "node")):
        axis = figure.add_subplot(grid[0, column])
        final = histories[name]["final_particles"]
        axis.scatter(
            final[:, 0], final[:, 1], s=5, alpha=0.35,
            color=colors[name], edgecolors="none", rasterized=True,
        )
        axis.scatter(
            *STABLE_EQUILIBRIUM, s=52, color="#d62728", marker="o",
            zorder=5, label="stable equilibrium",
        )
        axis.scatter(
            *UNSTABLE_EQUILIBRIUM, s=78, color="#ffbf00", marker="X",
            edgecolors="#222222", linewidths=0.7, zorder=6,
            label="unstable origin",
        )
        axis.set(
            xlim=(-0.4, 1.0), ylim=(-0.4, 1.0),
            xlabel=r"$x_1$", ylabel=r"$x_2$",
            title=f"{name.upper()} basis, N={particles}, t=1",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        if column == 0:
            axis.legend(frameon=False, loc="upper left")

    residual_axis = figure.add_subplot(grid[1, 0])
    for name in ("mlp", "node"):
        history = histories[name]
        residual_axis.plot(
            history["times"][1:], history["projection_residuals"],
            color=colors[name], linewidth=1.7, label=name.upper(),
        )
    residual_axis.set(
        xlabel="physical game time",
        ylabel="relative tangent-projection residual",
        xlim=(0.0, 1.0),
    )
    residual_axis.grid(alpha=0.25)
    residual_axis.legend(frameon=False)

    table_axis = figure.add_subplot(grid[1, 1])
    table_axis.axis("off")
    table_values = [
        [
            str(row["architecture"]),
            str(int(row["trainable_tangent_parameters"])),
            str(int(row["final_retained_rank"])),
            f"{float(row['mean_projection_residual']):.3f}",
            f"{float(row['median_distance_to_stable']):.3f}",
            f"{float(row['runtime_seconds']):.1f}s",
        ]
        for row in rows
    ]
    table = table_axis.table(
        cellText=table_values,
        colLabels=("basis", "M", "rank", "mean residual", "median dist.", "runtime"),
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.65)

    figure.suptitle(
        "Two-player Neural--DTB: MLP versus Neural-ODE basis generator",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _readme(
    cli: argparse.Namespace,
    rows: list[dict[str, float | int | str]],
) -> str:
    lines = [
        "# Two-player MLP versus NODE DTB comparison",
        "",
        "## Simple matched setup",
        "",
        f"- Particles: `{cli.particles}`, uniform on `[0,1]^2` with seed `{cli.seed}`.",
        "- Physical dynamics: 50 Euler steps of size `0.02`, ending at `t=1`.",
        f"- Both basis networks: width `{cli.width}`, depth `{cli.depth}`, `tanh`.",
        f"- Selected tangent coordinates: `m={cli.basis_size}`.",
        f"- NODE: `{cli.node_inner_steps}` fixed RK4 steps over internal depth `[0,{cli.node_integration_time:g}]`.",
        "- The NODE parameters stay fixed. Its terminal-flow parameter sensitivities are the DTB basis.",
        "- No direct SDE baseline is run in this small architecture comparison.",
        "",
        "## Results",
        "",
        "| Basis | Parameters M | Mean residual | Final residual | Median distance to (0.5,0.5) | Within 0.15 | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | {int(row['trainable_tangent_parameters'])} "
            f"| {float(row['mean_projection_residual']):.6f} "
            f"| {float(row['final_projection_residual']):.6f} "
            f"| {float(row['median_distance_to_stable']):.6f} "
            f"| {100 * float(row['fraction_within_0.15_of_stable']):.1f}% "
            f"| {float(row['runtime_seconds']):.3f} s |"
        )
    lines.extend(
        [
            "",
            "The runtime includes the DTB solve, diagnostics, and each model's individual figures.",
            "This is one seed and should be treated as a small mechanism check, not an architecture ranking.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
