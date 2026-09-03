"""Coordinate snapshots, tangent-projection plots, and mathematical metric tables.

Accepts NumPy arrays or PyTorch tensors shaped (time, particle, coordinate).
Coordinate pairs use ONE-BASED player numbers, e.g. [(1, 2), (3, 4)].

Example (dimension is inferred for 3D, 5D, 6D, or 10D games)::

    plots = save_game_visualizations(
        trajectory, h=H, equilibria=known_equilibria, output_dir="figures",
        state_diagnostics=result["states"],
        projection_diagnostics=result["projections"],
    )

The default coordinate planes are (x1, x2) and, when available, (x3, x4).
Diagnostics are plotted for the projection error and coefficient norm; all metrics
are rendered as Markdown math tables and saved to dtb_metrics.md.
Trajectory values and the recorded diagnostics are read without modification.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _trajectory(value):
    data = _numpy(value)
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("trajectory must have nonempty shape (time, particles, dimension)")
    if not np.isfinite(data).all():
        raise ValueError("trajectory contains non-finite values")
    return data


def _references(equilibria, dim):
    points = np.empty((0, dim)) if equilibria is None else _numpy(equilibria)
    if points.ndim != 2 or points.shape[1] != dim or not np.isfinite(points).all():
        raise ValueError("equilibria must have finite shape (number_of_equilibria, dimension)")
    return points


def _snapshot_indices(count, snapshot_steps):
    if snapshot_steps is None:
        return np.unique(np.linspace(0, count - 1, min(5, count), dtype=int))
    steps = np.asarray(snapshot_steps)
    if (steps.ndim != 1 or steps.size == 0 or not np.issubdtype(steps.dtype, np.integer)
            or np.any(steps < 0) or np.any(steps >= count)):
        raise ValueError("snapshot_steps must be nonempty, valid integer time indices")
    return np.unique(steps)


def _limits(values):
    low, high = float(np.min(values)), float(np.max(values))
    padding = 0.06 * max(high - low, 1e-3)
    return low - padding, high + padding


def plot_coordinate_snapshots(trajectory, times, snapshot_steps, *,
                              coordinate_pairs=None, equilibria=None,
                              trail_particles=6, trail_steps=10):
    """Rows are coordinate planes; columns are times. Axis limits are fixed per row."""
    data = _trajectory(trajectory)
    dim = data.shape[-1]
    if coordinate_pairs is None:
        coordinate_pairs = [(i, i + 1) for i in (1, 3) if i + 1 <= dim]
    pairs = [tuple(pair) for pair in coordinate_pairs]
    if not pairs or any(len(pair) != 2 or pair[0] == pair[1]
                        or any(not isinstance(i, (int, np.integer)) or not 1 <= i <= dim
                               for i in pair) for pair in pairs):
        raise ValueError("coordinate_pairs must contain distinct, valid ONE-BASED player indices")
    known = _references(equilibria, dim)
    fig, axes = plt.subplots(len(pairs), len(snapshot_steps), squeeze=False,
                             figsize=(3.0 * len(snapshot_steps), 3.1 * len(pairs)),
                             constrained_layout=True)
    tracked = np.unique(np.linspace(0, data.shape[1] - 1,
                                    min(trail_particles, data.shape[1]), dtype=int))
    for row, (first, second) in enumerate(pairs):
        a, b = first - 1, second - 1
        xlim = _limits(np.concatenate((data[:, :, a].ravel(), known[:, a])))
        ylim = _limits(np.concatenate((data[:, :, b].ravel(), known[:, b])))
        projected_known = np.unique(known[:, [a, b]], axis=0)
        for column, step in enumerate(snapshot_steps):
            axis = axes[row, column]
            axis.scatter(data[step, :, a], data[step, :, b], s=5, alpha=.38,
                         color="#2389bd", edgecolors="none", rasterized=True)
            for particle in tracked:
                segment = data[max(0, step - trail_steps):step + 1, particle]
                axis.plot(segment[:, a], segment[:, b], color="#17556c", alpha=.6, linewidth=.8)
            if len(known):
                axis.scatter(projected_known[:, 0], projected_known[:, 1], s=24,
                             marker="D", color="#d43e38", edgecolors="white", linewidths=.4,
                             label="Known equilibria", zorder=4)
            axis.set(xlim=xlim, ylim=ylim, xlabel=f"$x_{{{first}}}$", ylabel=f"$x_{{{second}}}$",
                     title=f"$t = {times[step]:.3g}$")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=.25)
    if len(known):
        axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Coordinate snapshots · same axes across time · short trails track fixed particles")
    return fig


def _math_number(value):
    value = float(value)
    if np.isnan(value):
        return r"\mathrm{NaN}"
    if np.isinf(value):
        return r"\infty" if value > 0 else r"-\infty"
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def format_dtb_metrics(states, projections):
    r"""Format recorded diagnostics as Markdown with LaTeX formulas and values.

    State records cover k=0,...,K; projection records cover k=0,...,K-1.
    Each table uses its own recorded timestamps, including the last solve at
    T-h. Values are taken directly from the records, without recomputation.
    """
    if len(states) == 0 or len(projections) == 0:
        raise ValueError("state and projection diagnostics must both be nonempty")

    def table(rows, records):
        first, last = records[0], records[-1]
        lines = [
            f"| Metric | Mathematical definition | First: $t={first['time']:.4g}$ "
            f"| Last: $t={last['time']:.4g}$ |",
            "| :--- | :--- | ---: | ---: |",
        ]
        for key, label, formula in rows:
            lines.append(f"| {label} | ${formula}$ | ${_math_number(first[key])}$ "
                         f"| ${_math_number(last[key])}$ |")
        return "\n".join(lines)

    state_rows = [
        ("game_drift_rms", "Game velocity (RMS)",
         r"v_k^b=\frac{\lVert\mathbf{g}_k\rVert_2}{\sqrt{N}}"),
        ("median_known_distance", "Median equilibrium distance",
         r"d_k^{50}=\operatorname{median}_{j}\,d_{k,j}"),
        ("p90_known_distance", "90th percentile equilibrium distance",
         r"d_k^{90}=Q_{0.9}(\{d_{k,j}\}_{j=1}^{N})"),
        ("minimum_coordinate", "Minimum coordinate",
         r"x_k^{\min}=\min_{j,i}x_{k,j,i}"),
        ("negative_coordinate_fraction", "Fraction of negative coordinates",
         r"f_k^-=\frac{1}{Nd}\sum_{j=1}^{N}\sum_{i=1}^{d}\mathbf{1}_{\{x_{k,j,i}<0\}}"),
    ]
    projection_rows = [
        ("projection_residual", "Relative projection residual",
         r"r_k=\frac{\lVert\mathbf{u}_k-\mathbf{g}_k\rVert_2}"
         r"{\max(\lVert\mathbf{g}_k\rVert_2,10^{-30})}"),
        ("projected_drift_rms", "Tangent velocity (RMS)",
         r"v_k^u=\frac{\lVert\mathbf{u}_k\rVert_2}{\sqrt{N}}"),
        ("alpha_norm", "Tangent coefficient norm", r"a_k=\lVert\alpha_k\rVert_2"),
        ("basis_seconds", "Jacobian time (seconds)",
         r"\Delta\tau_k^J=\tau_{k,\mathrm{solve}}-\tau_{k,\mathrm{basis}}"),
        ("solve_seconds", "SVD solve time (seconds)",
         r"\Delta\tau_k^{\mathrm{solve}}=\tau_{k,\mathrm{update}}-\tau_{k,\mathrm{solve}}"),
        ("update_seconds", "Map update time (seconds)",
         r"\Delta\tau_k^X=\tau_{k,\mathrm{end}}-\tau_{k,\mathrm{update}}"),
    ]
    notation = r"""For $N$ particles in $d$ dimensions and an $m$-vector selected tangent basis,
let $x_{k,j}=X_k(z_j)$ and $J_{k,j}^{S}=\partial_{\theta_S}f_{\theta_0}(x_{k,j})$.
Stack the particle velocities as
$\mathbf{g}_k=\operatorname{col}_{j=1}^{N}b(x_{k,j})$ and
$\mathbf{u}_k=\operatorname{col}_{j=1}^{N}(J_{k,j}^{S}\alpha_k)$, where
$\alpha_k\in\mathbb{R}^{m}$.
Distances use every coordinate and the supplied reference set $\mathcal{E}$:

$$d_{k,j}=\min_{e\in\mathcal{E}}\lVert x_{k,j}-e\rVert_2.$$

**State metrics**
"""
    notes = r"""The median follows `torch.median` (the lower middle value for even $N$);
$Q_{0.9}$ uses linear interpolation. Projection values describe the solve at
its input time $t_k$, so the last solve is at $T-h$, while the last state is at $T$.
The wall-clock timestamps $\tau$ mark the start of each named operation and the
end of the map update. Full time histories remain in the diagnostic CSV files.
"""
    return (notation + "\n" + table(state_rows, states)
            + "\n\n**Projection metrics**\n\n" + table(projection_rows, projections)
            + "\n\n" + notes)


def plot_projection_metrics(projections):
    r"""Plot the two diagnostics that directly describe the tangent projection."""
    if len(projections) == 0:
        raise ValueError("projection diagnostics must be nonempty")
    times = np.asarray([row["time"] for row in projections], dtype=float)
    residuals = np.asarray([row["projection_residual"] for row in projections], dtype=float)
    alpha_norms = np.asarray([row["alpha_norm"] for row in projections], dtype=float)
    if (not np.isfinite(times).all() or not np.isfinite(residuals).all()
            or not np.isfinite(alpha_norms).all() or np.any(np.diff(times) <= 0)
            or np.any(residuals < 0) or np.any(alpha_norms < 0)):
        raise ValueError("projection times and metrics must be finite, ordered, and nonnegative")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    markevery = max(1, len(times) // 10)
    axes[0].plot(times, residuals, color="#2374ab", linewidth=1.8,
                 marker="o", markersize=3, markevery=markevery)
    axes[0].set(title=("Relative projection error\n"
                       r"$r_k=\Vert\mathbf{u}_k-\mathbf{g}_k\Vert_2/"
                       r"\max(\Vert\mathbf{g}_k\Vert_2,10^{-30})$"),
                xlabel=r"Solve time $t_k$", ylabel=r"$r_k$")
    axes[1].plot(times, alpha_norms, color="#d87520", linewidth=1.8,
                 marker="o", markersize=3, markevery=markevery)
    axes[1].set(title="Selected tangent coefficient norm\n"
                     r"$a_k=\Vert\alpha_k\Vert_2$",
                xlabel=r"Solve time $t_k$", ylabel=r"$\Vert\alpha_k\Vert_2$")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle("Tangent-projection diagnostics")
    return fig


def save_game_visualizations(trajectory, *, h=1.0, times=None, output_dir,
                             coordinate_pairs=None, snapshot_steps=None,
                             equilibria=None, state_diagnostics=None,
                             projection_diagnostics=None, show=True):
    """Save coordinate snapshots, projection diagnostics, and mathematical metric tables.

    `times` supplies nonuniform timestamps; otherwise use `h` between states.
    `show=False` saves files without displaying a figure or importing IPython.
    The returned paths list the artifacts generated by this call.
    """
    data = _trajectory(trajectory)
    if times is None:
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be positive and finite")
        times = np.arange(len(data)) * h
    times = _numpy(times)
    if times.shape != (len(data),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must match the trajectory and be strictly increasing")
    steps = _snapshot_indices(len(data), snapshot_steps)
    known = _references(equilibria, data.shape[-1])
    if (state_diagnostics is None) != (projection_diagnostics is None):
        raise ValueError("provide both state_diagnostics and projection_diagnostics, or neither")
    metrics = (None if state_diagnostics is None
               else format_dtb_metrics(state_diagnostics, projection_diagnostics))
    figure = plot_coordinate_snapshots(data, times, steps, coordinate_pairs=coordinate_pairs,
                                       equilibria=known)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"coordinate_snapshots": folder / "coordinate_snapshots.png"}
    try:
        figure.savefig(paths["coordinate_snapshots"], dpi=160, bbox_inches="tight")
        if show:
            plt.show()
    finally:
        plt.close(figure)

    if metrics is not None:
        figure = plot_projection_metrics(projection_diagnostics)
        paths["projection_metrics"] = folder / "projection_metrics.png"
        try:
            figure.savefig(paths["projection_metrics"], dpi=160, bbox_inches="tight")
            if show:
                plt.show()
        finally:
            plt.close(figure)
        paths["dtb_metrics"] = folder / "dtb_metrics.md"
        paths["dtb_metrics"].write_text(metrics, encoding="utf-8")
        if show:
            from IPython.display import Markdown, display
            display(Markdown(metrics))

    return {"paths": paths, "snapshot_steps": steps, "metrics_markdown": metrics}
