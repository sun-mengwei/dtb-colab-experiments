"""Reusable coordinate, state-distribution, and equilibrium plots for game trajectories.

Accepts NumPy arrays or PyTorch tensors shaped (time, particle, coordinate).
No PCA, trajectory updates, or game-specific velocity functions are applied.
Coordinate pairs use ONE-BASED player numbers, e.g. [(1, 2), (3, 4)].

Example (the dimension is inferred, including for 3D, 5D, 6D, or 10D games)::

    plots = save_game_visualizations(
        trajectory, h=H, equilibria=known_equilibria,
        equilibrium_names=known_names, output_dir="figures",
        equilibrium_radius=0.05,
    )

The default coordinate planes are (x1, x2) and, when available, (x3, x4).
Pass coordinate_pairs to select other planes. All coordinates are retained in
the state heatmaps and equilibrium distances. Occupancy measures proximity to
the supplied reference points, not proof of convergence or basin membership.
For constrained games, pass feasible_mask with shape (time, particle).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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


def _references(equilibria, dim, names=None):
    points = np.empty((0, dim)) if equilibria is None else _numpy(equilibria)
    if points.ndim != 2 or points.shape[1] != dim or not np.isfinite(points).all():
        raise ValueError("equilibria must have finite shape (number_of_equilibria, dimension)")
    labels = ([f"equilibrium {i + 1}" for i in range(len(points))]
              if names is None else list(names))
    if len(labels) != len(points):
        raise ValueError("equilibrium_names must match the number of equilibria")
    return points, labels


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


@dataclass
class EquilibriumOccupancy:
    """Mutually exclusive assignments; -1 and the final count column mean unassigned."""

    names: list[str]
    assignments: np.ndarray
    nearest_distance: np.ndarray
    counts: np.ndarray
    fractions: np.ndarray
    pair_fractions: np.ndarray
    families: dict[str, np.ndarray]


def compute_equilibrium_occupancy(trajectory, equilibria, *, names=None,
                                  radius=0.05, feasible_mask=None):
    """Assign to the nearest reference ONLY within a full-dimensional Euclidean radius.

    Ties select the first reference. An optional feasibility mask rejects
    inadmissible particles without changing their coordinates. Fractions use
    all particles as the denominator, including unassigned particles.
    Two-coordinate support references are grouped by their nonzero indices;
    multiple references with the same support are aggregated in the pair map.
    """
    data = _trajectory(trajectory)
    nt, nparticles, dim = data.shape
    points, labels = _references(equilibria, dim, names)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    feasible = (np.ones((nt, nparticles), dtype=bool) if feasible_mask is None
                else _numpy(feasible_mask))
    if feasible.shape != (nt, nparticles) or feasible.dtype != np.bool_:
        raise ValueError("feasible_mask must be Boolean with shape (time, particles)")
    assignments = np.full((nt, nparticles), -1, dtype=int)
    distances = np.full((nt, nparticles), np.inf)
    counts = np.zeros((nt, len(points) + 1), dtype=int)
    # Evaluate one time slice at a time to avoid a (time, particle, equilibrium, dim) array.
    for step, current in enumerate(data):
        if len(points):
            squared = np.sum((current[:, None, :].astype(float)
                              - points[None, :, :]) ** 2, axis=-1)
            closest = squared.argmin(axis=1)
            distances[step] = np.sqrt(squared[np.arange(nparticles), closest])
            accepted = (distances[step] <= radius) & feasible[step]
            assignments[step, accepted] = closest[accepted]
        categories = np.where(assignments[step] < 0, len(points), assignments[step])
        counts[step] = np.bincount(categories, minlength=len(points) + 1)
    fractions = counts / nparticles
    pair_fractions = np.zeros((nt, dim, dim))
    families = {}
    for index, point in enumerate(points):
        support = np.flatnonzero(np.abs(point) > 1e-8)
        if len(support) == 2:
            first, second = support
            pair_fractions[:, first, second] += fractions[:, index]
            family = "Two-coordinate support"
        elif len(support) == 0:
            family = "Origin"
        elif np.allclose(point, point[0], atol=1e-8, rtol=0):
            family = "Equal-coordinate equilibria"
        else:
            family = "Other equilibria"
        families.setdefault(family, np.zeros(nt))
        families[family] += fractions[:, index]
    families["Unassigned"] = fractions[:, -1]
    return EquilibriumOccupancy(labels, assignments, distances, counts,
                                fractions, pair_fractions, families)


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
    known, _ = _references(equilibria, dim)
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
                     title=f"t = {times[step]:.3g}")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=.25)
    if len(known):
        axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Coordinate snapshots · same axes across time · short trails track fixed particles")
    return fig


def _particle_order(final_points, max_particles):
    """A deterministic nearest-neighbor walk groups similar final states for display."""
    if max_particles < 1:
        raise ValueError("max_heatmap_particles must be positive")
    ids = np.unique(np.linspace(0, len(final_points) - 1,
                               min(max_particles, len(final_points)), dtype=int))
    points = final_points[ids].astype(float)
    available = np.ones(len(ids), dtype=bool)
    current = int(np.lexsort(points.T[::-1])[0])
    order = []
    for _ in range(len(ids)):
        order.append(ids[current])
        available[current] = False
        distance = np.sum((points - points[current]) ** 2, axis=1)
        distance[~available] = np.inf
        current = int(distance.argmin())
    return np.asarray(order)


def plot_state_heatmaps(trajectory, times, snapshot_steps, *, max_particles=512):
    """Rows are fixed particles, columns are all players; share ordering and color scale."""
    data = _trajectory(trajectory)
    order = _particle_order(data[-1], max_particles)
    dim = data.shape[-1]
    if data.min() < 0:
        scale = max(float(np.abs(data).max()), 1e-12)
        norm, cmap = Normalize(-scale, scale), "RdBu_r"
    else:
        norm, cmap = Normalize(0, max(float(data.max()), 1e-12)), "viridis"
    fig, axes = plt.subplots(1, len(snapshot_steps), squeeze=False,
                             figsize=(3.0 * len(snapshot_steps), 5), constrained_layout=True)
    for axis, step in zip(axes.flat, snapshot_steps):
        image = axis.imshow(data[step, order], aspect="auto", interpolation="nearest",
                            cmap=cmap, norm=norm)
        axis.set(title=f"t = {times[step]:.3g}", xlabel="Player", ylabel="Particle (fixed order)")
        axis.set_xticks(np.arange(dim), np.arange(1, dim + 1), fontsize=8)
        ticks = np.unique(np.linspace(0, len(order) - 1, min(5, len(order)), dtype=int))
        axis.set_yticks(ticks, order[ticks] + 1)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Player quantity", shrink=.85)
    fig.suptitle(f"All-player states · {len(order)} of {data.shape[1]} particles · ordered once by final-state similarity")
    return fig, order


def plot_pair_occupancy(occupancy, times, snapshot_steps, *, radius):
    """Upper-triangular maps aggregate references with exactly two nonzero coordinates."""
    values = occupancy.pair_fractions
    dim = values.shape[-1]
    maximum = float(values.max()) or 1.0
    fig, axes = plt.subplots(1, len(snapshot_steps), squeeze=False,
                             figsize=(3.1 * len(snapshot_steps), 3.8), constrained_layout=True)
    mask = np.tril(np.ones((dim, dim), dtype=bool))
    cmap = plt.get_cmap("YlGnBu").with_extremes(bad="#eeeeee")
    for axis, step in zip(axes.flat, snapshot_steps):
        image = axis.imshow(np.ma.array(values[step], mask=mask), vmin=0, vmax=maximum,
                            cmap=cmap, interpolation="nearest")
        axis.set(title=f"t = {times[step]:.3g}", xlabel="Second support player", ylabel="First support player")
        axis.set_xticks(np.arange(dim), np.arange(1, dim + 1), fontsize=8)
        axis.set_yticks(np.arange(dim), np.arange(1, dim + 1), fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Fraction of all particles", shrink=.85)
    fig.suptitle(f"Pair-equilibrium neighborhoods · full-state distance ≤ {radius:g} · fixed color scale")
    return fig


def plot_occupancy_over_time(occupancy, times, *, radius):
    """Keep unassigned mass visible; never force every particle into an equilibrium."""
    fig, axis = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    palette = {"Two-coordinate support": "#208c9b", "Origin": "#6f66ad",
               "Equal-coordinate equilibria": "#eeb54b", "Other equilibria": "#d97d58",
               "Unassigned": "#d8dde2"}
    labels = list(occupancy.families)
    if len(times) == 1:
        bottom = 0.0
        for name, fractions in occupancy.families.items():
            axis.bar(times[0], fractions[0], bottom=bottom, color=palette[name], label=name)
            bottom += fractions[0]
    else:
        axis.stackplot(times, *occupancy.families.values(), labels=labels,
                       colors=[palette[name] for name in labels], alpha=.95)
    axis.set(xlabel="Time", ylabel="Fraction of all particles", ylim=(0, 1),
             title=f"Equilibrium neighborhood occupancy · full-state radius {radius:g}")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=min(3, len(labels)), fontsize=8)
    axis.grid(axis="y", alpha=.2)
    return fig


def plot_dtb_diagnostics(states, projections):
    """Use the existing DTB diagnostics without importing or rerunning the solver."""
    state_times = [row["time"] for row in states]
    step_times = [row["time"] for row in projections]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].semilogy(step_times, [max(row["projection_residual"], 1e-15) for row in projections])
    axes[0, 0].set_title("Tangent projection residual")
    axes[0, 1].semilogy(state_times, [max(row["game_drift_rms"], 1e-15) for row in states], label="Game velocity")
    axes[0, 1].semilogy(step_times, [max(row["projected_drift_rms"], 1e-15) for row in projections], label="Tangent velocity")
    axes[0, 1].set_title("RMS velocity"); axes[0, 1].legend()
    for key, label in [("median_known_distance", "Median"), ("p90_known_distance", "90th percentile")]:
        axes[0, 2].semilogy(state_times, [max(row[key], 1e-15) for row in states], label=label)
    axes[0, 2].set_title("Distance to known equilibrium subset"); axes[0, 2].legend()
    axes[1, 0].plot(state_times, [row["negative_coordinate_fraction"] for row in states])
    axes[1, 0].set_title("Fraction of negative coordinates")
    axes[1, 1].plot(step_times, [row["alpha_norm"] for row in projections])
    axes[1, 1].set_title("Tangent coefficient norm")
    for key, label in [("basis_seconds", "Jacobian"), ("solve_seconds", "SVD solve"), ("update_seconds", "Map update")]:
        axes[1, 2].plot(step_times, [row[key] for row in projections], label=label)
    axes[1, 2].set_title("Seconds per step"); axes[1, 2].legend()
    for axis in axes.flat:
        axis.set_xlabel("Time"); axis.grid(alpha=.25)
    return fig


def save_game_visualizations(trajectory, *, h=1.0, times=None, output_dir,
                             coordinate_pairs=None, snapshot_steps=None,
                             equilibria=None, equilibrium_names=None,
                             equilibrium_radius=0.05, feasible_mask=None,
                             max_heatmap_particles=512, state_diagnostics=None,
                             projection_diagnostics=None, show=True):
    """Save the coordinate atlas, heatmaps, occupancy figures/data, and optional DTB diagnostics.

    `times` can supply nonuniform timestamps; otherwise use `h` between states.
    The heatmap includes all particles up to max_heatmap_particles, then a
    deterministic evenly spaced subset. Its particle IDs are saved in NPZ.
    Returned metadata permits checking/reusing assignments without reading plots.
    No trajectory values are changed. All files are written under output_dir.
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
    known, names = _references(equilibria, data.shape[-1], equilibrium_names)
    occupancy = compute_equilibrium_occupancy(data, known, names=names,
                                              radius=equilibrium_radius, feasible_mask=feasible_mask)
    if (state_diagnostics is None) != (projection_diagnostics is None):
        raise ValueError("provide both state_diagnostics and projection_diagnostics, or neither")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}

    def save(figure, name):
        path = folder / f"{name}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        paths[name] = path
        if show:
            plt.show()
        plt.close(figure)

    if data.shape[-1] >= 2:
        save(plot_coordinate_snapshots(data, times, steps, coordinate_pairs=coordinate_pairs,
                                        equilibria=known), "coordinate_snapshots")
    heatmap, order = plot_state_heatmaps(data, times, steps, max_particles=max_heatmap_particles)
    save(heatmap, "state_heatmaps")
    if len(known):
        if np.any(np.count_nonzero(np.abs(known) > 1e-8, axis=1) == 2):
            save(plot_pair_occupancy(occupancy, times, steps, radius=equilibrium_radius),
                 "equilibrium_pair_occupancy")
        save(plot_occupancy_over_time(occupancy, times, radius=equilibrium_radius),
             "equilibrium_occupancy_over_time")
    if state_diagnostics is not None:
        save(plot_dtb_diagnostics(state_diagnostics, projection_diagnostics), "dtb_diagnostics")

    csv_path = folder / "equilibrium_occupancy.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "equilibrium_index", "equilibrium", "particles", "fraction", "radius"])
        for step, timestamp in enumerate(times):
            for index, name in enumerate(names + ["Unassigned"]):
                writer.writerow([timestamp, index if index < len(names) else -1, name,
                                 occupancy.counts[step, index], occupancy.fractions[step, index],
                                 equilibrium_radius])
    paths["equilibrium_occupancy_csv"] = csv_path
    data_path = folder / "visualization_data.npz"
    np.savez_compressed(data_path, times=times, snapshot_steps=steps, heatmap_particle_ids=order,
                        equilibrium_assignments=occupancy.assignments,
                        nearest_equilibrium_distance=occupancy.nearest_distance,
                        equilibrium_counts=occupancy.counts, equilibrium_fractions=occupancy.fractions,
                        equilibrium_names=np.asarray(names + ["Unassigned"], dtype=str),
                        equilibrium_radius=equilibrium_radius, pair_fractions=occupancy.pair_fractions)
    paths["visualization_data"] = data_path
    return {"paths": paths, "occupancy": occupancy, "heatmap_particle_ids": order,
            "snapshot_steps": steps}
