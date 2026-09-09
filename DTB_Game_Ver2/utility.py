"""Utilities for stochastic neural-DTB game experiments.

The notebooks keep the game definition and time-stepping loop visible.  This
module contains the reusable pieces that would otherwise obscure that loop:

* exact initial scores for simple reference laws;
* spatial derivatives of the projected neural tangent velocity;
* physical-coordinate tangent derivatives for an evolving solution map;
* the explicit Lagrangian score update;
* matched DTB/Euler--Maruyama plots and run serialization.

Coordinate pairs in plotting functions are one-based, matching player labels.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, jvp, vmap

from dtb import solution_map, u_at
from run_game_dtb import map_at


def sample_initial_with_score(
    count: int,
    dim: int,
    *,
    law: str,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    gaussian_mean: float = 0.5,
    gaussian_std: float = 0.15,
    smoothing_std: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``x_0`` and return its analytical score and log density.

    Supported laws are:

    ``gaussian``
        Independent ``N(gaussian_mean, gaussian_std**2)`` coordinates.
    ``uniform``
        Exact samples from ``[0,1]^d`` with the interior score set to zero.
        The uniform density has a distributional boundary score, which a
        particlewise finite vector cannot represent.
    ``smoothed_uniform``
        ``U([0,1]^d) + N(0, smoothing_std**2 I)``.  Its score is smooth and
        analytical, so this is the boundary-aware approximation to the paper's
        uniform initial cloud.
    """

    if count < 1 or dim < 1:
        raise ValueError("count and dim must be positive")
    if gaussian_std <= 0 or smoothing_std <= 0:
        raise ValueError("initial-law standard deviations must be positive")

    shape = (count, dim)
    if law == "gaussian":
        noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        particles = gaussian_mean + gaussian_std * noise
        score = -(particles - gaussian_mean) / gaussian_std**2
        log_normalizer = math.log(2.0 * math.pi * gaussian_std**2)
        log_density = -0.5 * (
            ((particles - gaussian_mean) / gaussian_std).square() + log_normalizer
        ).sum(dim=-1)
        return particles, score, log_density

    uniforms = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    if law == "uniform":
        return uniforms, torch.zeros_like(uniforms), torch.zeros(count, device=device, dtype=dtype)

    if law != "smoothed_uniform":
        raise ValueError("law must be 'gaussian', 'uniform', or 'smoothed_uniform'")

    particles = uniforms + smoothing_std * torch.randn(
        shape, device=device, dtype=dtype, generator=generator
    )
    upper = particles / smoothing_std
    lower = (particles - 1.0) / smoothing_std
    density_1d = (torch.special.ndtr(upper) - torch.special.ndtr(lower)).clamp_min(
        torch.finfo(dtype).tiny
    )
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    pdf_upper = inv_sqrt_2pi * torch.exp(-0.5 * upper.square())
    pdf_lower = inv_sqrt_2pi * torch.exp(-0.5 * lower.square())
    score = (pdf_upper - pdf_lower) / (smoothing_std * density_1d)
    log_density = torch.log(density_1d).sum(dim=-1)
    return particles, score, log_density


def tangent_velocity_spatial_terms(
    theta_flat: torch.Tensor,
    selected: torch.Tensor,
    alpha: torch.Tensor,
    particles: torch.Tensor,
    model: nn.Module,
    structure,
    *,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Evaluate ``u``, ``D_x u``, ``div(u)``, and ``grad(div(u))``.

    The tangent velocity is

    ``u(x) = partial_theta_selected f_theta(x) @ alpha``.

    A parameter-direction JVP constructs ``u`` first.  Spatial derivatives
    are then taken only for this final direction, avoiding one second-spatial-
    derivative calculation per selected parameter.

    Returned shapes are ``(N,d)``, ``(N,d,d)``, ``(N,)``, and ``(N,d)``.
    The spatial Jacobian convention is
    ``grad_u[n,a,b] = partial u_a / partial x_b``.
    """

    if theta_flat.ndim != 1 or selected.ndim != 1 or alpha.ndim != 1:
        raise ValueError("theta_flat, selected, and alpha must be one-dimensional")
    if selected.numel() != alpha.numel():
        raise ValueError("selected and alpha must have equal length")
    if particles.ndim != 2 or particles.shape[0] == 0:
        raise ValueError("particles must have nonempty shape (N,d)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    theta = theta_flat.detach().clone()
    direction = torch.zeros_like(theta).index_copy(0, selected, alpha.detach())

    def tangent_one(x_single: torch.Tensor) -> torch.Tensor:
        def model_at(parameters: torch.Tensor) -> torch.Tensor:
            return map_at(
                parameters,
                x_single.unsqueeze(0),
                model,
                structure,
            ).squeeze(0)

        return jvp(model_at, (theta,), (direction,))[1]

    spatial_jacobian_one = jacrev(tangent_one)

    def divergence_one(x_single: torch.Tensor) -> torch.Tensor:
        return torch.trace(spatial_jacobian_one(x_single))

    gradient_divergence_one = jacrev(divergence_one)
    velocity_batch = vmap(tangent_one)
    jacobian_batch = vmap(spatial_jacobian_one)
    divergence_batch = vmap(divergence_one)
    gradient_divergence_batch = vmap(gradient_divergence_one)

    velocities: list[torch.Tensor] = []
    spatial_jacobians: list[torch.Tensor] = []
    divergences: list[torch.Tensor] = []
    gradient_divergences: list[torch.Tensor] = []
    for start in range(0, particles.shape[0], chunk_size):
        chunk = particles[start : start + chunk_size]
        velocities.append(velocity_batch(chunk))
        spatial_jacobians.append(jacobian_batch(chunk))
        divergences.append(divergence_batch(chunk))
        gradient_divergences.append(gradient_divergence_batch(chunk))

    return (
        torch.cat(velocities, dim=0),
        torch.cat(spatial_jacobians, dim=0),
        torch.cat(divergences, dim=0),
        torch.cat(gradient_divergences, dim=0),
    )


def map_tangent_spatial_terms(
    theta_flat: torch.Tensor,
    selected: torch.Tensor,
    alpha: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    structure,
    *,
    theta_initial: torch.Tensor,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Return ``u``, ``D_x u``, ``div_x(u)``, and ``grad_x(div_x(u))``.

    Here ``x = T_theta(z)`` uses :func:`dtb.solution_map` and the fixed labels
    ``z``. With ``v(z) = partial_theta_selected T_theta(z) @ alpha`` and
    ``F = D_z T_theta``, the physical velocity satisfies ``u(T_theta(z))=v(z)``.
    Thus ``D_x u = (D_z v) F^{-1}`` and
    ``grad_x(div_x u) = F^{-T} grad_z(trace((D_z v) F^{-1}))``.

    The map must be locally invertible at the labels. Linear solves apply
    the chain rule without explicitly forming inverses. Returned shapes are
    ``(N,d)``, ``(N,d,d)``, ``(N,)``, and ``(N,d)``.
    """
    if theta_flat.ndim != 1 or selected.ndim != 1 or alpha.ndim != 1:
        raise ValueError("theta_flat, selected, and alpha must be one-dimensional")
    if selected.numel() != alpha.numel():
        raise ValueError("selected and alpha must have equal length")
    if theta_initial.shape != theta_flat.shape:
        raise ValueError("theta_initial and theta_flat must have equal shape")
    if labels.ndim != 2 or labels.shape[0] == 0:
        raise ValueError("labels must have nonempty shape (N,d)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    theta = theta_flat.detach()
    direction = torch.zeros_like(theta).index_copy(0, selected, alpha.detach())

    def map_one(label):
        return solution_map(
            theta, label.unsqueeze(0), model, structure, theta_initial=theta_initial,
        ).squeeze(0)

    def tangent_one(label):
        def network_at(parameters):
            return u_at(parameters, label.unsqueeze(0), model, structure).squeeze(0)

        return jvp(network_at, (theta,), (direction,))[1]

    def divergence_with_aux(label):
        deformation = jacrev(map_one)(label)
        grad_z_u = jacrev(tangent_one)(label)
        grad_x_u = torch.linalg.solve(deformation.T, grad_z_u.T).T
        return torch.trace(grad_x_u), (grad_x_u, deformation)

    def terms_one(label):
        grad_z_div, (grad_x_u, deformation) = jacrev(
            divergence_with_aux, has_aux=True,
        )(label)
        grad_x_div = torch.linalg.solve(deformation.T, grad_z_div)
        return tangent_one(label), grad_x_u, torch.trace(grad_x_u), grad_x_div

    chunks = [
        vmap(terms_one)(labels[start:start + chunk_size])
        for start in range(0, len(labels), chunk_size)
    ]
    return tuple(torch.cat([chunk[i] for chunk in chunks]) for i in range(4))


def euler_score_update(
    score: torch.Tensor,
    spatial_jacobian: torch.Tensor,
    gradient_divergence: torch.Tensor,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Advance the score along a projected-velocity characteristic.

    With ``D_x u[a,b] = partial u_a / partial x_b``, this computes

    ``q_next = q - h * ((D_x u)^T q + grad(div(u)))``.

    The two right-hand-side terms are also returned for diagnostics.
    """

    if score.ndim != 2 or spatial_jacobian.shape != (
        score.shape[0], score.shape[1], score.shape[1]
    ):
        raise ValueError("score and spatial_jacobian shapes are inconsistent")
    if gradient_divergence.shape != score.shape:
        raise ValueError("gradient_divergence must have the score shape")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    transported = torch.einsum("nab,na->nb", spatial_jacobian, score)
    next_score = score - step_size * (transported + gradient_divergence)
    return next_score, transported, gradient_divergence


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_history(value, name: str) -> np.ndarray:
    data = _numpy(value)
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError(f"{name} must have nonempty shape (time, particles, dimension)")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains non-finite values")
    return data


def _plot_limits(values: np.ndarray) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    padding = 0.06 * max(high - low, 1e-3)
    return low - padding, high + padding


def plot_dtb_em_snapshots(
    dtb_history,
    em_history,
    times,
    snapshot_steps: Sequence[int],
    *,
    coordinate_pairs: Sequence[tuple[int, int]],
    equilibria=None,
    stable_mask=None,
    output_dir: str | Path | None = None,
    filename_prefix: str = "dtb_vs_em",
) -> dict[tuple[int, int], plt.Figure]:
    """Plot matched DTB and Euler--Maruyama rows at identical times and axes."""
    return _plot_state_snapshots(
        (("Neural-DTB", dtb_history, "#2389bd"),
         ("Euler--Maruyama", em_history, "#e1812c")),
        times, snapshot_steps, coordinate_pairs=coordinate_pairs,
        equilibria=equilibria, stable_mask=stable_mask, output_dir=output_dir,
        filename_prefix=filename_prefix, title="Matched state snapshots",
    )


def plot_dtb_snapshots(
    dtb_history,
    times,
    snapshot_steps: Sequence[int],
    *,
    coordinate_pairs: Sequence[tuple[int, int]],
    equilibria=None,
    stable_mask=None,
    output_dir: str | Path | None = None,
    filename_prefix: str = "dtb",
) -> dict[tuple[int, int], plt.Figure]:
    """Plot one row of DTB particle snapshots with shared axes across times."""
    return _plot_state_snapshots(
        (("Neural-DTB", dtb_history, "#2389bd"),),
        times, snapshot_steps, coordinate_pairs=coordinate_pairs,
        equilibria=equilibria, stable_mask=stable_mask, output_dir=output_dir,
        filename_prefix=filename_prefix, title="Neural-DTB state snapshots",
    )


def _plot_state_snapshots(
    histories, times, snapshot_steps, *, coordinate_pairs, equilibria,
    stable_mask, output_dir, filename_prefix, title,
) -> dict[tuple[int, int], plt.Figure]:
    histories = [(label, _validate_history(data, label), color)
                 for label, data, color in histories]
    shape = histories[0][1].shape
    if any(data.shape != shape for _, data, _ in histories):
        raise ValueError("Particle histories must have equal shape")
    times = _numpy(times)
    if times.shape != (shape[0],):
        raise ValueError("times must have one entry per stored state")

    steps = np.asarray(snapshot_steps, dtype=int)
    if steps.ndim != 1 or len(steps) == 0 or np.any(steps < 0) or np.any(steps >= len(times)):
        raise ValueError("snapshot_steps contains an invalid state index")
    dim = shape[-1]
    references = np.empty((0, dim)) if equilibria is None else _numpy(equilibria)
    if references.ndim != 2 or references.shape[1] != dim:
        raise ValueError("equilibria must have shape (number, dimension)")
    stable = (
        np.ones(len(references), dtype=bool)
        if stable_mask is None
        else np.asarray(stable_mask, dtype=bool)
    )
    if stable.shape != (len(references),):
        raise ValueError("stable_mask must have one value per equilibrium")

    save_dir = None if output_dir is None else Path(output_dir)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[tuple[int, int], plt.Figure] = {}
    for first, second in coordinate_pairs:
        if first == second or not (1 <= first <= dim and 1 <= second <= dim):
            raise ValueError("coordinate pairs use distinct one-based indices")
        a, b = first - 1, second - 1
        x_values = np.concatenate([data[:, :, a].ravel() for _, data, _ in histories]
                                  + [references[:, a]])
        y_values = np.concatenate([data[:, :, b].ravel() for _, data, _ in histories]
                                  + [references[:, b]])
        xlim, ylim = _plot_limits(x_values), _plot_limits(y_values)

        fig, axes = plt.subplots(
            len(histories),
            len(steps),
            squeeze=False,
            figsize=(3.0 * len(steps), 3.0 * len(histories)),
            constrained_layout=True,
        )
        for column, step in enumerate(steps):
            for row, (label, history, color) in enumerate(histories):
                axis = axes[row, column]
                axis.scatter(
                    history[step, :, a],
                    history[step, :, b],
                    s=5,
                    alpha=0.38,
                    color=color,
                    edgecolors="none",
                    rasterized=True,
                )
                if len(references):
                    if np.any(stable):
                        axis.scatter(
                            references[stable, a], references[stable, b], s=30,
                            color="#d43e38", marker="D", edgecolors="white",
                            linewidths=0.4, label="stable reference", zorder=4,
                        )
                    if np.any(~stable):
                        axis.scatter(
                            references[~stable, a], references[~stable, b], s=42,
                            color="#ffbf00", marker="X", edgecolors="#333333",
                            linewidths=0.5, label="unstable reference", zorder=4,
                        )
                axis.set(
                    xlim=xlim,
                    ylim=ylim,
                    xlabel=rf"$x_{{{first}}}$",
                    ylabel=rf"$x_{{{second}}}$",
                    title=rf"$t={times[step]:.3g}$",
                )
                axis.set_aspect("equal", adjustable="box")
                axis.grid(alpha=0.25)
                if column == 0 and len(histories) > 1:
                    axis.text(
                        -0.22, 0.5, label, rotation=90, transform=axis.transAxes,
                        ha="center", va="center", fontweight="bold",
                    )
        handles, labels = axes[0, -1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8)
        fig.suptitle(
            rf"{title} on the $(x_{{{first}}},x_{{{second}}})$ plane"
        )
        if save_dir is not None:
            fig.savefig(
                save_dir / f"{filename_prefix}_x{first}_x{second}.png",
                dpi=180,
                bbox_inches="tight",
            )
        figures[(first, second)] = fig
    return figures


def plot_tangent_diagnostics(
    solve_times,
    relative_projection_error,
    alpha_norm,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the requested relative projection error and coefficient magnitude."""

    times = _numpy(solve_times)
    errors = _numpy(relative_projection_error)
    magnitudes = _numpy(alpha_norm)
    if times.ndim != 1 or errors.shape != times.shape or magnitudes.shape != times.shape:
        raise ValueError("solve times and diagnostics must be equal-length vectors")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(times, errors, color="#2478b5", linewidth=1.8)
    axes[0].set(
        title="Relative projection error",
        xlabel="Physical time",
        ylabel=r"$\|J_k^S\alpha_k-g_k\|_2/\max(\|g_k\|_2,10^{-30})$",
    )
    axes[1].plot(times, magnitudes, color="#df7f22", linewidth=1.8)
    axes[1].set(
        title=r"Selected tangent coefficient magnitude $\|\alpha_k\|_2$",
        xlabel="Physical time",
        ylabel=r"$\|\alpha_k\|_2$",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Stochastic neural-DTB tangent diagnostics")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def sliced_wasserstein_distance(
    first,
    second,
    *,
    projections: int = 128,
    seed: int = 0,
) -> float:
    """Compute a reproducible sliced 2-Wasserstein point-cloud discrepancy."""

    x, y = _numpy(first), _numpy(second)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("point clouds must have the same (particle, dimension) shape")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    x_proj = np.sort(x @ directions.T, axis=0)
    y_proj = np.sort(y @ directions.T, axis=0)
    return float(np.mean(np.sqrt(np.mean((x_proj - y_proj) ** 2, axis=0))))


def diagnostics_markdown(records: Sequence[Mapping[str, float]]) -> str:
    """Render first, mean, and final stochastic-DTB diagnostics with formulas."""

    if not records:
        raise ValueError("records must be nonempty")
    formulas = [
        (
            "relative_projection_error",
            "Relative projection error",
            r"r_k=\frac{\|J_k^S\alpha_k-g_k\|_2}{\max(\|g_k\|_2,10^{-30})}",
        ),
        ("alpha_norm", "Coefficient magnitude", r"a_k=\|\alpha_k\|_2"),
        ("score_rms", "Score RMS", r"s_k=\sqrt{N^{-1}\sum_i\|q_i^k\|_2^2}"),
        (
            "diffusion_rms",
            "Diffusion correction RMS",
            r"c_k=\frac{1}{2\sqrt N}\|Dq_k\|_2",
        ),
        ("target_rms", "Target velocity RMS", r"v_k^g=N^{-1/2}\|g_k\|_2"),
        ("tangent_rms", "Tangent velocity RMS", r"v_k^u=N^{-1/2}\|u_k\|_2"),
    ]

    def number(value: float) -> str:
        return f"{float(value):.4e}"

    lines = [
        "| Metric | Mathematical definition | First | Mean | Final |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for key, label, formula in formulas:
        values = np.asarray([record[key] for record in records], dtype=float)
        lines.append(
            f"| {label} | ${formula}$ | {number(values[0])} | "
            f"{number(values.mean())} | {number(values[-1])} |"
        )
    return "\n".join(lines)


def create_run_directory(
    root: str | Path,
    *,
    dim: int,
    network: str,
    activation: str,
    width: int,
    depth: int,
    parameter_count: int,
    basis_size: int,
    seed: int,
) -> Path:
    """Create a unique, descriptive folder for one saved experiment."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = (
        f"cournot_{dim}d_stochastic_{network}-{activation}_w{width}_d{depth}_"
        f"p{parameter_count}_basis{basis_size}_seed{seed}_{stamp}"
    )
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_run_data(
    output_dir: str | Path,
    *,
    config: Mapping,
    arrays: Mapping[str, np.ndarray],
    metrics_markdown: str,
) -> None:
    """Save the reproducibility configuration, numerical histories, and metric table."""

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, indent=2, sort_keys=True)
    np.savez_compressed(folder / "history.npz", **{key: _numpy(value) for key, value in arrays.items()})
    (folder / "dtb_metrics.md").write_text(metrics_markdown + "\n", encoding="utf-8")
