"""Sampling, serialization, and plotting helpers for DTB experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` and reject unavailable CUDA."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve the configured floating-point precision."""

    choices = {"float32": torch.float32, "float64": torch.float64}
    if name not in choices:
        raise ValueError("dtype must be 'float32' or 'float64'")
    return choices[name]


def warmup_cuda(device: torch.device, dtype: torch.dtype) -> None:
    """Initialize the CUDA context and cuBLAS handle on the main thread."""

    if device.type == "cuda":
        torch.cuda.init()
        value = torch.ones((2, 2), device=device, dtype=dtype)
        value @ value
        torch.cuda.synchronize()


def sample_initial_particles(
    count: int,
    dim: int,
    *,
    law: str,
    smoothing_std: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    uniform_low: float = 0.0,
    uniform_high: float = 1.0,
) -> torch.Tensor:
    """Draw reproducible CPU samples, then transfer the complete cloud."""

    particles, _ = sample_initial_with_score(
        count,
        dim,
        law=law,
        smoothing_std=smoothing_std,
        gaussian_mean=0.5,
        gaussian_std=0.15,
        dtype=dtype,
        device=device,
        generator=generator,
        uniform_low=uniform_low,
        uniform_high=uniform_high,
    )
    return particles


def sample_initial_with_score(
    count: int,
    dim: int,
    *,
    law: str,
    smoothing_std: float,
    gaussian_mean: float,
    gaussian_std: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    uniform_low: float = 0.0,
    uniform_high: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an initial cloud and its analytical density score.

    ``smoothed_uniform`` is ``U([low,high]^d) + N(0, smoothing_std^2 I)``.
    Its score represents the softened boundary. ``uniform`` returns the
    interior score zero; its distributional boundary score is not represented.
    """

    if count < 1 or dim < 1:
        raise ValueError("count and dim must be positive")
    if smoothing_std <= 0 or gaussian_std <= 0:
        raise ValueError("smoothing_std and gaussian_std must be positive")
    if not math.isfinite(uniform_low) or not math.isfinite(uniform_high):
        raise ValueError("uniform interval bounds must be finite")
    if uniform_high <= uniform_low:
        raise ValueError("uniform_high must be greater than uniform_low")
    if law not in {"gaussian", "uniform", "smoothed_uniform"}:
        raise ValueError("law must be 'gaussian', 'uniform', or 'smoothed_uniform'")
    if law == "gaussian":
        noise = torch.randn((count, dim), dtype=dtype, generator=generator)
        particles = gaussian_mean + gaussian_std * noise
        score = -(particles - gaussian_mean) / gaussian_std**2
        return particles.to(device), score.to(device)

    particles = uniform_low + (uniform_high - uniform_low) * torch.rand(
        (count, dim), dtype=dtype, generator=generator
    )
    if law == "uniform":
        return particles.to(device), torch.zeros_like(particles, device=device)

    particles += smoothing_std * torch.randn(
        (count, dim),
        dtype=dtype,
        generator=generator,
    )
    upper = (particles - uniform_low) / smoothing_std
    lower = (particles - uniform_high) / smoothing_std
    density = (torch.special.ndtr(upper) - torch.special.ndtr(lower)).clamp_min(
        torch.finfo(dtype).tiny
    )
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    pdf_upper = inv_sqrt_2pi * torch.exp(-0.5 * upper.square())
    pdf_lower = inv_sqrt_2pi * torch.exp(-0.5 * lower.square())
    score = (pdf_upper - pdf_lower) / (smoothing_std * density)
    return particles.to(device), score.to(device)


def make_time_grid(final_time: float, step_size: float) -> np.ndarray:
    """Create a uniform grid and require final_time to be a step multiple."""

    if final_time <= 0 or step_size <= 0:
        raise ValueError("final_time and step_size must be positive")
    step_count = int(round(final_time / step_size))
    if not np.isclose(step_count * step_size, final_time, rtol=0.0, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of step_size")
    return np.linspace(0.0, final_time, step_count + 1)


def snapshot_indices(
    times: np.ndarray,
    requested: Sequence[float],
) -> dict[int, float]:
    """Map grid indices to requested snapshot times."""

    result: dict[int, float] = {}
    for value in sorted(set(float(time) for time in requested)):
        index = int(np.argmin(np.abs(times - value)))
        if not np.isclose(times[index], value, rtol=0.0, atol=1e-10):
            raise ValueError(f"snapshot time {value:g} is not on the time grid")
        result[index] = float(times[index])
    if 0 not in result:
        result[0] = float(times[0])
    if len(times) - 1 not in result:
        result[len(times) - 1] = float(times[-1])
    return dict(sorted(result.items()))


def to_numpy(value) -> np.ndarray:
    """Detach a tensor if needed and return a NumPy array."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def write_json(path: str | Path, data: Mapping[str, object]) -> None:
    """Write indented JSON, converting paths and NumPy scalars."""

    def default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot encode {type(value).__name__}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(data), indent=2, sort_keys=True, default=default) + "\n")


def plot_cloud_snapshots(
    snapshots: Mapping[float, np.ndarray],
    *,
    coordinate_pairs: Sequence[tuple[int, int]] | None = None,
    title: str,
    color: str = "#176b87",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one-based coordinate pairs across saved particle snapshots."""

    if not snapshots:
        raise ValueError("snapshots must be nonempty")
    ordered = sorted((float(time), to_numpy(cloud)) for time, cloud in snapshots.items())
    dim = ordered[0][1].shape[1]
    if any(cloud.ndim != 2 or cloud.shape[1] != dim for _, cloud in ordered):
        raise ValueError("all snapshots must have shape (N, d) with the same d")
    pairs = list(coordinate_pairs or [(index, index + 1) for index in range(1, dim)])
    if not pairs:
        raise ValueError("at least one coordinate pair is required")
    for first, second in pairs:
        if first == second or not (1 <= first <= dim and 1 <= second <= dim):
            raise ValueError("coordinate pairs use distinct one-based indices")

    values = np.concatenate([cloud.ravel() for _, cloud in ordered])
    low, high = float(values.min()), float(values.max())
    padding = 0.04 * max(high - low, 1e-6)
    fig, axes = plt.subplots(
        len(pairs),
        len(ordered),
        figsize=(3.4 * len(ordered), 2.7 * len(pairs)),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row, (first, second) in enumerate(pairs):
        for column, (time, cloud) in enumerate(ordered):
            axis = axes[row, column]
            axis.scatter(
                cloud[:, first - 1],
                cloud[:, second - 1],
                s=3,
                alpha=0.35,
                color=color,
                linewidths=0,
                rasterized=True,
            )
            axis.set(
                xlabel=rf"$x_{first}$",
                ylabel=rf"$x_{second}$",
                xlim=(low - padding, high + padding),
                ylim=(low - padding, high + padding),
            )
            axis.set_aspect("equal", adjustable="box")
            if row == 0:
                axis.set_title(f"t = {time:g}")
    fig.suptitle(title)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_diagnostics(
    times,
    rms_residual,
    condition_number,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot tangent projection error and selected Jacobian conditioning."""

    times = to_numpy(times)
    residual = to_numpy(rms_residual)
    condition = to_numpy(condition_number)
    if residual.shape != times.shape or condition.shape != times.shape:
        raise ValueError("diagnostic arrays must have the time shape")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    axes[0].plot(times, residual, color="#176b87", linewidth=1.4)
    axes[0].set(xlabel="Time", ylabel="RMS projection residual", title="DTB projection error")
    axes[1].plot(times, condition, color="#7c3aed", linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Time",
        ylabel=r"$\kappa_2(J_{\mathrm{retained}})$",
        title="Retained Jacobian condition",
    )
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_configuration_diagnostics(
    state_times,
    trajectory_rms_error,
    projection_times,
    relative_projection_error,
    alpha_norm,
    condition_number,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    r"""Plot the main diagnostics for one experiment configuration.

    The panels show paired trajectory RMS error, relative tangent-projection
    error ``||J alpha - g||_2 / ||g||_2``, coefficient norm ``||alpha||_2``,
    and the selected Jacobian condition number.
    """

    state_times = to_numpy(state_times).astype(float, copy=False)
    trajectory = to_numpy(trajectory_rms_error).astype(float, copy=False)
    projection_times = to_numpy(projection_times).astype(float, copy=False)
    relative = to_numpy(relative_projection_error).astype(float, copy=False)
    coefficients = to_numpy(alpha_norm).astype(float, copy=False)
    condition = to_numpy(condition_number).astype(float, copy=False)
    if state_times.ndim != 1 or trajectory.shape != state_times.shape:
        raise ValueError("trajectory RMS error must have the state-time shape")
    if projection_times.ndim != 1 or projection_times.size < 1:
        raise ValueError("projection_times must be a nonempty vector")
    if any(
        values.shape != projection_times.shape
        for values in (relative, coefficients, condition)
    ):
        raise ValueError("projection diagnostics must have the projection-time shape")
    if not all(
        np.isfinite(values).all()
        for values in (
            state_times,
            trajectory,
            projection_times,
            relative,
            coefficients,
            condition,
        )
    ):
        raise ValueError("configuration diagnostics contain a nonfinite value")
    if (trajectory < 0).any() or (relative < 0).any() or (coefficients < 0).any():
        raise ValueError("error and coefficient-norm diagnostics cannot be negative")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), layout="constrained")
    axes[0, 0].plot(state_times, trajectory, color="#176b87", linewidth=1.4)
    axes[0, 0].set(
        xlabel="Time",
        ylabel=r"$E_{\mathrm{traj}}$",
        title="Trajectory RMS versus reference",
    )
    axes[0, 1].plot(projection_times, relative, color="#c2410c", linewidth=1.4)
    axes[0, 1].set(
        xlabel="Time",
        ylabel=r"$\|J\alpha-g\|_2/\|g\|_2$",
        title="Relative projection error",
    )
    axes[1, 0].plot(projection_times, coefficients, color="#0f766e", linewidth=1.4)
    axes[1, 0].set(
        xlabel="Time",
        ylabel=r"$\|\alpha\|_2$",
        title="Tangent coefficient norm",
    )
    axes[1, 1].plot(projection_times, condition, color="#7c3aed", linewidth=1.2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        xlabel="Time",
        ylabel=r"$\kappa_2(J_{\mathrm{retained}})$",
        title="Retained Jacobian condition",
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.2, which="both")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_stochastic_diagnostics(
    times,
    score_rms,
    diffusion_correction_rms,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot score magnitude and probability-flow diffusion correction."""

    times = to_numpy(times)
    score = to_numpy(score_rms)
    correction = to_numpy(diffusion_correction_rms)
    if score.shape != times.shape or correction.shape != times.shape:
        raise ValueError("stochastic diagnostic arrays must have the time shape")
    if not np.isfinite(score).all() or not np.isfinite(correction).all():
        raise ValueError("stochastic diagnostics contain a nonfinite value")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    axes[0].plot(times, score, color="#0f766e", linewidth=1.4)
    axes[0].set(xlabel="Time", ylabel="Score RMS", title="Transported density score")
    axes[1].plot(times, correction, color="#c2410c", linewidth=1.4)
    axes[1].set(
        xlabel="Time",
        ylabel="Correction RMS",
        title="Probability-flow diffusion correction",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def sliced_wasserstein_distance(
    first,
    second,
    *,
    projections: int = 128,
    seed: int = 0,
) -> float:
    """Return a reproducible sliced 2-Wasserstein point-cloud distance."""

    first = to_numpy(first)
    second = to_numpy(second)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("point clouds must have the same (N, d) shape")
    if projections < 1:
        raise ValueError("projections must be positive")
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(projections, first.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    first_projection = np.sort(first @ directions.T, axis=0)
    second_projection = np.sort(second @ directions.T, axis=0)
    return float(
        np.mean(np.sqrt(np.mean((first_projection - second_projection) ** 2, axis=0)))
    )


def plot_step_size_sweep(
    records,
    *,
    metric_label: str,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the standard seven-column step-size sweep table."""

    table = to_numpy(records)
    if table.ndim != 2 or table.shape[1] != 7 or table.shape[0] < 1:
        raise ValueError("records must have shape (step_sizes, 7)")
    order = np.argsort(table[:, 0])
    table = table[order]
    step_sizes = table[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    axes[0].loglog(
        step_sizes,
        table[:, 2],
        "o-",
        label="DTB vs finest reference",
    )
    positive_reference = table[:, 3] > 0
    if positive_reference.any():
        axes[0].loglog(
            step_sizes[positive_reference],
            table[positive_reference, 3],
            "s--",
            label="reference vs finest reference",
        )
    axes[0].set(
        xlabel="Step size",
        ylabel=metric_label,
        title="Final-cloud step-size comparison",
    )
    axes[0].legend()
    axes[1].loglog(step_sizes, table[:, 5], "o-", color="#176b87")
    axes[1].set(
        xlabel="Step size",
        ylabel="Final RMS projection residual",
        title="Tangent projection error",
    )
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_final_time_rms_sweep(
    step_sizes,
    rms_errors,
    *,
    final_time: float,
    output_path: str | Path | None = None,
) -> plt.Figure:
    r"""Plot final-time paired-particle RMS error against the step size.

    The plotted quantity is
    ``sqrt(mean_i(||X_DTB_i(T) - X_reference_i(T)||_2^2))``.
    """

    steps = to_numpy(step_sizes).astype(float, copy=False)
    errors = to_numpy(rms_errors).astype(float, copy=False)
    if steps.ndim != 1 or errors.shape != steps.shape or steps.size < 1:
        raise ValueError("step_sizes and rms_errors must be nonempty vectors")
    if not np.isfinite(steps).all() or not np.isfinite(errors).all():
        raise ValueError("step sizes and RMS errors must be finite")
    if (steps <= 0).any() or (errors < 0).any() or final_time <= 0:
        raise ValueError("step sizes and final_time must be positive; RMS cannot be negative")

    order = np.argsort(steps)
    fig, axis = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    if (errors > 0).all():
        axis.loglog(steps[order], errors[order], "o-", color="#176b87", linewidth=1.5)
    else:
        axis.semilogx(steps[order], errors[order], "o-", color="#176b87", linewidth=1.5)
    axis.set(
        xlabel="Step size $h$",
        ylabel="Final-time RMS error",
        title=fr"DTB versus reference at $T={final_time:g}$",
    )
    axis.grid(alpha=0.2, which="both")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_relative_projection_sweep(
    step_sizes,
    relative_errors,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    r"""Plot last-step relative projection error against the step size.

    Each value is ``||J_k alpha_k - g_k||_2 / ||g_k||_2`` at that run's last
    projection time ``t_k = T - h``.
    """

    steps = to_numpy(step_sizes).astype(float, copy=False)
    errors = to_numpy(relative_errors).astype(float, copy=False)
    if steps.ndim != 1 or errors.shape != steps.shape or steps.size < 1:
        raise ValueError("step_sizes and relative_errors must be nonempty vectors")
    if not np.isfinite(steps).all() or not np.isfinite(errors).all():
        raise ValueError("step sizes and relative projection errors must be finite")
    if (steps <= 0).any() or (errors < 0).any():
        raise ValueError("step sizes must be positive and relative errors nonnegative")

    order = np.argsort(steps)
    fig, axis = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    if (errors > 0).all():
        axis.loglog(steps[order], errors[order], "o-", color="#c2410c", linewidth=1.5)
    else:
        axis.semilogx(steps[order], errors[order], "o-", color="#c2410c", linewidth=1.5)
    axis.set(
        xlabel="Step size $h$",
        ylabel="Last-step relative projection error",
        title=r"$\|J_k\alpha_k-g_k\|_2/\|g_k\|_2$ versus step size",
    )
    axis.grid(alpha=0.2, which="both")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
