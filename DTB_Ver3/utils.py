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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an initial cloud and its analytical density score.

    ``smoothed_uniform`` is ``U([0,1]^d) + N(0, smoothing_std^2 I)``.
    Its score represents the softened boundary. ``uniform`` returns the
    interior score zero; its distributional boundary score is not represented.
    """

    if count < 1 or dim < 1:
        raise ValueError("count and dim must be positive")
    if smoothing_std <= 0 or gaussian_std <= 0:
        raise ValueError("smoothing_std and gaussian_std must be positive")
    if law not in {"gaussian", "uniform", "smoothed_uniform"}:
        raise ValueError("law must be 'gaussian', 'uniform', or 'smoothed_uniform'")
    if law == "gaussian":
        noise = torch.randn((count, dim), dtype=dtype, generator=generator)
        particles = gaussian_mean + gaussian_std * noise
        score = -(particles - gaussian_mean) / gaussian_std**2
        return particles.to(device), score.to(device)

    particles = torch.rand((count, dim), dtype=dtype, generator=generator)
    if law == "uniform":
        return particles.to(device), torch.zeros_like(particles, device=device)

    particles += smoothing_std * torch.randn(
        (count, dim),
        dtype=dtype,
        generator=generator,
    )
    upper = particles / smoothing_std
    lower = (particles - 1.0) / smoothing_std
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
    axes[1].set(xlabel="Time", ylabel=r"$\kappa_2(J)$", title="Selected Jacobian condition")
    for axis in axes:
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
