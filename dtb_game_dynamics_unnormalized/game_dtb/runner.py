"""Experiment runner, thesis presets, snapshots, and artifact generation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from .algorithm import DTBConfig, NeuralDTBGameDynamics
from .games import cournot_duopoly_drift, linear_quadratic_drift
from .models import TangentMLP
from .state import gaussian_particle_state, uniform_box_particle_state


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def diffusion_entry_from_noise(noise_std: float) -> float:
    """Convert SDE noise amplitude sigma into Fokker--Planck D=sigma^2.

    The target figure reports ``sigma_1=sigma_2=0.1``.  For
    ``dX=b(X)dt+sigma dW``, the algorithm's matrix is ``D=sigma sigma^T``;
    hence each diagonal entry is ``0.1^2=0.01``.
    """

    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")
    return noise_std * noise_std


def snapshot_schedule(text: str, steps: int, step_size: float) -> dict[int, float]:
    """Map requested times to Euler indices, requiring grid alignment."""

    if steps < 1 or step_size <= 0:
        raise ValueError("steps and step_size must be positive")
    final_time = steps * step_size
    if text == "auto":
        requested = np.linspace(0.0, final_time, 6).tolist()
    else:
        requested = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not requested:
        raise ValueError("at least one snapshot time is required")

    schedule: dict[int, float] = {}
    for time_value in requested:
        index = int(round(time_value / step_size))
        if index < 0 or index > steps:
            raise ValueError(f"snapshot time {time_value} lies outside [0,{final_time}]")
        aligned = index * step_size
        if not math.isclose(time_value, aligned, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"snapshot time {time_value} is not aligned with h={step_size}"
            )
        schedule[index] = aligned
    return dict(sorted(schedule.items()))


def run_experiment(args: Any) -> Path:
    """Run Neural--DTB and save figure-style snapshots and diagnostics."""

    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = snapshot_schedule(args.snapshot_times, args.steps, args.step_size)

    # BLOCK 1 -- Initialize x_i^0, log rho_0(x_i^0), and q_i^0.
    state_generator = torch.Generator(
        device=device.type if device.type == "cuda" else "cpu"
    )
    state_generator.manual_seed(args.seed)
    if args.initial_distribution == "uniform":
        state = uniform_box_particle_state(
            args.particles,
            args.dim,
            args.uniform_low,
            args.uniform_high,
            device=device,
            dtype=dtype,
            generator=state_generator,
        )
    elif args.initial_distribution == "gaussian":
        initial_std = (
            math.sqrt(args.initial_variance)
            if args.initial_variance is not None
            else args.initial_std
        )
        state = gaussian_particle_state(
            args.particles,
            args.dim,
            args.initial_mean,
            initial_std,
            device=device,
            dtype=dtype,
            generator=state_generator,
        )
    else:
        raise ValueError(f"unknown initial distribution: {args.initial_distribution}")

    initial_particles = state.particles.detach().cpu().numpy().copy()
    drift = _make_drift(args)

    # f_theta is not fitted to the solution; its selected parameter tangents
    # form the Eulerian dictionary used in Blocks 2--8 of the algorithm.
    model = TangentMLP(
        dim=args.dim,
        width=args.width,
        depth=args.depth,
        activation=args.activation,
        dtype=dtype,
    ).to(device)

    diffusion_entry = (
        args.diffusion_entry
        if args.diffusion_entry is not None
        else diffusion_entry_from_noise(args.noise_std)
    )
    diffusion = diffusion_entry * torch.eye(args.dim, device=device, dtype=dtype)
    config = DTBConfig(
        step_size=args.step_size,
        basis_size=args.basis_size,
        svd_rtol=args.svd_rtol,
        jacobian_chunk_size=args.jacobian_chunk,
        derivative_chunk_size=args.derivative_chunk,
        seed=args.seed,
    )
    method = NeuralDTBGameDynamics(model, drift, diffusion, config)

    times = [0.0]
    means = [state.particles.mean(dim=0).detach().cpu().numpy()]
    stds = [state.particles.std(dim=0).detach().cpu().numpy()]
    mean_log_density = [float(state.log_density.mean())]
    residuals: list[float] = []
    ranks: list[int] = []
    alpha_norms: list[float] = []
    mean_divergences: list[float] = []
    snapshot_particles: dict[int, np.ndarray] = {}
    if 0 in schedule:
        snapshot_particles[0] = initial_particles.copy()

    for step in range(args.steps):
        result = method.step(state, step)
        state = result.state
        completed = step + 1
        times.append(completed * args.step_size)
        means.append(state.particles.mean(dim=0).detach().cpu().numpy())
        stds.append(state.particles.std(dim=0).detach().cpu().numpy())
        mean_log_density.append(float(state.log_density.mean()))
        residuals.append(result.diagnostics.relative_residual)
        ranks.append(result.diagnostics.retained_rank)
        alpha_norms.append(result.alpha_norm)
        mean_divergences.append(result.mean_divergence)
        if completed in schedule:
            snapshot_particles[completed] = state.particles.detach().cpu().numpy().copy()
        if args.print_every > 0 and (
            completed % args.print_every == 0 or step == 0
        ):
            mean_text = np.array2string(means[-1], precision=4)
            print(
                f"step {completed:4d}/{args.steps} t={times[-1]:.4f} "
                f"mean={mean_text} residual={residuals[-1]:.3e} "
                f"rank={ranks[-1]} |alpha|={alpha_norms[-1]:.3e}"
            )

    missing = sorted(set(schedule) - set(snapshot_particles))
    if missing:
        raise RuntimeError(f"internal error: missing snapshots at steps {missing}")
    ordered_steps = list(schedule)
    ordered_snapshot_times = np.asarray([schedule[index] for index in ordered_steps])
    ordered_snapshots = np.stack([snapshot_particles[index] for index in ordered_steps])
    final_particles = state.particles.detach().cpu().numpy()

    baseline_snapshots: np.ndarray | None = None
    if args.run_sde_baseline:
        baseline_snapshots = simulate_euler_maruyama(
            initial_particles,
            drift,
            schedule,
            steps=args.steps,
            step_size=args.step_size,
            noise_std=args.noise_std,
            device=device,
            dtype=dtype,
            seed=args.seed + 10_000,
        )

    history_path = output_dir / "history.npz"
    arrays: dict[str, np.ndarray] = {
        "initial_particles": initial_particles,
        "final_particles": final_particles,
        "labels": state.labels.detach().cpu().numpy(),
        "final_log_density": state.log_density.detach().cpu().numpy(),
        "final_score": state.score.detach().cpu().numpy(),
        "times": np.asarray(times),
        "means": np.asarray(means),
        "stds": np.asarray(stds),
        "mean_log_density": np.asarray(mean_log_density),
        "projection_residuals": np.asarray(residuals),
        "retained_ranks": np.asarray(ranks),
        "alpha_norms": np.asarray(alpha_norms),
        "mean_divergences": np.asarray(mean_divergences),
        "snapshot_times": ordered_snapshot_times,
        "snapshot_particles": ordered_snapshots,
    }
    if baseline_snapshots is not None:
        arrays["sde_baseline_particles"] = baseline_snapshots
    np.savez(history_path, **arrays)

    metadata = vars(args).copy()
    metadata.update(
        {
            "resolved_device": str(device),
            "torch_version": torch.__version__,
            "trainable_parameters_M": method.parameter_count,
            "actual_basis_size_m": min(config.basis_size, method.parameter_count),
            "diffusion_matrix_diagonal": diffusion_entry,
            "dtb_config": asdict(config),
        }
    )
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    equilibria = _equilibria(args)
    _plot_snapshots(
        output_dir / "dtb_snapshots.png",
        ordered_snapshots,
        ordered_snapshot_times,
        equilibria,
        args.plot_low,
        args.plot_high,
        title=f"Neural--DTB: {args.initial_distribution} initial samples",
    )
    if baseline_snapshots is not None:
        _plot_snapshots(
            output_dir / "sde_baseline_snapshots.png",
            baseline_snapshots,
            ordered_snapshot_times,
            equilibria,
            args.plot_low,
            args.plot_high,
            title="Euler--Maruyama best-response baseline",
        )
    _plot_summary(
        output_dir / "diagnostics.png",
        np.asarray(times),
        np.asarray(means),
        np.asarray(residuals),
        np.asarray(alpha_norms),
        args.game,
    )
    print(f"saved {history_path}")
    print(f"saved {output_dir / 'dtb_snapshots.png'}")
    if baseline_snapshots is not None:
        print(f"saved {output_dir / 'sde_baseline_snapshots.png'}")
    print(f"saved {output_dir / 'diagnostics.png'}")
    return output_dir


def _make_drift(args: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    if args.game == "cournot":
        if args.dim != 2:
            raise ValueError("the Cournot example requires --dim 2")
        return lambda x: cournot_duopoly_drift(
            x, b=args.cournot_b, mu=args.cournot_mu
        )
    if args.game == "linear":
        if args.dim != 2:
            raise ValueError("the linear example requires --dim 2")
        return lambda x: linear_quadratic_drift(
            x,
            target=args.linear_target,
            contraction=args.linear_contraction,
            rotation=args.linear_rotation,
        )
    raise ValueError(f"unknown game: {args.game}")


def _equilibria(args: Any) -> np.ndarray:
    if args.game == "cournot" and args.cournot_b == 1.0 and args.cournot_mu == 2.0:
        return np.asarray([[0.0, 0.0], [0.5, 0.5]])
    if args.game == "linear":
        return np.asarray([[args.linear_target, args.linear_target]])
    return np.empty((0, 2))


def simulate_euler_maruyama(
    initial_particles: np.ndarray,
    drift: Callable[[torch.Tensor], torch.Tensor],
    schedule: dict[int, float],
    *,
    steps: int,
    step_size: float,
    noise_std: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> np.ndarray:
    """Direct SDE baseline for the same Fokker--Planck dynamics."""

    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    x = torch.as_tensor(initial_particles, device=device, dtype=dtype).clone()
    snapshots: dict[int, np.ndarray] = {}
    if 0 in schedule:
        snapshots[0] = initial_particles.copy()
    root_h = math.sqrt(step_size)
    for step in range(steps):
        noise = torch.randn(
            x.shape, device=device, dtype=dtype, generator=generator
        )
        x = x + step_size * drift(x) + noise_std * root_h * noise
        completed = step + 1
        if completed in schedule:
            snapshots[completed] = x.detach().cpu().numpy().copy()
    return np.stack([snapshots[index] for index in schedule])


def _plot_snapshots(
    path: Path,
    snapshots: np.ndarray,
    times: np.ndarray,
    equilibria: np.ndarray,
    plot_low: float,
    plot_high: float,
    *,
    title: str,
) -> None:
    """Create the 2-by-3 layout used by target Figures 4.2 and 4.3."""

    count = len(times)
    columns = min(3, count)
    rows = int(math.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.7 * columns, 3.5 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    labels = "abcdefghijklmnopqrstuvwxyz"
    for index, (axis, particles, time_value) in enumerate(
        zip(axes_array, snapshots, times)
    ):
        axis.scatter(
            particles[:, 0], particles[:, 1], s=4, alpha=0.35,
            color="#1786c7", edgecolors="none", rasterized=True,
        )
        if equilibria.size:
            axis.scatter(
                equilibria[:, 0], equilibria[:, 1], s=28,
                color="red", marker=".", zorder=5,
            )
        axis.set_xlim(plot_low, plot_high)
        axis.set_ylim(plot_low, plot_high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$x_1$")
        axis.set_ylabel(r"$x_2$")
        axis.text(
            0.5,
            -0.29,
            f"({labels[index]}) $t={time_value:g}$",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        axis.grid(alpha=0.45)
    for axis in axes_array[count:]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=12)
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.10, top=0.90, wspace=0.34, hspace=0.68
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_summary(
    path: Path,
    times: np.ndarray,
    means: np.ndarray,
    residuals: np.ndarray,
    alpha_norms: np.ndarray,
    game: str,
) -> None:
    """Plot mean strategies plus projection and coefficient diagnostics."""

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(times, means[:, 0], label="mean x1")
    axes[0].plot(times, means[:, 1], label="mean x2")
    axes[0].set_title(f"{game} particle means")
    axes[0].legend()
    axes[1].semilogy(times[1:], np.maximum(residuals, 1e-16))
    axes[1].set_title("Relative projection residual")
    axes[2].semilogy(times[1:], np.maximum(alpha_norms, 1e-16))
    axes[2].set_title(r"$\|\alpha_k\|_2$")
    for axis in axes:
        axis.set_xlabel("time")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
