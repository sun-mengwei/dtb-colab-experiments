"""Experiment runner, thesis presets, snapshots, and artifact generation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from .algorithm import DTBConfig, NeuralDTBGameDynamics
from .games import (
    cournot_duopoly_drift,
    cournot_five_player_drift,
    cournot_three_player_drift,
    linear_quadratic_drift,
)
from .models import TangentMLP, TangentMMNN, TangentNODE
from .state import gaussian_particle_state, uniform_box_particle_state


def dtb_progress_milestones(
    total_steps: int, report_count: int = 10
) -> tuple[int, ...]:
    """Return distinct, increasing step indices for percentage reports."""

    if total_steps < 1 or report_count < 1:
        raise ValueError("total_steps and report_count must be positive")
    return tuple(
        sorted(
            {
                math.ceil(total_steps * report_index / report_count)
                for report_index in range(1, report_count + 1)
            }
        )
    )


def projection_error_metrics(
    result: Any, particle_count: int
) -> tuple[float, float]:
    """Return relative residual and vector RMSE per particle."""

    relative = float(result.diagnostics.relative_residual)
    residual_norm = relative * float(result.target_velocity_norm)
    particle_rmse = residual_norm / math.sqrt(particle_count)
    return relative, particle_rmse


def _format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds_part:02d}s"
    if minutes:
        return f"{minutes}m {seconds_part:02d}s"
    return f"{seconds_part}s"


class DTBProgressReporter:
    """Print DTB projection quality and timing at ten progress milestones."""

    def __init__(
        self,
        *,
        total_steps: int,
        step_size: float,
        particle_count: int,
        basis_size: int,
        device: torch.device,
        report_count: int = 10,
    ) -> None:
        self.total_steps = total_steps
        self.step_size = step_size
        self.particle_count = particle_count
        self.basis_size = basis_size
        self.device = device
        self.milestones = frozenset(
            dtb_progress_milestones(total_steps, report_count)
        )
        self.reported: set[int] = set()
        self.started_at = time.perf_counter()

    def update(
        self,
        *,
        completed_step: int,
        result: Any,
        refit_count: int,
    ) -> bool:
        """Print one milestone report and return whether anything was printed."""

        if completed_step not in self.milestones or completed_step in self.reported:
            return False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        elapsed = time.perf_counter() - self.started_at
        fraction = completed_step / self.total_steps
        eta = elapsed * (1.0 - fraction) / fraction
        percentage = round(100.0 * fraction)
        relative, particle_rmse = projection_error_metrics(
            result, self.particle_count
        )
        rank = result.diagnostics.retained_rank
        physical_time = completed_step * self.step_size
        print(
            f"DTB {percentage:3d}% | step {completed_step}/{self.total_steps} "
            f"| t={physical_time:.4f} | projection: relative={relative:.3e}, "
            f"particle-RMSE={particle_rmse:.3e} | rank={rank}/{self.basis_size} "
            f"| refits={refit_count} | elapsed={_format_duration(elapsed)} "
            f"| ETA={_format_duration(eta)}"
        )
        self.reported.add(completed_step)
        return True


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
    refit_interval = getattr(args, "refit_interval", 0)
    if refit_interval > 0 and args.steps % refit_interval != 0:
        raise ValueError(
            "steps must be a multiple of refit_interval, matching the "
            "complete-block rule in the original DTB implementation"
        )

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

        def reference_sampler(
            count: int, generator: torch.Generator
        ) -> torch.Tensor:
            return (
                torch.rand(
                    count, args.dim, device=device, dtype=dtype,
                    generator=generator,
                )
                * (args.uniform_high - args.uniform_low)
                + args.uniform_low
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

        def reference_sampler(
            count: int, generator: torch.Generator
        ) -> torch.Tensor:
            return (
                torch.randn(
                    count, args.dim, device=device, dtype=dtype,
                    generator=generator,
                )
                * initial_std
                + args.initial_mean
            )
    else:
        raise ValueError(f"unknown initial distribution: {args.initial_distribution}")

    initial_particles = state.particles.detach().cpu().numpy().copy()
    drift = _make_drift(args)

    # The selected parameter tangents of f_theta form the Eulerian dictionary.
    # If resetting is enabled, accumulated tangent updates are periodically
    # compressed into f_theta on fresh samples from the reference law.
    architecture = getattr(args, "architecture", "mlp")
    if architecture == "mlp":
        model = TangentMLP(
            dim=args.dim,
            width=args.width,
            depth=args.depth,
            activation=args.activation,
            dtype=dtype,
        ).to(device)
    elif architecture == "mmnn":
        model = TangentMMNN(
            dim=args.dim,
            width=args.width,
            rank=getattr(args, "rank", 8),
            depth=args.depth,
            activation=args.activation,
            dtype=dtype,
        ).to(device)
    elif architecture == "node":
        model = TangentNODE(
            dim=args.dim,
            width=args.width,
            depth=args.depth,
            activation=args.activation,
            inner_steps=getattr(args, "node_inner_steps", 4),
            integration_time=getattr(args, "node_integration_time", 1.0),
            dtype=dtype,
        ).to(device)
    else:
        raise ValueError("architecture must be 'mlp', 'mmnn', or 'node'")

    initial_activation_diagnostics = model.tanh_diagnostics(state.particles)

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
        refit_interval=refit_interval,
        refit_optimizer_steps=getattr(args, "refit_optimizer_steps", 2_000),
        refit_learning_rate=getattr(args, "refit_learning_rate", 1e-3),
        refit_batch_size=getattr(args, "refit_batch_size", 2_048),
        refit_samples=getattr(args, "refit_samples", 10_000),
        refit_test_samples=getattr(args, "refit_test_samples", 4_000),
    )
    method = NeuralDTBGameDynamics(
        model, drift, diffusion, config, reference_sampler=reference_sampler
    )

    times = [0.0]
    means = [state.particles.mean(dim=0).detach().cpu().numpy()]
    stds = [state.particles.std(dim=0).detach().cpu().numpy()]
    mean_log_density = [float(state.log_density.mean())]
    residuals: list[float] = []
    ranks: list[int] = []
    alpha_norms: list[float] = []
    mean_divergences: list[float] = []
    target_velocity_norms: list[float] = []
    projected_velocity_norms: list[float] = []
    refit_performed: list[bool] = []
    refit_rmse_before: list[float] = []
    refit_rmse_after: list[float] = []
    refit_reasons: list[str] = []
    steps_in_tangent_block: list[int] = []
    snapshot_particles: dict[int, np.ndarray] = {}
    if 0 in schedule:
        snapshot_particles[0] = initial_particles.copy()
    progress_reporter = DTBProgressReporter(
        total_steps=args.steps,
        step_size=args.step_size,
        particle_count=state.particles.shape[0],
        basis_size=min(config.basis_size, method.parameter_count),
        device=device,
    )

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
        target_velocity_norms.append(result.target_velocity_norm)
        projected_velocity_norms.append(result.projected_velocity_norm)
        refit_performed.append(result.refit_performed)
        refit_rmse_before.append(result.refit_rmse_before)
        refit_rmse_after.append(result.refit_rmse_after)
        refit_reasons.append(result.refit_reason)
        steps_in_tangent_block.append(result.steps_in_tangent_block)
        if completed in schedule:
            snapshot_particles[completed] = state.particles.detach().cpu().numpy().copy()
        progress_reporter.update(
            completed_step=completed,
            result=result,
            refit_count=method.refit_count,
        )
        if result.refit_performed:
            print(
                f"  refit #{method.refit_count} ({result.refit_reason}) "
                f"RMSE {result.refit_rmse_before:.3e} -> "
                f"{result.refit_rmse_after:.3e} on fresh reference samples"
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
        "target_velocity_norms": np.asarray(target_velocity_norms),
        "projected_velocity_norms": np.asarray(projected_velocity_norms),
        "absolute_projection_errors": (
            np.asarray(target_velocity_norms) * np.asarray(residuals)
        ),
        "refit_performed": np.asarray(refit_performed, dtype=bool),
        "refit_rmse_before": np.asarray(refit_rmse_before),
        "refit_rmse_after": np.asarray(refit_rmse_after),
        "refit_reasons": np.asarray(refit_reasons),
        "steps_in_tangent_block": np.asarray(steps_in_tangent_block),
        "snapshot_times": ordered_snapshot_times,
        "snapshot_particles": ordered_snapshots,
    }
    if baseline_snapshots is not None:
        arrays["sde_baseline_particles"] = baseline_snapshots
    np.savez(history_path, **arrays)

    metadata = vars(args).copy()
    metadata.update(
        {
            "resolved_architecture": architecture,
            "mmnn_rank": (
                getattr(args, "rank", None) if architecture == "mmnn" else None
            ),
            "node_inner_steps": (
                getattr(args, "node_inner_steps", 4)
                if architecture == "node" else None
            ),
            "node_integration_time": (
                getattr(args, "node_integration_time", 1.0)
                if architecture == "node" else None
            ),
            "resolved_device": str(device),
            "torch_version": torch.__version__,
            "trainable_parameters_M": method.parameter_count,
            "tangent_parameter_names": [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
            "frozen_parameter_names": [
                name
                for name, parameter in model.named_parameters()
                if not parameter.requires_grad
            ],
            "actual_basis_size_m": min(config.basis_size, method.parameter_count),
            "diffusion_matrix_diagonal": diffusion_entry,
            "dtb_config": asdict(config),
            "refit_count": method.refit_count,
            "refit_rule": (
                "Source-matching DTB block reset: freeze theta_block and one "
                "random selected sub-basis for exactly L steps; accumulate "
                "s=sum(alpha); precompute f_theta_block+h*J_block*s on fresh "
                "samples from the initial reference law; fit all trainable "
                "parameters with Adam plus cosine LR; test on a fresh batch."
            ),
            "tanh_diagnostics_initial": initial_activation_diagnostics,
            "tanh_diagnostics_final": model.tanh_diagnostics(state.particles),
        }
    )
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    equilibria = _equilibria(args)
    equilibrium_is_stable = _equilibrium_stability(args)
    _plot_snapshots(
        output_dir / "dtb_snapshots.png",
        ordered_snapshots,
        ordered_snapshot_times,
        equilibria,
        equilibrium_is_stable,
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
            equilibrium_is_stable,
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
        refit_steps=np.flatnonzero(np.asarray(refit_performed)) + 1,
    )
    if method.refitting_enabled:
        _plot_refit_diagnostics(
            output_dir / "refit_diagnostics.png",
            np.asarray(times[1:]),
            np.asarray(residuals),
            np.asarray(ranks),
            np.asarray(target_velocity_norms),
            np.asarray(refit_performed),
            np.asarray(refit_rmse_before),
            np.asarray(refit_rmse_after),
        )
    if args.dim == 5:
        _plot_pairwise_final(
            output_dir / "pairwise_final.png",
            final_particles,
            equilibria,
            equilibrium_is_stable,
        )
    print(f"saved {history_path}")
    print(f"saved {output_dir / 'dtb_snapshots.png'}")
    if baseline_snapshots is not None:
        print(f"saved {output_dir / 'sde_baseline_snapshots.png'}")
    print(f"saved {output_dir / 'diagnostics.png'}")
    if method.refitting_enabled:
        print(f"saved {output_dir / 'refit_diagnostics.png'}")
    if args.dim == 5:
        print(f"saved {output_dir / 'pairwise_final.png'}")
    return output_dir


def _make_drift(args: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    if args.game == "cournot":
        if args.dim != 2:
            raise ValueError("the Cournot example requires --dim 2")
        return lambda x: cournot_duopoly_drift(
            x, b=args.cournot_b, mu=args.cournot_mu
        )
    if args.game == "cournot3":
        if args.dim != 3:
            raise ValueError("the three-player Cournot example requires --dim 3")
        return lambda x: cournot_three_player_drift(
            x, b=args.cournot_b, mu=args.cournot_mu
        )
    if args.game == "cournot5":
        if args.dim != 5:
            raise ValueError("the five-player Cournot example requires --dim 5")
        return lambda x: cournot_five_player_drift(
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
    if args.game == "cournot3" and args.cournot_b == 1.0 and args.cournot_mu == 2.0:
        return np.asarray(
            [
                [0.0, 0.0, 0.0],
                [3.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ]
        )
    if args.game == "cournot5" and args.cournot_b == 1.0 and args.cournot_mu == 2.0:
        symmetric = np.full(5, 7.0 / 32.0)
        one_zero = []
        for zero_index in range(5):
            equilibrium = np.full(5, 5.0 / 18.0)
            equilibrium[zero_index] = 0.0
            one_zero.append(equilibrium)
        return np.vstack([np.zeros(5), symmetric, np.asarray(one_zero)])
    if args.game == "linear":
        return np.asarray([[args.linear_target, args.linear_target]])
    return np.empty((0, 2))


def _equilibrium_stability(args: Any) -> np.ndarray:
    """Return a Boolean stability label aligned with ``_equilibria``."""

    if args.game == "cournot" and args.cournot_b == 1.0 and args.cournot_mu == 2.0:
        return np.asarray([False, True], dtype=bool)
    if args.game == "cournot3" and args.cournot_b == 1.0 and args.cournot_mu == 2.0:
        return np.asarray([False, True, True, True, True], dtype=bool)
    if args.game == "cournot5" and args.cournot_b == 1.0 and args.cournot_mu == 2.0:
        return np.zeros(7, dtype=bool)
    if args.game == "linear":
        return np.asarray([True], dtype=bool)
    return np.empty((0,), dtype=bool)


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
    equilibrium_is_stable: np.ndarray,
    plot_low: float,
    plot_high: float,
    *,
    title: str,
) -> None:
    """Create the 2-by-3 layout used by target Figures 4.2 and 4.3."""

    count = len(times)
    columns = min(3, count)
    rows = int(math.ceil(count / columns))
    dim = snapshots.shape[-1]
    if dim not in (2, 3, 5):
        raise ValueError("snapshot plotting currently supports d=2, d=3, or d=5")
    if equilibrium_is_stable.shape != (len(equilibria),):
        raise ValueError("equilibrium stability labels must align with equilibria")
    subplot_kw = {"projection": "3d"} if dim == 3 else {}
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.7 * columns, 3.7 * rows),
        subplot_kw=subplot_kw,
    )
    axes_array = np.asarray(axes).reshape(-1)
    labels = "abcdefghijklmnopqrstuvwxyz"
    point_count = snapshots.shape[1]
    point_size = 5.0 if point_count <= 1000 else 1.8
    point_alpha = 0.45 if point_count <= 1000 else 0.28
    for index, (axis, particles, time_value) in enumerate(
        zip(axes_array, snapshots, times)
    ):
        if dim == 3:
            axis.scatter(
                particles[:, 0],
                particles[:, 1],
                particles[:, 2],
                s=point_size,
                alpha=point_alpha,
                color="#1786c7",
                edgecolors="none",
                rasterized=True,
            )
            if equilibria.size:
                stable = equilibria[equilibrium_is_stable]
                unstable = equilibria[~equilibrium_is_stable]
                if stable.size:
                    axis.scatter(
                        stable[:, 0], stable[:, 1], stable[:, 2],
                        s=34, color="#d62728", marker="o", depthshade=False,
                        label=(
                            f"stable equilibria ({len(stable)})"
                            if index == 0 else "_nolegend_"
                        ),
                    )
                if unstable.size:
                    axis.scatter(
                        unstable[:, 0], unstable[:, 1], unstable[:, 2],
                        s=64, color="#ffbf00", edgecolors="#222222",
                        linewidths=0.7, marker="X", depthshade=False,
                        label=(
                            "unstable equilibrium (origin)"
                            if index == 0 else "_nolegend_"
                        ),
                    )
            axis.set_zlim(plot_low, plot_high)
            axis.set_zlabel(r"$x_3$", labelpad=1)
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=24, azim=-55)
            axis.text2D(
                0.5,
                -0.18,
                f"({labels[index]}) $t={time_value:g}$",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )
        elif dim == 5:
            opponents_mean = particles[:, 1:].mean(axis=1)
            axis.scatter(
                particles[:, 0], opponents_mean, s=point_size, alpha=point_alpha,
                color="#1786c7", edgecolors="none", rasterized=True,
            )
            if equilibria.size:
                projected = np.column_stack(
                    [equilibria[:, 0], equilibria[:, 1:].mean(axis=1)]
                )
                stable = projected[equilibrium_is_stable]
                unstable = projected[~equilibrium_is_stable]
                if stable.size:
                    axis.scatter(
                        stable[:, 0], stable[:, 1], s=38,
                        color="#d62728", marker="o", zorder=5,
                        label=(
                            f"stable equilibria ({len(stable)})"
                            if index == 0 else "_nolegend_"
                        ),
                    )
                if unstable.size:
                    axis.scatter(
                        unstable[:, 0], unstable[:, 1], s=68,
                        color="#ffbf00", edgecolors="#222222", linewidths=0.7,
                        marker="X", zorder=6,
                        label=(
                            f"known unstable equilibria ({len(unstable)}; projected)"
                            if index == 0 else "_nolegend_"
                        ),
                    )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(r"$x_1$")
            axis.set_ylabel(r"mean$(x_2,\ldots,x_5)$")
            axis.text(
                0.5,
                -0.29,
                f"({labels[index]}) $t={time_value:g}$",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )
        else:
            axis.scatter(
                particles[:, 0], particles[:, 1], s=4, alpha=0.35,
                color="#1786c7", edgecolors="none", rasterized=True,
            )
            if equilibria.size:
                stable = equilibria[equilibrium_is_stable]
                unstable = equilibria[~equilibrium_is_stable]
                if stable.size:
                    axis.scatter(
                        stable[:, 0], stable[:, 1], s=38,
                        color="#d62728", marker="o", zorder=5,
                        label=(
                            f"stable equilibria ({len(stable)})"
                            if index == 0 else "_nolegend_"
                        ),
                    )
                if unstable.size:
                    axis.scatter(
                        unstable[:, 0], unstable[:, 1], s=68,
                        color="#ffbf00", edgecolors="#222222", linewidths=0.7,
                        marker="X", zorder=6,
                        label=(
                            "unstable equilibrium (origin)"
                            if index == 0 else "_nolegend_"
                        ),
                    )
            axis.set_aspect("equal", adjustable="box")
            axis.text(
                0.5,
                -0.29,
                f"({labels[index]}) $t={time_value:g}$",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )
        axis.set_xlim(plot_low, plot_high)
        axis.set_ylim(plot_low, plot_high)
        if dim != 5:
            axis.set_xlabel(r"$x_1$")
            axis.set_ylabel(r"$x_2$")
        axis.grid(alpha=0.45)
    for axis in axes_array[count:]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=12)
    handles, legend_labels = axes_array[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=min(2, len(handles)),
            frameon=False,
        )
    figure.subplots_adjust(
        left=0.07, right=0.98, bottom=0.10, top=0.84, wspace=0.34, hspace=0.68
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_pairwise_final(
    path: Path,
    particles: np.ndarray,
    equilibria: np.ndarray,
    equilibrium_is_stable: np.ndarray,
) -> None:
    """Show every coordinate and coordinate pair for a five-dimensional run."""

    if particles.ndim != 2 or particles.shape[1] != 5:
        raise ValueError("pairwise plotting requires particles with shape (N,5)")
    if equilibrium_is_stable.shape != (len(equilibria),):
        raise ValueError("equilibrium stability labels must align with equilibria")

    figure, axes = plt.subplots(5, 5, figsize=(12, 12))
    for row in range(5):
        for column in range(5):
            axis = axes[row, column]
            if row == column:
                axis.hist(
                    particles[:, column], bins=45, density=True,
                    color="#1786c7", alpha=0.65,
                )
                for value in np.unique(equilibria[:, column]):
                    axis.axvline(value, color="#ffbf00", linewidth=1.0, alpha=0.85)
                axis.set_xlim(-0.4, 1.0)
            elif row > column:
                axis.scatter(
                    particles[:, column], particles[:, row],
                    s=2.0, alpha=0.20, color="#1786c7",
                    edgecolors="none", rasterized=True,
                )
                stable = equilibria[equilibrium_is_stable]
                unstable = equilibria[~equilibrium_is_stable]
                if stable.size:
                    axis.scatter(
                        stable[:, column], stable[:, row], s=30,
                        color="#d62728", marker="o", zorder=5,
                        label="stable equilibria",
                    )
                if unstable.size:
                    axis.scatter(
                        unstable[:, column], unstable[:, row], s=48,
                        color="#ffbf00", edgecolors="#222222", linewidths=0.6,
                        marker="X", zorder=6, label="known unstable equilibria",
                    )
                axis.set_xlim(-0.4, 1.0)
                axis.set_ylim(-0.4, 1.0)
            else:
                correlation = np.corrcoef(particles[:, column], particles[:, row])[0, 1]
                axis.text(
                    0.5, 0.5, rf"$\rho={correlation:.2f}$",
                    transform=axis.transAxes, ha="center", va="center",
                )
                axis.set_axis_off()

            if row == 4:
                axis.set_xlabel(rf"$x_{column + 1}$")
            if column == 0 and row != 0:
                axis.set_ylabel(rf"$x_{row + 1}$")
            axis.grid(alpha=0.20)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
            ncol=min(2, len(handles)), frameon=False,
        )
    figure.suptitle("Five-player Neural--DTB: final pairwise projections", y=0.995)
    figure.subplots_adjust(
        left=0.08, right=0.98, bottom=0.07, top=0.93, wspace=0.12, hspace=0.12
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
    *,
    refit_steps: np.ndarray | None = None,
) -> None:
    """Plot mean strategies plus projection and coefficient diagnostics."""

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for component in range(means.shape[1]):
        axes[0].plot(times, means[:, component], label=f"mean x{component + 1}")
    axes[0].set_title(f"{game} particle means")
    axes[0].legend()
    axes[1].semilogy(times[1:], np.maximum(residuals, 1e-16))
    axes[1].set_title("Relative projection residual")
    axes[2].semilogy(times[1:], np.maximum(alpha_norms, 1e-16))
    axes[2].set_title(r"$\|\alpha_k\|_2$")
    for axis in axes:
        axis.set_xlabel("time")
        axis.grid(alpha=0.25)
    if refit_steps is not None:
        for step in refit_steps:
            refit_time = times[int(step)]
            for axis in axes:
                axis.axvline(refit_time, color="#9467bd", alpha=0.20, linewidth=1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_refit_diagnostics(
    path: Path,
    step_times: np.ndarray,
    residuals: np.ndarray,
    ranks: np.ndarray,
    target_norms: np.ndarray,
    refit_mask: np.ndarray,
    rmse_before: np.ndarray,
    rmse_after: np.ndarray,
) -> None:
    """Plot approximation quality and each network block reset."""

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    event_times = step_times[refit_mask]
    axes[0, 0].plot(step_times, residuals, color="#1f77b4")
    axes[0, 0].set_title("Relative tangent projection residual")
    axes[0, 1].plot(step_times, ranks, color="#2ca02c")
    axes[0, 1].set_title("Truncated-SVD retained rank")
    axes[1, 0].plot(
        step_times, target_norms * residuals, color="#d62728"
    )
    axes[1, 0].set_title("Absolute projection error")
    if bool(refit_mask.any()):
        event_index = np.arange(1, int(refit_mask.sum()) + 1)
        axes[1, 1].plot(
            event_index, rmse_before[refit_mask], "o-", label="pre-fit train"
        )
        axes[1, 1].plot(
            event_index, rmse_after[refit_mask], "o-", label="post-fit fresh test"
        )
        axes[1, 1].legend()
    else:
        axes[1, 1].text(
            0.5, 0.5, "No refit was triggered", ha="center", va="center",
            transform=axes[1, 1].transAxes,
        )
    axes[1, 1].set_title("Network RMSE at block resets")
    axes[1, 1].set_xlabel("refit event")
    for axis in axes.reshape(-1)[:3]:
        for event_time in event_times:
            axis.axvline(
                event_time, color="#9467bd", alpha=0.28, linewidth=1.0
            )
        axis.set_xlabel("physical time")
    for axis in axes.reshape(-1):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
