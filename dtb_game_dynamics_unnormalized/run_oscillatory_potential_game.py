"""Controlled Neural--DTB sweep for an oscillatory multi-well potential game.

This driver reuses the existing DTB stepper without changing its tangent
construction, truncated-SVD solve, score transport, or periodic reset rule.
Only ``omega`` and ``gamma`` vary within a sweep.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from game_dtb.algorithm import DTBConfig, NeuralDTBGameDynamics
from game_dtb.models import TangentMLP, TangentMMNN, TangentNODE
from game_dtb.oscillatory_game import (
    OscillatoryGameParams,
    basin_masses,
    integrate_rk4,
    locate_oscillatory_equilibria,
    oscillatory_potential,
    oscillatory_vector_field,
    sample_uniform_initial_particles,
)
from game_dtb.runner import resolve_device
from game_dtb.state import ParticleState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled DTB frequency sweep for a two-player potential game"
    )
    # BLOCK 1 -- Only these game parameters vary inside one controlled sweep.
    parser.add_argument(
        "--omega-multipliers", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0],
        help="frequencies expressed as multiples of pi",
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.5)

    # BLOCK 2 -- Shared particle realization and physical time discretization.
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-seed", type=int, default=91)

    # BLOCK 3 -- Fixed neural tangent representation and SVD cutoff.
    parser.add_argument(
        "--architecture", choices=("mlp", "mmnn", "node"), default="mlp"
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--basis-size", type=int, default=128)
    parser.add_argument("--svd-rtol", type=float, default=1e-3)
    parser.add_argument("--activation", choices=("tanh", "gelu", "silu"), default="tanh")
    parser.add_argument("--node-inner-steps", type=int, default=4)
    parser.add_argument("--node-integration-time", type=float, default=1.0)
    parser.add_argument("--jacobian-chunk", type=int, default=128)
    parser.add_argument("--derivative-chunk", type=int, default=64)

    # BLOCK 4 -- Existing source-matching periodic reset; zero means fixed NN.
    parser.add_argument("--refit-interval", type=int, default=0)
    parser.add_argument("--refit-optimizer-steps", type=int, default=2_000)
    parser.add_argument("--refit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--refit-batch-size", type=int, default=2_048)
    parser.add_argument("--refit-samples", type=int, default=10_000)
    parser.add_argument("--refit-test-samples", type=int, default=4_000)

    # BLOCK 5 -- Reference accuracy and numerical equilibrium diagnostics.
    parser.add_argument("--reference-substeps", type=int, default=20)
    parser.add_argument("--skip-reference-check", action="store_true")
    parser.add_argument("--equilibrium-grid-size", type=int, default=None)
    parser.add_argument("--basin-tolerance", type=float, default=0.08)
    parser.add_argument("--max-unassigned-fraction", type=float, default=0.05)
    parser.add_argument("--potential-drop-tolerance", type=float, default=1e-4)

    # BLOCK 6 -- Runtime and output controls.
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="outputs/oscillatory_potential_game")
    parser.add_argument(
        "--smoke", action="store_true",
        help="run all requested cases with a tiny particle/time/tangent setup",
    )
    parser.add_argument("--print-every", type=int, default=40)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "particles", "steps", "width", "depth", "rank", "basis_size",
        "node_inner_steps", "jacobian_chunk", "derivative_chunk",
        "refit_optimizer_steps", "refit_batch_size", "refit_samples",
        "refit_test_samples", "reference_substeps",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if not 0.0 <= args.svd_rtol < 1.0:
        raise ValueError("svd_rtol must lie in [0,1)")
    if args.refit_interval < 0:
        raise ValueError("refit_interval must be nonnegative")
    if args.refit_interval and args.steps % args.refit_interval:
        raise ValueError("steps must be divisible by refit_interval")
    if any(value <= 0.0 for value in args.omega_multipliers):
        raise ValueError("omega multipliers must be positive")
    if any(value < 0.0 for value in args.gammas):
        raise ValueError("gammas must be nonnegative")
    if args.basin_tolerance <= 0.0:
        raise ValueError("basin_tolerance must be positive")
    if not 0.0 <= args.max_unassigned_fraction < 1.0:
        raise ValueError("max_unassigned_fraction must lie in [0,1)")


def apply_smoke_preset(args: argparse.Namespace) -> None:
    """Keep all requested frequencies/couplings but make each run very small."""

    args.particles = min(args.particles, 32)
    args.steps = min(args.steps, 4)
    args.width = min(args.width, 8)
    args.depth = min(args.depth, 2)
    args.rank = min(args.rank, 4)
    args.basis_size = min(args.basis_size, 16)
    args.jacobian_chunk = min(args.jacobian_chunk, 32)
    args.derivative_chunk = min(args.derivative_chunk, 16)
    args.reference_substeps = min(args.reference_substeps, 4)
    args.refit_interval = 0
    if args.equilibrium_grid_size is None:
        args.equilibrium_grid_size = 33
    args.print_every = 0


def make_model(args: argparse.Namespace, dtype: torch.dtype) -> torch.nn.Module:
    if args.architecture == "mlp":
        return TangentMLP(
            2, args.width, args.depth, args.activation, dtype=dtype
        )
    if args.architecture == "mmnn":
        return TangentMMNN(
            2, args.width, args.rank, args.depth, args.activation, dtype=dtype
        )
    return TangentNODE(
        2,
        args.width,
        args.depth,
        args.activation,
        args.node_inner_steps,
        args.node_integration_time,
        dtype=dtype,
    )


def model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initial_particle_state(
    initial_particles: np.ndarray, device: torch.device, dtype: torch.dtype
) -> ParticleState:
    labels = torch.as_tensor(initial_particles, device=device, dtype=dtype)
    state = ParticleState(
        particles=labels.clone(),
        log_density=torch.full(
            (len(labels),), -2.0 * math.log(2.0), device=device, dtype=dtype
        ),
        score=torch.zeros_like(labels),
        labels=labels.clone(),
    )
    state.validate()
    return state


def run_case(
    args: argparse.Namespace,
    initial_particles: np.ndarray,
    *,
    gamma: float,
    omega_multiplier: float,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    params = OscillatoryGameParams(
        lambda_=args.lambda_value,
        epsilon=args.epsilon,
        omega=omega_multiplier * np.pi,
        gamma=gamma,
    )
    params.validate()
    output_dir.mkdir(parents=True, exist_ok=True)

    # BLOCK A -- High-accuracy deterministic ODE reference on the same labels.
    start_total = time.perf_counter()
    reference_initial = torch.as_tensor(
        initial_particles, device=device, dtype=torch.float64
    )
    start_reference = time.perf_counter()
    reference_history_t = integrate_rk4(
        reference_initial,
        params,
        step_size=args.step_size,
        steps=args.steps,
        substeps_per_step=args.reference_substeps,
    )
    reference_runtime = time.perf_counter() - start_reference
    if args.skip_reference_check:
        reference_self_error = float("nan")
    else:
        refined_final = integrate_rk4(
            reference_initial,
            params,
            step_size=args.step_size,
            steps=args.steps,
            substeps_per_step=2 * args.reference_substeps,
            save_history=False,
        )
        reference_self_error = float(
            torch.sqrt(torch.mean((reference_history_t[-1] - refined_final).square()))
        )

    # BLOCK B -- Recreate the identical NN initialization and DTB RNG per case.
    torch.manual_seed(args.model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.model_seed)
    model = make_model(args, dtype).to(device)
    initialization_hash = model_state_hash(model)
    state = initial_particle_state(initial_particles, device, dtype)
    diffusion = torch.zeros((2, 2), device=device, dtype=dtype)

    def reference_sampler(count: int, generator: torch.Generator) -> torch.Tensor:
        return 2.0 * torch.rand(
            count, 2, device=device, dtype=dtype, generator=generator
        ) - 1.0

    config = DTBConfig(
        step_size=args.step_size,
        basis_size=args.basis_size,
        svd_rtol=args.svd_rtol,
        jacobian_chunk_size=args.jacobian_chunk,
        derivative_chunk_size=args.derivative_chunk,
        seed=args.seed,
        refit_interval=args.refit_interval,
        refit_optimizer_steps=args.refit_optimizer_steps,
        refit_learning_rate=args.refit_learning_rate,
        refit_batch_size=args.refit_batch_size,
        refit_samples=args.refit_samples,
        refit_test_samples=args.refit_test_samples,
    )
    method = NeuralDTBGameDynamics(
        model,
        drift=lambda values: oscillatory_vector_field(values, params),
        diffusion=diffusion,
        config=config,
        reference_sampler=reference_sampler,
    )
    actual_basis_size = min(args.basis_size, method.parameter_count)

    # BLOCK C -- Advance the unchanged DTB algorithm and record every metric.
    times = np.arange(args.steps + 1, dtype=float) * args.step_size
    dtb_history = [state.particles.detach().cpu().numpy().copy()]
    trajectory_rmse = [0.0]
    trajectory_relative = [0.0]
    dtb_mean_potential = [
        float(oscillatory_potential(state.particles, params).mean())
    ]
    reference_mean_potential = (
        oscillatory_potential(reference_history_t, params).mean(dim=1).cpu().numpy()
    )
    projection_residuals: list[float] = []
    velocity_rmse: list[float] = []
    sigma_max: list[float] = []
    sigma_min_retained: list[float] = []
    retained_rank: list[int] = []
    retained_condition: list[float] = []
    truncated_count: list[int] = []
    truncated_fraction: list[float] = []
    alpha_norm: list[float] = []
    refit_performed: list[bool] = []
    refit_rmse_before: list[float] = []
    refit_rmse_after: list[float] = []

    start_dtb = time.perf_counter()
    for step in range(args.steps):
        result = method.step(state, step)
        state = result.state
        reference_current = reference_history_t[step + 1]
        difference = state.particles.to(torch.float64) - reference_current
        absolute_error = torch.sqrt(torch.mean(difference.square().sum(dim=1)))
        relative_denominator = torch.sqrt(
            torch.sum(reference_current.square())
        ).clamp_min(1e-30)
        relative_error = torch.sqrt(torch.sum(difference.square())) / relative_denominator

        diagnostics = result.diagnostics
        condition = (
            diagnostics.sigma_max / diagnostics.sigma_min_retained
            if diagnostics.sigma_min_retained > 0.0 else float("inf")
        )
        truncated = actual_basis_size - diagnostics.retained_rank
        projection_residuals.append(diagnostics.relative_residual)
        velocity_rmse.append(
            diagnostics.relative_residual * result.target_velocity_norm
            / math.sqrt(args.particles)
        )
        sigma_max.append(diagnostics.sigma_max)
        sigma_min_retained.append(diagnostics.sigma_min_retained)
        retained_rank.append(diagnostics.retained_rank)
        retained_condition.append(condition)
        truncated_count.append(truncated)
        truncated_fraction.append(truncated / actual_basis_size)
        alpha_norm.append(result.alpha_norm)
        trajectory_rmse.append(float(absolute_error))
        trajectory_relative.append(float(relative_error))
        dtb_mean_potential.append(
            float(oscillatory_potential(state.particles, params).mean())
        )
        dtb_history.append(state.particles.detach().cpu().numpy().copy())
        refit_performed.append(result.refit_performed)
        refit_rmse_before.append(result.refit_rmse_before)
        refit_rmse_after.append(result.refit_rmse_after)
        completed = step + 1
        if args.print_every > 0 and (
            completed == 1
            or completed % args.print_every == 0
            or completed == args.steps
        ):
            print(
                f"  step {completed:4d}/{args.steps} "
                f"r_proj={projection_residuals[-1]:.3e} "
                f"E_traj={trajectory_rmse[-1]:.3e} "
                f"rank={retained_rank[-1]}/{actual_basis_size}"
            )
    dtb_runtime = time.perf_counter() - start_dtb
    dtb_history_array = np.asarray(dtb_history)
    reference_history = reference_history_t.detach().cpu().numpy()
    equilibria, equilibrium_stable, spectral_abscissa = (
        locate_oscillatory_equilibria(
            params, grid_size=args.equilibrium_grid_size
        )
    )
    stable_equilibria = equilibria[equilibrium_stable]
    effective_basin_tolerance = _effective_basin_tolerance(
        stable_equilibria, args.basin_tolerance
    )
    dtb_basin_masses, dtb_unassigned = basin_masses(
        dtb_history_array[-1],
        stable_equilibria,
        tolerance=effective_basin_tolerance,
    )
    reference_basin_masses, reference_unassigned = basin_masses(
        reference_history[-1],
        stable_equilibria,
        tolerance=effective_basin_tolerance,
    )
    basin_reliable = (
        dtb_unassigned <= args.max_unassigned_fraction
        and reference_unassigned <= args.max_unassigned_fraction
    )
    basin_error = (
        0.5 * float(np.abs(dtb_basin_masses - reference_basin_masses).sum())
        if basin_reliable else float("nan")
    )

    dtb_potential_drops = np.diff(np.asarray(dtb_mean_potential))
    reference_potential_drops = np.diff(reference_mean_potential)
    history_path = output_dir / "history.npz"
    np.savez_compressed(
        history_path,
        times=times,
        initial_particles=initial_particles,
        dtb_particles=dtb_history_array,
        reference_particles=reference_history,
        projection_residuals=np.asarray(projection_residuals),
        velocity_rmse=np.asarray(velocity_rmse),
        trajectory_rmse=np.asarray(trajectory_rmse),
        trajectory_relative=np.asarray(trajectory_relative),
        sigma_max=np.asarray(sigma_max),
        sigma_min_retained=np.asarray(sigma_min_retained),
        retained_rank=np.asarray(retained_rank),
        retained_condition=np.asarray(retained_condition),
        truncated_count=np.asarray(truncated_count),
        truncated_fraction=np.asarray(truncated_fraction),
        alpha_norm=np.asarray(alpha_norm),
        dtb_mean_potential=np.asarray(dtb_mean_potential),
        reference_mean_potential=reference_mean_potential,
        equilibria=equilibria,
        equilibrium_stable=equilibrium_stable,
        spectral_abscissa=spectral_abscissa,
        dtb_basin_masses=dtb_basin_masses,
        reference_basin_masses=reference_basin_masses,
        effective_basin_tolerance=np.asarray(effective_basin_tolerance),
        refit_performed=np.asarray(refit_performed, dtype=bool),
        refit_rmse_before=np.asarray(refit_rmse_before),
        refit_rmse_after=np.asarray(refit_rmse_after),
    )
    _save_equilibria_table(
        output_dir / "equilibria.csv",
        equilibria,
        equilibrium_stable,
        spectral_abscissa,
        dtb_basin_masses,
        reference_basin_masses,
    )

    case_config = {
        **vars(args),
        "game_parameters": asdict(params),
        "omega_multiplier": omega_multiplier,
        "resolved_device": str(device),
        "model_initialization_sha256": initialization_hash,
        "trainable_parameters": method.parameter_count,
        "actual_basis_size": actual_basis_size,
        "diffusion_matrix": [[0.0, 0.0], [0.0, 0.0]],
        "reference_method": "fixed-step vectorized classical RK4",
        "history": "history.npz",
    }
    (output_dir / "config.json").write_text(
        json.dumps(case_config, indent=2) + "\n"
    )

    _plot_landscape(
        output_dir / "potential_vector_field.png",
        params,
        equilibria,
        equilibrium_stable,
    )
    _plot_particle_comparison(
        output_dir / "particle_comparison.png",
        times,
        reference_history,
        dtb_history_array,
        equilibria,
        equilibrium_stable,
    )
    _plot_case_diagnostics(output_dir / "diagnostics.png", history_path)
    total_runtime = time.perf_counter() - start_total

    summary = {
        "gamma": gamma,
        "omega_multiplier_pi": omega_multiplier,
        "omega": params.omega,
        "particles": args.particles,
        "step_size": args.step_size,
        "final_time": args.steps * args.step_size,
        "mean_projection_residual": float(np.mean(projection_residuals)),
        "max_projection_residual": float(np.max(projection_residuals)),
        "final_projection_residual": float(projection_residuals[-1]),
        "final_velocity_rmse": float(velocity_rmse[-1]),
        "final_trajectory_rmse": float(trajectory_rmse[-1]),
        "mean_trajectory_rmse": float(np.mean(trajectory_rmse[1:])),
        "final_relative_trajectory_error": float(trajectory_relative[-1]),
        "equilibrium_count": len(equilibria),
        "stable_equilibrium_count": int(equilibrium_stable.sum()),
        "effective_basin_tolerance": effective_basin_tolerance,
        "basin_assignment_reliable": basin_reliable,
        "dtb_unassigned_fraction": dtb_unassigned,
        "reference_unassigned_fraction": reference_unassigned,
        "basin_mass_error": basin_error,
        "mean_retained_rank": float(np.mean(retained_rank)),
        "min_retained_rank": int(np.min(retained_rank)),
        "max_retained_rank": int(np.max(retained_rank)),
        "mean_truncated_fraction": float(np.mean(truncated_fraction)),
        "max_retained_condition": float(np.max(retained_condition)),
        "mean_retained_condition": float(np.mean(retained_condition)),
        "max_alpha_norm": float(np.max(alpha_norm)),
        "reference_self_error": reference_self_error,
        "min_dtb_potential_increment": float(dtb_potential_drops.min()),
        "min_reference_potential_increment": float(reference_potential_drops.min()),
        "dtb_substantial_potential_drop": bool(
            dtb_potential_drops.min() < -args.potential_drop_tolerance
        ),
        "reference_substantial_potential_drop": bool(
            reference_potential_drops.min() < -args.potential_drop_tolerance
        ),
        "reference_runtime_seconds": reference_runtime,
        "dtb_runtime_seconds": dtb_runtime,
        "total_runtime_seconds": total_runtime,
        "model_initialization_sha256": initialization_hash,
        "output_directory": str(output_dir),
    }
    return summary, history_path


def _effective_basin_tolerance(
    stable_equilibria: np.ndarray, requested_tolerance: float
) -> float:
    """Prevent basin-assignment balls from overlapping nearby stable roots."""

    if len(stable_equilibria) < 2:
        return requested_tolerance
    distances = np.linalg.norm(
        stable_equilibria[:, None, :] - stable_equilibria[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    return min(requested_tolerance, 0.45 * float(distances.min()))


def _save_equilibria_table(
    path: Path,
    equilibria: np.ndarray,
    stable: np.ndarray,
    spectral_abscissa: np.ndarray,
    dtb_masses: np.ndarray,
    reference_masses: np.ndarray,
) -> None:
    stable_index = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "equilibrium", "x1", "x2", "stable", "spectral_abscissa",
                "dtb_basin_mass", "reference_basin_mass",
            ),
        )
        writer.writeheader()
        for index, (root, is_stable, abscissa) in enumerate(
            zip(equilibria, stable, spectral_abscissa)
        ):
            row = {
                "equilibrium": index,
                "x1": root[0],
                "x2": root[1],
                "stable": bool(is_stable),
                "spectral_abscissa": abscissa,
                "dtb_basin_mass": "",
                "reference_basin_mass": "",
            }
            if is_stable:
                row["dtb_basin_mass"] = dtb_masses[stable_index]
                row["reference_basin_mass"] = reference_masses[stable_index]
                stable_index += 1
            writer.writerow(row)


def _plot_landscape(
    path: Path,
    params: OscillatoryGameParams,
    equilibria: np.ndarray,
    stable: np.ndarray,
) -> None:
    grid = np.linspace(-1.0, 1.0, 161)
    first, second = np.meshgrid(grid, grid)
    points = torch.as_tensor(
        np.stack((first, second), axis=-1), dtype=torch.float64
    )
    potential = oscillatory_potential(points, params).numpy()
    vector = oscillatory_vector_field(points, params).numpy()
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    contours = axis.contourf(first, second, potential, levels=32, cmap="viridis")
    stride = 8
    axis.quiver(
        first[::stride, ::stride],
        second[::stride, ::stride],
        vector[::stride, ::stride, 0],
        vector[::stride, ::stride, 1],
        color="white",
        alpha=0.65,
        scale=28,
    )
    if len(equilibria):
        marker_size = max(7.0, min(38.0, 3000.0 / len(equilibria)))
        if stable.any():
            axis.scatter(
                equilibria[stable, 0], equilibria[stable, 1], s=marker_size,
                color="#d62728", marker="o", label="stable equilibrium", zorder=5,
            )
        if (~stable).any():
            axis.scatter(
                equilibria[~stable, 0], equilibria[~stable, 1],
                s=1.35 * marker_size,
                color="#ffbf00", edgecolors="#222222", marker="X",
                label="unstable equilibrium", zorder=6,
            )
    axis.set(xlim=(-1, 1), ylim=(-1, 1), xlabel=r"$x_1$", ylabel=r"$x_2$")
    axis.set_aspect("equal")
    axis.set_title(
        rf"Potential and pseudo-gradient: $\omega={params.omega / np.pi:g}\pi$, "
        rf"$\gamma={params.gamma:g}$"
    )
    if len(equilibria):
        axis.legend(frameon=False, loc="upper right")
    figure.colorbar(contours, ax=axis, label=r"$\Phi_{\omega,\gamma}$")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_particle_comparison(
    path: Path,
    times: np.ndarray,
    reference: np.ndarray,
    dtb: np.ndarray,
    equilibria: np.ndarray,
    stable: np.ndarray,
) -> None:
    indices = [0, (len(times) - 1) // 2, len(times) - 1]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.4), sharex=True, sharey=True)
    marker_size = (
        max(5.0, min(28.0, 2200.0 / len(equilibria)))
        if len(equilibria) else 28.0
    )
    for row, (name, history) in enumerate((("Reference RK4", reference), ("Neural--DTB", dtb))):
        for column, index in enumerate(indices):
            axis = axes[row, column]
            axis.scatter(
                history[index, :, 0], history[index, :, 1], s=3.0,
                alpha=0.27, color="#1786c7", edgecolors="none", rasterized=True,
            )
            if len(equilibria):
                if stable.any():
                    axis.scatter(
                        equilibria[stable, 0], equilibria[stable, 1],
                        s=marker_size,
                        color="#d62728", marker="o", zorder=5,
                    )
                if (~stable).any():
                    axis.scatter(
                        equilibria[~stable, 0], equilibria[~stable, 1],
                        s=1.35 * marker_size,
                        color="#ffbf00", edgecolors="#222222", linewidths=0.5,
                        marker="X", zorder=6,
                    )
            axis.set_title(f"{name}, t={times[index]:g}")
            axis.set_xlim(-1.05, 1.05)
            axis.set_ylim(-1.05, 1.05)
            axis.set_aspect("equal")
            axis.grid(alpha=0.25)
            if row == 1:
                axis.set_xlabel(r"$x_1$")
            if column == 0:
                axis.set_ylabel(r"$x_2$")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_case_diagnostics(path: Path, history_path: Path) -> None:
    with np.load(history_path) as data:
        times = data["times"]
        figure, axes = plt.subplots(2, 3, figsize=(13, 7.8))
        axes[0, 0].plot(times[1:], data["projection_residuals"])
        axes[0, 0].set_title("Relative tangent projection residual")
        axes[0, 1].plot(times, data["trajectory_rmse"], label="absolute")
        axes[0, 1].plot(times, data["trajectory_relative"], label="relative")
        axes[0, 1].set_title("Trajectory error")
        axes[0, 1].legend(frameon=False)
        axes[0, 2].plot(times[1:], data["velocity_rmse"])
        axes[0, 2].set_title("Velocity approximation RMSE")
        axes[1, 0].plot(times[1:], data["retained_rank"], label="rank")
        axes[1, 0].set_title("Truncated-SVD retained rank")
        axes[1, 1].semilogy(times[1:], data["retained_condition"])
        axes[1, 1].set_title("Retained condition number")
        axes[1, 2].plot(times, data["reference_mean_potential"], label="reference")
        axes[1, 2].plot(times, data["dtb_mean_potential"], label="DTB")
        axes[1, 2].set_title("Mean potential")
        axes[1, 2].legend(frameon=False)
    for axis in axes.flat:
        axis.set_xlabel("physical time")
        axis.grid(alpha=0.28)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_group_diagnostics(
    path: Path, gamma: float, cases: list[tuple[dict[str, Any], Path]]
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for summary, history_path in cases:
        label = rf"${summary['omega_multiplier_pi']:g}\pi$"
        with np.load(history_path) as data:
            times = data["times"]
            axes[0, 0].plot(times[1:], data["projection_residuals"], label=label)
            axes[0, 1].plot(times, data["trajectory_rmse"], label=label)
            axes[1, 0].plot(times[1:], data["retained_rank"], label=label)
            axes[1, 1].semilogy(times[1:], data["retained_condition"], label=label)
    titles = (
        "Projection residual", "Trajectory RMSE",
        "Retained SVD rank", "Retained condition number",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("physical time")
        axis.grid(alpha=0.28)
        axis.legend(frameon=False, ncol=2)
    figure.suptitle(rf"Controlled frequency comparison, $\gamma={gamma:g}$")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_frequency_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for gamma in sorted({float(row["gamma"]) for row in summaries}):
        rows = sorted(
            (row for row in summaries if float(row["gamma"]) == gamma),
            key=lambda row: float(row["omega_multiplier_pi"]),
        )
        frequency = np.asarray([row["omega_multiplier_pi"] for row in rows])
        axes[0].plot(
            frequency, [row["mean_projection_residual"] for row in rows],
            marker="o", label=rf"$\gamma={gamma:g}$",
        )
        axes[1].plot(
            frequency, [row["final_trajectory_rmse"] for row in rows], marker="o"
        )
        basin = np.asarray([row["basin_mass_error"] for row in rows], dtype=float)
        finite = np.isfinite(basin)
        if finite.any():
            axes[2].plot(frequency[finite], basin[finite], marker="o")
    for axis, title, ylabel in zip(
        axes,
        ("Mean projection residual", "Final trajectory RMSE", "Basin-mass error"),
        (r"mean $r_{proj}$", r"$E_{traj}(T)$", r"$E_{basin}$"),
    ):
        axis.set_title(title)
        axis.set_xlabel(r"frequency multiplier in $\omega=m\pi$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.28)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_summary_tables(root: Path, summaries: list[dict[str, Any]]) -> None:
    fields = list(summaries[0])
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Oscillatory potential-game sweep summary",
        "",
        "| gamma | omega | mean projection residual | final trajectory RMSE | basin error | rank mean | condition max | runtime (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        basin = row["basin_mass_error"]
        basin_text = f"{basin:.3e}" if np.isfinite(basin) else "unreliable"
        lines.append(
            f"| {row['gamma']:.3g} | {row['omega_multiplier_pi']:g}pi | "
            f"{row['mean_projection_residual']:.3e} | "
            f"{row['final_trajectory_rmse']:.3e} | {basin_text} | "
            f"{row['mean_retained_rank']:.2f} | "
            f"{row['max_retained_condition']:.3e} | "
            f"{row['total_runtime_seconds']:.2f} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def safe_case_name(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke_preset(args)
    validate_args(args)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    initial_particles = sample_uniform_initial_particles(args.particles, args.seed)
    np.save(root / "initial_particles.npy", initial_particles)
    (root / "sweep_config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )

    summaries: list[dict[str, Any]] = []
    grouped_cases: dict[float, list[tuple[dict[str, Any], Path]]] = {}
    for gamma in args.gammas:
        grouped_cases[gamma] = []
        for omega_multiplier in args.omega_multipliers:
            case_dir = (
                root
                / f"gamma_{safe_case_name(gamma)}"
                / f"omega_{safe_case_name(omega_multiplier)}pi"
            )
            print(
                f"=== gamma={gamma:g}, omega={omega_multiplier:g}*pi, "
                f"N={args.particles} ==="
            )
            summary, history_path = run_case(
                args,
                initial_particles,
                gamma=gamma,
                omega_multiplier=omega_multiplier,
                output_dir=case_dir,
            )
            summaries.append(summary)
            grouped_cases[gamma].append((summary, history_path))
            print(
                f"  mean residual={summary['mean_projection_residual']:.3e}, "
                f"final trajectory RMSE={summary['final_trajectory_rmse']:.3e}"
            )

    hashes = {row["model_initialization_sha256"] for row in summaries}
    if len(hashes) != 1:
        raise RuntimeError("controlled comparison violated: NN initialization changed")
    for cases in grouped_cases.values():
        for _, history_path in cases:
            with np.load(history_path) as history:
                if not np.array_equal(
                    initial_particles.astype(history["dtb_particles"].dtype),
                    history["dtb_particles"][0],
                ):
                    raise RuntimeError(
                        "controlled comparison violated: initial particles changed"
                    )

    for gamma, cases in grouped_cases.items():
        _plot_group_diagnostics(
            root / f"gamma_{safe_case_name(gamma)}_frequency_comparison.png",
            gamma,
            cases,
        )
    _plot_frequency_summary(root / "final_error_vs_frequency.png", summaries)
    _write_summary_tables(root, summaries)
    print(f"saved controlled sweep under {root}")
    print(f"summary table: {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
