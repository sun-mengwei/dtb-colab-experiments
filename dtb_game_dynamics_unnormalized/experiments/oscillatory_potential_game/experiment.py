"""Controlled Neural--DTB sweep for an oscillatory potential game.

The experiment intentionally leaves the core DTB algorithm unchanged.  It
uses one particle realization, one neural-network initialization, and one DTB
configuration for every ``(gamma, omega)`` pair, while comparing the particle
paths with a tighter vectorized RK4 integration of the true game ODE.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from game_dtb.algorithm import DTBConfig, NeuralDTBGameDynamics
from game_dtb.games import (
    OscillatoryGameParams,
    oscillatory_game_jacobian,
    oscillatory_game_velocity,
    oscillatory_potential,
)
from game_dtb.models import TangentMLP, TangentMMNN, TangentNODE
from game_dtb.runner import DTBTrajectory, resolve_device, run_dtb_trajectory
from game_dtb.state import ParticleState, uniform_box_particle_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled Neural--DTB oscillatory potential-game sweep"
    )
    parser.add_argument(
        "--omega-multipliers", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0],
        help="frequencies represented as multiples of pi",
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.5)

    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-seed", type=int, default=91)

    parser.add_argument(
        "--architecture", choices=("mlp", "mmnn", "node"), default="mlp"
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--activation", choices=("tanh", "gelu", "silu"), default="tanh")
    parser.add_argument("--node-inner-steps", type=int, default=4)
    parser.add_argument("--node-integration-time", type=float, default=1.0)
    parser.add_argument("--basis-size", type=int, default=128)
    parser.add_argument("--svd-rtol", type=float, default=1e-3)
    parser.add_argument("--jacobian-chunk", type=int, default=128)
    parser.add_argument("--derivative-chunk", type=int, default=64)

    parser.add_argument("--refit-interval", type=int, default=0)
    parser.add_argument("--refit-optimizer-steps", type=int, default=2000)
    parser.add_argument("--refit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--refit-batch-size", type=int, default=2048)
    parser.add_argument("--refit-samples", type=int, default=10000)
    parser.add_argument("--refit-test-samples", type=int, default=4000)

    parser.add_argument("--reference-substeps", type=int, default=20)
    parser.add_argument("--max-reference-self-error", type=float, default=1e-8)
    parser.add_argument("--skip-reference-check", action="store_true")
    parser.add_argument("--equilibrium-grid-size", type=int, default=None)
    parser.add_argument("--basin-tolerance", type=float, default=0.08)
    parser.add_argument("--max-unassigned-fraction", type=float, default=0.05)
    parser.add_argument("--potential-drop-tolerance", type=float, default=1e-4)

    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="outputs/oscillatory_potential_game")
    parser.add_argument("--print-every", type=int, default=40)
    parser.add_argument(
        "--smoke", action="store_true",
        help="retain all requested cases but use a tiny end-to-end configuration",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_integers = (
        "particles", "steps", "width", "depth", "rank", "basis_size",
        "jacobian_chunk", "derivative_chunk", "node_inner_steps",
        "refit_optimizer_steps", "refit_batch_size", "refit_samples",
        "refit_test_samples", "reference_substeps",
    )
    for name in positive_integers:
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
    if args.max_reference_self_error <= 0.0:
        raise ValueError("max_reference_self_error must be positive")
    if not 0.0 <= args.max_unassigned_fraction < 1.0:
        raise ValueError("max_unassigned_fraction must lie in [0,1)")


def apply_smoke_preset(args: argparse.Namespace) -> None:
    """Make an eight-case workflow check inexpensive without changing its order."""

    args.particles = min(args.particles, 24)
    args.steps = min(args.steps, 3)
    args.width = min(args.width, 6)
    args.depth = min(args.depth, 1)
    args.rank = min(args.rank, 3)
    args.basis_size = min(args.basis_size, 10)
    args.jacobian_chunk = min(args.jacobian_chunk, 24)
    args.derivative_chunk = min(args.derivative_chunk, 12)
    args.reference_substeps = min(args.reference_substeps, 4)
    args.refit_interval = 0
    args.equilibrium_grid_size = args.equilibrium_grid_size or 33
    args.print_every = 0


def make_model(args: argparse.Namespace, dtype: torch.dtype) -> torch.nn.Module:
    """Initialize one of the existing tangent models with the shared settings."""

    if args.architecture == "mlp":
        return TangentMLP(
            dim=2, width=args.width, depth=args.depth,
            activation=args.activation, dtype=dtype,
        )
    if args.architecture == "mmnn":
        return TangentMMNN(
            dim=2, width=args.width, rank=args.rank, depth=args.depth,
            activation=args.activation, dtype=dtype,
        )
    return TangentNODE(
        dim=2, width=args.width, depth=args.depth,
        activation=args.activation, inner_steps=args.node_inner_steps,
        integration_time=args.node_integration_time, dtype=dtype,
    )


def model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def clone_state(
    source: ParticleState, device: torch.device, dtype: torch.dtype
) -> ParticleState:
    state = ParticleState(
        particles=source.particles.to(device=device, dtype=dtype).clone(),
        log_density=source.log_density.to(device=device, dtype=dtype).clone(),
        score=source.score.to(device=device, dtype=dtype).clone(),
        labels=source.labels.to(device=device, dtype=dtype).clone(),
    )
    state.validate()
    return state


@torch.no_grad()
def integrate_reference_rk4(
    initial_particles: torch.Tensor,
    params: OscillatoryGameParams,
    *,
    step_size: float,
    steps: int,
    substeps_per_step: int,
    save_history: bool = True,
) -> torch.Tensor:
    """Integrate the true game ODE with a tighter fixed-step RK4 scheme."""

    if step_size <= 0.0 or steps < 1 or substeps_per_step < 1:
        raise ValueError("step_size, steps, and substeps_per_step must be positive")
    state = initial_particles.clone()
    history = [state.clone()] if save_history else []
    reference_step = step_size / substeps_per_step
    for _ in range(steps):
        for _ in range(substeps_per_step):
            k1 = oscillatory_game_velocity(state, params)
            k2 = oscillatory_game_velocity(state + 0.5 * reference_step * k1, params)
            k3 = oscillatory_game_velocity(state + 0.5 * reference_step * k2, params)
            k4 = oscillatory_game_velocity(state + reference_step * k3, params)
            state = state + (reference_step / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
        if save_history:
            history.append(state.clone())
    return torch.stack(history) if save_history else state


def run_case(
    args: argparse.Namespace,
    initial_state_cpu: ParticleState,
    *,
    gamma: float,
    omega_multiplier: float,
    result_dir: Path,
    figure_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Run one controlled DTB/reference comparison and save its artifacts."""

    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    params = OscillatoryGameParams(
        lambda_=args.lambda_value,
        epsilon=args.epsilon,
        omega=omega_multiplier * math.pi,
        gamma=gamma,
    )
    params.validate()
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    start_total = time.perf_counter()

    reference_initial = initial_state_cpu.particles.to(dtype=torch.float64)
    start_reference = time.perf_counter()
    reference_tensor = integrate_reference_rk4(
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
        refined_final = integrate_reference_rk4(
            reference_initial,
            params,
            step_size=args.step_size,
            steps=args.steps,
            substeps_per_step=2 * args.reference_substeps,
            save_history=False,
        )
        reference_self_error = float(
            torch.sqrt(torch.mean((reference_tensor[-1] - refined_final).square()))
        )
        if reference_self_error > args.max_reference_self_error:
            raise RuntimeError(
                "reference integration did not pass its refinement check: "
                f"RMSE={reference_self_error:.3e} exceeds "
                f"{args.max_reference_self_error:.3e}; increase --reference-substeps"
            )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.model_seed)
        model = make_model(args, dtype)
    initialization_hash = model_state_hash(model)
    model = model.to(device)
    state = clone_state(initial_state_cpu, device, dtype)
    diffusion = torch.zeros((2, 2), device=device, dtype=dtype)

    def reference_sampler(
        count: int, generator: torch.Generator
    ) -> torch.Tensor:
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
        model=model,
        drift=lambda values: oscillatory_game_velocity(values, params),
        diffusion=diffusion,
        config=config,
        reference_sampler=reference_sampler,
    )
    start_dtb = time.perf_counter()
    trajectory = run_dtb_trajectory(
        method,
        state,
        args.steps,
        potential=lambda values: oscillatory_potential(values, params),
        print_every=args.print_every,
    )
    dtb_runtime = time.perf_counter() - start_dtb
    reference = reference_tensor.cpu().numpy()

    difference = trajectory.particles.astype(np.float64) - reference
    trajectory_rmse = np.sqrt(np.mean(np.sum(difference * difference, axis=2), axis=1))
    relative_denominator = np.sqrt(np.sum(reference * reference, axis=(1, 2)))
    trajectory_relative = np.sqrt(np.sum(difference * difference, axis=(1, 2))) / np.maximum(
        relative_denominator, 1e-30
    )
    reference_mean_potential = (
        oscillatory_potential(reference_tensor, params).mean(dim=1).cpu().numpy()
    )

    equilibria, stable, spectral_abscissa = locate_equilibria(
        params, grid_size=args.equilibrium_grid_size
    )
    stable_equilibria = equilibria[stable]
    effective_tolerance = effective_basin_tolerance(
        stable_equilibria, args.basin_tolerance
    )
    dtb_masses, dtb_unassigned = basin_masses(
        trajectory.particles[-1], stable_equilibria, effective_tolerance
    )
    reference_masses, reference_unassigned = basin_masses(
        reference[-1], stable_equilibria, effective_tolerance
    )
    basin_reliable = (
        dtb_unassigned <= args.max_unassigned_fraction
        and reference_unassigned <= args.max_unassigned_fraction
    )
    basin_error = (
        0.5 * float(np.abs(dtb_masses - reference_masses).sum())
        if basin_reliable else float("nan")
    )

    dtb_mean_potential = trajectory.mean_potential
    assert dtb_mean_potential is not None
    dtb_potential_increment = np.diff(dtb_mean_potential)
    reference_potential_increment = np.diff(reference_mean_potential)
    history_path = result_dir / "history.npz"
    np.savez_compressed(
        history_path,
        times=trajectory.times,
        initial_particles=initial_state_cpu.particles.numpy(),
        dtb_particles=trajectory.particles,
        reference_particles=reference,
        trajectory_rmse=trajectory_rmse,
        trajectory_relative=trajectory_relative,
        projection_residuals=trajectory.projection_residuals,
        velocity_rmse=trajectory.velocity_rmse,
        sigma_max=trajectory.sigma_max,
        sigma_min_retained=trajectory.sigma_min_retained,
        retained_rank=trajectory.retained_rank,
        retained_condition=trajectory.retained_condition,
        truncated_count=trajectory.truncated_count,
        truncated_fraction=trajectory.truncated_fraction,
        alpha_norm=trajectory.alpha_norm,
        target_velocity_norm=trajectory.target_velocity_norm,
        projected_velocity_norm=trajectory.projected_velocity_norm,
        mean_divergence=trajectory.mean_divergence,
        dtb_mean_potential=dtb_mean_potential,
        reference_mean_potential=reference_mean_potential,
        equilibria=equilibria,
        equilibrium_stable=stable,
        spectral_abscissa=spectral_abscissa,
        dtb_basin_masses=dtb_masses,
        reference_basin_masses=reference_masses,
        effective_basin_tolerance=np.asarray(effective_tolerance),
        refit_performed=trajectory.refit_performed,
        refit_rmse_before=trajectory.refit_rmse_before,
        refit_rmse_after=trajectory.refit_rmse_after,
    )
    save_equilibria_table(
        result_dir / "equilibria.csv", equilibria, stable, spectral_abscissa,
        dtb_masses, reference_masses,
    )

    case_config = {
        **vars(args),
        "game_parameters": asdict(params),
        "omega_multiplier_pi": omega_multiplier,
        "resolved_device": str(device),
        "model_initialization_sha256": initialization_hash,
        "trainable_parameters": method.parameter_count,
        "actual_basis_size": min(args.basis_size, method.parameter_count),
        "diffusion_matrix": [[0.0, 0.0], [0.0, 0.0]],
        "reference_method": "vectorized fixed-step classical RK4",
    }
    (result_dir / "config.json").write_text(
        json.dumps(case_config, indent=2) + "\n", encoding="utf-8"
    )

    plot_landscape(
        figure_dir / "potential_vector_field.png", params, equilibria, stable
    )
    plot_particle_comparison(
        figure_dir / "particle_comparison.png", trajectory.times,
        reference, trajectory.particles, equilibria, stable,
    )
    plot_case_diagnostics(
        figure_dir / "diagnostics.png", trajectory, trajectory_rmse,
        trajectory_relative, reference_mean_potential,
    )
    total_runtime = time.perf_counter() - start_total

    summary = {
        "gamma": gamma,
        "omega_multiplier_pi": omega_multiplier,
        "omega": params.omega,
        "particles": args.particles,
        "step_size": args.step_size,
        "final_time": args.steps * args.step_size,
        "mean_projection_residual": float(trajectory.projection_residuals.mean()),
        "max_projection_residual": float(trajectory.projection_residuals.max()),
        "final_projection_residual": float(trajectory.projection_residuals[-1]),
        "final_velocity_rmse": float(trajectory.velocity_rmse[-1]),
        "final_trajectory_rmse": float(trajectory_rmse[-1]),
        "mean_trajectory_rmse": float(trajectory_rmse[1:].mean()),
        "final_relative_trajectory_error": float(trajectory_relative[-1]),
        "equilibrium_count": len(equilibria),
        "stable_equilibrium_count": int(stable.sum()),
        "effective_basin_tolerance": effective_tolerance,
        "basin_assignment_reliable": basin_reliable,
        "dtb_unassigned_fraction": dtb_unassigned,
        "reference_unassigned_fraction": reference_unassigned,
        "basin_mass_error": basin_error,
        "mean_retained_rank": float(trajectory.retained_rank.mean()),
        "min_retained_rank": int(trajectory.retained_rank.min()),
        "max_retained_rank": int(trajectory.retained_rank.max()),
        "mean_truncated_fraction": float(trajectory.truncated_fraction.mean()),
        "mean_retained_condition": float(trajectory.retained_condition.mean()),
        "max_retained_condition": float(trajectory.retained_condition.max()),
        "max_alpha_norm": float(trajectory.alpha_norm.max()),
        "reference_self_error": reference_self_error,
        "min_dtb_potential_increment": float(dtb_potential_increment.min()),
        "min_reference_potential_increment": float(reference_potential_increment.min()),
        "dtb_substantial_potential_drop": bool(
            dtb_potential_increment.min() < -args.potential_drop_tolerance
        ),
        "reference_substantial_potential_drop": bool(
            reference_potential_increment.min() < -args.potential_drop_tolerance
        ),
        "reference_runtime_seconds": reference_runtime,
        "dtb_runtime_seconds": dtb_runtime,
        "total_runtime_seconds": total_runtime,
        "model_initialization_sha256": initialization_hash,
        "result_directory": str(result_dir),
        "figure_directory": str(figure_dir),
    }
    return summary, history_path


@torch.no_grad()
def locate_equilibria(
    params: OscillatoryGameParams,
    *,
    grid_size: int | None = None,
    residual_tolerance: float = 1e-9,
    duplicate_tolerance: float = 1e-6,
    max_iterations: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically enumerate roots in ``[-1,1]^2`` and classify stability."""

    params.validate()
    if grid_size is None:
        cycles_across_box = params.omega / math.pi
        grid_size = max(65, int(math.ceil(8.0 * cycles_across_box)) + 1)
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")

    grid = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float64)
    first, second = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack((first.reshape(-1), second.reshape(-1)), dim=1)
    for _ in range(max_iterations):
        field = oscillatory_game_velocity(points, params)
        jacobian = oscillatory_game_jacobian(points, params)
        a = jacobian[:, 0, 0]
        coupling = jacobian[:, 0, 1]
        d = jacobian[:, 1, 1]
        determinant = a * d - coupling.square()
        safe = determinant.abs() > 1e-10
        direction = torch.empty_like(points)
        direction[safe, 0] = (
            d[safe] * field[safe, 0] - coupling[safe] * field[safe, 1]
        ) / determinant[safe]
        direction[safe, 1] = (
            -coupling[safe] * field[safe, 0] + a[safe] * field[safe, 1]
        ) / determinant[safe]
        if bool((~safe).any()):
            gradient = torch.einsum("nji,nj->ni", jacobian[~safe], field[~safe])
            scale = torch.linalg.vector_norm(
                gradient, dim=1, keepdim=True
            ).clamp_min(1e-12)
            direction[~safe] = 0.05 * gradient / scale
        norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        direction = direction * torch.clamp(0.25 / norm.clamp_min(1e-12), max=1.0)
        points = (points - direction).clamp(-1.05, 1.05)

    residual = torch.linalg.vector_norm(
        oscillatory_game_velocity(points, params), dim=1
    )
    inside = (points.abs() <= 1.0 + duplicate_tolerance).all(dim=1)
    candidates = points[(residual <= residual_tolerance) & inside].cpu().numpy()
    if not len(candidates):
        return np.empty((0, 2)), np.empty((0,), dtype=bool), np.empty((0,))

    roots: list[np.ndarray] = []
    for candidate in candidates[np.lexsort((candidates[:, 1], candidates[:, 0]))]:
        if not any(
            np.linalg.norm(candidate - existing) <= duplicate_tolerance
            for existing in roots
        ):
            roots.append(candidate)
    root_array = np.asarray(roots)
    jacobians = oscillatory_game_jacobian(
        torch.as_tensor(root_array, dtype=torch.float64), params
    )
    eigenvalues = torch.linalg.eigvalsh(jacobians).cpu().numpy()
    spectral_abscissa = eigenvalues[:, -1]
    stable = spectral_abscissa < -1e-8
    return root_array, stable, spectral_abscissa


def effective_basin_tolerance(
    stable_equilibria: np.ndarray, requested: float
) -> float:
    """Keep nearest-equilibrium assignment balls from overlapping."""

    if len(stable_equilibria) < 2:
        return requested
    distances = np.linalg.norm(
        stable_equilibria[:, None, :] - stable_equilibria[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    return min(requested, 0.45 * float(distances.min()))


def basin_masses(
    particles: np.ndarray, stable_equilibria: np.ndarray, tolerance: float
) -> tuple[np.ndarray, float]:
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not len(stable_equilibria):
        return np.empty((0,)), 1.0
    distances = np.linalg.norm(
        particles[:, None, :] - stable_equilibria[None, :, :], axis=2
    )
    nearest = distances.argmin(axis=1)
    assigned = distances[np.arange(len(particles)), nearest] <= tolerance
    counts = np.bincount(nearest[assigned], minlength=len(stable_equilibria))
    return counts / len(particles), float(1.0 - assigned.mean())


def save_equilibria_table(
    path: Path,
    equilibria: np.ndarray,
    stable: np.ndarray,
    spectral_abscissa: np.ndarray,
    dtb_masses: np.ndarray,
    reference_masses: np.ndarray,
) -> None:
    stable_index = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
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
            row: dict[str, Any] = {
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


def plot_landscape(
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
    velocity = oscillatory_game_velocity(points, params).numpy()
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    contours = axis.contourf(first, second, potential, levels=32, cmap="viridis")
    stride = 8
    axis.quiver(
        first[::stride, ::stride], second[::stride, ::stride],
        velocity[::stride, ::stride, 0], velocity[::stride, ::stride, 1],
        color="white", alpha=0.65, scale=28,
    )
    plot_equilibria(axis, equilibria, stable)
    axis.set(
        xlim=(-1, 1), ylim=(-1, 1), xlabel=r"$x_1$", ylabel=r"$x_2$",
        title=(
            rf"Potential and velocity: $\omega={params.omega / math.pi:g}\pi$, "
            rf"$\gamma={params.gamma:g}$"
        ),
    )
    axis.set_aspect("equal")
    if len(equilibria):
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles, labels, frameon=False, loc="upper center", ncol=2,
            bbox_to_anchor=(0.47, 0.995),
        )
    figure.colorbar(contours, ax=axis, label=r"$\Phi_{\omega,\gamma}$")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_equilibria(
    axis: plt.Axes, equilibria: np.ndarray, stable: np.ndarray
) -> None:
    if not len(equilibria):
        return
    marker_size = max(6.0, min(36.0, 2600.0 / len(equilibria)))
    if stable.any():
        axis.scatter(
            equilibria[stable, 0], equilibria[stable, 1], s=marker_size,
            color="#d62728", marker="o", label="stable equilibrium", zorder=5,
        )
    if (~stable).any():
        axis.scatter(
            equilibria[~stable, 0], equilibria[~stable, 1],
            s=1.35 * marker_size, color="#ffbf00", edgecolors="#222222",
            marker="X", label="unstable equilibrium", zorder=6,
        )


def plot_particle_comparison(
    path: Path,
    times: np.ndarray,
    reference: np.ndarray,
    dtb: np.ndarray,
    equilibria: np.ndarray,
    stable: np.ndarray,
) -> None:
    indices = [0, (len(times) - 1) // 2, len(times) - 1]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.4), sharex=True, sharey=True)
    show_equilibria = len(equilibria) <= 100
    point_count = reference.shape[1]
    point_size = 12.0 if point_count <= 100 else 3.0
    point_alpha = 0.58 if point_count <= 100 else 0.27
    for row, (name, history) in enumerate(
        (("Reference RK4", reference), ("Neural--DTB", dtb))
    ):
        for column, index in enumerate(indices):
            axis = axes[row, column]
            axis.scatter(
                history[index, :, 0], history[index, :, 1], s=point_size,
                alpha=point_alpha, color="#1786c7", edgecolors="none",
                rasterized=True,
            )
            if show_equilibria:
                plot_equilibria(axis, equilibria, stable)
            axis.set_title(f"{name}, t={times[index]:g}")
            axis.set_xlim(-1.05, 1.05)
            axis.set_ylim(-1.05, 1.05)
            axis.set_aspect("equal")
            axis.grid(alpha=0.25)
            if row == 1:
                axis.set_xlabel(r"$x_1$")
            if column == 0:
                axis.set_ylabel(r"$x_2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    elif len(equilibria):
        figure.text(
            0.5, 0.985,
            f"{len(equilibria)} equilibrium markers omitted for particle readability; "
            "see the landscape panel",
            ha="center", va="top", fontsize=9,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_case_diagnostics(
    path: Path,
    trajectory: DTBTrajectory,
    trajectory_rmse: np.ndarray,
    trajectory_relative: np.ndarray,
    reference_mean_potential: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    axes[0, 0].plot(trajectory.times[1:], trajectory.projection_residuals)
    axes[0, 0].set_title("Relative tangent projection residual")
    axes[0, 1].plot(trajectory.times, trajectory_rmse, label="absolute")
    axes[0, 1].plot(trajectory.times, trajectory_relative, label="relative")
    axes[0, 1].set_title("Trajectory error")
    axes[0, 1].legend(frameon=False)
    axes[0, 2].plot(trajectory.times[1:], trajectory.velocity_rmse)
    axes[0, 2].set_title("Velocity approximation RMSE")
    axes[1, 0].plot(trajectory.times[1:], trajectory.retained_rank)
    axes[1, 0].set_title("Truncated-SVD retained rank")
    axes[1, 1].semilogy(trajectory.times[1:], trajectory.retained_condition)
    axes[1, 1].set_title("Retained condition number")
    axes[1, 2].plot(trajectory.times, reference_mean_potential, label="reference")
    assert trajectory.mean_potential is not None
    axes[1, 2].plot(trajectory.times, trajectory.mean_potential, label="DTB")
    axes[1, 2].set_title("Mean potential")
    axes[1, 2].legend(frameon=False)
    for axis in axes.flat:
        axis.set_xlabel("physical time")
        axis.grid(alpha=0.28)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_group_diagnostics(
    path: Path, gamma: float, cases: list[tuple[dict[str, Any], Path]]
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for summary, history_path in cases:
        label = rf"${summary['omega_multiplier_pi']:g}\pi$"
        with np.load(history_path) as history:
            times = history["times"]
            axes[0, 0].plot(times[1:], history["projection_residuals"], label=label)
            axes[0, 1].plot(times, history["trajectory_rmse"], label=label)
            axes[1, 0].plot(times[1:], history["retained_rank"], label=label)
            axes[1, 1].semilogy(
                times[1:], np.maximum(history["retained_condition"], 1.0), label=label
            )
    for axis, title in zip(
        axes.flat,
        (
            "Projection residual", "Trajectory RMSE",
            "Retained SVD rank", "Retained condition number",
        ),
    ):
        axis.set_title(title)
        axis.set_xlabel("physical time")
        axis.grid(alpha=0.28)
        axis.legend(frameon=False, ncol=2)
    figure.suptitle(rf"Controlled frequency comparison, $\gamma={gamma:g}$")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_frequency_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
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


def write_summary_tables(table_dir: Path, summaries: list[dict[str, Any]]) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    fields = list(summaries[0])
    with (table_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Oscillatory potential-game sweep summary",
        "",
        "| gamma | omega | mean projection residual | final trajectory RMSE | "
        "basin error | rank mean | condition max | runtime (s) |",
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
    (table_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.allclose(first, first[0]) or np.allclose(second, second[0]):
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def write_scientific_report(root: Path, summaries: list[dict[str, Any]]) -> None:
    projection = np.asarray([row["mean_projection_residual"] for row in summaries])
    trajectory = np.asarray([row["final_trajectory_rmse"] for row in summaries])
    condition = np.asarray([row["max_retained_condition"] for row in summaries])
    rank = np.asarray([row["mean_retained_rank"] for row in summaries])
    projection_correlation = _correlation(projection, trajectory)
    condition_correlation = _correlation(condition, trajectory)
    rank_correlation = _correlation(rank, trajectory)

    lines = [
        "# Automated numerical readout",
        "",
        "This report describes the saved sweep only; correlation is not treated as causation.",
        "",
        "## Frequency trends",
        "",
    ]
    for gamma in sorted({float(row["gamma"]) for row in summaries}):
        rows = sorted(
            (row for row in summaries if float(row["gamma"]) == gamma),
            key=lambda row: float(row["omega_multiplier_pi"]),
        )
        residuals = np.asarray([row["mean_projection_residual"] for row in rows])
        errors = np.asarray([row["final_trajectory_rmse"] for row in rows])
        residual_trend = (
            "monotone increasing"
            if np.all(np.diff(residuals) >= 0)
            else "not monotone"
        )
        error_trend = (
            "monotone increasing"
            if np.all(np.diff(errors) >= 0)
            else "not monotone"
        )
        lines.append(
            f"- gamma={gamma:g}: projection residual is {residual_trend}; "
            f"final trajectory error is {error_trend}."
        )
    lines.extend(
        [
            "",
            "## Cross-case diagnostics",
            "",
            "- Correlation(mean projection residual, final trajectory RMSE): "
            f"{projection_correlation:.4f}",
            "- Correlation(max retained condition, final trajectory RMSE): "
            f"{condition_correlation:.4f}",
            f"- Correlation(mean retained rank, final trajectory RMSE): {rank_correlation:.4f}",
            "",
            "## Basin and potential checks",
            "",
        ]
    )
    reliable = [row for row in summaries if row["basin_assignment_reliable"]]
    lines.append(
        f"- Reliable basin assignment: {len(reliable)}/{len(summaries)} cases."
    )
    lines.append(
        f"- DTB cases with a flagged potential decrease: "
        f"{sum(bool(row['dtb_substantial_potential_drop']) for row in summaries)}."
    )
    lines.append(
        f"- Reference cases with a flagged potential decrease: "
        f"{sum(bool(row['reference_substantial_potential_drop']) for row in summaries)}."
    )
    lines.extend(
        [
            "",
            "Interpret the smoke preset only as a workflow check. Scientific "
            "conclusions require the full particle/time configuration and a "
            "sufficiently small reference self-error.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_sweep(args: argparse.Namespace) -> Path:
    if args.smoke:
        apply_smoke_preset(args)
    validate_args(args)
    root = Path(args.output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"output root already contains files: {root}; choose a new --output-root"
        )
    result_root = root / "results"
    figure_root = root / "figures"
    table_root = root / "tables"
    result_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    initial_state = uniform_box_particle_state(
        args.particles,
        2,
        -1.0,
        1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=generator,
    )
    np.save(root / "initial_particles.npy", initial_state.particles.numpy())
    (root / "sweep_config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )

    summaries: list[dict[str, Any]] = []
    grouped: dict[float, list[tuple[dict[str, Any], Path]]] = {}
    for gamma in args.gammas:
        grouped[gamma] = []
        for omega_multiplier in args.omega_multipliers:
            relative = (
                Path(f"gamma_{safe_number(gamma)}")
                / f"omega_{safe_number(omega_multiplier)}pi"
            )
            print(
                f"=== gamma={gamma:g}, omega={omega_multiplier:g}*pi, "
                f"N={args.particles} ==="
            )
            summary, history_path = run_case(
                args,
                initial_state,
                gamma=gamma,
                omega_multiplier=omega_multiplier,
                result_dir=result_root / relative,
                figure_dir=figure_root / "cases" / relative,
            )
            summaries.append(summary)
            grouped[gamma].append((summary, history_path))
            print(
                f"  mean residual={summary['mean_projection_residual']:.3e}, "
                f"final trajectory RMSE={summary['final_trajectory_rmse']:.3e}"
            )

    hashes = {row["model_initialization_sha256"] for row in summaries}
    if len(hashes) != 1:
        raise RuntimeError("controlled comparison violated: NN initialization changed")
    for _, history_path in (case for cases in grouped.values() for case in cases):
        with np.load(history_path) as history:
            if not np.array_equal(initial_state.particles.numpy(), history["initial_particles"]):
                raise RuntimeError(
                    "controlled comparison violated: initial particles changed"
                )

    for gamma, cases in grouped.items():
        plot_group_diagnostics(
            figure_root / f"gamma_{safe_number(gamma)}_frequency_comparison.png",
            gamma,
            cases,
        )
    plot_frequency_summary(figure_root / "final_error_vs_frequency.png", summaries)
    write_summary_tables(table_root, summaries)
    write_scientific_report(root, summaries)
    print(f"saved controlled sweep under {root}")
    return root


def main(argv: list[str] | None = None) -> None:
    run_sweep(parse_args(argv))


if __name__ == "__main__":
    main()
