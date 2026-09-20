"""DTB particle evolution with supervised neural-map refitting.

This module implements an alternative to the parameter Euler update used by
``dtb_step``.  The physical particles still move with the projected DTB
velocity, but after every outer time step all trainable network parameters are
refitted to the new particle cloud.  The fit is unregularized and warm-starts
from the parameters obtained at the previous time step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .dtb import (
    TangentProjection,
    evaluate_dtb_projection,
    evaluate_model,
    flat_parameters,
)
from .experiment import rk4_flow
from .games import Game
from .utils import (
    make_time_grid,
    paired_rms,
    resolve_device,
    resolve_dtype,
    sample_uniform_box,
    snapshot_indices,
    to_numpy,
    warmup_cuda,
)


@dataclass(frozen=True)
class NetworkRefitConfig:
    """Settings for the unregularized fit ``T_theta(z) ~= X_target``."""

    learning_rate: float = 1e-3
    maximum_steps: int = 100
    relative_tolerance: float = 0.05
    absolute_tolerance: float = 1e-7

    def validate(self) -> None:
        """Validate the optimizer and stopping-rule parameters."""

        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("refit learning_rate must be positive and finite")
        if self.maximum_steps < 1:
            raise ValueError("refit maximum_steps must be positive")
        if (
            not np.isfinite(self.relative_tolerance)
            or self.relative_tolerance < 0
            or not np.isfinite(self.absolute_tolerance)
            or self.absolute_tolerance < 0
        ):
            raise ValueError("refit tolerances must be nonnegative and finite")


@dataclass(frozen=True)
class RefitDTBConfig:
    """Numerical settings for deterministic, fixed-label refit DTB."""

    particle_count: int = 2000
    initial_low: float = -1.0
    initial_high: float = 1.0
    step_size: float = 0.002
    final_time: float = 0.2
    snapshot_times: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    basis_size: int = 128
    svd_rtol: float = 1e-8
    jacobian_chunk: int = 512
    reference_step_size: float = 0.00025
    seed: int = 2026
    dtype: str = "float64"
    device: str = "auto"
    progress_reports: int = 5
    refit_interval_fraction: float = 0.10
    refit: NetworkRefitConfig = NetworkRefitConfig()

    def validate(self) -> None:
        """Validate time stepping, sampling, tangent, and refit settings."""

        if self.particle_count < 1 or self.basis_size < 1 or self.jacobian_chunk < 1:
            raise ValueError(
                "particle_count, basis_size, and jacobian_chunk must be positive"
            )
        if not np.isfinite(self.initial_low) or not np.isfinite(self.initial_high):
            raise ValueError("initial interval bounds must be finite")
        if self.initial_high <= self.initial_low:
            raise ValueError("initial_high must be greater than initial_low")
        if self.step_size <= 0 or self.final_time <= 0:
            raise ValueError("step_size and final_time must be positive")
        if self.reference_step_size <= 0:
            raise ValueError("reference_step_size must be positive")
        if not 0 <= self.svd_rtol < 1:
            raise ValueError("svd_rtol must lie in [0, 1)")
        if self.progress_reports < 0:
            raise ValueError("progress_reports cannot be negative")
        if not 0 < self.refit_interval_fraction <= 1:
            raise ValueError("refit_interval_fraction must lie in (0, 1]")
        self.refit.validate()

    def resolved_refit_interval(self, total_steps: int) -> int:
        """Return the number of outer steps between successive NN refits.

        A fractional interval is interpreted relative to the complete outer
        time grid.  For example, ``0.10`` with 100 outer steps gives an
        interval of 10 steps.  ``ceil`` prevents a requested fraction from
        producing refits more frequently than its nominal spacing.
        """

        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        return max(1, int(np.ceil(self.refit_interval_fraction * total_steps)))


@dataclass(frozen=True)
class NetworkRefitDiagnostics:
    """Diagnostics for one minimization of ``||T_theta(z)-X_target||``."""

    rms_before: float
    rms_after: float
    relative_rms_after: float
    stopping_tolerance: float
    optimizer_steps: int
    parameter_change_norm: float
    converged: bool


@dataclass(frozen=True)
class RefitDTBResult:
    """Complete trajectory and diagnostics returned by :func:`run_refit_dtb`."""

    config: RefitDTBConfig
    refit_stride: int
    times: np.ndarray
    projection_times: np.ndarray
    refit_times: np.ndarray
    initial_particles: np.ndarray
    final_particles: np.ndarray
    reference_final_particles: np.ndarray
    initial_parameters: np.ndarray
    final_parameters: np.ndarray
    final_selected_indices: np.ndarray
    selected_indices_history: tuple[np.ndarray, ...]
    projection_basis_size: np.ndarray
    dtb_snapshots: dict[float, np.ndarray]
    reference_snapshots: dict[float, np.ndarray]
    projection_error: np.ndarray
    relative_projection_error: np.ndarray
    alpha_norm: np.ndarray
    condition_number: np.ndarray
    retained_rank: np.ndarray
    trajectory_rms_error: np.ndarray
    refit_rms_before: np.ndarray
    refit_rms_after: np.ndarray
    relative_refit_error: np.ndarray
    refit_tolerance: np.ndarray
    refit_optimizer_steps: np.ndarray
    parameter_change_norm: np.ndarray
    refit_converged: np.ndarray
    final_projection_error: float
    final_relative_projection_error: float
    final_alpha_norm: float
    final_paired_rms: float
    elapsed_seconds: float


def _paired_rms_tensor(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    r"""Return ``sqrt(N^-1 sum_i ||first_i-second_i||_2^2)`` as a tensor."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired clouds must have equal matrix shapes")
    return (first - second).square().sum(dim=1).mean().sqrt()


def draw_tangent_subset(
    parameter_count: int,
    subset_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Draw the random coordinate set ``S_k`` used by the tangent matrix."""

    if parameter_count < 1 or not 1 <= subset_size <= parameter_count:
        raise ValueError("subset_size must lie between 1 and parameter_count")
    selected = torch.randperm(parameter_count, generator=generator)[:subset_size]
    return selected.sort().values.to(device)


def refit_network_to_particles(
    model: nn.Module,
    labels: torch.Tensor,
    target_particles: torch.Tensor,
    *,
    displacement_rms: float,
    config: NetworkRefitConfig,
) -> NetworkRefitDiagnostics:
    r"""Warm-start the unregularized fit ``min_theta ||T_theta(z)-X_target||``.

    The stopping threshold is

    ``max(absolute_tolerance, relative_tolerance * displacement_rms)``.

    No parameter norm or parameter-displacement penalty is added to the loss.
    The reported RMS uses Euclidean particle errors rather than a coordinatewise
    mean, so it is directly comparable with the paired trajectory RMS.
    """

    config.validate()
    if labels.shape != target_particles.shape or labels.ndim != 2:
        raise ValueError("labels and target_particles must have equal matrix shapes")
    if displacement_rms < 0 or not np.isfinite(displacement_rms):
        raise ValueError("displacement_rms must be nonnegative and finite")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")

    theta_before, _ = flat_parameters(model)
    stopping_tolerance = max(
        config.absolute_tolerance,
        config.relative_tolerance * displacement_rms,
    )
    was_training = model.training
    model.train()
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)

    with torch.no_grad():
        rms_before = float(_paired_rms_tensor(model(labels), target_particles).item())

    completed_steps = 0
    current_rms = rms_before
    while completed_steps < config.maximum_steps and current_rms > stopping_tolerance:
        prediction = model(labels)
        loss = (prediction - target_particles).square().sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        completed_steps += 1
        with torch.no_grad():
            current_rms = float(
                _paired_rms_tensor(model(labels), target_particles).item()
            )

    model.train(was_training)
    theta_after, _ = flat_parameters(model)
    tiny = torch.finfo(labels.dtype).tiny
    relative = current_rms / max(displacement_rms, tiny)
    return NetworkRefitDiagnostics(
        rms_before=rms_before,
        rms_after=current_rms,
        relative_rms_after=relative,
        stopping_tolerance=stopping_tolerance,
        optimizer_steps=completed_steps,
        parameter_change_norm=float(
            torch.linalg.vector_norm(theta_after - theta_before).item()
        ),
        converged=current_rms <= stopping_tolerance,
    )


def _progress_steps(total_steps: int, reports: int) -> set[int]:
    """Return outer-step indices at which progress should be printed."""

    if reports <= 0:
        return set()
    return {
        max(1, min(total_steps, int(round(index * total_steps / reports))))
        for index in range(1, reports + 1)
    }


def _record_projection(
    projection: TangentProjection,
    projection_error: list[float],
    relative_projection_error: list[float],
    alpha_norm: list[float],
    condition_number: list[float],
    retained_rank: list[int],
) -> None:
    """Append scalar diagnostics for ``P_J b = J alpha`` to history arrays."""

    projection_error.append(float(projection.rms_residual.item()))
    relative_projection_error.append(float(projection.relative_residual.item()))
    alpha_norm.append(float(torch.linalg.vector_norm(projection.alpha).item()))
    condition_number.append(projection.condition_number)
    retained_rank.append(projection.retained_rank)


def run_refit_dtb(
    game: Game,
    model: nn.Module,
    config: RefitDTBConfig | None = None,
) -> RefitDTBResult:
    r"""Run projected particle Euler steps followed by unregularized NN refits.

    Each outer step computes

    ``alpha_k = argmin_a ||J_{S_k}(theta_k,z)a-b(X_k)||_2``,

    ``X_{k+1} = X_k + h J_{S_k}(theta_k,z) alpha_k``,

    At scheduled refit steps, all network parameters are refitted using

    ``theta_{k+1} = argmin_theta ||T_theta(z)-X_{k+1}||_F^2``.

    The physical particles are never overwritten by the fitted network output.
    Scheduled refit steps use the full parameter tangent for their projection.
    Between refit events, ``theta`` remains fixed while the particle state and
    a randomly selected tangent-coordinate subset continue to evolve.
    """

    config = config or RefitDTBConfig()
    config.validate()
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)
    warmup_cuda(device, dtype)
    model = model.to(device=device, dtype=dtype)

    labels = sample_uniform_box(
        config.particle_count,
        game.dim,
        low=config.initial_low,
        high=config.initial_high,
        dtype=dtype,
        device=device,
        seed=config.seed,
    )
    particles = labels.detach().clone()
    theta, structure = flat_parameters(model)
    initial_theta = theta.detach().clone()
    initial_map = evaluate_model(theta, labels, model, structure).detach()
    identity_tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    if not torch.allclose(initial_map, particles, rtol=0.0, atol=identity_tolerance):
        raise ValueError(
            "refit DTB requires T_theta0(z)=z; use an identity-initialized residual model"
        )

    parameter_count = theta.numel()
    subset_size = min(config.basis_size, parameter_count)
    basis_generator = torch.Generator().manual_seed(config.seed + 1)
    times = make_time_grid(config.final_time, config.step_size)
    snapshots_by_index = snapshot_indices(times, config.snapshot_times)
    reference_particles = particles.detach().clone()
    dtb_snapshots = {float(times[0]): to_numpy(particles).copy()}
    reference_snapshots = {float(times[0]): to_numpy(reference_particles).copy()}

    projection_error: list[float] = []
    relative_projection_error: list[float] = []
    alpha_norm: list[float] = []
    condition_number: list[float] = []
    retained_rank: list[int] = []
    trajectory_rms_error: list[float] = [0.0]
    refit_rms_before: list[float] = []
    refit_rms_after: list[float] = []
    relative_refit_error: list[float] = []
    refit_tolerance: list[float] = []
    refit_optimizer_steps: list[int] = []
    parameter_change_norm: list[float] = []
    refit_converged: list[bool] = []
    selected_history: list[np.ndarray] = []
    projection_basis_size: list[int] = []

    total_steps = len(times) - 1
    refit_stride = config.resolved_refit_interval(total_steps)
    reports = _progress_steps(total_steps, config.progress_reports)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()

    print(
        f"Refit DTB run: game={game.name}, device={device}, dtype={dtype}, "
        f"particles={config.particle_count}, steps={total_steps}, "
        f"parameters={parameter_count}, random_subset_size={subset_size}, "
        f"refit_projection_size={parameter_count}, "
        f"refit_stride={refit_stride} outer steps, "
        f"refit_max_steps={config.refit.maximum_steps}",
        flush=True,
    )
    refit_times: list[float] = []
    last_refit_particles = particles.detach().clone()
    for step in range(total_steps):
        current_time = float(times[step])
        outer_step = float(times[step + 1] - times[step])
        state_index = step + 1
        refit_due = (
            state_index % refit_stride == 0
            or state_index == total_steps
        )
        if refit_due:
            selected = torch.arange(parameter_count, device=device)
        else:
            selected = draw_tangent_subset(
                parameter_count,
                subset_size,
                basis_generator,
                device,
            )
        selected_history.append(to_numpy(selected).copy())
        projection_basis_size.append(int(selected.numel()))
        target_velocity = game.velocity(particles, current_time)
        if target_velocity.shape != particles.shape or not torch.isfinite(
            target_velocity
        ).all():
            raise ValueError("game velocity must be finite and match the particle shape")

        projection = evaluate_dtb_projection(
            theta,
            selected,
            labels,
            target_velocity,
            model,
            structure,
            chunk_size=config.jacobian_chunk,
            svd_rtol=config.svd_rtol,
        )
        next_particles = (particles + outer_step * projection.velocity).detach()
        refit = None
        next_theta = theta
        if refit_due:
            displacement_since_refit = paired_rms(
                next_particles,
                last_refit_particles,
            )
            refit = refit_network_to_particles(
                model,
                labels,
                next_particles,
                displacement_rms=displacement_since_refit,
                config=config.refit,
            )
            next_theta, next_structure = flat_parameters(model)
            if next_structure != structure:
                raise RuntimeError("model parameter structure changed during refitting")
            refit_times.append(float(times[state_index]))
            refit_rms_before.append(refit.rms_before)
            refit_rms_after.append(refit.rms_after)
            relative_refit_error.append(refit.relative_rms_after)
            refit_tolerance.append(refit.stopping_tolerance)
            refit_optimizer_steps.append(refit.optimizer_steps)
            parameter_change_norm.append(refit.parameter_change_norm)
            refit_converged.append(refit.converged)
            last_refit_particles = next_particles.detach().clone()

        reference_particles = rk4_flow(
            game,
            reference_particles,
            current_time,
            outer_step,
            maximum_step=config.reference_step_size,
        )
        theta = next_theta
        particles = next_particles
        trajectory_rms_error.append(paired_rms(particles, reference_particles))
        _record_projection(
            projection,
            projection_error,
            relative_projection_error,
            alpha_norm,
            condition_number,
            retained_rank,
        )
        if state_index in snapshots_by_index:
            snapshot_time = snapshots_by_index[state_index]
            dtb_snapshots[snapshot_time] = to_numpy(particles).copy()
            reference_snapshots[snapshot_time] = to_numpy(reference_particles).copy()
        if state_index in reports:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            eta = elapsed * (total_steps - state_index) / state_index
            refit_text = (
                "refit=not scheduled"
                if refit is None
                else f"refit={refit.rms_after:.3e} | inner={refit.optimizer_steps}"
            )
            print(
                f"{100.0 * state_index / total_steps:5.1f}% | "
                f"step {state_index}/{total_steps} | t={times[state_index]:g} | "
                f"projection={projection.relative_residual.item():.3e} | "
                f"{refit_text} | "
                f"elapsed={elapsed / 60:.1f} min | ETA={eta / 60:.1f} min",
                flush=True,
            )

    final_selected = torch.arange(parameter_count, device=device)
    final_velocity = game.velocity(particles, float(times[-1]))
    final_projection = evaluate_dtb_projection(
        theta,
        final_selected,
        labels,
        final_velocity,
        model,
        structure,
        chunk_size=config.jacobian_chunk,
        svd_rtol=config.svd_rtol,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    final_rms = paired_rms(particles, reference_particles)
    print(
        f"Complete in {elapsed_seconds / 60:.1f} min | "
        f"DTB/RK4 paired RMS={final_rms:.4e} | "
        f"final projection={final_projection.relative_residual.item():.4e}",
        flush=True,
    )

    return RefitDTBResult(
        config=config,
        refit_stride=refit_stride,
        times=times,
        projection_times=times[:-1].copy(),
        refit_times=np.asarray(refit_times),
        initial_particles=to_numpy(labels).copy(),
        final_particles=to_numpy(particles).copy(),
        reference_final_particles=to_numpy(reference_particles).copy(),
        initial_parameters=to_numpy(initial_theta).copy(),
        final_parameters=to_numpy(theta).copy(),
        final_selected_indices=to_numpy(final_selected).copy(),
        selected_indices_history=tuple(selected_history),
        projection_basis_size=np.asarray(projection_basis_size),
        dtb_snapshots=dtb_snapshots,
        reference_snapshots=reference_snapshots,
        projection_error=np.asarray(projection_error),
        relative_projection_error=np.asarray(relative_projection_error),
        alpha_norm=np.asarray(alpha_norm),
        condition_number=np.asarray(condition_number),
        retained_rank=np.asarray(retained_rank),
        trajectory_rms_error=np.asarray(trajectory_rms_error),
        refit_rms_before=np.asarray(refit_rms_before),
        refit_rms_after=np.asarray(refit_rms_after),
        relative_refit_error=np.asarray(relative_refit_error),
        refit_tolerance=np.asarray(refit_tolerance),
        refit_optimizer_steps=np.asarray(refit_optimizer_steps),
        parameter_change_norm=np.asarray(parameter_change_norm),
        refit_converged=np.asarray(refit_converged, dtype=bool),
        final_projection_error=float(final_projection.rms_residual.item()),
        final_relative_projection_error=float(final_projection.relative_residual.item()),
        final_alpha_norm=float(torch.linalg.vector_norm(final_projection.alpha).item()),
        final_paired_rms=final_rms,
        elapsed_seconds=elapsed_seconds,
    )
