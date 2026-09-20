"""Reusable mathematical diagnostics for DTB experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dtb import ParameterStructure, TangentProjection, dtb_step, project_velocity
from .experiment import rk4_flow
from .games import OscillatoryNonpotentialGame
from .utils import paired_rms, to_numpy


def projection_metrics(
    tangent_matrix: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    rtol: float,
) -> dict[str, TangentProjection | float]:
    r"""Project ``q`` onto ``range(J)`` and report representation metrics.

    The package projection solves ``alpha*=argmin_alpha ||J alpha-q||_2^2``.
    This function returns that projection together with
    ``E_repr=||J alpha*-q||/||q||``, ``R_osc=1-E_repr^2``, and
    ``||alpha*||_2``.
    """

    projection = project_velocity(tangent_matrix, target_velocity, rtol=rtol)
    error = float(projection.relative_residual.item())
    return {
        "projection": projection,
        "representation_error": error,
        "captured_energy": 1.0 - error**2,
        "alpha_norm": float(torch.linalg.vector_norm(projection.alpha).item()),
    }


def run_oscillatory_dynamic_diagnostic(
    game: OscillatoryNonpotentialGame,
    model: nn.Module,
    theta_initial: torch.Tensor,
    structure: ParameterStructure,
    selected: torch.Tensor,
    initial_particles: torch.Tensor,
    *,
    step_size: float,
    final_time: float,
    jacobian_chunk: int,
    svd_rtol: float,
    rk4_maximum_step: float,
) -> dict[str, np.ndarray]:
    r"""Run matched DTB, Euler, and refined-RK4 trajectory diagnostics.

    DTB advances ``X[k+1]=X[k]+h J[k] alpha[k]`` through :func:`dtb_step`.
    Euler advances ``X_E[k+1]=X_E[k]+h b(X_E[k])``. Refined RK4 approximates
    the exact flow. At each step the result records:

    - tangent projection residuals;
    - ``E_step = RMS(X_DTB[k+1] - Phi_h(X_DTB[k]))``;
    - paired DTB/RK4 and Euler/RK4 trajectory errors;
    - ``mu=lambda_max((Db+Db.T)/2)`` along the DTB particles.
    """

    if step_size <= 0 or final_time <= 0:
        raise ValueError("step_size and final_time must be positive")
    if jacobian_chunk < 1 or not 0 <= svd_rtol < 1:
        raise ValueError("jacobian_chunk must be positive and svd_rtol in [0, 1)")
    steps = int(round(final_time / step_size))
    if not np.isclose(steps * step_size, final_time, atol=1e-12, rtol=0.0):
        raise ValueError("final_time must be an integer multiple of step_size")

    theta = theta_initial.detach().clone()
    dtb_state = initial_particles.detach().clone()
    euler_state = initial_particles.detach().clone()
    reference_state = initial_particles.detach().clone()
    fixed_labels = initial_particles.detach().clone()

    times = [0.0]
    dtb_trajectory_error = [0.0]
    euler_trajectory_error = [0.0]
    projection_error: list[float] = []
    relative_projection_error: list[float] = []
    one_step_error: list[float] = []
    mean_growth: list[float] = []
    max_growth: list[float] = []
    positive_growth_fraction: list[float] = []

    for step in range(steps):
        time_value = step * step_size
        old_dtb = dtb_state
        target = game.velocity(old_dtb, time_value)
        theta, dtb_state, projection = dtb_step(
            theta,
            selected,
            old_dtb,
            target,
            model,
            structure,
            step_size=step_size,
            chunk_size=jacobian_chunk,
            svd_rtol=svd_rtol,
            tangent_inputs=fixed_labels,
        )

        local_reference = rk4_flow(
            game,
            old_dtb,
            time_value,
            step_size,
            maximum_step=rk4_maximum_step,
        )
        one_step_error.append(paired_rms(dtb_state, local_reference))

        euler_state = (
            euler_state + step_size * game.velocity(euler_state, time_value)
        ).detach()
        reference_state = rk4_flow(
            game,
            reference_state,
            time_value,
            step_size,
            maximum_step=rk4_maximum_step,
        )

        projection_error.append(float(projection.rms_residual.item()))
        relative_projection_error.append(float(projection.relative_residual.item()))
        dtb_trajectory_error.append(paired_rms(dtb_state, reference_state))
        euler_trajectory_error.append(paired_rms(euler_state, reference_state))

        growth = game.symmetric_growth_rate(old_dtb)
        mean_growth.append(float(growth.mean().item()))
        max_growth.append(float(growth.max().item()))
        positive_growth_fraction.append(
            float((growth > 0).to(growth.dtype).mean().item())
        )
        times.append((step + 1) * step_size)

    return {
        "times": np.asarray(times),
        "projection_times": np.asarray(times[:-1]),
        "projection_error": np.asarray(projection_error),
        "relative_projection_error": np.asarray(relative_projection_error),
        "one_step_error": np.asarray(one_step_error),
        "dtb_trajectory_error": np.asarray(dtb_trajectory_error),
        "euler_trajectory_error": np.asarray(euler_trajectory_error),
        "mean_growth": np.asarray(mean_growth),
        "max_growth": np.asarray(max_growth),
        "positive_growth_fraction": np.asarray(positive_growth_fraction),
        "dtb_final": to_numpy(dtb_state),
        "euler_final": to_numpy(euler_state),
        "rk4_final": to_numpy(reference_state),
    }
