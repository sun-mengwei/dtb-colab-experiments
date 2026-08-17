"""Oscillatory two-player potential game and reference calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class OscillatoryGameParams:
    """Parameters of the oscillatory multi-well potential game."""

    lambda_: float = 0.5
    epsilon: float = 0.5
    omega: float = 2.0 * np.pi
    gamma: float = 0.0

    def validate(self) -> None:
        if self.lambda_ <= 0.0:
            raise ValueError("lambda_ must be positive")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative")
        if self.omega <= 0.0:
            raise ValueError("omega must be positive")
        if self.gamma < 0.0:
            raise ValueError("gamma must be nonnegative")


def sample_uniform_initial_particles(count: int, seed: int) -> np.ndarray:
    """Draw one reproducible particle set shared by every sweep case."""

    if count < 1:
        raise ValueError("count must be positive")
    generator = np.random.default_rng(seed)
    return generator.uniform(-1.0, 1.0, size=(count, 2))


def oscillatory_potential(
    x: torch.Tensor, params: OscillatoryGameParams
) -> torch.Tensor:
    r"""Evaluate the common potential ``Phi_(omega,gamma)`` pointwise."""

    _validate_points(x)
    params.validate()
    x1, x2 = x[..., 0], x[..., 1]
    confinement = -0.5 * params.lambda_ * (x1.square() + x2.square())
    coupling = -0.5 * params.gamma * (x1 - x2).square()
    wells = (params.epsilon / params.omega) * (
        torch.cos(params.omega * x1) + torch.cos(params.omega * x2)
    )
    return confinement + coupling + wells


def oscillatory_vector_field(
    x: torch.Tensor, params: OscillatoryGameParams
) -> torch.Tensor:
    r"""Return the pseudo-gradient ``b=grad(Phi)`` with shape ``(...,2)``."""

    _validate_points(x)
    params.validate()
    x1, x2 = x[..., 0], x[..., 1]
    first = (
        -params.lambda_ * x1
        - params.gamma * (x1 - x2)
        - params.epsilon * torch.sin(params.omega * x1)
    )
    second = (
        -params.lambda_ * x2
        - params.gamma * (x2 - x1)
        - params.epsilon * torch.sin(params.omega * x2)
    )
    return torch.stack((first, second), dim=-1)


def oscillatory_vector_field_jacobian(
    x: torch.Tensor, params: OscillatoryGameParams
) -> torch.Tensor:
    r"""Return the analytic Jacobian ``Db`` with shape ``(...,2,2)``."""

    _validate_points(x)
    params.validate()
    diagonal = (
        -params.lambda_
        - params.gamma
        - params.epsilon * params.omega * torch.cos(params.omega * x)
    )
    result = torch.zeros(
        *x.shape[:-1], 2, 2, device=x.device, dtype=x.dtype
    )
    result[..., 0, 0] = diagonal[..., 0]
    result[..., 1, 1] = diagonal[..., 1]
    result[..., 0, 1] = params.gamma
    result[..., 1, 0] = params.gamma
    return result


def integrate_rk4(
    initial_particles: torch.Tensor,
    params: OscillatoryGameParams,
    *,
    step_size: float,
    steps: int,
    substeps_per_step: int,
    save_history: bool = True,
) -> torch.Tensor:
    """Integrate the true particle ODE with fixed-step vectorized RK4."""

    if step_size <= 0.0 or steps < 1 or substeps_per_step < 1:
        raise ValueError("step_size, steps, and substeps_per_step must be positive")
    state = initial_particles.clone()
    history = [state.clone()] if save_history else []
    reference_h = step_size / substeps_per_step
    for _ in range(steps):
        for _ in range(substeps_per_step):
            k1 = oscillatory_vector_field(state, params)
            k2 = oscillatory_vector_field(state + 0.5 * reference_h * k1, params)
            k3 = oscillatory_vector_field(state + 0.5 * reference_h * k2, params)
            k4 = oscillatory_vector_field(state + reference_h * k3, params)
            state = state + (reference_h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if save_history:
            history.append(state.clone())
    return torch.stack(history) if save_history else state


def locate_oscillatory_equilibria(
    params: OscillatoryGameParams,
    *,
    grid_size: int | None = None,
    residual_tolerance: float = 1e-9,
    duplicate_tolerance: float = 1e-6,
    max_iterations: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate roots in ``[-1,1]^2`` and classify linear stability.

    A dense grid supplies Newton starting points.  This is a numerical root
    enumeration, so the returned set should not be interpreted as a proof of
    completeness.
    """

    params.validate()
    if grid_size is None:
        cycles_across_domain = params.omega / np.pi
        grid_size = max(65, int(np.ceil(8.0 * cycles_across_domain)) + 1)
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")

    grid = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float64)
    first, second = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack((first.reshape(-1), second.reshape(-1)), dim=1)

    for _ in range(max_iterations):
        field = oscillatory_vector_field(points, params)
        jacobian = oscillatory_vector_field_jacobian(points, params)
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
            gradient = torch.einsum(
                "nji,nj->ni", jacobian[~safe], field[~safe]
            )
            scale = torch.linalg.vector_norm(gradient, dim=1, keepdim=True).clamp_min(1e-12)
            direction[~safe] = 0.05 * gradient / scale
        direction_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        direction = direction * torch.clamp(0.25 / direction_norm.clamp_min(1e-12), max=1.0)
        points = (points - direction).clamp(-1.05, 1.05)

    residuals = torch.linalg.vector_norm(
        oscillatory_vector_field(points, params), dim=1
    )
    inside = (points.abs() <= 1.0 + duplicate_tolerance).all(dim=1)
    converged = points[(residuals <= residual_tolerance) & inside]
    candidates = converged.detach().cpu().numpy()
    if not len(candidates):
        return np.empty((0, 2)), np.empty((0,), dtype=bool), np.empty((0,))

    order = np.lexsort((candidates[:, 1], candidates[:, 0]))
    roots: list[np.ndarray] = []
    for candidate in candidates[order]:
        if not any(
            np.linalg.norm(candidate - root) <= duplicate_tolerance
            for root in roots
        ):
            roots.append(candidate)
    root_array = np.asarray(roots)
    jacobians = oscillatory_vector_field_jacobian(
        torch.as_tensor(root_array, dtype=torch.float64), params
    )
    eigenvalues = torch.linalg.eigvalsh(jacobians).numpy()
    spectral_abscissa = eigenvalues[:, -1]
    stable = spectral_abscissa < -1e-8
    return root_array, stable, spectral_abscissa


def basin_masses(
    particles: np.ndarray,
    stable_equilibria: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    """Assign particles to their nearest stable root inside ``tolerance``."""

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


def _validate_points(x: torch.Tensor) -> None:
    if x.ndim < 1 or x.shape[-1] != 2:
        raise ValueError("oscillatory game points must have shape (...,2)")
