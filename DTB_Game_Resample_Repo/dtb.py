"""Reusable score-aware Deep Tangent Bundle primitives.

The module keeps the DTB workflow explicit:

1. flatten a neural map's trainable parameters;
2. evaluate the map with ``torch.func.functional_call``;
3. differentiate only a selected parameter sub-basis;
4. project a drift/diffusion/score velocity by truncated SVD;
5. transport particles, log density, and score by explicit Euler; and
6. periodically refit the neural network to the current pushforward map.

The public functions work with any vector-valued ``torch.nn.Module`` whose
input and output dimensions agree with the particle dimension.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

Drift = Callable[[torch.Tensor], torch.Tensor]
MapTarget = torch.Tensor | Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Flat-parameter utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterStructure:
    """Names, shapes, and sizes needed to reconstruct trainable parameters."""

    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    sizes: tuple[int, ...]

    @property
    def total(self) -> int:
        return sum(self.sizes)


def flat_params(model: nn.Module) -> tuple[torch.Tensor, ParameterStructure]:
    """Return a detached flat vector and the model-specific unflatten recipe."""

    names: list[str] = []
    shapes: list[torch.Size] = []
    sizes: list[int] = []
    pieces: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        shapes.append(parameter.shape)
        sizes.append(parameter.numel())
        pieces.append(parameter.detach().reshape(-1).clone())

    if not pieces:
        raise ValueError("model has no trainable parameters")
    structure = ParameterStructure(tuple(names), tuple(shapes), tuple(sizes))
    return torch.cat(pieces), structure


def unflatten(flat: torch.Tensor, structure: ParameterStructure) -> dict[str, torch.Tensor]:
    """Reconstruct a trainable parameter dictionary without copying data."""

    if flat.ndim != 1 or flat.numel() != structure.total:
        raise ValueError(
            f"flat must have shape ({structure.total},), got {tuple(flat.shape)}"
        )
    output: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape, size in zip(
        structure.names, structure.shapes, structure.sizes
    ):
        output[name] = flat[offset : offset + size].reshape(shape)
        offset += size
    return output


def write_flat_into_model(
    model: nn.Module, flat: torch.Tensor, structure: ParameterStructure
) -> None:
    """Copy a flat vector into the corresponding live trainable parameters."""

    values = unflatten(flat, structure)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            if name not in parameters:
                raise KeyError(f"model no longer contains parameter {name!r}")
            parameters[name].copy_(value)


# ---------------------------------------------------------------------------
# Functional map evaluation and parameter derivatives
# ---------------------------------------------------------------------------


def evaluate_map(
    theta_flat: torch.Tensor,
    points: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
) -> torch.Tensor:
    """Evaluate ``model(points)`` at a supplied flat parameter vector.

    Frozen parameters and buffers are taken from the live module.  The live
    trainable parameters are not mutated, which makes this function safe to
    differentiate with ``jacrev``.
    """

    state = dict(model.named_parameters())
    state.update(dict(model.named_buffers()))
    state.update(unflatten(theta_flat, structure))
    return functional_call(model, state, (points,))


def _evaluate_one(
    theta_flat: torch.Tensor,
    point: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
) -> torch.Tensor:
    """Single-point vector form used inside ``vmap`` and ``jacrev``."""

    value = evaluate_map(theta_flat, point.unsqueeze(0), model, structure)
    return value.squeeze(0).reshape(-1)


def select_parameter_subset(
    parameter_count: int,
    basis_size: int | None,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose sorted DTB coordinate indices without replacement."""

    if parameter_count < 1:
        raise ValueError("parameter_count must be positive")
    size = parameter_count if basis_size is None else min(basis_size, parameter_count)
    if size < 1:
        raise ValueError("basis_size must be positive")
    # A CPU generator works consistently for CPU, CUDA, and MPS experiments.
    selected = torch.randperm(parameter_count, generator=generator)[:size]
    return selected.sort().values.to(device)


@dataclass(frozen=True)
class DTBBasis:
    """Map evaluations and selected parameter tangent columns."""

    values: torch.Tensor  # (N, d_out)
    jacobian: torch.Tensor  # (N, d_out, m)
    matrix: torch.Tensor  # (N*d_out, m)
    selected: torch.Tensor  # (m,)


def dtb_basis_matrix(
    theta_flat: torch.Tensor,
    selected: torch.Tensor,
    points: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    *,
    chunk_size: int = 256,
) -> DTBBasis:
    """Build ``J = partial_theta f_theta`` for selected parameter columns.

    The full ``(N,d,M)`` Jacobian is never materialized.  Only the selected
    coordinates enter ``jacrev``; all other coordinates remain constants.
    """

    if points.ndim != 2:
        raise ValueError("points must have shape (N,d)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if selected.ndim != 1 or selected.numel() == 0:
        raise ValueError("selected must be a nonempty one-dimensional tensor")
    if selected.min() < 0 or selected.max() >= theta_flat.numel():
        raise ValueError("selected contains an out-of-range parameter index")
    if selected.unique().numel() != selected.numel():
        raise ValueError("selected parameter indices must be unique")

    theta_selected = theta_flat[selected].detach().clone()
    theta_constant = theta_flat.detach().clone()

    def selected_map(theta_subset: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        full = theta_constant.index_copy(0, selected, theta_subset)
        return _evaluate_one(full, point, model, structure)

    jacobian_one = jacrev(selected_map, argnums=0)
    values: list[torch.Tensor] = []
    jacobians: list[torch.Tensor] = []
    for start in range(0, points.shape[0], chunk_size):
        point_chunk = points[start : start + chunk_size]
        values.append(vmap(selected_map, in_dims=(None, 0))(theta_selected, point_chunk))
        jacobians.append(
            vmap(jacobian_one, in_dims=(None, 0))(theta_selected, point_chunk)
        )

    value_tensor = torch.cat(values, dim=0)
    jacobian_tensor = torch.cat(jacobians, dim=0)
    matrix = jacobian_tensor.reshape(
        value_tensor.shape[0] * value_tensor.shape[1], selected.numel()
    )
    return DTBBasis(value_tensor, jacobian_tensor, matrix, selected)


# ---------------------------------------------------------------------------
# Score term, velocity, and projection
# ---------------------------------------------------------------------------


def diffusion_score_term(score: torch.Tensor, diffusion: torch.Tensor) -> torch.Tensor:
    """Return the Fokker--Planck score contribution ``-0.5 D q``."""

    if score.ndim != 2:
        raise ValueError("score must have shape (N,d)")
    dim = score.shape[1]
    if diffusion.shape != (dim, dim):
        raise ValueError(f"diffusion must have shape ({dim},{dim})")
    return -0.5 * (score @ diffusion.T)


def form_velocity(
    particles: torch.Tensor,
    score: torch.Tensor,
    drift: Drift,
    diffusion: torch.Tensor,
) -> torch.Tensor:
    """Form ``v=b(x)-0.5 D q`` for every particle."""

    drift_values = drift(particles)
    if drift_values.shape != particles.shape:
        raise ValueError("drift must return a tensor with the particle shape")
    return drift_values + diffusion_score_term(score, diffusion)


@dataclass(frozen=True)
class ProjectionResult:
    """Minimum-norm SVD projection and useful numerical diagnostics."""

    alpha: torch.Tensor
    rank: int
    relative_residual: float
    normal_equation_residual: float
    singular_values: torch.Tensor
    gram: torch.Tensor
    rhs: torch.Tensor


def svd_projection(
    basis: torch.Tensor,
    target: torch.Tensor,
    *,
    rtol: float = 1e-6,
) -> ProjectionResult:
    """Solve ``min_alpha ||J alpha-v||`` with a relative SVD cutoff.

    ``basis`` may have shape ``(N,d,m)`` or already be stacked as ``(N*d,m)``.
    ``target`` may similarly have shape ``(N,d)`` or ``(N*d,)``.
    """

    if not 0 <= rtol < 1:
        raise ValueError("rtol must satisfy 0 <= rtol < 1")
    matrix = basis.reshape(-1, basis.shape[-1])
    vector = target.reshape(-1)
    if matrix.shape[0] != vector.numel():
        raise ValueError("basis and target have inconsistent row counts")

    gram = matrix.T @ matrix
    rhs = matrix.T @ vector
    U, singular_values, Vh = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values[0]) == 0.0:
        keep = torch.zeros_like(singular_values, dtype=torch.bool)
        alpha = torch.zeros(
            matrix.shape[1], device=matrix.device, dtype=matrix.dtype
        )
    else:
        keep = singular_values > rtol * singular_values[0]
        coefficients = (U[:, keep].T @ vector) / singular_values[keep]
        alpha = Vh[keep].T @ coefficients

    residual = torch.linalg.norm(matrix @ alpha - vector)
    relative = residual / (torch.linalg.norm(vector) + 1e-30)
    normal_residual = torch.linalg.norm(gram @ alpha - rhs)
    normal_relative = normal_residual / (torch.linalg.norm(rhs) + 1e-30)
    return ProjectionResult(
        alpha=alpha.detach(),
        rank=int(keep.sum()),
        relative_residual=float(relative),
        normal_equation_residual=float(normal_relative),
        singular_values=singular_values.detach(),
        gram=gram.detach(),
        rhs=rhs.detach(),
    )


# ---------------------------------------------------------------------------
# Spatial derivatives and score-aware Euler update
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionTerms:
    """Projected action and the spatial terms required by score transport."""

    velocity: torch.Tensor  # (N,d)
    jacobian: torch.Tensor  # (N,d,d), entry (a,b)=partial_b u_a
    divergence: torch.Tensor  # (N,)
    gradient_divergence: torch.Tensor  # (N,d)


def projected_action_terms(
    theta_flat: torch.Tensor,
    selected: torch.Tensor,
    alpha: torch.Tensor,
    particles: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    *,
    chunk_size: int = 64,
) -> ActionTerms:
    """Evaluate ``u=J alpha``, ``grad u``, ``div u``, and ``grad div u``."""

    if alpha.shape != selected.shape:
        raise ValueError("alpha and selected must have the same shape")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    theta_selected = theta_flat[selected].detach().clone()
    theta_constant = theta_flat.detach().clone()
    alpha_constant = alpha.detach().clone()

    def selected_map(theta_subset: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        full = theta_constant.index_copy(0, selected, theta_subset)
        return _evaluate_one(full, point, model, structure)

    parameter_jacobian_one = jacrev(selected_map, argnums=0)

    def action_one(point: torch.Tensor) -> torch.Tensor:
        return parameter_jacobian_one(theta_selected, point) @ alpha_constant

    spatial_jacobian_one = jacrev(action_one)

    def divergence_one(point: torch.Tensor) -> torch.Tensor:
        spatial_jacobian = spatial_jacobian_one(point)
        if spatial_jacobian.shape[0] != spatial_jacobian.shape[1]:
            raise ValueError("score transport requires equal input and output dimensions")
        return torch.trace(spatial_jacobian)

    gradient_divergence_one = jacrev(divergence_one)
    velocities: list[torch.Tensor] = []
    jacobians: list[torch.Tensor] = []
    divergences: list[torch.Tensor] = []
    gradient_divergences: list[torch.Tensor] = []
    for start in range(0, particles.shape[0], chunk_size):
        point_chunk = particles[start : start + chunk_size]
        velocities.append(vmap(action_one)(point_chunk))
        jacobians.append(vmap(spatial_jacobian_one)(point_chunk))
        divergences.append(vmap(divergence_one)(point_chunk))
        gradient_divergences.append(vmap(gradient_divergence_one)(point_chunk))

    return ActionTerms(
        torch.cat(velocities, dim=0),
        torch.cat(jacobians, dim=0),
        torch.cat(divergences, dim=0),
        torch.cat(gradient_divergences, dim=0),
    )


@dataclass(frozen=True)
class ParticleState:
    """Reference labels and their transported density state."""

    labels: torch.Tensor
    particles: torch.Tensor
    log_density: torch.Tensor
    score: torch.Tensor

    def validate(self) -> None:
        if self.particles.ndim != 2:
            raise ValueError("particles must have shape (N,d)")
        n, dim = self.particles.shape
        if self.labels.shape != (n, dim):
            raise ValueError("labels must have shape (N,d)")
        if self.log_density.shape != (n,):
            raise ValueError("log_density must have shape (N,)")
        if self.score.shape != (n, dim):
            raise ValueError("score must have shape (N,d)")
        if not all(
            torch.isfinite(value).all()
            for value in (self.particles, self.log_density, self.score)
        ):
            raise ValueError("particle state contains a non-finite value")


def gaussian_log_density_and_score(
    points: torch.Tensor, mean: torch.Tensor, std: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``log N(mean,std^2 I)`` and its exact score."""

    if std <= 0:
        raise ValueError("std must be positive")
    centered = points - mean
    dim = points.shape[-1]
    log_density = (
        -0.5 * centered.square().sum(dim=-1) / std**2
        - dim * math.log(std)
        - 0.5 * dim * math.log(2.0 * math.pi)
    )
    score = -centered / std**2
    return log_density, score


def gaussian_particle_state(
    count: int,
    mean: torch.Tensor,
    std: float,
    *,
    generator: torch.Generator | None = None,
) -> ParticleState:
    """Sample labels, set ``X_0(z)=z``, and initialize density and score."""

    if count < 1:
        raise ValueError("count must be positive")
    # Draw on CPU when a CPU generator is supplied, then move to the target.
    if generator is None:
        noise = torch.randn(
            count, mean.numel(), device=mean.device, dtype=mean.dtype
        )
    else:
        noise = torch.randn(
            count, mean.numel(), generator=generator, dtype=mean.dtype
        ).to(mean.device)
    labels = mean + std * noise
    log_density, score = gaussian_log_density_and_score(labels, mean, std)
    state = ParticleState(labels, labels.clone(), log_density, score)
    state.validate()
    return state


@dataclass(frozen=True)
class ResetDiagnostics:
    """Map-refit error at a scheduled neural-network reset."""

    step: int
    rmse_before: float
    rmse_after: float
    optimizer_steps: int


@dataclass(frozen=True)
class DTBStepResult:
    """New state, tangent projection, and optional reset information."""

    state: ParticleState
    basis: DTBBasis
    projection: ProjectionResult
    action: ActionTerms
    target_velocity: torch.Tensor
    reset: ResetDiagnostics | None = None


def dtb_step(
    state: ParticleState,
    theta_flat: torch.Tensor,
    selected: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    drift: Drift,
    diffusion: torch.Tensor,
    step_size: float,
    *,
    svd_rtol: float = 1e-6,
    jacobian_chunk_size: int = 256,
    derivative_chunk_size: int = 64,
) -> DTBStepResult:
    """Apply one complete score-aware Neural--DTB Euler step."""

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    state.validate()
    target_velocity = form_velocity(
        state.particles, state.score, drift, diffusion
    )
    basis = dtb_basis_matrix(
        theta_flat,
        selected,
        state.particles,
        model,
        structure,
        chunk_size=jacobian_chunk_size,
    )
    projection = svd_projection(
        basis.jacobian, target_velocity, rtol=svd_rtol
    )
    action = projected_action_terms(
        theta_flat,
        selected,
        projection.alpha,
        state.particles,
        model,
        structure,
        chunk_size=derivative_chunk_size,
    )

    # action.jacobian[n,a,b] = partial_b u_a, hence A^T q below.
    transported_score = torch.einsum(
        "nab,na->nb", action.jacobian, state.score
    )
    next_state = ParticleState(
        labels=state.labels,
        particles=(state.particles + step_size * action.velocity).detach(),
        log_density=(
            state.log_density - step_size * action.divergence
        ).detach(),
        score=(
            state.score
            - step_size
            * (transported_score + action.gradient_divergence)
        ).detach(),
    )
    next_state.validate()
    return DTBStepResult(
        next_state, basis, projection, action, target_velocity.detach()
    )


# ---------------------------------------------------------------------------
# Periodic neural-map reset
# ---------------------------------------------------------------------------


def _map_target_values(
    reference_points: torch.Tensor, current_map: MapTarget
) -> torch.Tensor:
    if callable(current_map):
        with torch.no_grad():
            values = current_map(reference_points).detach()
    else:
        values = current_map.detach()
    if values.ndim != 2 or values.shape[0] != reference_points.shape[0]:
        raise ValueError("current_map values must have shape (N,d_out)")
    return values


def fit_model_to_map(
    model: nn.Module,
    reference_points: torch.Tensor,
    current_map: MapTarget,
    *,
    optimizer_steps: int = 500,
    learning_rate: float = 1e-3,
    batch_size: int = 512,
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    """Refit all trainable parameters to the current pushforward map.

    ``current_map`` may be precomputed values or a callable.  Returning both
    errors makes reset quality visible to the caller instead of hiding it.
    """

    if min(optimizer_steps, batch_size) < 1:
        raise ValueError("optimizer_steps and batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    target = _map_target_values(reference_points, current_map)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")

    with torch.no_grad():
        rmse_before = torch.mean((model(reference_points) - target).square()).sqrt()
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimizer_steps
    )
    was_training = model.training
    model.train()
    count = reference_points.shape[0]
    actual_batch = min(batch_size, count)
    for _ in range(optimizer_steps):
        indices = torch.randint(
            0, count, (actual_batch,), generator=generator
        ).to(reference_points.device)
        prediction = model(reference_points[indices])
        loss = torch.mean((prediction - target[indices]).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.train(was_training)

    with torch.no_grad():
        rmse_after = torch.mean((model(reference_points) - target).square()).sqrt()
    return float(rmse_before), float(rmse_after)


@dataclass(frozen=True)
class PeriodicReset:
    """Simple source-matching reset schedule for a neural pushforward map."""

    interval: int = 0
    optimizer_steps: int = 500
    learning_rate: float = 1e-3
    batch_size: int = 512

    def due(self, completed_steps: int) -> bool:
        return self.interval > 0 and completed_steps % self.interval == 0

    def maybe_reset(
        self,
        model: nn.Module,
        completed_steps: int,
        reference_points: torch.Tensor,
        current_map: MapTarget,
        *,
        generator: torch.Generator | None = None,
    ) -> ResetDiagnostics | None:
        """Refit only on scheduled steps; otherwise return ``None``."""

        if not self.due(completed_steps):
            return None
        before, after = fit_model_to_map(
            model,
            reference_points,
            current_map,
            optimizer_steps=self.optimizer_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            generator=generator,
        )
        return ResetDiagnostics(
            completed_steps, before, after, self.optimizer_steps
        )


class NeuralDTB:
    """Concise stateful driver with fixed tangent blocks and periodic resets."""

    def __init__(
        self,
        model: nn.Module,
        drift: Drift,
        diffusion: torch.Tensor,
        *,
        step_size: float,
        basis_size: int | None = None,
        svd_rtol: float = 1e-6,
        jacobian_chunk_size: int = 256,
        derivative_chunk_size: int = 64,
        reset: PeriodicReset | None = None,
        seed: int = 0,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if diffusion.ndim != 2 or diffusion.shape[0] != diffusion.shape[1]:
            raise ValueError("diffusion must be square")
        self.model = model
        self.drift = drift
        self.diffusion = diffusion
        self.step_size = step_size
        self.basis_size = basis_size
        self.svd_rtol = svd_rtol
        self.jacobian_chunk_size = jacobian_chunk_size
        self.derivative_chunk_size = derivative_chunk_size
        self.reset_schedule = reset or PeriodicReset(interval=0)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.theta: torch.Tensor
        self.structure: ParameterStructure
        self.selected: torch.Tensor
        self._refresh_tangent_block()

    def _refresh_tangent_block(self) -> None:
        """Freeze the current network parameters and draw a fresh sub-basis."""

        self.theta, self.structure = flat_params(self.model)
        self.selected = select_parameter_subset(
            self.theta.numel(),
            self.basis_size,
            device=self.theta.device,
            generator=self.generator,
        )

    def step(self, state: ParticleState, completed_steps: int) -> DTBStepResult:
        """Advance once and perform a scheduled source-matching reset."""

        result = dtb_step(
            state,
            self.theta,
            self.selected,
            self.model,
            self.structure,
            self.drift,
            self.diffusion,
            self.step_size,
            svd_rtol=self.svd_rtol,
            jacobian_chunk_size=self.jacobian_chunk_size,
            derivative_chunk_size=self.derivative_chunk_size,
        )
        reset_diagnostics = self.reset_schedule.maybe_reset(
            self.model,
            completed_steps,
            result.state.labels,
            result.state.particles,
            generator=self.generator,
        )
        if reset_diagnostics is not None:
            self._refresh_tangent_block()
            result = replace(result, reset=reset_diagnostics)
        return result

    def evaluate_network_map(self, reference_points: torch.Tensor) -> torch.Tensor:
        """Evaluate the frozen neural map associated with the active block."""

        return evaluate_map(
            self.theta, reference_points, self.model, self.structure
        )


__all__ = [
    "ActionTerms",
    "DTBBasis",
    "DTBStepResult",
    "NeuralDTB",
    "ParameterStructure",
    "ParticleState",
    "PeriodicReset",
    "ProjectionResult",
    "ResetDiagnostics",
    "diffusion_score_term",
    "dtb_basis_matrix",
    "dtb_step",
    "evaluate_map",
    "fit_model_to_map",
    "flat_params",
    "form_velocity",
    "gaussian_log_density_and_score",
    "gaussian_particle_state",
    "projected_action_terms",
    "select_parameter_subset",
    "svd_projection",
    "unflatten",
    "write_flat_into_model",
]
