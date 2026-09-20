"""Core Deep Tangent Bundle projection and update operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, jvp, vmap

ParameterStructure = list[tuple[str, tuple[int, ...]]]


def flat_parameters(model: nn.Module) -> tuple[torch.Tensor, ParameterStructure]:
    """Return a detached flat trainable vector and its named shape structure."""

    parts: list[torch.Tensor] = []
    structure: ParameterStructure = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parts.append(parameter.detach().reshape(-1).clone())
            structure.append((name, tuple(parameter.shape)))
    if not parts:
        raise ValueError("model has no trainable parameters")
    return torch.cat(parts), structure


def unflatten_parameters(
    flat: torch.Tensor,
    structure: Sequence[tuple[str, tuple[int, ...]]],
) -> dict[str, torch.Tensor]:
    """Reconstruct named parameter tensors from a flat vector."""

    parameters: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape in structure:
        count = 1
        for size in shape:
            count *= size
        parameters[name] = flat[offset : offset + count].reshape(shape)
        offset += count
    if offset != flat.numel():
        raise ValueError("flat parameter size does not match structure")
    return parameters


def evaluate_model(
    theta: torch.Tensor,
    particles: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
) -> torch.Tensor:
    """Evaluate ``model`` functionally without modifying the live module."""

    state = dict(model.named_parameters())
    state.update(dict(model.named_buffers()))
    state.update(unflatten_parameters(theta, structure))
    return functional_call(model, state, (particles,))


def subset_tangent_selection(
    theta: torch.Tensor,
    selected: torch.Tensor,
    particles: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return a parameter-subset tangent in tensor and matrix forms.

    The outputs have shapes ``(N, d, m)`` and ``(N*d, m)``. ``chunk_size``
    splits only the particle dimension and therefore changes memory and speed,
    not the mathematical tangent basis.
    """

    if theta.ndim != 1 or selected.ndim != 1:
        raise ValueError("theta and selected must be one-dimensional")
    if particles.ndim != 2 or particles.shape[0] < 1:
        raise ValueError("particles must have nonempty shape (N, d)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if selected.numel() < 1 or selected.numel() > theta.numel():
        raise ValueError("selected has an invalid size")
    if selected.device != theta.device or particles.device != theta.device:
        raise ValueError("theta, selected, and particles must use the same device")
    if selected.min() < 0 or selected.max() >= theta.numel():
        raise ValueError("selected contains an invalid parameter index")
    if torch.unique(selected).numel() != selected.numel():
        raise ValueError("selected parameter indices must be unique")

    theta_selected = theta[selected].detach().clone()
    theta_frozen = theta.detach().clone()

    def model_one(selected_values: torch.Tensor, particle: torch.Tensor) -> torch.Tensor:
        full = theta_frozen.index_copy(0, selected, selected_values)
        return evaluate_model(
            full,
            particle.unsqueeze(0),
            model,
            structure,
        ).squeeze(0)

    jacobian_one = jacrev(model_one, argnums=0)
    jacobian_batch = vmap(jacobian_one, in_dims=(None, 0))
    chunks = [
        jacobian_batch(theta_selected, particles[start : start + chunk_size])
        for start in range(0, particles.shape[0], chunk_size)
    ]
    tangent = torch.cat(chunks, dim=0)
    expected_shape = (particles.shape[0], particles.shape[1], selected.numel())
    if tangent.shape != expected_shape:
        raise ValueError(
            f"model tangent has shape {tuple(tangent.shape)}; expected {expected_shape}"
        )
    matrix = tangent.reshape(particles.shape[0] * particles.shape[1], selected.numel())
    return tangent, matrix


def full_tangent_matrix(
    model: nn.Module,
    particles: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, ParameterStructure, torch.Tensor, torch.Tensor]:
    r"""Return the full sampled neural parameter Jacobian.

    For trainable parameter vector ``theta`` and samples ``z_i``, this computes

    ``J(theta, Z) = [D_theta T_theta(z_1); ...; D_theta T_theta(z_N)]``.

    The returned tuple is ``(theta, structure, selected, J)`` where
    ``selected = (0, ..., len(theta)-1)``.  Use ``J / sqrt(N)`` when the
    normalized empirical inner product is required.
    """

    theta, structure = flat_parameters(model)
    selected = torch.arange(theta.numel(), device=theta.device)
    _, matrix = subset_tangent_selection(
        theta,
        selected,
        particles,
        model,
        structure,
        chunk_size=chunk_size,
    )
    return theta, structure, selected, matrix.detach()


def active_tangent_columns(
    matrix: torch.Tensor,
    selected: torch.Tensor,
    *,
    relative_tolerance: float = 1e-13,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Remove structurally zero columns without changing ``range(J)``.

    The retained coordinate set is
    ``S = {j : ||J[:,j]|| > tolerance * max_l ||J[:,l]||}``.
    Identity-initialized residual networks can contain exactly zero tangent
    coordinates because a zero output layer screens earlier parameters.
    """

    if matrix.ndim != 2 or selected.ndim != 1:
        raise ValueError("matrix must be two-dimensional and selected one-dimensional")
    if matrix.shape[1] != selected.numel():
        raise ValueError("matrix columns must match selected coordinates")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be nonnegative")
    norms = torch.linalg.vector_norm(matrix, dim=0)
    if norms.numel() == 0 or norms.max() <= 0:
        raise FloatingPointError("the tangent matrix has no active columns")
    keep = norms > relative_tolerance * norms.max()
    if not keep.any():
        raise FloatingPointError("the tangent matrix has no active columns")
    return matrix[:, keep], selected[keep]


@dataclass(frozen=True)
class TangentProjection:
    """Result of projecting a physical velocity onto a DTB tangent space."""

    alpha: torch.Tensor
    velocity: torch.Tensor
    rms_residual: torch.Tensor
    relative_residual: torch.Tensor
    singular_values: torch.Tensor
    retained_rank: int
    condition_number: float


def project_velocity(
    tangent_matrix: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    rtol: float,
) -> TangentProjection:
    """Solve the selected tangent least-squares problem by truncated SVD."""

    if tangent_matrix.ndim != 2 or target_velocity.ndim != 2:
        raise ValueError("tangent_matrix and target_velocity must be matrices")
    if tangent_matrix.shape[0] != target_velocity.numel():
        raise ValueError("tangent matrix rows must match the flattened target")
    if not 0 <= rtol < 1:
        raise ValueError("rtol must lie in [0, 1)")

    particle_count = target_velocity.shape[0]
    scale = particle_count**0.5
    normalized_tangent = tangent_matrix / scale
    normalized_target = target_velocity.reshape(-1) / scale
    left, singular_values, right_h = torch.linalg.svd(
        normalized_tangent,
        full_matrices=False,
    )
    if singular_values.numel() == 0 or singular_values[0] <= 0:
        raise FloatingPointError("tangent matrix has no positive singular values")
    retained = singular_values > rtol * singular_values[0]
    retained_rank = int(retained.sum().item())
    if retained_rank == 0:
        raise FloatingPointError("no tangent singular value was retained")

    alpha = right_h[retained].T @ (
        (left[:, retained].T @ normalized_target) / singular_values[retained]
    )
    velocity = (tangent_matrix @ alpha).reshape_as(target_velocity)
    difference = velocity - target_velocity
    rms_residual = difference.square().sum(dim=1).mean().sqrt()
    relative_residual = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        target_velocity
    ).clamp_min(torch.finfo(target_velocity.dtype).tiny)
    retained_values = singular_values[retained]
    minimum_retained = float(retained_values[-1].item())
    condition = float(singular_values[0].item()) / minimum_retained
    return TangentProjection(
        alpha=alpha,
        velocity=velocity,
        rms_residual=rms_residual,
        relative_residual=relative_residual,
        singular_values=singular_values,
        retained_rank=retained_rank,
        condition_number=condition,
    )


def dtb_step(
    theta: torch.Tensor,
    selected: torch.Tensor,
    particles: torch.Tensor,
    target_velocity: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    *,
    step_size: float,
    chunk_size: int,
    svd_rtol: float,
    tangent_inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, TangentProjection]:
    r"""Advance particles and parameters with the same projected increment.

    This performs

    ``X_{k+1} = X_k + h J_{S_k}(theta_k, X_k) alpha_k`` and
    ``theta_{k+1}[S_k] = theta_k[S_k] + h alpha_k``.

    The updated neural map is never evaluated to replace ``X_{k+1}``. There
    are no resets or refits. By default the tangent is evaluated at the
    current particles. Pass fixed ``tangent_inputs`` to evaluate the moving
    parameter basis at immutable reference labels instead.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    basis_inputs = particles if tangent_inputs is None else tangent_inputs
    if basis_inputs.shape != particles.shape:
        raise ValueError("tangent_inputs must have the particle shape")
    _, tangent_matrix = subset_tangent_selection(
        theta,
        selected,
        basis_inputs,
        model,
        structure,
        chunk_size=chunk_size,
    )
    projection = project_velocity(tangent_matrix, target_velocity, rtol=svd_rtol)
    next_particles = (particles + step_size * projection.velocity).detach()
    next_theta = theta.detach().clone()
    next_theta[selected] += step_size * projection.alpha.detach()
    if not torch.isfinite(next_particles).all() or not torch.isfinite(next_theta).all():
        raise FloatingPointError("DTB update produced a nonfinite value")
    return next_theta, next_particles, projection


def tangent_spatial_terms(
    theta: torch.Tensor,
    selected: torch.Tensor,
    alpha: torch.Tensor,
    particles: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Evaluate ``u``, ``D_x u``, ``div(u)``, and ``grad(div(u))``.

    Here ``u(x) = partial_theta_selected f_theta(x) alpha``. The returned
    spatial Jacobian follows ``grad_u[n,a,b] = partial u_a / partial x_b``.
    These terms evolve the score of the density transported by the projected
    DTB velocity.
    """

    if theta.ndim != 1 or selected.ndim != 1 or alpha.ndim != 1:
        raise ValueError("theta, selected, and alpha must be one-dimensional")
    if selected.numel() != alpha.numel():
        raise ValueError("selected and alpha must have equal length")
    if particles.ndim != 2 or particles.shape[0] < 1:
        raise ValueError("particles must have nonempty shape (N, d)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    parameters = theta.detach().clone()
    direction = torch.zeros_like(parameters).index_copy(0, selected, alpha.detach())

    def tangent_one(particle: torch.Tensor) -> torch.Tensor:
        def model_at(candidate: torch.Tensor) -> torch.Tensor:
            return evaluate_model(
                candidate,
                particle.unsqueeze(0),
                model,
                structure,
            ).squeeze(0)

        return jvp(model_at, (parameters,), (direction,))[1]

    spatial_jacobian_one = jacrev(tangent_one)

    def divergence_one(particle: torch.Tensor) -> torch.Tensor:
        return torch.trace(spatial_jacobian_one(particle))

    gradient_divergence_one = jacrev(divergence_one)
    velocity_batch = vmap(tangent_one)
    jacobian_batch = vmap(spatial_jacobian_one)
    divergence_batch = vmap(divergence_one)
    gradient_divergence_batch = vmap(gradient_divergence_one)

    velocities: list[torch.Tensor] = []
    spatial_jacobians: list[torch.Tensor] = []
    divergences: list[torch.Tensor] = []
    gradient_divergences: list[torch.Tensor] = []
    for start in range(0, particles.shape[0], chunk_size):
        chunk = particles[start : start + chunk_size]
        velocities.append(velocity_batch(chunk))
        spatial_jacobians.append(jacobian_batch(chunk))
        divergences.append(divergence_batch(chunk))
        gradient_divergences.append(gradient_divergence_batch(chunk))
    return (
        torch.cat(velocities, dim=0),
        torch.cat(spatial_jacobians, dim=0),
        torch.cat(divergences, dim=0),
        torch.cat(gradient_divergences, dim=0),
    )


def advance_score(
    score: torch.Tensor,
    spatial_jacobian: torch.Tensor,
    gradient_divergence: torch.Tensor,
    *,
    step_size: float,
) -> torch.Tensor:
    r"""Euler-step the score transported by a velocity field.

    With ``q = grad(log rho)`` and velocity ``u``, this computes

    ``q_next = q - h ((D_x u)^T q + grad(div(u)))``.
    """

    if score.ndim != 2:
        raise ValueError("score must have shape (N, d)")
    expected = (score.shape[0], score.shape[1], score.shape[1])
    if spatial_jacobian.shape != expected:
        raise ValueError("score and spatial_jacobian shapes are inconsistent")
    if gradient_divergence.shape != score.shape:
        raise ValueError("gradient_divergence must have the score shape")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    transported = torch.einsum("nab,na->nb", spatial_jacobian, score)
    next_score = score - step_size * (transported + gradient_divergence)
    if not torch.isfinite(next_score).all():
        raise FloatingPointError("score update produced a nonfinite value")
    return next_score.detach()
