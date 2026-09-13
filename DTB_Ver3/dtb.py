"""Core Deep Tangent Bundle projection and update operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

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


def restricted_tangent_matrix(
    theta: torch.Tensor,
    selected: torch.Tensor,
    particles: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the selected parameter tangent in tensor and matrix forms.

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
    minimum = float(singular_values[-1].item())
    condition = float(singular_values[0].item()) / minimum if minimum > 0 else float("inf")
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
) -> tuple[torch.Tensor, torch.Tensor, TangentProjection]:
    r"""Advance particles and parameters with the same projected increment.

    This performs

    ``X_{k+1} = X_k + h J(theta_k, X_k) alpha_k`` and
    ``theta_{k+1}[S] = theta_k[S] + h alpha_k``.

    The updated neural map is never evaluated to replace ``X_{k+1}``. There
    are no resets or refits.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    _, tangent_matrix = restricted_tangent_matrix(
        theta,
        selected,
        particles,
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
