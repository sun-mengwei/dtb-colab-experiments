"""Unnormalized neural tangent projection from the supplied algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.func import jacrev, vmap

from .parameters import ParameterStructure, functional_model_call


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Numerical diagnostics for the truncated-SVD solve."""

    retained_rank: int
    sigma_max: float
    sigma_min_retained: float
    relative_residual: float


def _field_one(
    flat: torch.Tensor,
    x_single: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
) -> torch.Tensor:
    """Evaluate ``f_theta(x)`` for one spatial point."""

    return functional_model_call(
        model, flat, structure, x_single.unsqueeze(0)
    ).squeeze(0)


def selected_parameter_jacobian(
    theta_flat: torch.Tensor,
    selected: torch.Tensor,
    particles: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    *,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute ``J_i = d f_theta(x_i) / d theta_selected``.

    Returns a tensor of shape ``(N, d, m)``.  The implementation reuses the
    original DTB restricted-coordinate trick: only the selected coordinates
    enter ``jacrev``; the remaining coordinates are constants.  Therefore the
    full ``(N, d, M)`` Jacobian is never materialized.
    """

    if selected.ndim != 1 or selected.numel() == 0:
        raise ValueError("selected must be a nonempty 1-D index tensor")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    theta_selected = theta_flat[selected].detach().clone()
    theta_constant = theta_flat.detach().clone()

    def field_from_subset(theta_subset: torch.Tensor, x_single: torch.Tensor) -> torch.Tensor:
        full = theta_constant.index_copy(0, selected, theta_subset)
        return _field_one(full, x_single, model, structure)

    jacobian_one = jacrev(field_from_subset, argnums=0)
    jacobian_batch = vmap(jacobian_one, in_dims=(None, 0))

    chunks: List[torch.Tensor] = []
    for start in range(0, particles.shape[0], chunk_size):
        chunks.append(jacobian_batch(theta_selected, particles[start : start + chunk_size]))
    return torch.cat(chunks, dim=0)


def stack_unnormalized_system(
    jacobians: torch.Tensor, target_velocity: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``J_i`` and ``v_i`` with no ``1/N`` or ``1/sqrt(N)`` factor."""

    if jacobians.ndim != 3:
        raise ValueError("jacobians must have shape (N, d, m)")
    n, dim, m = jacobians.shape
    if target_velocity.shape != (n, dim):
        raise ValueError("target_velocity must have shape (N, d)")
    return jacobians.reshape(n * dim, m), target_velocity.reshape(n * dim)


def truncated_svd_solve(
    stacked_jacobian: torch.Tensor,
    stacked_velocity: torch.Tensor,
    *,
    rtol: float = 1e-7,
) -> tuple[torch.Tensor, ProjectionDiagnostics]:
    """Compute the minimum-norm unnormalized least-squares solution.

    The retained singular values obey the supplied rule
    ``sigma_j > rtol * sigma_1``.  This is the explicit version of the
    original ``jform_solve(..., method='svd_gpu')`` pattern.
    """

    if stacked_jacobian.ndim != 2 or stacked_velocity.ndim != 1:
        raise ValueError("expected a matrix and a vector")
    if stacked_jacobian.shape[0] != stacked_velocity.numel():
        raise ValueError("stacked system has inconsistent row counts")
    if not 0 <= rtol < 1:
        raise ValueError("rtol must satisfy 0 <= rtol < 1")

    u, singular_values, vh = torch.linalg.svd(stacked_jacobian, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values[0]) == 0.0:
        alpha = torch.zeros(
            stacked_jacobian.shape[1],
            device=stacked_jacobian.device,
            dtype=stacked_jacobian.dtype,
        )
        residual = torch.linalg.norm(stacked_velocity)
        relative = residual / (torch.linalg.norm(stacked_velocity) + 1e-30)
        diagnostics = ProjectionDiagnostics(0, 0.0, 0.0, float(relative))
        return alpha, diagnostics

    keep = singular_values > rtol * singular_values[0]
    rank = int(keep.sum())
    # V_r Sigma_r^{-1} U_r^T V, written without forming diagonal matrices.
    coefficients = (u[:, keep].T @ stacked_velocity) / singular_values[keep]
    alpha = vh[keep, :].T @ coefficients

    residual = stacked_jacobian @ alpha - stacked_velocity
    relative = torch.linalg.norm(residual) / (torch.linalg.norm(stacked_velocity) + 1e-30)
    diagnostics = ProjectionDiagnostics(
        retained_rank=rank,
        sigma_max=float(singular_values[0]),
        sigma_min_retained=float(singular_values[keep][-1]),
        relative_residual=float(relative),
    )
    return alpha, diagnostics
