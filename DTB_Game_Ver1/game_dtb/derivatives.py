"""Spatial derivatives of the projected tangent velocity."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.func import jacrev, vmap

from .parameters import ParameterStructure
from .projection import _field_one


def tangent_velocity_and_spatial_terms(
    theta_flat: torch.Tensor,
    selected: torch.Tensor,
    alpha: torch.Tensor,
    particles: torch.Tensor,
    model: nn.Module,
    structure: ParameterStructure,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate ``u``, ``grad u``, ``div u``, and ``grad(div u)``.

    Shapes are ``(N,d)``, ``(N,d,d)``, ``(N,)``, and ``(N,d)``.  Nested
    ``jacrev`` calls implement the exact score update from the algorithm;
    this is the most expensive part of a step.
    """

    if alpha.shape != selected.shape:
        raise ValueError("alpha and selected must have the same length")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    theta_selected = theta_flat[selected].detach().clone()
    theta_constant = theta_flat.detach().clone()
    alpha_constant = alpha.detach().clone()

    def field_from_subset(theta_subset: torch.Tensor, x_single: torch.Tensor) -> torch.Tensor:
        full = theta_constant.index_copy(0, selected, theta_subset)
        return _field_one(full, x_single, model, structure)

    # u(x) = (d f_theta(x) / d theta_selected) alpha.
    parameter_jacobian = jacrev(field_from_subset, argnums=0)

    def tangent_one(x_single: torch.Tensor) -> torch.Tensor:
        return parameter_jacobian(theta_selected, x_single) @ alpha_constant

    spatial_jacobian_one = jacrev(tangent_one)

    def divergence_one(x_single: torch.Tensor) -> torch.Tensor:
        return torch.trace(spatial_jacobian_one(x_single))

    gradient_divergence_one = jacrev(divergence_one)

    tangent_batch = vmap(tangent_one)
    spatial_jacobian_batch = vmap(spatial_jacobian_one)
    divergence_batch = vmap(divergence_one)
    gradient_divergence_batch = vmap(gradient_divergence_one)

    velocities: List[torch.Tensor] = []
    spatial_jacobians: List[torch.Tensor] = []
    divergences: List[torch.Tensor] = []
    gradient_divergences: List[torch.Tensor] = []
    for start in range(0, particles.shape[0], chunk_size):
        x_chunk = particles[start : start + chunk_size]
        velocities.append(tangent_batch(x_chunk))
        spatial_jacobians.append(spatial_jacobian_batch(x_chunk))
        divergences.append(divergence_batch(x_chunk))
        gradient_divergences.append(gradient_divergence_batch(x_chunk))

    return (
        torch.cat(velocities, dim=0),
        torch.cat(spatial_jacobians, dim=0),
        torch.cat(divergences, dim=0),
        torch.cat(gradient_divergences, dim=0),
    )
