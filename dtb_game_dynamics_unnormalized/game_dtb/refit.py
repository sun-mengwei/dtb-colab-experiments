"""Periodic tangent-network refitting for the game-dynamics DTB solver.

The original Deep Tangent Bundle implementation periodically compresses the
accumulated first-order tangent update into a fresh neural representation.
Here the same block-reset idea is applied on the *current particles*: the
network is fitted to ``f_theta + h J_theta sum(alpha)`` and the next DTB block
is linearized around the fitted parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .parameters import ParameterStructure, functional_model_call
from .projection import selected_parameter_jacobian


@dataclass(frozen=True)
class RefitDiagnostics:
    """Training error before and after one block reset."""

    rmse_before: float
    rmse_after: float
    optimizer_steps: int


def accumulated_tangent_teacher(
    model: nn.Module,
    theta_base: torch.Tensor,
    structure: ParameterStructure,
    selected: torch.Tensor,
    accumulated_alpha: torch.Tensor,
    particles: torch.Tensor,
    *,
    step_size: float,
    jacobian_chunk_size: int,
) -> torch.Tensor:
    """Evaluate the first-order neural state accumulated in one DTB block."""

    with torch.no_grad():
        base_values = functional_model_call(
            model, theta_base, structure, particles
        ).detach()
    jacobians = selected_parameter_jacobian(
        theta_base,
        selected,
        particles,
        model,
        structure,
        chunk_size=jacobian_chunk_size,
    )
    tangent_increment = torch.einsum(
        "ndm,m->nd", jacobians, accumulated_alpha
    )
    return (base_values + step_size * tangent_increment).detach()


def fit_model_to_current_particles(
    model: nn.Module,
    particles: torch.Tensor,
    targets: torch.Tensor,
    *,
    optimizer_steps: int,
    learning_rate: float,
    batch_size: int,
    generator: torch.Generator,
) -> RefitDiagnostics:
    """Fit trainable parameters to block targets on the current samples.

    Parameters with ``requires_grad=False`` are deliberately excluded.  This
    means that the frozen random features in an MMNN remain non-trainable.
    """

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("the tangent model has no trainable parameters")
    if optimizer_steps < 1 or learning_rate <= 0 or batch_size < 1:
        raise ValueError("refit steps, learning rate, and batch size must be positive")
    if particles.shape != targets.shape:
        raise ValueError("refit particles and targets must have the same shape")

    was_training = model.training
    model.train()
    starting_parameters = [parameter.detach().clone() for parameter in trainable]
    with torch.no_grad():
        rmse_before = torch.mean((model(particles) - targets).square()).sqrt()

    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimizer_steps
    )
    sample_count = particles.shape[0]
    effective_batch = min(batch_size, sample_count)
    for _ in range(optimizer_steps):
        indices = torch.randperm(
            sample_count, device=particles.device, generator=generator
        )[:effective_batch]
        prediction = model(particles[indices])
        loss = torch.mean((prediction - targets[indices]).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        rmse_after = torch.mean((model(particles) - targets).square()).sqrt()
        # A refit is a representation refresh, so it must never make the
        # block target fit worse.  Restore the pre-refit network if a short or
        # overly aggressive optimizer run overshoots.
        if rmse_after > rmse_before:
            for parameter, starting_value in zip(trainable, starting_parameters):
                parameter.copy_(starting_value)
            rmse_after = rmse_before.clone()
    model.train(was_training)
    return RefitDiagnostics(
        rmse_before=float(rmse_before),
        rmse_after=float(rmse_after),
        optimizer_steps=optimizer_steps,
    )
