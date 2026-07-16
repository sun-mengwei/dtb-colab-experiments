"""Particle, log-density, and score state for the discrete scheme."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParticleState:
    """All quantities transported by one explicit Euler step.

    Shapes:
        particles: ``(N, d)``
        log_density: ``(N,)``
        score: ``(N, d)``
        labels: ``(N, d)`` reference samples ``z_i``
    """

    particles: torch.Tensor
    log_density: torch.Tensor
    score: torch.Tensor
    labels: torch.Tensor

    def validate(self) -> None:
        n, dim = self.particles.shape
        if self.log_density.shape != (n,):
            raise ValueError("log_density must have shape (N,)")
        if self.score.shape != (n, dim):
            raise ValueError("score must have shape (N, d)")
        if self.labels.shape != (n, dim):
            raise ValueError("labels must have shape (N, d)")


def gaussian_particle_state(
    n: int,
    dim: int,
    mean: float,
    std: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> ParticleState:
    """Initialize a known Gaussian density and its exact score.

    This is the algorithm file's Gaussian initialization, generalized from
    ``N(0, I_d)`` to ``N(mean, std^2 I_d)``.
    """

    if n < 1 or dim < 1 or std <= 0:
        raise ValueError("n, dim, and std must be positive")
    labels = torch.randn(
        n, dim, device=device, dtype=dtype, generator=generator
    ) * std + mean
    particles = labels.clone()
    centered = particles - mean
    log_density = (
        -0.5 * centered.square().sum(dim=1) / (std * std)
        - dim * math.log(std)
        - 0.5 * dim * math.log(2.0 * math.pi)
    )
    score = -centered / (std * std)
    state = ParticleState(particles, log_density, score, labels)
    state.validate()
    return state


def uniform_box_particle_state(
    n: int,
    dim: int,
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> ParticleState:
    """Initialize the Figure 4.2 uniform distribution on a box.

    The density is constant and its score is zero in the box interior.  The
    score is not classically defined on the boundary, but samples hit that
    measure-zero set with probability zero.  This is the particle-level
    interpretation used by the replication preset.
    """

    if n < 1 or dim < 1 or not high > low:
        raise ValueError("n and dim must be positive and high must exceed low")
    labels = (
        torch.rand(n, dim, device=device, dtype=dtype, generator=generator)
        * (high - low)
        + low
    )
    particles = labels.clone()
    log_density_value = -dim * math.log(high - low)
    log_density = torch.full(
        (n,), log_density_value, device=device, dtype=dtype
    )
    score = torch.zeros_like(particles)
    state = ParticleState(particles, log_density, score, labels)
    state.validate()
    return state
