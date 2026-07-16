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
