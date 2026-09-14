"""Game definitions used by the DTB experiment driver.

A new game only needs a dimension, a name, and a ``velocity(x, t)`` method.
``FunctionalGame`` is the shortest route for notebook-defined examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, runtime_checkable

import torch


@runtime_checkable
class Game(Protocol):
    """Minimal interface consumed by :func:`run_experiment`."""

    dim: int
    name: str

    def velocity(self, particles: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        """Return the physical velocity with the same shape as ``particles``."""

    def metadata(self) -> Mapping[str, object]:
        """Return JSON-serializable game parameters."""


@runtime_checkable
class Diffusion(Protocol):
    r"""Noise interface for an Itô SDE ``dX = b(X,t)dt + Sigma(X,t)dW``."""

    name: str

    def noise_matrix(self, particles: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        """Return ``Sigma`` with shape ``(N, d, brownian_dim)``."""

    def covariance_divergence(
        self,
        particles: torch.Tensor,
        time: float = 0.0,
    ) -> torch.Tensor:
        r"""Return ``div(Sigma Sigma^T)`` with shape ``(N, d)``."""

    def metadata(self) -> Mapping[str, object]:
        """Return JSON-serializable diffusion parameters."""


@dataclass(frozen=True)
class ConstantDiffusion:
    """Constant additive noise with a user-supplied matrix ``Sigma``."""

    matrix: tuple[tuple[float, ...], ...]
    name: str = "constant_diffusion"

    def __post_init__(self) -> None:
        if not self.matrix or not self.matrix[0]:
            raise ValueError("matrix must be nonempty")
        width = len(self.matrix[0])
        if any(len(row) != width for row in self.matrix):
            raise ValueError("matrix rows must have equal length")
        values = torch.as_tensor(self.matrix, dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("matrix must contain finite values")

    @classmethod
    def isotropic(cls, dim: int, amplitude: float) -> "ConstantDiffusion":
        """Create ``Sigma = amplitude * I_dim``."""

        if dim < 1 or amplitude < 0:
            raise ValueError("dim must be positive and amplitude must be nonnegative")
        matrix = tuple(
            tuple(amplitude if row == column else 0.0 for column in range(dim))
            for row in range(dim)
        )
        return cls(matrix=matrix, name=f"isotropic_sigma_{amplitude:g}")

    @property
    def dim(self) -> int:
        return len(self.matrix)

    @property
    def brownian_dim(self) -> int:
        return len(self.matrix[0])

    def noise_matrix(self, particles: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        del time
        _check_particles(particles, self.dim)
        matrix = torch.as_tensor(self.matrix, device=particles.device, dtype=particles.dtype)
        return matrix.unsqueeze(0).expand(particles.shape[0], -1, -1)

    def covariance_divergence(
        self,
        particles: torch.Tensor,
        time: float = 0.0,
    ) -> torch.Tensor:
        del time
        _check_particles(particles, self.dim)
        return torch.zeros_like(particles)

    def metadata(self) -> Mapping[str, object]:
        matrix = torch.as_tensor(self.matrix, dtype=torch.float64)
        covariance = matrix @ matrix.T
        return {
            "name": self.name,
            "kind": "constant",
            "matrix": [list(row) for row in self.matrix],
            "covariance": covariance.tolist(),
        }


@dataclass(frozen=True)
class FunctionalDiffusion:
    """Wrap arbitrary state/time-dependent diffusion functions.

    ``noise_fn`` returns ``Sigma(x,t)`` with shape ``(N,d,r)``.
    ``covariance_divergence_fn`` returns ``div(Sigma Sigma^T)``. Supplying the
    latter explicitly keeps the probability-flow calculation general without
    hiding an additional high-order automatic-differentiation pass.
    """

    dim: int
    noise_fn: Callable[[torch.Tensor, float], torch.Tensor]
    covariance_divergence_fn: Callable[[torch.Tensor, float], torch.Tensor]
    name: str = "functional_diffusion"

    def __post_init__(self) -> None:
        if self.dim < 1 or not self.name:
            raise ValueError("dim must be positive and name must be nonempty")

    def noise_matrix(self, particles: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        _check_particles(particles, self.dim)
        matrix = self.noise_fn(particles, time)
        if matrix.ndim != 3 or matrix.shape[:2] != particles.shape or matrix.shape[2] < 1:
            raise ValueError("noise_fn must return shape (N, d, brownian_dim)")
        if matrix.device != particles.device or matrix.dtype != particles.dtype:
            raise ValueError("noise_fn output must use the particle device and dtype")
        if not torch.isfinite(matrix).all():
            raise FloatingPointError("noise_fn returned a nonfinite value")
        return matrix

    def covariance_divergence(
        self,
        particles: torch.Tensor,
        time: float = 0.0,
    ) -> torch.Tensor:
        _check_particles(particles, self.dim)
        divergence = self.covariance_divergence_fn(particles, time)
        if divergence.shape != particles.shape:
            raise ValueError("covariance_divergence_fn must return the particle shape")
        if divergence.device != particles.device or divergence.dtype != particles.dtype:
            raise ValueError(
                "covariance_divergence_fn output must use the particle device and dtype"
            )
        if not torch.isfinite(divergence).all():
            raise FloatingPointError("covariance_divergence_fn returned a nonfinite value")
        return divergence

    def metadata(self) -> Mapping[str, object]:
        return {"name": self.name, "kind": "functional", "dim": self.dim}


@dataclass(frozen=True)
class FunctionalGame:
    """Wrap a velocity function as a game without adding package code."""

    dim: int
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor]
    name: str = "custom_game"

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if not self.name:
            raise ValueError("name must be nonempty")

    def velocity(self, particles: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        _check_particles(particles, self.dim)
        result = self.velocity_fn(particles, time)
        if result.shape != particles.shape:
            raise ValueError("velocity_fn must return the particle shape")
        if result.device != particles.device or result.dtype != particles.dtype:
            raise ValueError("velocity_fn output must use the particle device and dtype")
        return result

    def metadata(self) -> Mapping[str, object]:
        return {"name": self.name, "dim": self.dim, "kind": "functional"}


@dataclass(frozen=True)
class CournotGame:
    r"""Cournot best-response dynamics in one symmetric block.

    For ``s_i = sum_{j != i} x_j``, the dynamics are

    ``dx_i/dt = 2 b (max(mu s_i (1-s_i), 0) - x_i)``.
    """

    dim: int = 5
    b: float = 2.0
    mu: float = 7.0 / 4.0
    name: str = "cournot_5d_b2_mu7_4"

    def __post_init__(self) -> None:
        if self.dim < 2:
            raise ValueError("a Cournot block needs at least two players")
        if self.b <= 0 or self.mu <= 0:
            raise ValueError("b and mu must be positive")

    def velocity(self, particles: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        del time
        _check_particles(particles, self.dim)
        rivals = particles.sum(dim=-1, keepdim=True) - particles
        response = (self.mu * rivals * (1.0 - rivals)).clamp_min(0.0)
        return 2.0 * self.b * (response - particles)

    def metadata(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "kind": "cournot",
            "dim": self.dim,
            "b": self.b,
            "mu": self.mu,
        }


@dataclass(frozen=True)
class BlockCournotGame:
    """Independent Cournot blocks with a different ``mu`` in each block."""

    block_mus: tuple[float, ...] = (7.0 / 4.0, 33.0 / 20.0)
    block_size: int = 5
    b: float = 2.0
    name: str = "cournot_two_block_10d"

    def __post_init__(self) -> None:
        if not self.block_mus or any(mu <= 0 for mu in self.block_mus):
            raise ValueError("block_mus must contain positive values")
        if self.block_size < 2 or self.b <= 0:
            raise ValueError("block_size must be at least two and b must be positive")

    @property
    def dim(self) -> int:
        return self.block_size * len(self.block_mus)

    def velocity(self, particles: torch.Tensor, time: float = 0.0) -> torch.Tensor:
        del time
        _check_particles(particles, self.dim)
        blocks = []
        for block, mu in enumerate(self.block_mus):
            start = block * self.block_size
            values = particles[..., start : start + self.block_size]
            rivals = values.sum(dim=-1, keepdim=True) - values
            response = (mu * rivals * (1.0 - rivals)).clamp_min(0.0)
            blocks.append(2.0 * self.b * (response - values))
        return torch.cat(blocks, dim=-1)

    def metadata(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "kind": "block_cournot",
            "dim": self.dim,
            "block_size": self.block_size,
            "block_mus": list(self.block_mus),
            "b": self.b,
        }


def _check_particles(particles: torch.Tensor, dim: int) -> None:
    if particles.ndim != 2 or particles.shape[0] < 1 or particles.shape[1] != dim:
        raise ValueError(f"particles must have nonempty shape (N, {dim})")
