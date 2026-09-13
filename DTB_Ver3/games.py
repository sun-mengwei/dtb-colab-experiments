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
