"""Target functions for approximation and PDE experiments."""

from __future__ import annotations

import math
from typing import Callable, Dict

import torch


TensorFn = Callable[[torch.Tensor], torch.Tensor]


def smooth(x: torch.Tensor) -> torch.Tensor:
    """Low-frequency smooth target; most networks should fit this easily."""
    z = x[:, 0]
    return torch.sin(math.pi * z) + 0.25 * torch.cos(2.0 * math.pi * z)


def oscillatory(x: torch.Tensor) -> torch.Tensor:
    """High-frequency target; exposes spectral-bias differences."""
    z = x[:, 0]
    return 0.7 * torch.sin(8.0 * math.pi * z) + 0.3 * torch.sin(21.0 * math.pi * z)


def localized(x: torch.Tensor) -> torch.Tensor:
    """Two narrow Gaussian bumps; tests whether a model can localize detail."""
    z = x[:, 0]
    bump_left = torch.exp(-90.0 * (z + 0.45) ** 2)
    bump_right = 0.65 * torch.exp(-180.0 * (z - 0.35) ** 2)
    return bump_left - bump_right


def multiscale(x: torch.Tensor) -> torch.Tensor:
    """Large smooth trend plus small high-frequency signal."""
    z = x[:, 0]
    return torch.sin(math.pi * z) + 0.15 * torch.sin(35.0 * math.pi * z)


def piecewise(x: torch.Tensor) -> torch.Tensor:
    """Discontinuous target; useful for seeing Gibbs-like overshoot."""
    z = x[:, 0]
    return torch.where(z < 0.0, -0.65 + 0.2 * z, 0.55 + 0.35 * z)


APPROX_TARGETS: Dict[str, TensorFn] = {
    "smooth": smooth,
    "oscillatory": oscillatory,
    "localized": localized,
    "multiscale": multiscale,
    "piecewise": piecewise,
}


def poisson_exact(x: torch.Tensor) -> torch.Tensor:
    """Exact solution on [0, 1] with zero boundary values."""
    z = x[:, 0]
    return torch.sin(math.pi * z) + 0.25 * torch.sin(4.0 * math.pi * z)


def poisson_rhs(x: torch.Tensor) -> torch.Tensor:
    """Right-hand side f for -u'' = f using poisson_exact."""
    z = x[:, 0]
    return (
        (math.pi ** 2) * torch.sin(math.pi * z)
        + 0.25 * (4.0 * math.pi) ** 2 * torch.sin(4.0 * math.pi * z)
    )
