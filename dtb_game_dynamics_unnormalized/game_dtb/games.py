"""Example game drifts ``b(x)`` for the distributional dynamics."""

from __future__ import annotations

import torch


def cournot_duopoly_drift(
    x: torch.Tensor, b: float = 1.0, mu: float = 2.0
) -> torch.Tensor:
    """Non-potential two-player Cournot drift reused from ``run_game_dtb.py``.

    For ``b=1`` and ``mu=2``, the stable Nash equilibrium is ``(0.5, 0.5)``.
    The method in the supplied algorithm is unconstrained, so this function
    does not clip strategies to a box.
    """

    if x.shape[-1] != 2:
        raise ValueError("the Cournot game requires dim=2")
    x1, x2 = x[..., 0], x[..., 1]
    drift1 = -2.0 * b * x1 + 2.0 * b * mu * x2 - 2.0 * b * mu * x2.square()
    drift2 = -2.0 * b * x2 + 2.0 * b * mu * x1 - 2.0 * b * mu * x1.square()
    return torch.stack((drift1, drift2), dim=-1)


def linear_quadratic_drift(
    x: torch.Tensor,
    target: float = 0.5,
    contraction: float = 1.0,
    rotation: float = 0.35,
) -> torch.Tensor:
    """Stable two-player linear-quadratic drift with a rotational component.

    The rotation makes the field non-gradient when nonzero, while contraction
    drives the mean strategy toward ``(target, target)``.  This smooth game is
    useful as a first diagnostic before the nonlinear Cournot experiment.
    """

    if x.shape[-1] != 2:
        raise ValueError("the linear-quadratic example requires dim=2")
    centered = x - target
    first = -contraction * centered[..., 0] + rotation * centered[..., 1]
    second = -rotation * centered[..., 0] - contraction * centered[..., 1]
    return torch.stack((first, second), dim=-1)
