"""Example game drifts ``b(x)`` for the distributional dynamics."""

from __future__ import annotations

import torch


def cournot_multiplayer_payoff(
    x: torch.Tensor, b: float = 1.0, mu: float = 2.0, d: float = 0.0
) -> torch.Tensor:
    """Equilibrium-consistent payoff for the multiplayer Cournot game.

    Let ``r_i=sum_{j != i} x_j``.  Interpreting the two aggregate terms in
    the displayed cost as the opponents' total ``r_i`` gives

    ``Pi_i = -d - b*x_i^2 + 2*b*mu*x_i*r_i*(1-r_i)``.

    Its own-action gradient is exactly ``cournot_multiplayer_drift``.  Using
    the total ``S=sum_j x_j`` in those cost terms instead would give a
    different gradient and would not admit the five equilibria reported with
    the three-player example.
    """

    if x.shape[-1] < 2:
        raise ValueError("the multiplayer Cournot game needs at least two players")
    opponents_total = x.sum(dim=-1, keepdim=True) - x
    return (
        -d
        - b * x.square()
        + 2.0 * b * mu * x * opponents_total * (1.0 - opponents_total)
    )


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
    return cournot_multiplayer_drift(x, b=b, mu=mu)


def cournot_three_player_drift(
    x: torch.Tensor, b: float = 1.0, mu: float = 2.0
) -> torch.Tensor:
    """Three-player non-potential Cournot payoff-gradient drift.

    For player ``i``, let ``r_i=sum_{j != i} x_j``.  Then

    ``b_i(x) = d Pi_i/d x_i = -2 b x_i + 2 b mu r_i - 2 b mu r_i^2``.

    With ``b=1`` and ``mu=2`` the known equilibria in the supplied source are
    ``(0,0,0)``, ``(3/8,3/8,3/8)``, and the three permutations of
    ``(1/2,1/2,0)``.
    """

    if x.shape[-1] != 3:
        raise ValueError("the three-player Cournot game requires dim=3")
    return cournot_multiplayer_drift(x, b=b, mu=mu)


def cournot_multiplayer_drift(
    x: torch.Tensor, b: float = 1.0, mu: float = 2.0
) -> torch.Tensor:
    """Own-action payoff gradient, also ``2b*(best_response-x_i)``."""

    if x.shape[-1] < 2:
        raise ValueError("the multiplayer Cournot game needs at least two players")
    opponents_total = x.sum(dim=-1, keepdim=True) - x
    return (
        -2.0 * b * x
        + 2.0 * b * mu * opponents_total
        - 2.0 * b * mu * opponents_total.square()
    )


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
