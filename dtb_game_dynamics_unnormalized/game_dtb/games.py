"""Example game drifts ``b(x)`` for the distributional dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OscillatoryGameParams:
    r"""Parameters for the two-player oscillatory potential game.

    The common payoff is

    .. math::

       \Phi(x) = -\frac{\lambda}{2}\lVert x\rVert^2
       -\frac{\gamma}{2}(x_1-x_2)^2
       +\frac{\epsilon}{\omega}\sum_i \cos(\omega x_i).

    Both players ascend this potential, so the deterministic game velocity is
    its gradient.  ``lambda_`` is named with a trailing underscore because
    ``lambda`` is a Python keyword.
    """

    lambda_: float = 0.5
    epsilon: float = 0.5
    omega: float = 2.0 * math.pi
    gamma: float = 0.0

    def validate(self) -> None:
        values = (self.lambda_, self.epsilon, self.omega, self.gamma)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("oscillatory game parameters must be finite")
        if self.lambda_ <= 0.0:
            raise ValueError("lambda_ must be positive")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative")
        if self.omega <= 0.0:
            raise ValueError("omega must be positive")
        if self.gamma < 0.0:
            raise ValueError("gamma must be nonnegative")


def oscillatory_potential(
    x: torch.Tensor, params: OscillatoryGameParams
) -> torch.Tensor:
    r"""Evaluate the identical-interest payoff ``Phi`` at ``(..., 2)`` points."""

    _validate_two_player_points(x, "oscillatory potential game")
    params.validate()
    first, second = x.unbind(dim=-1)
    confinement = -0.5 * params.lambda_ * (first.square() + second.square())
    coupling = -0.5 * params.gamma * (first - second).square()
    wells = (params.epsilon / params.omega) * (
        torch.cos(params.omega * first) + torch.cos(params.omega * second)
    )
    return confinement + coupling + wells


def oscillatory_game_velocity(
    x: torch.Tensor, params: OscillatoryGameParams
) -> torch.Tensor:
    r"""Return the game pseudo-gradient ``b=grad(Phi)`` with shape ``(..., 2)``.

    This is the drift/velocity supplied to :class:`NeuralDTBGameDynamics` for
    the deterministic experiment (the diffusion matrix is zero).
    """

    _validate_two_player_points(x, "oscillatory potential game")
    params.validate()
    first, second = x.unbind(dim=-1)
    first_velocity = (
        -params.lambda_ * first
        - params.gamma * (first - second)
        - params.epsilon * torch.sin(params.omega * first)
    )
    second_velocity = (
        -params.lambda_ * second
        - params.gamma * (second - first)
        - params.epsilon * torch.sin(params.omega * second)
    )
    return torch.stack((first_velocity, second_velocity), dim=-1)


def oscillatory_game_jacobian(
    x: torch.Tensor, params: OscillatoryGameParams
) -> torch.Tensor:
    r"""Return the analytic game Jacobian ``Db`` with shape ``(..., 2, 2)``."""

    _validate_two_player_points(x, "oscillatory potential game")
    params.validate()
    diagonal = (
        -params.lambda_
        - params.gamma
        - params.epsilon * params.omega * torch.cos(params.omega * x)
    )
    result = torch.zeros(
        *x.shape[:-1], 2, 2, device=x.device, dtype=x.dtype
    )
    result[..., 0, 0] = diagonal[..., 0]
    result[..., 1, 1] = diagonal[..., 1]
    result[..., 0, 1] = params.gamma
    result[..., 1, 0] = params.gamma
    return result


def _validate_two_player_points(x: torch.Tensor, game_name: str) -> None:
    if x.ndim < 1 or x.shape[-1] != 2:
        raise ValueError(f"the {game_name} requires points with shape (..., 2)")


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


def cournot_five_player_drift(
    x: torch.Tensor, b: float = 1.0, mu: float = 2.0
) -> torch.Tensor:
    """Five-player nonnegative best-response dynamics from Section 4.7.4.

    The listed equilibria with one zero component satisfy the constrained
    first-order condition rather than the unconstrained gradient equation.
    Therefore the best response ``mu*r_i*(1-r_i)`` is projected onto the
    nonnegative strategy set before forming ``2b*(BR_i-x_i)``.
    """

    if x.shape[-1] != 5:
        raise ValueError("the five-player Cournot game requires dim=5")
    opponents_total = x.sum(dim=-1, keepdim=True) - x
    best_response = torch.clamp_min(
        mu * opponents_total * (1.0 - opponents_total), 0.0
    )
    return 2.0 * b * (best_response - x)


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
