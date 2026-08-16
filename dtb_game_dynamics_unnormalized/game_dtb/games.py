"""Example game drifts ``b(x)`` for the distributional dynamics."""

from __future__ import annotations

import torch


def nonlinear_network_payoff(
    x: torch.Tensor,
    interaction_matrix: torch.Tensor,
    bias: torch.Tensor,
    mu: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    r"""Payoffs for a smooth, nonconcave directed network game.

    Player ``i`` has payoff

    ``Pi_i = r_i*x_i + 0.5*mu_i*x_i^2 - 0.25*x_i^4``
    ``       + beta_i*x_i*tanh(sum_j G_ij*x_j)``.

    The diagonal of ``G`` must be zero.  Consequently the network aggregate
    does not change when player ``i`` differentiates with respect to its own
    action, and the own-action payoff gradient is exactly
    ``nonlinear_network_drift``.  A nonsymmetric ``G`` makes this a
    non-potential game in general.
    """

    _validate_network_game_inputs(x, interaction_matrix, bias, mu, beta)
    aggregate = x @ interaction_matrix.T
    return (
        bias * x
        + 0.5 * mu * x.square()
        - 0.25 * x.pow(4)
        + beta * x * torch.tanh(aggregate)
    )


def nonlinear_network_drift(
    x: torch.Tensor,
    interaction_matrix: torch.Tensor,
    bias: torch.Tensor,
    mu: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    r"""Stacked payoff-gradient dynamics for the nonlinear network game.

    The equilibrium equation is

    ``0 = r_i + mu_i*x_i - x_i^3 + beta_i*tanh((Gx)_i)``.

    It can have many roots, while the directed interactions make dynamical
    stability depend on the spectrum of a nonsymmetric Jacobian.
    """

    _validate_network_game_inputs(x, interaction_matrix, bias, mu, beta)
    aggregate = x @ interaction_matrix.T
    return bias + mu * x - x.pow(3) + beta * torch.tanh(aggregate)


def nonlinear_network_jacobian(
    x: torch.Tensor,
    interaction_matrix: torch.Tensor,
    mu: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    r"""Analytic state Jacobian of ``nonlinear_network_drift``.

    For one point this returns shape ``(d,d)``; for a batch with shape
    ``(...,d)`` it returns ``(...,d,d)``.
    """

    dim = x.shape[-1]
    zero_bias = torch.zeros(dim, device=x.device, dtype=x.dtype)
    _validate_network_game_inputs(
        x, interaction_matrix, zero_bias, mu, beta
    )
    aggregate = x @ interaction_matrix.T
    diagonal = torch.diag_embed(mu - 3.0 * x.square())
    coupling_scale = beta * (1.0 - torch.tanh(aggregate).square())
    return diagonal + torch.diag_embed(coupling_scale) @ interaction_matrix


def _validate_network_game_inputs(
    x: torch.Tensor,
    interaction_matrix: torch.Tensor,
    bias: torch.Tensor,
    mu: torch.Tensor,
    beta: torch.Tensor,
) -> None:
    """Check the shapes and zero-self-interaction assumption once per call."""

    if x.ndim < 1 or x.shape[-1] < 2:
        raise ValueError("the nonlinear network game needs at least two players")
    dim = x.shape[-1]
    if interaction_matrix.shape != (dim, dim):
        raise ValueError(f"interaction_matrix must have shape ({dim},{dim})")
    for name, value in (("bias", bias), ("mu", mu), ("beta", beta)):
        if value.shape not in (torch.Size([]), torch.Size([dim])):
            raise ValueError(f"{name} must be scalar or have shape ({dim},)")
    diagonal = torch.diagonal(interaction_matrix)
    if bool(torch.any(diagonal != 0)):
        raise ValueError("interaction_matrix must have a zero diagonal")


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
