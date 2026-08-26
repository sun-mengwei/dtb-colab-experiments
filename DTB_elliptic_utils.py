"""Shared PyTorch utilities for the DTB--Deep Ritz elliptic experiments.

We approximate the zero-Dirichlet problem

    -div(A grad u) + c u = f  in Omega,       u = 0 on boundary(Omega)

with a selected parameter-tangent trial function

    u_{theta,alpha}(x) = D_theta T_theta(x)[I] alpha.

Notation:
    theta             (p,)        flattened network parameters
    I / indices       (r,)        selected parameter coordinates, r << p
    alpha             (r,)        coefficients in the selected tangent space
    direction         (p,)        alpha embedded at coordinates I
    points / x        (N,d)       spatial quadrature points
    J / features      (N,r)       J_ni = partial T_theta(x_n)/partial theta_Ii
    B                 (N,d,r)     B_nki = partial J_ni/partial x_k
    A / diffusion     (d,d) or    symmetric diffusion tensor, constant or
                      (N,d,d)     evaluated at the quadrature points
    c / reaction      scalar or   reaction coefficient, constant or sampled
                      (N,)
    G / stiffness     (r,r)       Ritz (Gram/stiffness) matrix
    b / load          (r,)        Ritz load vector

The empirical inner objective is

    E_theta(alpha) = 1/2 alpha^T G_theta alpha - b_theta^T alpha,

where

    G_ij = integral_Omega [(grad J_i)^T A grad J_j + c J_i J_j] dx,
    b_i  = integral_Omega f J_i dx.

Strategy:
  * Patch only the selected coordinates before applying ``jacrev``; this
    avoids constructing the full (N,p) parameter Jacobian when r << p.
  * Approximate volume integrals by uniform Monte Carlo/Sobol averages.
  * Stabilize the small inner system with an optional scale-aware ridge.
  * Differentiate the outer energy with ``alpha`` fixed (envelope step), so
    no dense parameter Hessian or derivative through the linear solve is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch.func import functional_call, grad, jacrev, jvp, vmap

Tensor = torch.Tensor
ScalarFunction = Callable[[Tensor], Tensor]
TrialFunction = Callable[[Tensor, Tensor], Tensor]


# -------------------------- models and metadata --------------------------


class MLP(nn.Module):
    """Small scalar-valued MLP ``R^d -> R`` used by all notebooks."""

    def __init__(
        self,
        dimension: int,
        width: int = 24,
        depth: int = 2,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if dimension < 1 or width < 1 or depth < 1:
            raise ValueError("dimension, width, and depth must be positive")
        activations = {"tanh": nn.Tanh, "gelu": nn.GELU, "sine": None}
        if activation not in activations:
            raise ValueError(f"unknown activation: {activation}")

        layers: list[nn.Module] = []
        incoming = dimension
        for _ in range(depth):
            layers.append(nn.Linear(incoming, width))
            layers.append(Sine() if activation == "sine" else activations[activation]())
            incoming = width
        layers.append(nn.Linear(incoming, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the network at points with trailing dimension ``d``."""

        return self.network(x).squeeze(-1)


class Sine(nn.Module):
    """Elementwise sine activation."""

    def forward(self, x: Tensor) -> Tensor:
        """Return ``sin(x)`` elementwise."""

        return torch.sin(x)


@dataclass(frozen=True)
class ParameterSpec:
    """Names, tensor shapes, and flattened sizes needed to reconstruct theta."""

    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    sizes: tuple[int, ...]


@dataclass(frozen=True)
class RitzSolution:
    """Inner coefficients and numerical diagnostics for the stabilized solve."""

    alpha: Tensor
    ridge_shift: float
    relative_residual: float


# ------------------------- flat-parameter utilities -------------------------


def flatten_parameters(model: nn.Module) -> tuple[Tensor, ParameterSpec]:
    """Return trainable parameters as ``theta=(p,)`` without modifying ``model``."""

    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not named:
        raise ValueError("model has no trainable parameters")
    theta = torch.cat([value.detach().reshape(-1) for _, value in named]).clone()
    return theta, ParameterSpec(
        names=tuple(name for name, _ in named),
        shapes=tuple(value.shape for _, value in named),
        sizes=tuple(value.numel() for _, value in named),
    )


def unflatten_parameters(theta: Tensor, spec: ParameterSpec) -> dict[str, Tensor]:
    """Reconstruct the named parameter tensors encoded by flat ``theta``."""

    if theta.numel() != sum(spec.sizes):
        raise ValueError("flat parameter size does not match the specification")
    pieces = torch.split(theta, spec.sizes)
    return {
        name: piece.reshape(shape)
        for name, shape, piece in zip(spec.names, spec.shapes, pieces)
    }


def call_flat(model: nn.Module, spec: ParameterSpec, theta: Tensor, x: Tensor) -> Tensor:
    """Evaluate ``model(x)`` using differentiable flat parameters ``theta``."""

    parameters = dict(model.named_parameters())
    parameters.update(unflatten_parameters(theta, spec))
    return functional_call(model, parameters, (x,))


# ---------------------- boundary trial and sampling ----------------------


def box_boundary_factor(x: Tensor, normalize: bool = True) -> Tensor:
    """Hard Dirichlet factor on ``[-1,1]^d``.

    The optional ``(3/2)^d`` scaling makes the factor have unit mean for
    uniform points.  It changes conditioning but not its zero boundary trace.
    """

    factor = torch.prod(1.0 - x.square(), dim=-1)
    if normalize:
        factor = factor * (1.5 ** x.shape[-1])
    return factor


def make_box_trial(
    model: nn.Module,
    spec: ParameterSpec,
    *,
    normalize_boundary: bool = True,
) -> TrialFunction:
    """Return ``T_theta(x)=rho(x) N_theta(x)`` with exact zero boundary trace."""

    def trial(theta: Tensor, x: Tensor) -> Tensor:
        return box_boundary_factor(x, normalize_boundary) * call_flat(
            model, spec, theta, x
        )

    return trial


def sample_box(
    count: int,
    dimension: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Draw ``count`` scrambled Sobol points in ``[-1,1]^dimension``."""

    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    return 2.0 * engine.draw(count).to(dtype=dtype, device=device) - 1.0


def sample_box_boundary(
    points_per_face: int,
    dimension: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Sample both faces normal to each axis; return ``(2*d*m,d)`` points."""

    clouds = []
    for coordinate in range(dimension):
        base = sample_box(
            points_per_face, dimension, seed + coordinate, dtype=dtype, device=device
        )
        for side in (-1.0, 1.0):
            face = base.clone()
            face[:, coordinate] = side
            clouds.append(face)
    return torch.cat(clouds)


# ------------------------- selected tangent basis -------------------------


def select_parameter_indices(
    parameter_count: int,
    tangent_dimension: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Choose ``r`` reproducible coordinates from ``p`` without replacement."""

    if not 1 <= tangent_dimension <= parameter_count:
        raise ValueError("tangent_dimension must lie in [1, parameter_count]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(parameter_count, generator=generator)[:tangent_dimension]
    return torch.sort(indices).values.to(device=device)


def expand_direction(alpha: Tensor, indices: Tensor, parameter_count: int) -> Tensor:
    """Embed ``alpha=(r,)`` at ``indices`` in a zero vector of shape ``(p,)``."""

    direction = torch.zeros(parameter_count, dtype=alpha.dtype, device=alpha.device)
    return direction.index_copy(0, indices, alpha)


def selected_tangent_basis(
    trial: TrialFunction,
    theta: Tensor,
    points: Tensor,
    indices: Tensor,
    *,
    chunk_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """Evaluate the selected tangent features ``J`` and their gradients ``B``.

    For ``I=(I_1,...,I_r)``, this returns

        J_ni  = partial T_theta(x_n) / partial theta_{I_i},       shape (N,r),
        B_nki = partial J_ni / partial x_k,                       shape (N,d,r).

    Differentiating the patched ``r``-vector directly avoids constructing the
    full ``(N,p)`` parameter Jacobian.  Point chunks limit transform memory;
    they do not change the mathematical result.
    """

    selected = theta.index_select(0, indices)

    def selected_trial(values: Tensor, point: Tensor) -> Tensor:
        full = theta.index_copy(0, indices, values)
        return trial(full, point)

    jac_one = jacrev(selected_trial, argnums=0)
    spatial_jac_one = jacrev(jac_one, argnums=1)
    feature_chunks, spatial_chunks = [], []
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        feature_chunks.append(
            vmap(jac_one, in_dims=(None, 0))(selected, chunk)
        )  # (N,r)
        raw = vmap(spatial_jac_one, in_dims=(None, 0))(selected, chunk)  # (N,r,d)
        spatial_chunks.append(raw.transpose(1, 2))
    return torch.cat(feature_chunks), torch.cat(spatial_chunks)


def _diffusion_energy(gradients: Tensor, diffusion: Tensor | None) -> Tensor:
    """Return pointwise ``grad(u)^T A grad(u)`` for constant or sampled ``A``."""

    if diffusion is None:
        return torch.sum(gradients.square(), dim=-1)
    if diffusion.ndim == 2:
        return torch.einsum("nd,de,ne->n", gradients, diffusion, gradients)
    if diffusion.ndim == 3:
        return torch.einsum("nd,nde,ne->n", gradients, diffusion, gradients)
    raise ValueError("diffusion must have shape (d,d) or (N,d,d)")


# -------------------------- Ritz assembly and solve --------------------------


def assemble_ritz_system(
    features: Tensor,
    feature_gradients: Tensor,
    forcing: Tensor,
    volume: float,
    *,
    diffusion: Tensor | None = None,
    reaction: float | Tensor = 0.0,
) -> tuple[Tensor, Tensor]:
    """Assemble the empirical elliptic Ritz matrix ``G`` and load ``b``.

    With tangent features ``J_i`` and their spatial gradients ``B_i``,

        G_ij = integral [(B_i)^T A B_j + c J_i J_j] dx,
        b_i  = integral f J_i dx,
        E_theta(alpha) = 1/2 alpha^T G alpha - b^T alpha.

    ``features`` has shape ``(N,r)`` and ``feature_gradients`` has shape
    ``(N,d,r)``.  ``diffusion`` may be constant ``(d,d)`` or sampled
    ``(N,d,d)``; ``reaction`` may be scalar or sampled ``(N,)``.  Uniform
    quadrature uses the weight ``volume/N``.  The final symmetrization removes
    only floating-point asymmetry; it does not make an indefinite ``A`` valid.
    """

    count = features.shape[0]
    # ``stiffness`` is G: first assemble integral (grad J_i)^T A grad J_j.
    if diffusion is None:
        stiffness = torch.einsum(
            "ndi,ndj->ij", feature_gradients, feature_gradients
        )
    elif diffusion.ndim == 2:
        stiffness = torch.einsum(
            "ndi,de,nej->ij", feature_gradients, diffusion, feature_gradients
        )
    elif diffusion.ndim == 3:
        stiffness = torch.einsum(
            "ndi,nde,nej->ij", feature_gradients, diffusion, feature_gradients
        )
    else:
        raise ValueError("diffusion must have shape (d,d) or (N,d,d)")

    # Add integral c J_i J_j to G when the reaction coefficient is nonzero.
    reaction_values = torch.as_tensor(
        reaction, dtype=features.dtype, device=features.device
    )
    if bool(torch.any(reaction_values != 0.0)):
        if reaction_values.ndim == 0:
            stiffness = stiffness + reaction_values * (features.T @ features)
        else:
            stiffness = stiffness + torch.einsum(
                "n,ni,nj->ij", reaction_values, features, features
            )

    stiffness = (volume / count) * stiffness
    stiffness = 0.5 * (stiffness + stiffness.T)
    # b_i = integral f J_i, with the same uniform quadrature weight.
    load = (volume / count) * (features.T @ forcing)
    return stiffness, load


def solve_ritz_system(
    stiffness: Tensor,
    load: Tensor,
    *,
    ridge_relative: float = 1.0e-8,
) -> RitzSolution:
    """Solve ``(G + lambda I) alpha = b`` and report its relative residual.

    The shift is scale aware:

        lambda = ridge_relative * max(trace(G)/r, machine_epsilon).

    ``torch.linalg.solve`` is preferable to forming ``G^{-1}``: it is more
    accurate and uses less work.  Set ``ridge_relative=0`` only when ``G`` is
    reliably nonsingular and well conditioned; otherwise the direct solve may
    fail or amplify sampling error.  A positive ridge stabilizes redundant or
    nearly dependent tangent features, but biases ``alpha`` and changes the
    inner problem.  For genuinely rank-deficient systems where a minimum-norm
    solution is desired, an SVD/pseudoinverse solver is the appropriate
    alternative rather than a large ridge.

    ``relative_residual`` measures ``||(G+lambda I)alpha-b||/||b||`` for the
    regularized system, not the unshifted residual ``||G alpha-b||/||b||``.
    """

    if ridge_relative < 0.0:
        raise ValueError("ridge_relative must be nonnegative")
    scale = (torch.trace(stiffness) / stiffness.shape[0]).clamp_min(
        torch.finfo(stiffness.dtype).eps
    )
    shift = ridge_relative * scale
    system = stiffness + shift * torch.eye(
        stiffness.shape[0], dtype=stiffness.dtype, device=stiffness.device
    )
    alpha = torch.linalg.solve(system, load)
    denominator = torch.linalg.norm(load).clamp_min(torch.finfo(load.dtype).eps)
    residual = torch.linalg.norm(system @ alpha - load) / denominator
    return RitzSolution(
        alpha=alpha,
        ridge_shift=float(shift),
        relative_residual=float(residual),
    )


def solve_tangent_ritz(
    trial: TrialFunction,
    theta: Tensor,
    indices: Tensor,
    points: Tensor,
    forcing: Tensor,
    volume: float,
    *,
    diffusion: Tensor | None = None,
    reaction: float | Tensor = 0.0,
    ridge_relative: float = 1.0e-8,
    chunk_size: int = 256,
) -> tuple[RitzSolution, Tensor, Tensor]:
    """Build ``(J,B)``, assemble ``(G,b)``, and solve the inner Ritz problem."""

    features, gradients = selected_tangent_basis(
        trial, theta, points, indices, chunk_size=chunk_size
    )
    stiffness, load = assemble_ritz_system(
        features,
        gradients,
        forcing,
        volume,
        diffusion=diffusion,
        reaction=reaction,
    )
    solution = solve_ritz_system(
        stiffness, load, ridge_relative=ridge_relative
    )
    return solution, stiffness, load


# ------------------- tangent evaluation and outer update -------------------


def tangent_values_and_gradients(
    trial: TrialFunction,
    theta: Tensor,
    direction: Tensor,
    points: Tensor,
    *,
    chunk_size: int = 512,
) -> tuple[Tensor, Tensor]:
    """Evaluate ``u=D_theta T_theta[direction]`` and ``grad_x u``.

    Returns values with shape ``(N,)`` and spatial gradients with shape
    ``(N,d)``.  ``jvp`` computes the tangent action without materializing the
    full parameter Jacobian.
    """

    def value_one(parameters: Tensor, point: Tensor) -> Tensor:
        return jvp(
            lambda candidate: trial(candidate, point),
            (parameters,),
            (direction,),
        )[1]

    gradient_one = jacrev(value_one, argnums=1)
    value_chunks, gradient_chunks = [], []
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        value_chunks.append(vmap(value_one, in_dims=(None, 0))(theta, chunk))
        gradient_chunks.append(vmap(gradient_one, in_dims=(None, 0))(theta, chunk))
    return torch.cat(value_chunks), torch.cat(gradient_chunks)


def predict_tangent(
    trial: TrialFunction,
    theta: Tensor,
    direction: Tensor,
    points: Tensor,
    *,
    chunk_size: int = 1024,
) -> Tensor:
    """Evaluate only ``u(x)=D_theta T_theta(x)[direction]`` at ``points``."""

    def value_one(point: Tensor) -> Tensor:
        return jvp(
            lambda candidate: trial(candidate, point),
            (theta,),
            (direction,),
        )[1]

    return torch.cat(
        [
            vmap(value_one)(points[start : start + chunk_size])
            for start in range(0, points.shape[0], chunk_size)
        ]
    )


def direct_ritz_energy(
    trial: TrialFunction,
    theta: Tensor,
    direction: Tensor,
    points: Tensor,
    forcing: Tensor,
    volume: float,
    *,
    diffusion: Tensor | None = None,
    reaction: float | Tensor = 0.0,
) -> Tensor:
    """Evaluate the empirical elliptic energy of a tangent direction.

    ``E(u) = integral [1/2 grad(u)^T A grad(u) + 1/2 c u^2 - f u] dx``.
    """

    values, gradients = tangent_values_and_gradients(
        trial, theta, direction, points
    )
    density = 0.5 * _diffusion_energy(gradients, diffusion) - forcing * values
    reaction_values = torch.as_tensor(reaction, dtype=values.dtype, device=values.device)
    if bool(torch.any(reaction_values != 0.0)):
        density = density + 0.5 * reaction_values * values.square()
    return volume * torch.mean(density)


def envelope_gradient(
    trial: TrialFunction,
    theta: Tensor,
    direction: Tensor,
    points: Tensor,
    forcing: Tensor,
    volume: float,
    *,
    diffusion: Tensor | None = None,
    reaction: float | Tensor = 0.0,
) -> Tensor:
    """Differentiate the outer Ritz energy while holding ``direction`` fixed.

    This is the envelope gradient when the inner problem is solved exactly.
    With a non-negligible ridge or inner residual, it is instead the gradient
    of the unregularized energy evaluated at the approximate inner solution;
    the ridge dependence itself is intentionally not differentiated.
    """

    fixed_direction = direction.detach()
    objective = lambda parameters: direct_ritz_energy(
        trial,
        parameters,
        fixed_direction,
        points,
        forcing,
        volume,
        diffusion=diffusion,
        reaction=reaction,
    )
    return grad(objective)(theta)


def armijo_update(
    objective: Callable[[Tensor], Tensor],
    theta: Tensor,
    gradient: Tensor,
    *,
    initial_step: float = 1.0e-2,
    contraction: float = 0.5,
    sufficient_decrease: float = 1.0e-4,
    max_backtracks: int = 16,
) -> tuple[Tensor, float]:
    """Apply Armijo backtracking to the fixed-inner-direction outer energy.

    The callback is expected to keep the current inner direction fixed.  Thus
    acceptance guarantees decrease of this local envelope surrogate, not of an
    objective that re-solves the inner system at every trial point.
    """

    before = float(objective(theta))
    norm_squared = float(torch.sum(gradient.square()))
    step = initial_step
    for _ in range(max_backtracks):
        candidate = (theta - step * gradient).detach()
        if float(objective(candidate)) <= before - sufficient_decrease * step * norm_squared:
            return candidate, step
        step *= contraction
    return theta.detach(), 0.0


# ------------------- manufactured problems and diagnostics -------------------


def manufactured_forcing(
    exact_solution: ScalarFunction,
    points: Tensor,
    *,
    diffusion: Tensor | None = None,
    reaction: float = 0.0,
) -> Tensor:
    """Compute ``f=-tr(A Hessian(u))+c u`` for an exact solution and constant A.

    This equals ``-div(A grad u)+c u`` only for spatially constant ``A``.  A
    variable diffusion tensor also contributes derivatives of ``A``.
    """

    hessian = vmap(jacrev(jacrev(exact_solution)))(points)
    if diffusion is None:
        elliptic = -torch.diagonal(hessian, dim1=-2, dim2=-1).sum(-1)
    elif diffusion.ndim == 2:
        elliptic = -torch.einsum("de,nde->n", diffusion, hessian)
    else:
        raise ValueError("manufactured_forcing currently expects constant diffusion")
    return elliptic + reaction * vmap(exact_solution)(points)


def relative_l2_error(
    trial: TrialFunction,
    theta: Tensor,
    direction: Tensor,
    points: Tensor,
    exact_solution: ScalarFunction,
) -> float:
    """Return ``||u_DTB-u_exact||_2 / ||u_exact||_2`` on held-out points."""

    prediction = predict_tangent(trial, theta, direction, points)
    reference = vmap(exact_solution)(points)
    return float(
        torch.linalg.norm(prediction - reference)
        / torch.linalg.norm(reference).clamp_min(torch.finfo(reference.dtype).eps)
    )


def matrix_ritz_energy(stiffness: Tensor, load: Tensor, alpha: Tensor) -> Tensor:
    """Evaluate ``1/2 alpha^T G alpha - b^T alpha`` from assembled ``G,b``."""

    return 0.5 * alpha @ stiffness @ alpha - load @ alpha


def summarize_history(history: Sequence[dict[str, float]]) -> None:
    """Print one compact line per monitored outer iteration."""

    for row in history:
        print(
            f"step={int(row['step']):3d}  F(theta)={row['F_theta']: .3e}  "
            f"rel-L2={row['relative_l2']:.3e}  "
            f"inner-res={row['inner_residual']:.1e}  "
            f"||alpha||={row['alpha_norm']:.2e}"
        )
