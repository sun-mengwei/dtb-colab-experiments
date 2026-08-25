"""Shared PyTorch utilities for the DTB--Deep Ritz Colab experiments.

The trial solution is a selected parameter-tangent vector

    u_{theta, alpha}(x) = D_theta T_theta(x)[I] alpha.

The inner problem is a small Ritz/Galerkin solve in ``alpha``.  The outer
step differentiates the Ritz energy with ``alpha`` fixed (the envelope
gradient), so no dense parameter Hessian is formed.
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


class MLP(nn.Module):
    """Small scalar MLP used by all notebooks."""

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
        return self.network(x).squeeze(-1)


class Sine(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


@dataclass(frozen=True)
class ParameterSpec:
    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    sizes: tuple[int, ...]


@dataclass(frozen=True)
class RitzSolution:
    alpha: Tensor
    ridge_shift: float
    relative_residual: float


def flatten_parameters(model: nn.Module) -> tuple[Tensor, ParameterSpec]:
    """Flatten all trainable parameters without modifying ``model``."""

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
    if theta.numel() != sum(spec.sizes):
        raise ValueError("flat parameter size does not match the specification")
    pieces = torch.split(theta, spec.sizes)
    return {
        name: piece.reshape(shape)
        for name, shape, piece in zip(spec.names, spec.shapes, pieces)
    }


def call_flat(model: nn.Module, spec: ParameterSpec, theta: Tensor, x: Tensor) -> Tensor:
    """Evaluate ``model`` with a differentiable flat parameter vector."""

    parameters = dict(model.named_parameters())
    parameters.update(unflatten_parameters(theta, spec))
    return functional_call(model, parameters, (x,))


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
    """Return ``T_theta = rho * N_theta`` with an exact zero boundary trace."""

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
    """Scrambled Sobol points in ``[-1,1]^d``."""

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
    """Sample both faces normal to every coordinate axis."""

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


def select_parameter_indices(
    parameter_count: int,
    tangent_dimension: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Select a reproducible coordinate subset of ``theta`` without replacement."""

    if not 1 <= tangent_dimension <= parameter_count:
        raise ValueError("tangent_dimension must lie in [1, parameter_count]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(parameter_count, generator=generator)[:tangent_dimension]
    return torch.sort(indices).values.to(device=device)


def expand_direction(alpha: Tensor, indices: Tensor, parameter_count: int) -> Tensor:
    """Embed selected tangent coefficients in the full parameter space."""

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
    """Return selected features ``J`` and spatial gradients ``B``.

    Shapes are ``J=(N,r)`` and ``B=(N,d,r)``.  Differentiating with respect
    to the selected vector directly avoids constructing the full ``(N,p)``
    parameter Jacobian when ``r << p``.
    """

    selected = theta.index_select(0, indices)

    def selected_trial(values: Tensor, point: Tensor) -> Tensor:
        full = theta.index_copy(0, indices, values)
        return trial(full, point)

    feature_one = jacrev(selected_trial, argnums=0)
    spatial_one = jacrev(feature_one, argnums=1)
    feature_chunks, spatial_chunks = [], []
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        feature_chunks.append(vmap(feature_one, in_dims=(None, 0))(selected, chunk))
        raw = vmap(spatial_one, in_dims=(None, 0))(selected, chunk)  # (N,r,d)
        spatial_chunks.append(raw.transpose(1, 2))
    return torch.cat(feature_chunks), torch.cat(spatial_chunks)


def _diffusion_energy(gradients: Tensor, diffusion: Tensor | None) -> Tensor:
    if diffusion is None:
        return torch.sum(gradients.square(), dim=-1)
    if diffusion.ndim == 2:
        return torch.einsum("nd,de,ne->n", gradients, diffusion, gradients)
    if diffusion.ndim == 3:
        return torch.einsum("nd,nde,ne->n", gradients, diffusion, gradients)
    raise ValueError("diffusion must have shape (d,d) or (N,d,d)")


def assemble_ritz_system(
    features: Tensor,
    feature_gradients: Tensor,
    forcing: Tensor,
    volume: float,
    *,
    diffusion: Tensor | None = None,
    reaction: float | Tensor = 0.0,
) -> tuple[Tensor, Tensor]:
    """Monte Carlo/Sobol assembly of the symmetric elliptic Ritz system."""

    count = features.shape[0]
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

    reaction_values = torch.as_tensor(reaction, dtype=features.dtype, device=features.device)
    if bool(torch.any(reaction_values != 0.0)):
        if reaction_values.ndim == 0:
            stiffness = stiffness + reaction_values * (features.T @ features)
        else:
            stiffness = stiffness + torch.einsum(
                "n,ni,nj->ij", reaction_values, features, features
            )

    stiffness = (volume / count) * stiffness
    stiffness = 0.5 * (stiffness + stiffness.T)
    load = (volume / count) * (features.T @ forcing)
    return stiffness, load


def solve_ritz_system(
    stiffness: Tensor,
    load: Tensor,
    *,
    ridge_relative: float = 1.0e-8,
) -> RitzSolution:
    """Solve a mean-diagonal-scaled ridge system and report its residual."""

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
    """Build the selected tangent basis, assemble, and solve the inner problem."""

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


def tangent_values_and_gradients(
    trial: TrialFunction,
    theta: Tensor,
    direction: Tensor,
    points: Tensor,
    *,
    chunk_size: int = 512,
) -> tuple[Tensor, Tensor]:
    """Evaluate ``D_theta T_theta direction`` and its spatial gradient."""

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
    """Evaluate only the tangent solution values."""

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
    """Empirical elliptic energy of a tangent direction."""

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
    """Outer gradient with the inner coefficient/direction held fixed."""

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
    """Backtracking update for the fixed-inner-coefficient outer energy."""

    before = float(objective(theta))
    norm_squared = float(torch.sum(gradient.square()))
    step = initial_step
    for _ in range(max_backtracks):
        candidate = (theta - step * gradient).detach()
        if float(objective(candidate)) <= before - sufficient_decrease * step * norm_squared:
            return candidate, step
        step *= contraction
    return theta.detach(), 0.0


def manufactured_forcing(
    exact_solution: ScalarFunction,
    points: Tensor,
    *,
    diffusion: Tensor | None = None,
    reaction: float = 0.0,
) -> Tensor:
    """Compute ``-tr(A Hessian(u)) + c u`` for constant ``A``."""

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
    """Relative held-out L2 error used by the concise notebook histories."""

    prediction = predict_tangent(trial, theta, direction, points)
    reference = vmap(exact_solution)(points)
    return float(
        torch.linalg.norm(prediction - reference)
        / torch.linalg.norm(reference).clamp_min(torch.finfo(reference.dtype).eps)
    )


def matrix_ritz_energy(stiffness: Tensor, load: Tensor, alpha: Tensor) -> Tensor:
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
