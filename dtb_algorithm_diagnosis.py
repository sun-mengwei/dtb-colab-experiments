"""Diagnostics for the single-mode DTB--Ritz Poisson experiment.

The module separates four effects that are easy to conflate in high dimension:

* spatial quadrature size ``N``;
* selected tangent dimension ``R``;
* flattened-random versus layer-balanced parameter selection;
* whether outer updates materially change the tangent basis and its span.

It is intentionally importable from Colab so experiment cells stay concise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import torch

from DTB_elliptic_utils import (
    MLP,
    ParameterSpec,
    armijo_update,
    direct_ritz_energy,
    envelope_gradient,
    expand_direction,
    flatten_parameters,
    make_box_trial,
    matrix_ritz_energy,
    predict_tangent,
    sample_box,
    select_parameter_indices,
    selected_tangent_basis,
    solve_tangent_ritz,
    tangent_values_and_gradients,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class SingleModeConfig:
    """Configuration for one single-mode DTB diagnostic run."""

    dimension: int = 10
    train_count: int = 768
    validation_count: int = 4096
    tangent_dimension: int = 32
    selection: str = "random"
    outer_steps: int = 0
    width: int = 16
    depth: int = 2
    ridge_relative: float = 1.0e-4
    initial_step: float = 1.0
    model_seed: int = 110
    quadrature_seed: int = 20
    validation_seed: int = 2026
    selection_seed: int = 710
    probe_count: int = 256
    probe_seed: int = 3030


def single_mode_exact(points: Tensor) -> Tensor:
    """Return ``prod_j cos(pi*x_j/2)`` for a batch ``(N,d)``."""

    return torch.prod(torch.cos(0.5 * math.pi * points), dim=-1)


def single_mode_gradients(points: Tensor) -> Tensor:
    """Return exact spatial gradients for a batch ``(N,d)``."""

    dimension = points.shape[-1]
    cosines = torch.cos(0.5 * math.pi * points)
    columns = []
    for coordinate in range(dimension):
        factors = cosines.clone()
        factors[:, coordinate] = -0.5 * math.pi * torch.sin(
            0.5 * math.pi * points[:, coordinate]
        )
        columns.append(torch.prod(factors, dim=1))
    return torch.stack(columns, dim=1)


def single_mode_forcing(points: Tensor) -> Tensor:
    """Return the Poisson forcing ``f=(d*pi^2/4)u_exact``."""

    dimension = points.shape[-1]
    return (dimension * math.pi**2 / 4.0) * single_mode_exact(points)


def _parameter_blocks(spec: ParameterSpec) -> list[dict[str, object]]:
    blocks = []
    start = 0
    for name, size in zip(spec.names, spec.sizes):
        end = start + size
        blocks.append(
            {
                "name": name,
                "layer": name.rsplit(".", 1)[0],
                "start": start,
                "end": end,
                "is_bias": name.endswith(".bias"),
            }
        )
        start = end
    return blocks


def layer_balanced_parameter_indices(
    spec: ParameterSpec,
    tangent_dimension: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Select nearly equal counts per layer and retain a bias per layer.

    This avoids the dimension-dependent bias of sampling uniformly from one
    flattened vector, whose first-layer weight block grows with input dimension.
    """

    blocks = _parameter_blocks(spec)
    layers = list(dict.fromkeys(str(block["layer"]) for block in blocks))
    layer_indices = []
    layer_biases = []
    for layer in layers:
        members = [block for block in blocks if block["layer"] == layer]
        indices = torch.cat(
            [torch.arange(int(block["start"]), int(block["end"])) for block in members]
        )
        biases = torch.cat(
            [
                torch.arange(int(block["start"]), int(block["end"]))
                for block in members
                if bool(block["is_bias"])
            ]
        )
        layer_indices.append(indices)
        layer_biases.append(biases)

    parameter_count = sum(spec.sizes)
    if not 1 <= tangent_dimension <= parameter_count:
        raise ValueError("tangent_dimension must lie in [1, parameter_count]")

    allocations = [0] * len(layers)
    for _ in range(tangent_dimension):
        available = [
            index
            for index, candidates in enumerate(layer_indices)
            if allocations[index] < candidates.numel()
        ]
        chosen_layer = min(available, key=lambda index: (allocations[index], index))
        allocations[chosen_layer] += 1

    selected = []
    for layer_index, (candidates, biases, count) in enumerate(
        zip(layer_indices, layer_biases, allocations)
    ):
        if count == 0:
            continue
        generator = torch.Generator(device="cpu").manual_seed(seed + layer_index)
        guaranteed = biases[-1:] if biases.numel() else torch.empty(0, dtype=torch.long)
        keep = ~torch.isin(candidates, guaranteed)
        pool = candidates[keep]
        remainder = count - guaranteed.numel()
        sampled = pool[torch.randperm(pool.numel(), generator=generator)[:remainder]]
        selected.append(torch.cat([guaranteed, sampled]))
    return torch.sort(torch.cat(selected)).values.to(device=device)


def select_tangent_indices(
    parameter_count: int,
    spec: ParameterSpec,
    tangent_dimension: int,
    strategy: str,
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Select tangent coordinates with ``random`` or ``layer_balanced``."""

    if strategy == "random":
        return select_parameter_indices(
            parameter_count, tangent_dimension, seed, device=device
        )
    if strategy == "layer_balanced":
        return layer_balanced_parameter_indices(
            spec, tangent_dimension, seed, device=device
        )
    raise ValueError("selection must be 'random' or 'layer_balanced'")


def selection_counts(indices: Tensor, spec: ParameterSpec) -> dict[str, int]:
    """Count selected coordinates by neural-network layer."""

    counts: dict[str, int] = {}
    for block in _parameter_blocks(spec):
        layer = str(block["layer"])
        selected = (indices >= int(block["start"])) & (indices < int(block["end"]))
        counts[layer] = counts.get(layer, 0) + int(selected.sum())
    return counts


def exact_empirical_energy(points: Tensor) -> float:
    """Evaluate the exact solution with the same empirical Ritz quadrature."""

    values = single_mode_exact(points)
    gradients = single_mode_gradients(points)
    forcing = single_mode_forcing(points)
    volume = 2.0 ** points.shape[-1]
    return float(
        volume * torch.mean(0.5 * gradients.square().sum(1) - forcing * values)
    )


def effective_sample_size(weights: Tensor) -> float:
    """Return ``(sum w)^2/sum(w^2)`` for nonnegative diagnostic weights."""

    denominator = weights.square().sum().clamp_min(torch.finfo(weights.dtype).eps)
    return float(weights.sum().square() / denominator)


def _orthonormal_span(features: Tensor, tolerance: float = 1.0e-10) -> Tensor:
    left, singular_values, _ = torch.linalg.svd(features, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values[0]) == 0.0:
        return left[:, :0]
    rank = int((singular_values > tolerance * singular_values[0]).sum())
    return left[:, :rank]


def tangent_span_diagnostics(
    initial_features: Tensor,
    current_features: Tensor,
) -> tuple[float, float]:
    """Return raw relative feature change and maximum principal-angle sine."""

    denominator = torch.linalg.norm(initial_features).clamp_min(
        torch.finfo(initial_features.dtype).eps
    )
    relative_change = float(
        torch.linalg.norm(current_features - initial_features) / denominator
    )
    initial_span = _orthonormal_span(initial_features)
    current_span = _orthonormal_span(current_features)
    if initial_span.shape[1] != current_span.shape[1] or initial_span.shape[1] == 0:
        return relative_change, 1.0
    cosines = torch.linalg.svdvals(initial_span.T @ current_span).clamp(0.0, 1.0)
    maximum_sine = float(torch.sqrt((1.0 - cosines.min().square()).clamp_min(0.0)))
    return relative_change, maximum_sine


def _stiffness_diagnostics(stiffness: Tensor) -> tuple[int, float]:
    eigenvalues = torch.linalg.eigvalsh(stiffness)
    maximum = eigenvalues.max()
    positive = eigenvalues[eigenvalues > 1.0e-10 * maximum]
    if positive.numel() == 0:
        return 0, math.inf
    return positive.numel(), float(positive.max() / positive.min())


def run_single_mode_configuration(
    config: SingleModeConfig,
    *,
    device: torch.device | str = "cpu",
    evaluation_points: dict[str, Tensor] | None = None,
) -> dict[str, object]:
    """Run one configuration and return accuracy, conditioning, and basis drift.

    When ``evaluation_points`` is supplied, predictions are retained at every
    outer iteration. This supports controlled slice and plane comparisons
    without rerunning nominally identical optimization configurations.
    """

    torch.manual_seed(config.model_seed)
    model = MLP(
        config.dimension, width=config.width, depth=config.depth
    ).to(device)
    theta, spec = flatten_parameters(model)
    theta = theta.to(device)
    trial = make_box_trial(model, spec, normalize_boundary=True)
    indices = select_tangent_indices(
        theta.numel(),
        spec,
        config.tangent_dimension,
        config.selection,
        config.selection_seed,
        device=device,
    )
    volume = 2.0 ** config.dimension
    train = sample_box(
        config.train_count, config.dimension, config.quadrature_seed, device=device
    )
    validation = sample_box(
        config.validation_count,
        config.dimension,
        config.validation_seed,
        device=device,
    )
    probe = sample_box(
        config.probe_count, config.dimension, config.probe_seed, device=device
    )
    forcing = single_mode_forcing(train)
    validation_reference = single_mode_exact(validation)
    theta_initial = theta.clone()
    initial_features: Tensor | None = None
    initial_prediction: Tensor | None = None
    history: list[dict[str, float]] = []
    prepared_evaluations = {
        name: points.to(device=device, dtype=theta.dtype)
        for name, points in (evaluation_points or {}).items()
    }
    evaluation_history = {
        name: {
            "points": points.detach().cpu(),
            "reference": single_mode_exact(points).detach().cpu(),
            "predictions": [],
        }
        for name, points in prepared_evaluations.items()
    }

    for outer_step in range(config.outer_steps + 1):
        solution, stiffness, load = solve_tangent_ritz(
            trial,
            theta,
            indices,
            train,
            forcing,
            volume,
            ridge_relative=config.ridge_relative,
        )
        direction = expand_direction(solution.alpha, indices, theta.numel())
        validation_prediction = predict_tangent(trial, theta, direction, validation)
        relative_l2 = float(
            torch.linalg.norm(validation_prediction - validation_reference)
            / torch.linalg.norm(validation_reference).clamp_min(
                torch.finfo(validation_reference.dtype).eps
            )
        )
        probe_features, _ = selected_tangent_basis(trial, theta, probe, indices)
        probe_prediction = predict_tangent(trial, theta, direction, probe)
        for name, points in prepared_evaluations.items():
            evaluation_history[name]["predictions"].append(
                predict_tangent(trial, theta, direction, points).detach().cpu()
            )
        if initial_features is None:
            initial_features = probe_features.detach()
            initial_prediction = probe_prediction.detach()
        feature_change, subspace_sine = tangent_span_diagnostics(
            initial_features, probe_features
        )
        prediction_change = float(
            torch.linalg.norm(probe_prediction - initial_prediction)
            / torch.linalg.norm(initial_prediction).clamp_min(
                torch.finfo(probe_prediction.dtype).eps
            )
        )
        rank, condition = _stiffness_diagnostics(stiffness)
        load_norm = torch.linalg.norm(load).clamp_min(torch.finfo(load.dtype).eps)
        unregularized_residual = float(
            torch.linalg.norm(stiffness @ solution.alpha - load) / load_norm
        )
        theta_change = float(
            torch.linalg.norm(theta - theta_initial)
            / torch.linalg.norm(theta_initial).clamp_min(torch.finfo(theta.dtype).eps)
        )
        center = torch.zeros(1, config.dimension, dtype=theta.dtype, device=device)
        row = {
            "outer_step": float(outer_step),
            "train_F": float(matrix_ritz_energy(stiffness, load, solution.alpha)),
            "relative_l2": relative_l2,
            "center_value": float(predict_tangent(trial, theta, direction, center)[0]),
            "alpha_norm": float(torch.linalg.norm(solution.alpha)),
            "regularized_residual": solution.relative_residual,
            "unregularized_residual": unregularized_residual,
            "ridge_shift": solution.ridge_shift,
            "effective_rank": float(rank),
            "condition_number": condition,
            "relative_theta_change": theta_change,
            "relative_feature_change": feature_change,
            "maximum_subspace_sine": subspace_sine,
            "relative_prediction_change": prediction_change,
            "gradient_norm": math.nan,
            "accepted_step": math.nan,
        }
        history.append(row)
        if outer_step == config.outer_steps:
            break

        gradient = envelope_gradient(trial, theta, direction, train, forcing, volume)
        objective = lambda candidate: direct_ritz_energy(
            trial, candidate, direction.detach(), train, forcing, volume
        )
        theta, accepted_step = armijo_update(
            objective,
            theta,
            gradient,
            initial_step=config.initial_step,
        )
        row["gradient_norm"] = float(torch.linalg.norm(gradient))
        row["accepted_step"] = accepted_step

    validation_values, validation_gradients = tangent_values_and_gradients(
        trial, theta, direction, validation
    )
    validation_exact_gradients = single_mode_gradients(validation)
    relative_h1 = float(
        torch.linalg.norm(validation_gradients - validation_exact_gradients)
        / torch.linalg.norm(validation_exact_gradients).clamp_min(
            torch.finfo(validation_gradients.dtype).eps
        )
    )
    validation_forcing = single_mode_forcing(validation)
    validation_energy = float(
        volume
        * torch.mean(
            0.5 * validation_gradients.square().sum(1)
            - validation_forcing * validation_values
        )
    )
    exact_values_train = single_mode_exact(train)
    exact_energy = -config.dimension * math.pi**2 / 8.0
    summary = dict(history[-1])
    summary.update(
        {
            "relative_h1": relative_h1,
            "validation_F": validation_energy,
            "exact_continuous_F": exact_energy,
            "exact_train_F": exact_empirical_energy(train),
            "exact_validation_F": exact_empirical_energy(validation),
            "u_squared_effective_sample_size": effective_sample_size(
                exact_values_train.square()
            ),
            "central_sample_count": float(
                (train.abs().amax(dim=1) < 0.5).sum()
            ),
            "parameter_count": float(theta.numel()),
            "sample_to_tangent_ratio": config.train_count
            / config.tangent_dimension,
        }
    )
    return {
        "config": asdict(config),
        "summary": summary,
        "history": history,
        "layer_counts": selection_counts(indices, spec),
        "selected_indices": indices.detach().cpu().tolist(),
        "evaluations": evaluation_history,
    }


def run_configuration_sweep(
    configurations: Iterable[SingleModeConfig],
    *,
    device: torch.device | str = "cpu",
    evaluation_points: dict[str, Tensor] | None = None,
) -> list[dict[str, object]]:
    """Run configurations sequentially and return their result dictionaries."""

    return [
        run_single_mode_configuration(
            config,
            device=device,
            evaluation_points=evaluation_points,
        )
        for config in configurations
    ]


def print_compact_summary(result: dict[str, object]) -> None:
    """Print the most useful scalar diagnostics for one result."""

    config = result["config"]
    summary = result["summary"]
    print(
        f"selection={config['selection']:>14s}  "
        f"R={config['tangent_dimension']:3d}  N={config['train_count']:5d}  "
        f"L2={summary['relative_l2']:.3e}  H1={summary['relative_h1']:.3e}  "
        f"u(0)={summary['center_value']:.3f}  "
        f"cond(G)={summary['condition_number']:.2e}"
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print_compact_summary(run_single_mode_configuration(SingleModeConfig()))
