"""Post-process a nonlinear network-game run without assuming known equilibria."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .games import nonlinear_network_drift, nonlinear_network_jacobian


def analyze_network_run(
    output_dir: str | Path,
    *,
    seed_count: int = 32,
    root_tolerance: float = 1e-7,
    duplicate_tolerance: float = 1e-3,
    max_iterations: int = 80,
) -> list[dict[str, object]]:
    """Find, classify, save, and plot equilibrium candidates.

    Terminal particles provide starting points only.  Damped Newton refinement
    solves ``b(x)=0``; the Jacobian spectrum classifies dynamical stability,
    while each player's own payoff curvature checks the local-Nash second-order
    condition.  The routine does not claim to enumerate every equilibrium.
    """

    output_path = Path(output_dir)
    history_path = output_path / "history.npz"
    with np.load(history_path) as history:
        initial_particles = history["initial_particles"].astype(float)
        final_particles = history["final_particles"].astype(float)
        matrix = history["network_interaction_matrix"].astype(float)
        bias = history["network_bias"].astype(float)
        mu = history["network_mu"].astype(float)
        beta = history["network_beta"].astype(float)

    tensors = tuple(
        torch.as_tensor(value, dtype=torch.float64)
        for value in (matrix, bias, mu, beta)
    )
    matrix_t, bias_t, mu_t, beta_t = tensors

    terminal_tensor = torch.as_tensor(final_particles, dtype=torch.float64)
    terminal_drift = nonlinear_network_drift(
        terminal_tensor, matrix_t, bias_t, mu_t, beta_t
    )
    terminal_norms = torch.linalg.vector_norm(terminal_drift, dim=1).numpy()
    seeds = _select_diverse_low_residual_seeds(
        final_particles, terminal_norms, seed_count
    )

    roots: list[np.ndarray] = []
    root_residuals: list[float] = []
    for seed in seeds:
        root, residual = refine_network_equilibrium(
            seed,
            matrix,
            bias,
            mu,
            beta,
            tolerance=root_tolerance,
            max_iterations=max_iterations,
        )
        if residual > root_tolerance:
            continue
        if any(np.linalg.norm(root - known) <= duplicate_tolerance for known in roots):
            continue
        roots.append(root)
        root_residuals.append(residual)

    records: list[dict[str, object]] = []
    for index, (root, residual) in enumerate(zip(roots, root_residuals)):
        root_t = torch.as_tensor(root, dtype=torch.float64)
        jacobian = nonlinear_network_jacobian(
            root_t, matrix_t, mu_t, beta_t
        )
        spectral_abscissa = float(torch.linalg.eigvals(jacobian).real.max())
        own_curvature = mu - 3.0 * root**2
        record: dict[str, object] = {
            "candidate": index,
            "residual_norm": residual,
            "spectral_abscissa": spectral_abscissa,
            "dynamically_stable": spectral_abscissa < -1e-6,
            "local_nash_second_order": bool(np.all(own_curvature < -1e-6)),
            "maximum_own_curvature": float(own_curvature.max()),
        }
        record.update({f"x_{coordinate + 1}": value for coordinate, value in enumerate(root)})
        records.append(record)

    _save_candidate_table(output_path / "equilibrium_candidates.csv", records, final_particles.shape[1])
    np.savez(
        output_path / "equilibrium_candidates.npz",
        candidates=(
            np.asarray(roots) if roots else np.empty((0, final_particles.shape[1]))
        ),
        residual_norms=np.asarray(root_residuals),
        spectral_abscissas=np.asarray(
            [record["spectral_abscissa"] for record in records]
        ),
        dynamically_stable=np.asarray(
            [record["dynamically_stable"] for record in records], dtype=bool
        ),
        local_nash_second_order=np.asarray(
            [record["local_nash_second_order"] for record in records], dtype=bool
        ),
    )
    _plot_network_analysis(
        output_path / "network_equilibrium_analysis.png",
        initial_particles,
        final_particles,
        terminal_norms,
        records,
        roots,
        matrix,
        bias,
        mu,
        beta,
    )
    return records


def refine_network_equilibrium(
    seed: np.ndarray,
    interaction_matrix: np.ndarray,
    bias: np.ndarray,
    mu: np.ndarray,
    beta: np.ndarray,
    *,
    tolerance: float = 1e-7,
    max_iterations: int = 80,
) -> tuple[np.ndarray, float]:
    """Refine one candidate with damped Newton steps and a residual line search."""

    point = torch.as_tensor(seed, dtype=torch.float64).clone()
    matrix_t, bias_t, mu_t, beta_t = (
        torch.as_tensor(value, dtype=torch.float64)
        for value in (interaction_matrix, bias, mu, beta)
    )
    for _ in range(max_iterations):
        drift = nonlinear_network_drift(point, matrix_t, bias_t, mu_t, beta_t)
        residual = float(torch.linalg.vector_norm(drift))
        if residual <= tolerance:
            break
        jacobian = nonlinear_network_jacobian(point, matrix_t, mu_t, beta_t)
        try:
            direction = torch.linalg.solve(jacobian, -drift)
        except RuntimeError:
            direction = torch.linalg.lstsq(jacobian, -drift).solution

        accepted = False
        step = 1.0
        for _ in range(14):
            proposal = point + step * direction
            proposal_drift = nonlinear_network_drift(
                proposal, matrix_t, bias_t, mu_t, beta_t
            )
            if float(torch.linalg.vector_norm(proposal_drift)) < residual:
                point = proposal
                accepted = True
                break
            step *= 0.5
        if not accepted:
            gradient = jacobian.T @ drift
            gradient_norm = torch.linalg.vector_norm(gradient)
            if float(gradient_norm) == 0.0:
                break
            point = point - min(1e-2, residual / float(gradient_norm)) * gradient

    final_drift = nonlinear_network_drift(point, matrix_t, bias_t, mu_t, beta_t)
    return point.numpy(), float(torch.linalg.vector_norm(final_drift))


def _select_diverse_low_residual_seeds(
    particles: np.ndarray, residuals: np.ndarray, count: int
) -> np.ndarray:
    if count < 1:
        raise ValueError("seed_count must be positive")
    candidate_count = min(len(particles), max(count, 8 * count))
    pool_indices = np.argsort(residuals)[:candidate_count]
    pool = particles[pool_indices]
    selected = [0]
    min_distance = np.linalg.norm(pool - pool[0], axis=1)
    while len(selected) < min(count, len(pool)):
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.linalg.norm(pool - pool[next_index], axis=1)
        min_distance = np.minimum(min_distance, distance)
    return pool[np.asarray(selected)]


def _save_candidate_table(path: Path, records: list[dict[str, object]], dim: int) -> None:
    fields = [
        "candidate",
        "residual_norm",
        "spectral_abscissa",
        "dynamically_stable",
        "local_nash_second_order",
        "maximum_own_curvature",
        *[f"x_{index + 1}" for index in range(dim)],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot_network_analysis(
    path: Path,
    initial_particles: np.ndarray,
    final_particles: np.ndarray,
    terminal_norms: np.ndarray,
    records: list[dict[str, object]],
    roots: list[np.ndarray],
    matrix: np.ndarray,
    bias: np.ndarray,
    mu: np.ndarray,
    beta: np.ndarray,
) -> None:
    """Plot a fixed terminal-PCA view and diagnostics for discovered roots."""

    center = final_particles.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(final_particles - center, full_matrices=False)
    basis = right_vectors[:2].T
    initial_projection = (initial_particles - center) @ basis
    final_projection = (final_particles - center) @ basis
    root_array = np.asarray(roots) if roots else np.empty((0, final_particles.shape[1]))
    root_projection = (root_array - center) @ basis if len(root_array) else np.empty((0, 2))

    initial_t = torch.as_tensor(initial_particles, dtype=torch.float64)
    matrix_t, bias_t, mu_t, beta_t = (
        torch.as_tensor(value, dtype=torch.float64)
        for value in (matrix, bias, mu, beta)
    )
    initial_norms = torch.linalg.vector_norm(
        nonlinear_network_drift(initial_t, matrix_t, bias_t, mu_t, beta_t), dim=1
    ).numpy()

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes[0, 0].scatter(
        final_projection[:, 0], final_projection[:, 1], s=5, alpha=0.25,
        color="#1786c7", edgecolors="none", rasterized=True, label="terminal particles",
    )
    if records:
        stable = np.asarray([bool(record["dynamically_stable"]) for record in records])
        if stable.any():
            axes[0, 0].scatter(
                root_projection[stable, 0], root_projection[stable, 1], s=70,
                color="#d62728", marker="o", label="stable candidate",
            )
        if (~stable).any():
            axes[0, 0].scatter(
                root_projection[~stable, 0], root_projection[~stable, 1], s=90,
                color="#ffbf00", edgecolors="#222222", marker="X",
                label="unstable candidate",
            )
    axes[0, 0].set_title("Terminal cloud and candidates (terminal PCA)")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(
        initial_projection[:, 0], initial_projection[:, 1], s=4, alpha=0.16,
        color="#7f7f7f", edgecolors="none", rasterized=True, label="initial",
    )
    axes[0, 1].scatter(
        final_projection[:, 0], final_projection[:, 1], s=4, alpha=0.22,
        color="#1786c7", edgecolors="none", rasterized=True, label="terminal",
    )
    axes[0, 1].set_title("Distribution movement in the same PCA plane")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    axes[0, 1].legend(frameon=False)

    positive_norms = np.concatenate(
        [initial_norms[initial_norms > 0], terminal_norms[terminal_norms > 0]]
    )
    lower = max(float(positive_norms.min()), 1e-12) if positive_norms.size else 1e-12
    upper = float(positive_norms.max()) if positive_norms.size else 1.0
    if upper <= lower:
        upper = 10.0 * lower
    bins = np.logspace(
        np.log10(lower),
        np.log10(upper),
        40,
    )
    axes[1, 0].hist(np.maximum(initial_norms, lower), bins=bins, alpha=0.45, label="initial")
    axes[1, 0].hist(np.maximum(terminal_norms, lower), bins=bins, alpha=0.55, label="terminal")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title(r"Stationarity diagnostic $\|b(x)\|_2$")
    axes[1, 0].set_xlabel("drift norm")
    axes[1, 0].set_ylabel("particle count")
    axes[1, 0].legend(frameon=False)

    if records:
        indices = np.arange(len(records))
        spectral = np.asarray([record["spectral_abscissa"] for record in records])
        colors = ["#d62728" if value < 0 else "#ffbf00" for value in spectral]
        axes[1, 1].bar(indices, spectral, color=colors)
        axes[1, 1].axhline(0.0, color="#222222", linewidth=1.0)
        axes[1, 1].set_xlabel("candidate index")
        axes[1, 1].set_ylabel(r"$\max\,\mathrm{Re}\,\lambda(Db)$")
        axes[1, 1].set_title("Local dynamical stability")
    else:
        axes[1, 1].text(
            0.5, 0.5, "No root met the requested residual tolerance",
            ha="center", va="center", transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_axis_off()

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Nonlinear directed network game: discovered candidates, not a complete enumeration"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
