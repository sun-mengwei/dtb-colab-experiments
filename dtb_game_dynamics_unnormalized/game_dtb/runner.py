"""Experiment runner and artifact generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .algorithm import DTBConfig, NeuralDTBGameDynamics
from .games import cournot_duopoly_drift, linear_quadratic_drift
from .models import TangentMLP
from .state import gaussian_particle_state


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def run_experiment(args: Any) -> Path:
    """Run an experiment and save data, diagnostics, and a summary figure."""

    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # BLOCK 1 — Initialize z_i, x_i^0, log rho_0(x_i^0), and q_i^0.
    state_generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    state_generator.manual_seed(args.seed)
    state = gaussian_particle_state(
        args.particles,
        args.dim,
        args.initial_mean,
        args.initial_std,
        device=device,
        dtype=dtype,
        generator=state_generator,
    )
    initial_particles = state.particles.detach().cpu().numpy().copy()

    # Build f_theta and the game drift.  The f_theta values are not fitted;
    # only their selected parameter derivatives form the tangent basis.
    model = TangentMLP(
        dim=args.dim,
        width=args.width,
        depth=args.depth,
        activation=args.activation,
        dtype=dtype,
    ).to(device)
    if args.game == "cournot":
        if args.dim != 2:
            raise ValueError("the Cournot example requires --dim 2")
        drift = lambda x: cournot_duopoly_drift(x, b=args.cournot_b, mu=args.cournot_mu)
    elif args.game == "linear":
        if args.dim != 2:
            raise ValueError("the linear example requires --dim 2")
        drift = lambda x: linear_quadratic_drift(
            x,
            target=args.linear_target,
            contraction=args.linear_contraction,
            rotation=args.linear_rotation,
        )
    else:
        raise ValueError(f"unknown game: {args.game}")

    diffusion = args.diffusion * torch.eye(args.dim, device=device, dtype=dtype)
    config = DTBConfig(
        step_size=args.step_size,
        basis_size=args.basis_size,
        svd_rtol=args.svd_rtol,
        jacobian_chunk_size=args.jacobian_chunk,
        derivative_chunk_size=args.derivative_chunk,
        seed=args.seed,
    )
    method = NeuralDTBGameDynamics(model, drift, diffusion, config)

    times = [0.0]
    means = [state.particles.mean(dim=0).detach().cpu().numpy()]
    stds = [state.particles.std(dim=0).detach().cpu().numpy()]
    mean_log_density = [float(state.log_density.mean())]
    residuals: list[float] = []
    ranks: list[int] = []
    alpha_norms: list[float] = []
    mean_divergences: list[float] = []

    for step in range(args.steps):
        result = method.step(state, step)
        state = result.state
        times.append((step + 1) * args.step_size)
        means.append(state.particles.mean(dim=0).detach().cpu().numpy())
        stds.append(state.particles.std(dim=0).detach().cpu().numpy())
        mean_log_density.append(float(state.log_density.mean()))
        residuals.append(result.diagnostics.relative_residual)
        ranks.append(result.diagnostics.retained_rank)
        alpha_norms.append(result.alpha_norm)
        mean_divergences.append(result.mean_divergence)
        if args.print_every > 0 and ((step + 1) % args.print_every == 0 or step == 0):
            mean_text = np.array2string(means[-1], precision=4)
            print(
                f"step {step + 1:4d}/{args.steps} t={times[-1]:.4f} "
                f"mean={mean_text} residual={residuals[-1]:.3e} "
                f"rank={ranks[-1]} |alpha|={alpha_norms[-1]:.3e}"
            )

    final_particles = state.particles.detach().cpu().numpy()
    history_path = output_dir / "history.npz"
    np.savez(
        history_path,
        initial_particles=initial_particles,
        final_particles=final_particles,
        labels=state.labels.detach().cpu().numpy(),
        final_log_density=state.log_density.detach().cpu().numpy(),
        final_score=state.score.detach().cpu().numpy(),
        times=np.asarray(times),
        means=np.asarray(means),
        stds=np.asarray(stds),
        mean_log_density=np.asarray(mean_log_density),
        projection_residuals=np.asarray(residuals),
        retained_ranks=np.asarray(ranks),
        alpha_norms=np.asarray(alpha_norms),
        mean_divergences=np.asarray(mean_divergences),
    )

    metadata = vars(args).copy()
    metadata.update(
        {
            "resolved_device": str(device),
            "torch_version": torch.__version__,
            "trainable_parameters_M": method.parameter_count,
            "actual_basis_size_m": min(config.basis_size, method.parameter_count),
            "dtb_config": asdict(config),
        }
    )
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _plot_summary(
        output_dir / "summary.png",
        initial_particles,
        final_particles,
        np.asarray(times),
        np.asarray(means),
        np.asarray(residuals),
        args.game,
    )
    print(f"saved {history_path}")
    print(f"saved {output_dir / 'summary.png'}")
    return output_dir


def _plot_summary(
    path: Path,
    initial_particles: np.ndarray,
    final_particles: np.ndarray,
    times: np.ndarray,
    means: np.ndarray,
    residuals: np.ndarray,
    game: str,
) -> None:
    """Create a compact visual diagnostic for a two-dimensional run."""

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].scatter(initial_particles[:, 0], initial_particles[:, 1], s=14, alpha=0.55)
    axes[0].set_title("Initial particles")
    axes[1].scatter(final_particles[:, 0], final_particles[:, 1], s=14, alpha=0.55)
    axes[1].set_title("Final particles")
    for axis in axes[:2]:
        axis.set_xlabel("strategy x1")
        axis.set_ylabel("strategy x2")
        axis.grid(alpha=0.2)

    axes[2].plot(times, means[:, 0], label="mean x1")
    axes[2].plot(times, means[:, 1], label="mean x2")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("particle mean")
    axes[2].set_title(f"{game} mean; final residual={residuals[-1]:.2e}")
    axes[2].grid(alpha=0.2)
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
