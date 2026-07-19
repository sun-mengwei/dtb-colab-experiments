"""Compute the supplied three-player Cournot game with Neural--DTB.

The default configuration is Colab-friendly. Add ``--paper-scale`` for a
denser point cloud and larger tangent basis.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from game_dtb.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-scale", action="store_true")
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="override the number of hidden Linear+tanh blocks",
    )
    parser.add_argument("--architecture", choices=("mlp", "mmnn"), default="mlp")
    parser.add_argument("--rank", type=int, default=8, help="MMNN component rank")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--svd-rtol",
        type=float,
        default=None,
        help="override the relative SVD cutoff (use 1e-3 for large-N runs)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/three_player_replication")
    parser.add_argument("--skip-sde-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def experiment_args(cli: argparse.Namespace) -> SimpleNamespace:
    if cli.paper_scale:
        particles = cli.particles or 5000
        basis_size, width, depth = 256, 64, 3
        steps, step_size = 200, 0.005
        jacobian_chunk, derivative_chunk = 256, 128
    else:
        particles = cli.particles or 256
        basis_size, width, depth = 128, 32, 2
        steps, step_size = 200, 0.005
        jacobian_chunk, derivative_chunk = 128, 64

    if cli.depth is not None:
        if cli.depth < 1:
            raise ValueError("depth must be positive")
        depth = cli.depth
    if cli.width is not None:
        if cli.width < 1:
            raise ValueError("width must be positive")
        width = cli.width
    if cli.rank < 1:
        raise ValueError("rank must be positive")

    return SimpleNamespace(
        game="cournot3",
        dim=3,
        particles=particles,
        steps=steps,
        step_size=step_size,
        basis_size=basis_size,
        svd_rtol=1e-4 if cli.svd_rtol is None else cli.svd_rtol,
        noise_std=0.1,
        diffusion_entry=None,
        initial_distribution="uniform",
        uniform_low=0.0,
        uniform_high=1.0,
        initial_mean=0.5,
        initial_std=0.15,
        initial_variance=0.03,
        width=width,
        depth=depth,
        architecture=cli.architecture,
        rank=cli.rank,
        activation="tanh",
        jacobian_chunk=jacobian_chunk,
        derivative_chunk=derivative_chunk,
        linear_target=0.5,
        linear_contraction=1.0,
        linear_rotation=0.35,
        cournot_b=1.0,
        cournot_mu=2.0,
        snapshot_times="0,0.2,0.4,0.6,0.8,1.0",
        plot_low=-0.4,
        plot_high=1.0,
        run_sde_baseline=not cli.skip_sde_baseline,
        device=cli.device,
        dtype="float32",
        seed=cli.seed,
        print_every=max(1, steps // 5),
        output_dir=cli.output_dir,
    )


def main() -> None:
    cli = parse_args()
    print("=== Three-player non-potential Cournot game ===")
    print("payoff convention: opponent total r_i=sum_{j != i} x_j")
    print("known equilibria: origin, (3/8)^3, permutations of (1/2,1/2,0)")
    run_experiment(experiment_args(cli))


if __name__ == "__main__":
    main()
