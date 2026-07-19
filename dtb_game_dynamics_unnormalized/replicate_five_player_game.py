"""Compute the supplied five-player Cournot game with Neural--DTB.

The default run is intentionally small enough for Colab.  The thesis reports
nonnegative quantities, so the five-player drift uses the projected best
response needed by the five boundary equilibria in Section 4.7.4.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from game_dtb.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="number of hidden Linear+tanh blocks",
    )
    parser.add_argument("--basis-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--svd-rtol", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/five_player_replication")
    parser.add_argument("--skip-sde-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def experiment_args(cli: argparse.Namespace) -> SimpleNamespace:
    if cli.particles < 1:
        raise ValueError("particles must be positive")
    if cli.depth < 1:
        raise ValueError("depth must be positive")
    if cli.basis_size < 1 or cli.width < 1:
        raise ValueError("basis-size and width must be positive")

    steps = 200
    return SimpleNamespace(
        # BLOCK 1 -- Five player game and an initial uniform cloud on [0,1]^5.
        game="cournot5",
        dim=5,
        particles=cli.particles,
        steps=steps,
        step_size=0.005,
        basis_size=cli.basis_size,
        svd_rtol=cli.svd_rtol,
        noise_std=0.1,
        diffusion_entry=None,
        initial_distribution="uniform",
        uniform_low=0.0,
        uniform_high=1.0,
        initial_mean=0.5,
        initial_std=0.15,
        initial_variance=0.03,
        # BLOCK 2 -- Random tangent dictionary produced by the requested MLP.
        width=cli.width,
        depth=cli.depth,
        activation="tanh",
        jacobian_chunk=128,
        derivative_chunk=64,
        linear_target=0.5,
        linear_contraction=1.0,
        linear_rotation=0.35,
        cournot_b=1.0,
        cournot_mu=2.0,
        # BLOCKS 3--8 -- Advance the DTB state and retain thesis time slices.
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
    print("=== Five-player non-potential Cournot game ===")
    print("strategy constraint: x_i >= 0 (projected best response)")
    print("known equilibria: origin, (7/32)^5, and five permutations of")
    print("                    (0,5/18,5/18,5/18,5/18)")
    print("stability: all seven reported equilibria are unstable")
    run_experiment(experiment_args(cli))


if __name__ == "__main__":
    main()
