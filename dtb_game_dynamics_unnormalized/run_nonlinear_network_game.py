"""Run a high-dimensional nonlinear directed network game with Neural--DTB."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from game_dtb.network_analysis import analyze_network_run
from game_dtb.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-dimensional nonlinear network-game DTB experiment"
    )
    # BLOCK 1 -- Game dimension and reproducible directed interaction network.
    parser.add_argument("--dim", type=int, default=20, help="number of scalar players")
    parser.add_argument("--network-density", type=float, default=0.25)
    parser.add_argument("--network-scale", type=float, default=0.8)
    parser.add_argument("--network-seed", type=int, default=17)
    parser.add_argument("--network-bias-std", type=float, default=0.15)
    parser.add_argument("--network-mu", type=float, default=1.0)
    parser.add_argument("--network-beta", type=float, default=0.8)

    # BLOCK 2 -- Particle cloud, physical time grid, and diffusion amplitude.
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.05)

    # BLOCK 3 -- Neural tangent dictionary and truncated-SVD solve.
    parser.add_argument("--architecture", choices=("mlp", "mmnn"), default="mlp")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--basis-size", type=int, default=128)
    parser.add_argument("--svd-rtol", type=float, default=1e-3)

    # BLOCK 4 -- Optional source-matching periodic tangent-network reset.
    parser.add_argument("--refit-interval", type=int, default=0)
    parser.add_argument("--refit-optimizer-steps", type=int, default=2_000)
    parser.add_argument("--refit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--refit-batch-size", type=int, default=2_048)
    parser.add_argument("--refit-samples", type=int, default=10_000)
    parser.add_argument("--refit-test-samples", type=int, default=4_000)

    # BLOCK 5 -- Candidate discovery is diagnostic, not a complete enumeration.
    parser.add_argument("--candidate-seeds", type=int, default=32)
    parser.add_argument("--root-tolerance", type=float, default=1e-7)
    parser.add_argument("--skip-equilibrium-analysis", action="store_true")
    parser.add_argument("--skip-sde-baseline", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/nonlinear_network_game")
    return parser.parse_args()


def experiment_args(cli: argparse.Namespace) -> SimpleNamespace:
    if cli.dim < 2:
        raise ValueError("dim must be at least two")
    for name in ("particles", "steps", "width", "depth", "basis_size", "rank"):
        if getattr(cli, name) < 1:
            raise ValueError(f"{name} must be positive")
    if cli.step_size <= 0.0:
        raise ValueError("step_size must be positive")

    return SimpleNamespace(
        # Game and physical evolution.
        game="network",
        dim=cli.dim,
        particles=cli.particles,
        steps=cli.steps,
        step_size=cli.step_size,
        noise_std=cli.noise_std,
        diffusion_entry=None,
        network_density=cli.network_density,
        network_scale=cli.network_scale,
        network_seed=cli.network_seed,
        network_bias_std=cli.network_bias_std,
        network_mu=cli.network_mu,
        network_beta=cli.network_beta,
        initial_distribution="uniform",
        uniform_low=-1.5,
        uniform_high=1.5,
        initial_mean=0.0,
        initial_std=0.75,
        initial_variance=None,
        # Tangent bundle.
        architecture=cli.architecture,
        width=cli.width,
        depth=cli.depth,
        rank=cli.rank,
        activation="tanh",
        basis_size=cli.basis_size,
        svd_rtol=cli.svd_rtol,
        jacobian_chunk=128,
        derivative_chunk=64,
        # Periodic reset.
        refit_interval=cli.refit_interval,
        refit_optimizer_steps=cli.refit_optimizer_steps,
        refit_learning_rate=cli.refit_learning_rate,
        refit_batch_size=cli.refit_batch_size,
        refit_samples=cli.refit_samples,
        refit_test_samples=cli.refit_test_samples,
        # Unused parameters retained for the shared runner interface.
        linear_target=0.5,
        linear_contraction=1.0,
        linear_rotation=0.35,
        cournot_b=1.0,
        cournot_mu=2.0,
        # Output and comparison.
        snapshot_times="auto",
        plot_low=-2.0,
        plot_high=2.0,
        run_sde_baseline=not cli.skip_sde_baseline,
        device=cli.device,
        dtype=cli.dtype,
        seed=cli.seed,
        print_every=max(1, cli.steps // 5),
        output_dir=cli.output_dir,
    )


def main() -> None:
    cli = parse_args()
    print("=== Nonlinear directed network game ===")
    print(f"players/dimension: {cli.dim}")
    print("equilibria: unknown; candidates are refined from terminal DTB particles")
    print("stable iff max Re eigenvalue of Db is negative")
    output_dir = run_experiment(experiment_args(cli))
    if not cli.skip_equilibrium_analysis:
        records = analyze_network_run(
            output_dir,
            seed_count=cli.candidate_seeds,
            root_tolerance=cli.root_tolerance,
        )
        stable_count = sum(bool(record["dynamically_stable"]) for record in records)
        nash_count = sum(bool(record["local_nash_second_order"]) for record in records)
        print(
            f"found {len(records)} distinct candidate roots: "
            f"{stable_count} dynamically stable, {nash_count} passing the "
            "local-Nash second-order test"
        )
        print(f"saved {output_dir / 'equilibrium_candidates.csv'}")
        print(f"saved {output_dir / 'network_equilibrium_analysis.png'}")


if __name__ == "__main__":
    main()
