"""Reproduce the Figure 4.2 and 4.3 game-dynamics experiments.

The default is a Colab-friendly run.  Add ``--paper-scale`` for denser point
clouds and a larger tangent basis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from game_dtb.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-scale", action="store_true")
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="outputs/thesis_replication")
    parser.add_argument("--skip-sde-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def experiment_args(
    initial_distribution: str,
    output_dir: Path,
    cli: argparse.Namespace,
) -> SimpleNamespace:
    if cli.paper_scale:
        particles = cli.particles or 5000
        basis_size, width, steps, step_size = 256, 64, 100, 0.01
        jacobian_chunk, derivative_chunk = 256, 128
    else:
        particles = cli.particles or 256
        basis_size, width, steps, step_size = 64, 16, 50, 0.02
        jacobian_chunk, derivative_chunk = 128, 64

    return SimpleNamespace(
        game="cournot",
        dim=2,
        particles=particles,
        steps=steps,
        step_size=step_size,
        basis_size=basis_size,
        svd_rtol=1e-5,
        noise_std=0.1,
        diffusion_entry=None,
        initial_distribution=initial_distribution,
        uniform_low=0.0,
        uniform_high=1.0,
        initial_mean=0.5,
        initial_std=0.15,
        initial_variance=0.03,
        width=width,
        depth=2 if not cli.paper_scale else 3,
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
        output_dir=str(output_dir),
    )


def main() -> None:
    cli = parse_args()
    output_root = Path(cli.output_root)
    print("=== Figure 4.2: uniform initial samples ===")
    run_experiment(
        experiment_args("uniform", output_root / "figure_4_2_uniform", cli)
    )
    print("=== Figure 4.3: Gaussian initial samples ===")
    run_experiment(
        experiment_args("gaussian", output_root / "figure_4_3_gaussian", cli)
    )
    print(f"replication artifacts saved under {output_root}")


if __name__ == "__main__":
    main()
