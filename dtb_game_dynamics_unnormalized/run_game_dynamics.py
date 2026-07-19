"""Command-line entry point for unnormalized Neural--DTB game dynamics."""

from __future__ import annotations

import argparse

from game_dtb.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neural--DTB distributional game dynamics with score transport"
    )
    parser.add_argument(
        "--game", choices=("linear", "cournot", "cournot3", "cournot5"), default="cournot"
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--particles", type=int, default=256, help="particle count N")
    parser.add_argument("--steps", type=int, default=50, help="Euler step count K")
    parser.add_argument("--step-size", type=float, default=0.02, help="Euler step h")
    parser.add_argument("--basis-size", type=int, default=64, help="selected basis size m")
    parser.add_argument("--svd-rtol", type=float, default=1e-5)

    # In the thesis caption sigma_i=0.1 is an SDE noise amplitude.  The
    # algorithm uses D=sigma sigma^T, so the default diagonal is 0.01.
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument(
        "--diffusion-entry",
        type=float,
        default=None,
        help="override each diagonal entry of D; otherwise use noise_std**2",
    )

    parser.add_argument(
        "--initial-distribution", choices=("uniform", "gaussian"), default="gaussian"
    )
    parser.add_argument("--uniform-low", type=float, default=0.0)
    parser.add_argument("--uniform-high", type=float, default=1.0)
    parser.add_argument("--initial-mean", type=float, default=0.5)
    parser.add_argument("--initial-std", type=float, default=0.15)
    parser.add_argument(
        "--initial-variance",
        type=float,
        default=0.03,
        help="Gaussian covariance entry; overrides --initial-std",
    )

    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--architecture", choices=("mlp", "mmnn"), default="mlp")
    parser.add_argument(
        "--rank", type=int, default=8, help="MMNN intermediate component count"
    )
    parser.add_argument("--activation", choices=("tanh", "gelu", "silu"), default="tanh")
    parser.add_argument("--jacobian-chunk", type=int, default=128)
    parser.add_argument("--derivative-chunk", type=int, default=64)
    parser.add_argument("--linear-target", type=float, default=0.5)
    parser.add_argument("--linear-contraction", type=float, default=1.0)
    parser.add_argument("--linear-rotation", type=float, default=0.35)
    parser.add_argument("--cournot-b", type=float, default=1.0)
    parser.add_argument("--cournot-mu", type=float, default=2.0)

    parser.add_argument(
        "--snapshot-times",
        default="0,0.2,0.4,0.6,0.8,1.0",
        help="comma-separated Euler-grid times, or 'auto'",
    )
    parser.add_argument("--plot-low", type=float, default=-0.4)
    parser.add_argument("--plot-high", type=float, default=1.0)
    parser.add_argument("--run-sde-baseline", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/game_replication")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
