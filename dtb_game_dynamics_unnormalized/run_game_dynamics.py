"""Command-line entry point for the unnormalized Neural--DTB scheme."""

from __future__ import annotations

import argparse

from game_dtb.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neural--DTB distributional game dynamics with score transport"
    )
    parser.add_argument("--game", choices=("linear", "cournot"), default="linear")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--particles", type=int, default=64, help="particle count N")
    parser.add_argument("--steps", type=int, default=10, help="Euler step count K")
    parser.add_argument("--step-size", type=float, default=0.01, help="Euler step h")
    parser.add_argument("--basis-size", type=int, default=64, help="selected basis size m")
    parser.add_argument("--svd-rtol", type=float, default=1e-7)
    parser.add_argument("--diffusion", type=float, default=0.05, help="D = value * I")
    parser.add_argument("--initial-mean", type=float, default=0.25)
    parser.add_argument("--initial-std", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--activation", choices=("tanh", "gelu", "silu"), default="tanh")
    parser.add_argument("--jacobian-chunk", type=int, default=128)
    parser.add_argument("--derivative-chunk", type=int, default=64)
    parser.add_argument("--linear-target", type=float, default=0.5)
    parser.add_argument("--linear-contraction", type=float, default=1.0)
    parser.add_argument("--linear-rotation", type=float, default=0.35)
    parser.add_argument("--cournot-b", type=float, default=1.0)
    parser.add_argument("--cournot-mu", type=float, default=2.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/smoke")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
