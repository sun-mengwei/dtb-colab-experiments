"""Train small neural networks to approximate 1-D functions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models import build_model, count_trainable
from targets import APPROX_TARGETS


@dataclass
class Result:
    target: str
    model: str
    params: int
    grid_rmse: float
    test_rmse: float
    max_abs_error: float


def choose_device(name: str) -> torch.device:
    """Resolve a user-facing device name into a torch.device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def sample_interval(n: int, device: torch.device) -> torch.Tensor:
    """Sample n points uniformly from the approximation domain [-1, 1]."""
    return 2.0 * torch.rand(n, 1, device=device) - 1.0


def train_one(target_name: str, model_name: str, args: argparse.Namespace,
              device: torch.device) -> tuple[nn.Module, Result]:
    """Train one model on one target function and return metrics.

    This function is reused by both the command-line script and the notebook.
    Keeping the training loop here prevents the notebook from silently drifting
    away from the scripted experiments.
    """
    target = APPROX_TARGETS[target_name]

    # First seed controls deterministic model initialization.
    torch.manual_seed(args.seed)

    # All model families share the same factory interface, so switching from
    # FCNN to MMNN is just a command-line/notebook argument change.
    model = build_model(
        model_name,
        in_dim=1,
        tiny_hidden=args.tiny_hidden,
        hidden=args.hidden,
        fcnn_depth=args.fcnn_depth,
        width=args.width,
        rank=args.rank,
        mmnn_depth=args.mmnn_depth,
        activation=args.activation,
        random_activation=args.random_activation,
    ).to(device)

    # Reset the seed after model construction so FCNN/MMNN consume the same
    # training samples, mini-batch indices, and random test samples. Without
    # this, wider models would advance the global RNG further during init.
    torch.manual_seed(args.seed)

    # Draw a fixed training set once. Mini-batches are sampled from this pool
    # so each model sees the same target distribution under the same seed.
    x_train = sample_interval(args.n_train, device)
    with torch.no_grad():
        y_train = target(x_train)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    for step in range(args.steps):
        # Stochastic mini-batch training; the target is known analytically, so
        # there is no dataset file to load.
        idx = torch.randint(0, args.n_train, (args.batch_size,), device=device)
        xb = x_train[idx]
        yb = y_train[idx]

        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

        if args.verbose and (step == 0 or (step + 1) % args.log_every == 0):
            print(
                f"{target_name:11s} {model_name:5s} "
                f"step={step + 1:5d} "
                f"batch_rmse={math.sqrt(float(loss.detach())):.3e}"
            )

    # Evaluate two ways:
    #   grid_rmse/max_abs_error show visual fit on a dense deterministic grid;
    #   test_rmse estimates generalization on new random points.
    with torch.no_grad():
        x_grid = torch.linspace(-1.0, 1.0, args.n_grid, device=device).unsqueeze(1)
        grid_err = model(x_grid) - target(x_grid)
        x_test = sample_interval(args.n_test, device)
        test_err = model(x_test) - target(x_test)

    result = Result(
        target=target_name,
        model=model_name,
        params=count_trainable(model),
        grid_rmse=float(torch.sqrt(torch.mean(grid_err ** 2))),
        test_rmse=float(torch.sqrt(torch.mean(test_err ** 2))),
        max_abs_error=float(torch.max(torch.abs(grid_err))),
    )
    return model, result


def plot_target(target_name: str, fitted: Dict[str, nn.Module],
                args: argparse.Namespace, device: torch.device) -> str:
    """Save target/prediction overlays and pointwise error curves."""
    target = APPROX_TARGETS[target_name]
    x = torch.linspace(-1.0, 1.0, args.n_grid, device=device).unsqueeze(1)
    with torch.no_grad():
        y_true_tensor = target(x).detach().cpu()
        preds = {
            name: model(x).detach().cpu()
            for name, model in fitted.items()
        }

    # Convert through Python lists instead of Tensor.numpy(). Some PyTorch/
    # NumPy version combinations raise "Numpy is not available" even when
    # Matplotlib itself can plot lists just fine.
    x_cpu = x[:, 0].detach().cpu().tolist()
    y_true = y_true_tensor.tolist()
    fig, (ax_fit, ax_err) = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_fit.plot(x_cpu, y_true, color="black", linewidth=2.2, label="target")
    for name, pred in preds.items():
        ax_fit.plot(x_cpu, pred.tolist(), linewidth=1.6, label=name)
        ax_err.plot(x_cpu, (pred - y_true_tensor).tolist(),
                    linewidth=1.2, label=name)

    ax_fit.set_title(target_name)
    ax_fit.set_ylabel("u(x)")
    ax_fit.grid(alpha=0.25)
    ax_fit.legend(fontsize=8)
    ax_err.axhline(0.0, color="black", linewidth=0.8)
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("error")
    ax_err.grid(alpha=0.25)
    fig.tight_layout()

    path = os.path.join(args.out_dir, f"{target_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_summary(results: Iterable[Result], args: argparse.Namespace) -> str:
    """Save grouped bar charts for FCNN/MMNN-style comparison.

    The per-target PNGs answer "where does each model miss?" This summary
    answers "which model wins on which target, and by how much?"
    """
    rows = list(results)
    if not rows:
        raise ValueError("cannot plot a summary with no results")

    targets = list(dict.fromkeys(r.target for r in rows))
    models = list(dict.fromkeys(r.model for r in rows))
    x = [float(i) for i in range(len(targets))]
    bar_width = min(0.8 / max(len(models), 1), 0.22)
    offsets = [
        (i - (len(models) - 1) / 2.0) * bar_width
        for i in range(len(models))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for ax, metric_name, ylabel in [
        (axes[0], "grid_rmse", "grid RMSE"),
        (axes[1], "max_abs_error", "max |error|"),
    ]:
        for offset, model_name in zip(offsets, models):
            values = []
            for target_name in targets:
                match = next(
                    r for r in rows
                    if r.target == target_name and r.model == model_name
                )
                values.append(getattr(match, metric_name))
            ax.bar([pos + offset for pos in x], values, width=bar_width,
                   label=model_name)

        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x, targets, rotation=20, ha="right")

    axes[0].set_title("Dense-grid accuracy")
    axes[1].set_title("Worst grid error")
    axes[1].legend(fontsize=8)
    fig.suptitle("Approximation comparison across target functions")
    fig.tight_layout()

    path = os.path.join(args.out_dir, "metrics_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_csv(results: Iterable[Result], path: str) -> None:
    rows = [asdict(r) for r in results]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Command-line configuration for reproducible approximation experiments."""
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+",
                   default=["smooth", "oscillatory", "localized"],
                   choices=sorted(APPROX_TARGETS))
    p.add_argument("--models", nargs="+",
                   default=["tiny", "fcnn", "mcnn", "mmnn"],
                   choices=["tiny", "fcnn", "mcnn", "mmnn"])
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_train", type=int, default=4096)
    p.add_argument("--n_test", type=int, default=4096)
    p.add_argument("--n_grid", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--tiny_hidden", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fcnn_depth", type=int, default=3)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--mmnn_depth", type=int, default=3)
    p.add_argument("--activation", choices=["tanh", "relu", "gelu", "sin"],
                   default="tanh")
    p.add_argument("--random_activation",
                   choices=["tanh", "relu", "gelu", "sin"], default="gelu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out_dir", default="runs/approx")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log_every", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    device = choose_device(args.device)
    print(f"device: {device}")

    results: List[Result] = []
    for target_name in args.targets:
        fitted: Dict[str, nn.Module] = {}
        for model_name in args.models:
            model, result = train_one(target_name, model_name, args, device)
            fitted[model_name] = model
            results.append(result)
            print(
                f"{target_name:11s} {model_name:5s} "
                f"params={result.params:6d} "
                f"grid_rmse={result.grid_rmse:.3e} "
                f"test_rmse={result.test_rmse:.3e} "
                f"max_grid_err={result.max_abs_error:.3e}"
            )

        plot_path = plot_target(target_name, fitted, args, device)
        print(f"saved {plot_path}")

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    write_csv(results, csv_path)
    print(f"saved {csv_path}")
    summary_path = plot_metric_summary(results, args)
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
