"""A first PDE experiment: solve 1-D Poisson with a PINN loss.

Problem:
  -u''(x) = f(x),  x in (0, 1)
   u(0) = u(1) = 0

The right-hand side is chosen from a known exact solution in targets.py.
This gives a small bridge from function approximation to PDE solving.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models import build_model, count_trainable
from targets import poisson_exact, poisson_rhs


@dataclass
class PDEResult:
    model: str
    params: int
    grid_rmse: float
    max_abs_error: float
    final_pde_loss: float
    final_bc_loss: float


def choose_device(name: str) -> torch.device:
    """Resolve a user-facing device name into a torch.device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def second_derivative(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute u_xx(x) with PyTorch autograd for the PINN residual."""
    # Clone x because autograd needs to track derivatives with respect to the
    # input coordinate, not only with respect to model parameters.
    x = x.detach().clone().requires_grad_(True)
    u = model(x)

    # First derivative du/dx.
    du = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    # Second derivative d2u/dx2. The model has one input coordinate, so we use
    # the first column of each gradient tensor.
    d2u = torch.autograd.grad(
        du[:, 0], x, grad_outputs=torch.ones_like(du[:, 0]), create_graph=True
    )[0][:, 0]
    return d2u


def train(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, PDEResult]:
    """Train a PINN on 1-D Poisson and return the fitted model plus metrics."""
    torch.manual_seed(args.seed)
    model = build_model(
        args.model,
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

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    final_pde_loss = math.nan
    final_bc_loss = math.nan
    for step in range(args.steps):
        # Interior points enforce the differential equation. Boundary points
        # enforce u(0)=u(1)=0.
        x_int = torch.rand(args.n_interior, 1, device=device)
        x_bc = torch.tensor([[0.0], [1.0]], device=device)

        # PINN loss = equation residual + weighted boundary-condition penalty.
        u_xx = second_derivative(model, x_int)
        residual = -u_xx - poisson_rhs(x_int)
        pde_loss = torch.mean(residual ** 2)
        bc_loss = torch.mean(model(x_bc) ** 2)
        loss = pde_loss + args.bc_weight * bc_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        final_pde_loss = float(pde_loss.detach())
        final_bc_loss = float(bc_loss.detach())
        if args.verbose and (step == 0 or (step + 1) % args.log_every == 0):
            print(
                f"step={step + 1:5d} "
                f"pde_loss={final_pde_loss:.3e} "
                f"bc_loss={final_bc_loss:.3e}"
            )

    # Evaluate against the known exact solution on a dense grid.
    with torch.no_grad():
        x_grid = torch.linspace(0.0, 1.0, args.n_grid, device=device).unsqueeze(1)
        err = model(x_grid) - poisson_exact(x_grid)
        result = PDEResult(
            model=args.model,
            params=count_trainable(model),
            grid_rmse=float(torch.sqrt(torch.mean(err ** 2))),
            max_abs_error=float(torch.max(torch.abs(err))),
            final_pde_loss=final_pde_loss,
            final_bc_loss=final_bc_loss,
        )
    return model, result


def plot_solution(model: nn.Module, args: argparse.Namespace,
                  device: torch.device) -> str:
    """Save the exact solution, prediction, and pointwise error."""
    x = torch.linspace(0.0, 1.0, args.n_grid, device=device).unsqueeze(1)
    with torch.no_grad():
        exact_tensor = poisson_exact(x).detach().cpu()
        pred_tensor = model(x).detach().cpu()

    # Use lists rather than Tensor.numpy() so plotting still works if the
    # PyTorch/NumPy bridge is broken in the active environment.
    x_cpu = x[:, 0].detach().cpu().tolist()
    exact = exact_tensor.tolist()
    pred = pred_tensor.tolist()
    fig, (ax_sol, ax_err) = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_sol.plot(x_cpu, exact, color="black", linewidth=2.2, label="exact")
    ax_sol.plot(x_cpu, pred, linewidth=1.8, label=args.model)
    ax_sol.set_title("1-D Poisson PINN")
    ax_sol.set_ylabel("u(x)")
    ax_sol.legend(fontsize=8)
    ax_sol.grid(alpha=0.25)
    ax_err.axhline(0.0, color="black", linewidth=0.8)
    ax_err.plot(x_cpu, (pred_tensor - exact_tensor).tolist(), linewidth=1.4)
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("error")
    ax_err.grid(alpha=0.25)
    fig.tight_layout()

    path = os.path.join(args.out_dir, f"poisson_{args.model}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fcnn",
                   choices=["tiny", "fcnn", "mcnn", "mmnn"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n_interior", type=int, default=256)
    p.add_argument("--n_grid", type=int, default=1000)
    p.add_argument("--bc_weight", type=float, default=100.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--tiny_hidden", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--fcnn_depth", type=int, default=3)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--rank", type=int, default=24)
    p.add_argument("--mmnn_depth", type=int, default=3)
    p.add_argument("--activation", choices=["tanh", "relu", "gelu", "sin"],
                   default="tanh")
    p.add_argument("--random_activation",
                   choices=["tanh", "relu", "gelu", "sin"], default="gelu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out_dir", default="runs/poisson")
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
    model, result = train(args, device)
    print(json.dumps(asdict(result), indent=2))
    plot_path = plot_solution(model, args, device)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
