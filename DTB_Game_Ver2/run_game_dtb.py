"""
Deep Tangent Bundle (DTB) time stepping for game dynamics.

This is a separate, game-focused analogue of run_5d_ac.py.  The current
implementation targets deterministic best-response dynamics

    d x / dt = mu(x),

where x is a strategy profile and mu_i(x) = d Pi_i / d x_i.  A neural map
T_theta transports initial strategy samples z ~ lambda to current strategy
samples x(t) = T_theta(t)(z).  Inside each DTB block we freeze theta, project
the velocity field onto the tangent space spanned by dT_theta / dtheta, and
accumulate the coefficients:

    T^{k+1}(z) = T^k(z) + h * dT_theta(z)/dtheta * alpha^k,
    alpha^k = argmin_alpha || dT_theta(z)/dtheta * alpha - mu(T^k(z)) ||_2.

Every L steps, the accumulated map is refit back into T_theta.  This is the
same block/reset idea as the Allen-Cahn reproduction, but with a vector-valued
pushforward map instead of a scalar PDE solution.

Default game: the 2D non-potential Cournot duopoly from thesis Chapter 4,
with b = 1 and mu = 2:

    dx1/dt = -2 b x1 + 2 b mu x2 - 2 b mu x2^2
    dx2/dt = -2 b x2 + 2 b mu x1 - 2 b mu x1^2.

The stable Nash equilibrium for this parameter set is (0.5, 0.5).

Diffusion note:
    The stochastic thesis model adds -0.5 * D grad_x log rho(x,t) to the
    velocity.  That needs either an invertible flow density model or a score
    estimator.  This file deliberately starts with the deterministic case so
    the DTB mechanics are clear and testable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

from dtb import device, flat_params, jform_solve, unflatten, write_flat_into_model


# ------------------------------- game --------------------------------

def cournot_duopoly_velocity(x: torch.Tensor, b: float = 1.0,
                             mu: float = 2.0) -> torch.Tensor:
    """Best-response velocity for the thesis 2D non-potential Cournot game.

    x: (..., 2), where x[..., 0] = x1 and x[..., 1] = x2.
    """
    if x.shape[-1] != 2:
        raise ValueError("cournot_duopoly_velocity expects x.shape[-1] == 2")
    x1 = x[..., 0]
    x2 = x[..., 1]
    v1 = -2.0 * b * x1 + 2.0 * b * mu * x2 - 2.0 * b * mu * x2 ** 2
    v2 = -2.0 * b * x2 + 2.0 * b * mu * x1 - 2.0 * b * mu * x1 ** 2
    return torch.stack((v1, v2), dim=-1)


def make_velocity(args) -> Callable[[torch.Tensor], torch.Tensor]:
    if args.game == "cournot2":
        return lambda x: cournot_duopoly_velocity(x, b=args.cournot_b,
                                                  mu=args.cournot_mu)
    raise ValueError(f"unknown game {args.game!r}")


def sample_initial(n: int, dim: int, low: float, high: float,
                   dtype: torch.dtype, dev: torch.device) -> torch.Tensor:
    """Uniform initial strategy distribution lambda on [low, high]^dim."""
    return torch.rand(n, dim, dtype=dtype, device=dev) * (high - low) + low


# ---------------------------- neural map -----------------------------

class ResidualMLPMap(nn.Module):
    """Vector-valued pushforward map T_theta(z) = z + net_theta(z).

    By default the last layer is initialized to zero, so T_theta starts
    exactly as the identity map. Set zero_init_output=False to retain the
    standard MLP initialization when net is used as a tangent field.
    """

    def __init__(self, dim: int = 2, width: int = 64, depth: int = 3,
                 activation: str = "tanh", dtype=torch.float32, *,
                 zero_init_output: bool = True):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        act = {
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }[activation]

        layers: List[nn.Module] = []
        d_in = dim
        for _ in range(depth):
            layers.append(nn.Linear(d_in, width, dtype=dtype))
            layers.append(act())
            d_in = width
        layers.append(nn.Linear(d_in, dim, dtype=dtype))
        self.net = nn.Sequential(*layers)

        last = self.net[-1]
        if zero_init_output and isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------- vector DTB primitives -----------------------

def map_at(theta_flat: torch.Tensor, z: torch.Tensor, model: nn.Module,
           structure) -> torch.Tensor:
    """Evaluate vector map T_theta(z) for a batch z using flat parameters."""
    trainable = unflatten(theta_flat, structure)
    params_and_buffers = dict(model.named_parameters())
    params_and_buffers.update(dict(model.named_buffers()))
    params_and_buffers.update(trainable)
    return functional_call(model, params_and_buffers, (z,))


def _map_one(theta_flat: torch.Tensor, z_single: torch.Tensor,
             model: nn.Module, structure) -> torch.Tensor:
    return map_at(theta_flat, z_single.unsqueeze(0), model, structure).squeeze(0)


def game_dtb_basis_matrices(theta_flat: torch.Tensor, sel: torch.Tensor,
                            z: torch.Tensor, model: nn.Module, structure,
                            chunk: int = 2000
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return vector map values and restricted tangent basis.

    Outputs:
        y_vals:  (N, d)       T_theta(z_i)
        J_tens:  (N, d, m)    d T_theta(z_i)_a / d theta_sel_j
        J_flat:  (N*d, m)     J_tens flattened for least squares
    """
    theta_sel = theta_flat[sel].detach().clone()
    theta_frozen_part = theta_flat.detach().clone()

    def map_fn_one(theta_s, z_single):
        full = theta_frozen_part.clone()
        full = full.index_copy(0, sel, theta_s)
        return _map_one(full, z_single, model, structure)

    jac_map = jacrev(map_fn_one, argnums=0)
    map_batched = vmap(map_fn_one, in_dims=(None, 0))
    jac_batched = vmap(jac_map, in_dims=(None, 0))

    y_chunks: List[torch.Tensor] = []
    j_chunks: List[torch.Tensor] = []
    for i in range(0, z.shape[0], chunk):
        zb = z[i:i + chunk]
        y_chunks.append(map_batched(theta_sel, zb))
        j_chunks.append(jac_batched(theta_sel, zb))

    y_vals = torch.cat(y_chunks, dim=0)
    J_tens = torch.cat(j_chunks, dim=0)
    J_flat = J_tens.reshape(z.shape[0] * z.shape[1], sel.numel())
    return y_vals, J_tens, J_flat


class CurrentGameMap:
    """Callable z -> accumulated DTB map at the end of a block.

    T_accum(z) = T_theta_block(z) + h * dT_theta_block(z)/dtheta_sel * s.
    """

    def __init__(self, theta_block_flat: torch.Tensor, sel: torch.Tensor,
                 s: torch.Tensor, h: float, model: nn.Module, structure,
                 chunk: int):
        self.theta = theta_block_flat
        self.sel = sel
        self.s = s
        self.h = h
        self.model = model
        self.structure = structure
        self.chunk = chunk

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        y_vals, J_tens, _ = game_dtb_basis_matrices(
            self.theta, self.sel, z, self.model, self.structure,
            chunk=self.chunk
        )
        return y_vals + self.h * torch.einsum("ndm,m->nd", J_tens, self.s)


def fit_map_to_target(model: nn.Module,
                      target_fn: Callable[[torch.Tensor], torch.Tensor],
                      dim: int, low: float, high: float,
                      n_samples: int = 10_000, steps: int = 2000,
                      lr: float = 1e-3, batch_size: int = 2048,
                      verbose: bool = False) -> float:
    """Fit T_theta(z) to a vector-valued target map on z ~ lambda."""
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        z_train = sample_initial(n_samples, dim, low, high, dtype, dev)
        y_train = target_fn(z_train).detach()

    opt = torch.optim.Adam([p for p in model.parameters()
                            if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss_fn = nn.MSELoss()

    for it in range(steps):
        idx = torch.randint(0, n_samples, (batch_size,), device=dev)
        zb = z_train[idx]
        yb = y_train[idx]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(zb), yb)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and it % max(1, steps // 10) == 0:
            print(f"    fit it={it:5d}  mse={loss.item():.3e}")

    with torch.no_grad():
        z_test = sample_initial(4000, dim, low, high, dtype, dev)
        rmse = torch.sqrt(torch.mean((model(z_test) - target_fn(z_test)) ** 2))
    return float(rmse)


# ------------------------------- driver ------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--game", type=str, default="cournot2",
                   choices=["cournot2"])
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.001)
    p.add_argument("--L", type=int, default=50, help="DTB reset cadence")
    p.add_argument("--N", type=int, default=2000,
                   help="collocation samples per DTB step")
    p.add_argument("--m", type=int, default=1500,
                   help="random DTB parameter sub-basis size")
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--domain_low", type=float, default=0.0)
    p.add_argument("--domain_high", type=float, default=1.0)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--activation", type=str, default="tanh",
                   choices=["tanh", "gelu", "relu", "silu"])
    p.add_argument("--fit_steps", type=int, default=2000)
    p.add_argument("--fit_lr", type=float, default=1e-3)
    p.add_argument("--fit_batch", type=int, default=2048)
    p.add_argument("--fit_samples", type=int, default=10_000)
    p.add_argument("--solver", type=str, default="lstsq",
                   choices=["svd_gpu", "svd", "lstsq"])
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--cournot_b", type=float, default=1.0)
    p.add_argument("--cournot_mu", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="game_cournot2")
    p.add_argument("--snap_dir", type=str, default="snapshots")
    p.add_argument("--verbose_fit", action="store_true")
    p.add_argument("--diag", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.game == "cournot2" and args.dim != 2:
        raise SystemExit("--game cournot2 requires --dim 2")

    torch.manual_seed(args.seed)
    dev = device()
    dtype = torch.float32
    velocity = make_velocity(args)

    K = int(round(args.T / args.h))
    if abs(K * args.h - args.T) > 1e-9:
        raise SystemExit("T must be a multiple of h")
    n_blocks = K // args.L
    if n_blocks * args.L != K:
        raise SystemExit("K = T/h must be a multiple of L")

    out_dir = os.path.join(args.snap_dir, f"run_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    model = ResidualMLPMap(
        dim=args.dim, width=args.width, depth=args.depth,
        activation=args.activation, dtype=dtype,
    ).to(dev)
    M = count_trainable(model)
    m = min(args.m, M)
    _, structure, _ = flat_params(model)
    torch.save({"args": vars(args), "structure": structure},
               os.path.join(out_dir, "meta.pt"))

    print(f"device: {dev}   dtype: {dtype}")
    print(f"game={args.game}  dim={args.dim}  K={K}  L={args.L}  blocks={n_blocks}")
    print(f"trainable params: {M}  DTB basis size: {m}")

    theta_flat, _, _ = flat_params(model)
    torch.save(theta_flat.detach().cpu(),
               os.path.join(out_dir, "theta_t0000.pt"))

    log = {
        "step": [], "t": [], "x_mean": [], "x_std": [],
        "velocity_norm": [], "alpha_norm": [], "s_norm": [],
        "resid": [], "reset_rmse": [], "wall_s": [], "block": [],
    }

    t_total0 = time.time()
    for blk in range(n_blocks):
        block_t0 = time.time()
        theta_block, _, _ = flat_params(model)
        theta_block = theta_block.to(dev)
        sel = torch.randperm(M, device=dev)[:m].sort().values
        s = torch.zeros(m, dtype=dtype, device=dev)
        avg_resid = 0.0

        for step in range(args.L):
            step_t0 = time.time()
            z = sample_initial(args.N, args.dim, args.domain_low,
                               args.domain_high, dtype, dev)
            y_vals, J_tens, J_flat = game_dtb_basis_matrices(
                theta_block, sel, z, model, structure, chunk=args.chunk
            )
            x_k = y_vals + args.h * torch.einsum("ndm,m->nd", J_tens, s)
            v = velocity(x_k)
            g = v.reshape(args.N * args.dim)
            alpha = jform_solve(J_flat, g, rtol=args.rtol,
                                method=args.solver)

            with torch.no_grad():
                resid = float(torch.linalg.norm(J_flat @ alpha - g) /
                              (torch.linalg.norm(g) + 1e-30))
                avg_resid += resid
                x_mean = x_k.mean(dim=0).detach().cpu().tolist()
                x_std = x_k.std(dim=0).detach().cpu().tolist()

            global_step = blk * args.L + step
            log["step"].append(global_step)
            log["t"].append((global_step + 1) * args.h)
            log["block"].append(blk + 1)
            log["x_mean"].append(x_mean)
            log["x_std"].append(x_std)
            log["velocity_norm"].append(float(v.norm()))
            log["alpha_norm"].append(float(alpha.norm()))
            log["s_norm"].append(float(s.norm()))
            log["resid"].append(resid)
            log["reset_rmse"].append(float("nan"))
            log["wall_s"].append(time.time() - step_t0)

            if args.diag and step % max(1, args.L // 5) == 0:
                print(f"  step {step:4d}/{args.L}  "
                      f"mean={x_mean}  |v|={float(v.norm()):.3e}  "
                      f"|alpha|={float(alpha.norm()):.3e}  resid={resid:.3e}")
            s = s + alpha

        avg_resid /= args.L
        target_fn = CurrentGameMap(theta_block, sel, s, args.h, model,
                                   structure, chunk=args.chunk)
        rmse_reset = fit_map_to_target(
            model, target_fn, dim=args.dim, low=args.domain_low,
            high=args.domain_high, n_samples=args.fit_samples,
            steps=args.fit_steps, lr=args.fit_lr, batch_size=args.fit_batch,
            verbose=args.verbose_fit,
        )
        log["reset_rmse"][-1] = rmse_reset

        t_end = (blk + 1) * args.L * args.h
        theta_now, _, _ = flat_params(model)
        idx_str = f"{int(round(t_end * 1000)):04d}"
        torch.save(theta_now.detach().cpu(),
                   os.path.join(out_dir, f"theta_t{idx_str}.pt"))

        print(f"block {blk + 1:3d}/{n_blocks}  t={t_end:6.3f}  "
              f"avg residual={avg_resid:.3e}  reset RMSE={rmse_reset:.3e}  "
              f"({time.time() - block_t0:.1f}s)")

    torch.save(log, os.path.join(out_dir, "log.pt"))
    print(f"=== done in {(time.time() - t_total0) / 60:.1f} min ===")
    print(f"snapshots in {out_dir}")


if __name__ == "__main__":
    main()
