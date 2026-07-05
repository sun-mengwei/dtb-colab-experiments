"""
DTB / Forward-Euler with periodic-reset solve of the 5-D Allen-Cahn
equation (arXiv:2509.00957, sec 3, Algorithm 3 + Proposition 2.4).

  d_t u = nu * Lap(u) + u - u^3,   z in [-1,1]^5,   periodic BCs
  u(0, z) = w(z)                                     from problem.py
  nu = 0.01,   T = 2.0,   h = 0.01,   K = 200 steps
  L = 20    : reset cadence (10 blocks)
  N = 20000 : sample points per step (Monte-Carlo collocation)
  m = 6000  : randomly selected DTB basis vectors from B(f, theta)

Outputs:
  snapshots/run_<tag>/theta_t<XX>.pt   trainable flat-vector at t = 0.0,
                                       0.2, 0.4, ..., 2.0
  snapshots/run_<tag>/meta.pt          architecture meta + frozen W,b
"""

from __future__ import annotations

import argparse
import math
import os
import time
import json

import torch

from network import PeriodicMMNN, count_trainable
from dtb import (device, flat_params, write_flat_into_model,
                 dtb_basis_matrices, jform_solve, fit_model_to_target)
from problem import initial_condition, ac_rhs, NU


# --------------------------------- args -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # PDE / time stepping
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--h", type=float, default=0.01)
    p.add_argument("--L", type=int, default=20, help="reset cadence")
    p.add_argument("--nu", type=float, default=NU)
    # Sampling
    p.add_argument("--N", type=int, default=20_000,
                   help="MC collocation points per step")
    p.add_argument("--m", type=int, default=6000,
                   help="random DTB basis size")
    p.add_argument("--chunk", type=int, default=1000,
                   help="vmap chunk for Jacobian sweep")
    # Architecture
    p.add_argument("--width", type=int, default=366)
    p.add_argument("--rank", type=int, default=25)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--k_per_dim", type=int, default=40)
    p.add_argument("--activation", type=str, default="gelu")
    # Reset/init fit
    p.add_argument("--fit_steps", type=int, default=4000)
    p.add_argument("--fit_lr", type=float, default=5e-3)
    p.add_argument("--fit_batch", type=int, default=4000)
    p.add_argument("--fit_samples", type=int, default=20_000)
    # Solver (unbiased least squares only -- no Tikhonov)
    p.add_argument("--solver", type=str, default="lstsq",
                   choices=["svd_gpu", "svd", "lstsq"])
    p.add_argument("--rtol", type=float, default=1e-8,
                   help="truncation ratio for the SVD pseudo-inverse "
                        "(applied to lstsq via rcond as well). Default "
                        "1e-8 keeps essentially all singular values; "
                        "use 1e-2 for a 2% cutoff if you want to "
                        "regularize.")
    # Bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="paper")
    p.add_argument("--snap_dir", type=str, default="snapshots")
    p.add_argument("--verbose_fit", action="store_true")
    p.add_argument("--diag", action="store_true",
                   help="print per-step diagnostics")
    p.add_argument("--log_svdvals", action="store_true",
                   help="record per-step sigma_min/sigma_max of J_sel "
                        "(adds ~3 s/step)")
    return p.parse_args()


# --------------------------- target function ------------------------

class CurrentSolution:
    """Callable z -> u^k(z) at the end of a block.

      u(z) = f_theta(z) + h * J_sel(theta; z) @ s

    All quantities frozen to the values supplied at construction time.
    Used as the target for the periodic-reset fit.
    """

    def __init__(self, theta_block_flat, sel, s, h, model, structure,
                 chunk: int):
        self.theta = theta_block_flat
        self.sel = sel
        self.s = s
        self.h = h
        self.model = model
        self.structure = structure
        self.chunk = chunk

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        u_vals, J_sel, _, _ = dtb_basis_matrices(
            self.theta, self.sel, z, self.model, self.structure,
            chunk=self.chunk
        )
        return u_vals + self.h * (J_sel @ self.s)


# --------------------------------- main -----------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    dtype = torch.float32
    print(f"device: {dev}   dtype: {dtype}")

    # K = T/h must be a multiple of L.
    K = int(round(args.T / args.h))
    assert abs(K * args.h - args.T) < 1e-9, "T must be a multiple of h"
    n_blocks = K // args.L
    assert n_blocks * args.L == K, "K must be a multiple of L"
    print(f"K={K} steps, L={args.L}, n_blocks={n_blocks}")

    # Output directory.
    out_dir = os.path.join(args.snap_dir, f"run_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    # ----------- build the network and structure metadata ----------
    model = PeriodicMMNN(
        dim_in=5, k_per_dim=args.k_per_dim,
        width=args.width, rank=args.rank, depth=args.depth,
        activation=args.activation, dtype=dtype,
    ).to(dev)
    M = count_trainable(model)
    print(f"trainable params: {M}")

    # Cache the (name, shape) structure of the trainable params; we
    # rebuild flat tensors using this throughout.
    _, structure, _ = flat_params(model)

    # Save frozen W, b so the snapshot is reproducible.
    frozen = {name: p.detach().cpu().clone()
              for name, p in model.named_parameters()
              if not p.requires_grad}
    torch.save({
        "args": vars(args),
        "structure": structure,
        "frozen": frozen,
    }, os.path.join(out_dir, "meta.pt"))

    # --------------- initial fit: f_theta ~= w(z) -------------------
    print("=== initial fit to IC ===")
    rmse0 = fit_model_to_target(
        model, initial_condition, d_in=5,
        n_samples=args.fit_samples, steps=args.fit_steps,
        lr=args.fit_lr, batch_size=args.fit_batch,
        verbose=args.verbose_fit,
    )
    print(f"initial-fit RMSE = {rmse0:.3e}")
    # Snapshot t = 0.
    theta_flat, _, _ = flat_params(model)
    torch.save(theta_flat.detach().cpu(),
               os.path.join(out_dir, "theta_t000.pt"))

    # Per-step log (parallel to run_ngm.py's `log` dict).
    log = {
        "step": [], "t": [], "u_max": [], "u_rms": [],
        "alpha_norm": [], "s_norm": [], "J_norm": [], "g_norm": [],
        "sigma_min": [], "sigma_max": [], "resid": [], "wall_s": [],
        "block": [],
    }

    # --------------- block loop -------------------------------------
    t_total0 = time.time()
    for blk in range(n_blocks):
        block_t0 = time.time()
        # Refresh flat params at the start of this block (after reset).
        theta_block, _, _ = flat_params(model)
        theta_block = theta_block.to(dev)
        # Fresh random sub-basis for this block.
        sel = torch.randperm(M, device=dev)[:args.m].sort().values

        # Accumulator s starts at 0.
        s = torch.zeros(args.m, dtype=dtype, device=dev)
        avg_resid = 0.0
        for step in range(args.L):
            step_t0 = time.time()
            # Fresh MC samples.
            z = (torch.rand(args.N, 5, dtype=dtype, device=dev) * 2.0 - 1.0)
            u_vals, J_sel, lap_vals, Lap_J_sel = dtb_basis_matrices(
                theta_block, sel, z, model, structure, chunk=args.chunk
            )
            # Current solution and its Laplacian at z.
            u_k = u_vals + args.h * (J_sel @ s)
            lap_k = lap_vals + args.h * (Lap_J_sel @ s)
            # RHS: F[u^k] = nu * Lap(u^k) + u^k - u^k^3.
            g = ac_rhs(u_k, lap_k, nu=args.nu)
            # Solve J_sel alpha = g (least squares).
            alpha = jform_solve(J_sel, g, rtol=args.rtol,
                                method=args.solver)
            # Per-step diagnostics.
            with torch.no_grad():
                num = float(torch.linalg.norm(J_sel @ alpha - g))
                den = float(torch.linalg.norm(g)) + 1e-30
                avg_resid += num / den
                # Singular values via svdvals (no U, V). On GPU fp32
                # this costs ~3s; skip with --no_svdvals if too slow.
                if args.log_svdvals:
                    S = torch.linalg.svdvals(J_sel)
                    sigma_max = float(S[0]); sigma_min = float(S[-1])
                else:
                    sigma_max = sigma_min = float("nan")

            global_step = blk * args.L + step
            global_t = (global_step + 1) * args.h
            log["step"].append(global_step)
            log["t"].append(global_t)
            log["block"].append(blk + 1)
            log["u_max"].append(float(u_k.abs().max()))
            log["u_rms"].append(float((u_k ** 2).mean().sqrt()))
            log["alpha_norm"].append(float(alpha.norm()))
            log["s_norm"].append(float(s.norm()))
            log["J_norm"].append(float(J_sel.norm()))
            log["g_norm"].append(float(g.norm()))
            log["sigma_min"].append(sigma_min)
            log["sigma_max"].append(sigma_max)
            log["resid"].append(num / den)
            log["wall_s"].append(time.time() - step_t0)

            if args.diag and step % max(1, args.L // 5) == 0:
                print(f"  step {step:3d}/{args.L}  "
                      f"|u_k| max={float(u_k.abs().max()):.3e}  "
                      f"|alpha|={float(alpha.norm()):.3e}  "
                      f"|s|={float(s.norm()):.3e}  "
                      f"|J|={float(J_sel.norm()):.3e}  "
                      f"|g|={float(g.norm()):.3e}  "
                      f"resid={num/den:.3e}")
            s = s + alpha

        avg_resid /= args.L
        # Snapshot of the solution function at t = (blk+1)*L*h.
        target_fn = CurrentSolution(theta_block, sel, s, args.h, model,
                                    structure, chunk=args.chunk)

        # ----------- periodic reset -----------------------------
        # Fit the live model to target_fn (this writes new theta into
        # the model in-place; frozen W, b are kept).
        rmse_reset = fit_model_to_target(
            model, target_fn, d_in=5,
            n_samples=args.fit_samples, steps=args.fit_steps,
            lr=args.fit_lr, batch_size=args.fit_batch,
            verbose=args.verbose_fit,
        )
        t_end = (blk + 1) * args.L * args.h
        # Save snapshot.
        theta_now, _, _ = flat_params(model)
        idx_str = f"{int(round(t_end * 100)):03d}"  # t * 100, zero-padded
        snap_path = os.path.join(out_dir, f"theta_t{idx_str}.pt")
        torch.save(theta_now.detach().cpu(), snap_path)

        block_dt = time.time() - block_t0
        print(f"block {blk+1:2d}/{n_blocks}  t={t_end:5.2f}  "
              f"avg DTB residual={avg_resid:.3e}  reset RMSE={rmse_reset:.3e}"
              f"  ({block_dt:.1f}s)")

    t_total = time.time() - t_total0
    torch.save(log, os.path.join(out_dir, "log.pt"))
    print(f"=== done in {t_total/60:.1f} min ===")
    print(f"snapshots in {out_dir}")


if __name__ == "__main__":
    main()
