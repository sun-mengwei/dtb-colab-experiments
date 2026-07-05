"""
Neural-Galerkin baseline (Bruna et al., 2023 / paper's "forward update"
variant of Algorithm 3 with no reset).

  alpha^k  = Jac(theta^k; z)^+  *  F[u_{theta^k}](z)        (J-form solve)
  theta^{k+1}  =  theta^k  +  h * alpha^k                   (forward update)
  -- no DTB accumulation, no Proposition 2.4 reset

Network, samples, basis size and PDE are *identical* to `run_5d_ac.py`,
so the only manipulated variables are the update rule (forward vs DTB)
and the absence of reset.  Expectation: NGM diverges -- as soon as the
Gram matrix becomes ill-conditioned the parameter ODE becomes stiff /
singular and the Euler step over-shoots.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from network import PeriodicMMNN, count_trainable
from dtb import (device, flat_params, write_flat_into_model,
                 dtb_basis_matrices, jform_solve, fit_model_to_target)
from problem import initial_condition, ac_rhs, NU


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--h", type=float, default=0.01)
    p.add_argument("--nu", type=float, default=NU)
    p.add_argument("--N", type=int, default=20_000)
    p.add_argument("--m", type=int, default=6000)
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--width", type=int, default=366)
    p.add_argument("--rank", type=int, default=25)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--k_per_dim", type=int, default=40)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--fit_steps", type=int, default=6000)
    p.add_argument("--fit_lr", type=float, default=3e-3)
    p.add_argument("--fit_batch", type=int, default=4000)
    p.add_argument("--fit_samples", type=int, default=20_000)
    p.add_argument("--solver", type=str, default="lstsq",
                   choices=["svd_gpu", "svd", "lstsq"])
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="ngm")
    p.add_argument("--snap_dir", type=str, default="snapshots")
    p.add_argument("--snap_every", type=int, default=20,
                   help="save a theta snapshot every N steps")
    # Safety nets.
    p.add_argument("--blowup_thresh", type=float, default=20.0,
                   help="stop if max |u| exceeds this")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = device()
    dtype = torch.float32

    K = int(round(args.T / args.h))
    print(f"NGM: K={K} steps, h={args.h}, T={args.T}, "
          f"N={args.N}, m={args.m}")

    out_dir = os.path.join(args.snap_dir, f"run_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    model = PeriodicMMNN(
        dim_in=5, k_per_dim=args.k_per_dim,
        width=args.width, rank=args.rank, depth=args.depth,
        activation=args.activation, dtype=dtype,
    ).to(dev)
    M = count_trainable(model)
    _, structure, _ = flat_params(model)
    print(f"trainable params: {M}")

    frozen = {n: p.detach().cpu().clone()
              for n, p in model.named_parameters() if not p.requires_grad}
    torch.save({"args": vars(args), "structure": structure,
                "frozen": frozen}, os.path.join(out_dir, "meta.pt"))

    # ---- initial fit ----
    print("=== initial fit to IC ===")
    rmse0 = fit_model_to_target(
        model, initial_condition, d_in=5,
        n_samples=args.fit_samples, steps=args.fit_steps,
        lr=args.fit_lr, batch_size=args.fit_batch,
    )
    print(f"initial-fit RMSE = {rmse0:.3e}")

    # Save t=0 snapshot.
    theta_flat, _, _ = flat_params(model)
    theta_flat = theta_flat.to(dev)
    torch.save(theta_flat.detach().cpu(),
               os.path.join(out_dir, "theta_t000.pt"))

    # Fixed random sub-basis for the whole NGM run.
    sel = torch.randperm(M, device=dev)[:args.m].sort().values
    torch.save(sel.detach().cpu(), os.path.join(out_dir, "sel.pt"))

    # ---- per-step diagnostics ----
    log = {
        "step": [], "t": [], "u_max": [], "u_rms": [],
        "alpha_norm": [], "J_norm": [], "g_norm": [],
        "sigma_min": [], "sigma_max": [], "resid": [],
        "wall_s": [], "status": [],
    }

    blew_up = False
    t_total0 = time.time()
    for k in range(K):
        step_t0 = time.time()
        z = (torch.rand(args.N, 5, dtype=dtype, device=dev) * 2.0 - 1.0)
        u_vals, J_sel, lap_vals, Lap_J_sel = dtb_basis_matrices(
            theta_flat, sel, z, model, structure, chunk=args.chunk
        )
        # NGM evaluates u and Lap u at the CURRENT theta directly
        # (no DTB accumulation).
        u_k = u_vals
        lap_k = lap_vals
        g = ac_rhs(u_k, lap_k, nu=args.nu)
        alpha = jform_solve(J_sel, g, rtol=args.rtol, method=args.solver)

        # Diagnostics.
        with torch.no_grad():
            num = float(torch.linalg.norm(J_sel @ alpha - g))
            den = float(torch.linalg.norm(g)) + 1e-30
            S = torch.linalg.svdvals(J_sel)
            sigma_max = float(S[0]); sigma_min = float(S[-1])
            u_max = float(u_k.abs().max())
            u_rms = float((u_k ** 2).mean().sqrt())

        log["step"].append(k)
        log["t"].append((k + 1) * args.h)
        log["u_max"].append(u_max)
        log["u_rms"].append(u_rms)
        log["alpha_norm"].append(float(alpha.norm()))
        log["J_norm"].append(float(J_sel.norm()))
        log["g_norm"].append(float(g.norm()))
        log["sigma_min"].append(sigma_min)
        log["sigma_max"].append(sigma_max)
        log["resid"].append(num / den)
        log["wall_s"].append(time.time() - step_t0)
        log["status"].append("ok")

        # Forward update of selected params (the NGM step).
        with torch.no_grad():
            theta_flat = theta_flat.clone()
            theta_flat[sel] = theta_flat[sel] + args.h * alpha
        write_flat_into_model(model, theta_flat, structure)

        if (k + 1) % args.snap_every == 0 or k == K - 1:
            torch.save(theta_flat.detach().cpu(),
                       os.path.join(out_dir,
                                    f"theta_t{int(round((k+1)*args.h*100)):03d}.pt"))

        if k < 5 or (k + 1) % 5 == 0:
            print(f"step {k+1:4d}/{K}  t={(k+1)*args.h:5.2f}  "
                  f"|u|_max={u_max:.3e}  |alpha|={float(alpha.norm()):.3e}  "
                  f"sigma_min={sigma_min:.2e}  resid={num/den:.2e}")

        if not (u_max == u_max) or u_max > args.blowup_thresh:
            print(f"!! blowup at step {k+1} t={(k+1)*args.h:.3f}: "
                  f"|u|_max={u_max:.3e}")
            log["status"][-1] = "blowup"
            blew_up = True
            torch.save(theta_flat.detach().cpu(),
                       os.path.join(out_dir, "theta_at_blowup.pt"))
            break

    torch.save(log, os.path.join(out_dir, "log.pt"))
    total = time.time() - t_total0
    print(f"=== NGM finished in {total:.1f}s,  blew up = {blew_up} ===")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
