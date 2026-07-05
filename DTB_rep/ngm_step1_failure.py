"""
One-step failure-mode demonstration for NGM, mirroring Table 1 of
arXiv:2509.00957: at t = 0 (right after the initial fit), compute the
J-form alpha at a range of pseudo-inverse truncation cutoffs `rtol`
and report ||alpha||.

The paper's claim: as rtol shrinks (more singular values kept), ||alpha||
blows up because J is ill-conditioned and the small singular values
amplify the projected RHS.

Two basis sizes:
  m = 6000     (random sub-basis, as the DTB / our NGM uses)
  m = M_full   (full parameter Galerkin -- the classical NGM)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from network import PeriodicMMNN, count_trainable
from dtb import flat_params, dtb_basis_matrices, write_flat_into_model, device
from problem import initial_condition, ac_rhs, NU


def alpha_at_rtol(J: torch.Tensor, g: torch.Tensor, rtol: float):
    """Truncated-SVD pseudo-inverse on GPU fp32; keeps singular values
    above rtol * sigma_max.  Returns alpha, n_kept, sigma_min_kept."""
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    cutoff = rtol * float(S[0])
    keep = S > cutoff
    Sinv = torch.where(keep, 1.0 / S, torch.zeros_like(S))
    alpha = Vh.T @ (Sinv * (U.T @ g))
    return alpha, int(keep.sum().item()), float(S[keep][-1].item()) if keep.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="snapshots/run_paper2",
                    help="directory holding meta.pt and theta_t000.pt")
    ap.add_argument("--N", type=int, default=20_000)
    ap.add_argument("--m", type=int, default=6000,
                    help="random sub-basis size; use 0 for full M")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtols", type=float, nargs="+",
                    default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8,
                             1e-10, 1e-12])
    args = ap.parse_args()

    dev = device()
    torch.manual_seed(args.seed)

    meta = torch.load(os.path.join(args.run_dir, "meta.pt"),
                      weights_only=False)
    a = meta["args"]
    structure = meta["structure"]
    frozen = meta["frozen"]
    model = PeriodicMMNN(
        dim_in=5, k_per_dim=a["k_per_dim"],
        width=a["width"], rank=a["rank"], depth=a["depth"],
        activation=a["activation"], dtype=torch.float32,
    ).to(dev)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in frozen:
                p.copy_(frozen[name].to(dev))
    flat = torch.load(os.path.join(args.run_dir, "theta_t000.pt"),
                      weights_only=False).to(dev)
    write_flat_into_model(model, flat, structure)
    theta_flat, _, _ = flat_params(model)
    theta_flat = theta_flat.to(dev)
    M = theta_flat.numel()
    m = M if args.m == 0 else args.m
    print(f"M={M} (total trainable),  m={m} (sub-basis used)")

    sel = (torch.arange(M, device=dev) if m == M
           else torch.randperm(M, device=dev)[:m].sort().values)

    z = (torch.rand(args.N, 5, device=dev) * 2.0 - 1.0)
    print(f"computing J_sel and Lap_J_sel at t=0 on N={args.N} points...")
    u_vals, J_sel, lap_vals, _ = dtb_basis_matrices(
        theta_flat, sel, z, model, structure, chunk=args.chunk
    )
    g = ac_rhs(u_vals, lap_vals, nu=NU)

    # SVD once -- reuse across rtols.
    print(f"computing SVD of J ({J_sel.shape})...")
    U, S, Vh = torch.linalg.svd(J_sel, full_matrices=False)
    smax = float(S[0])
    smin = float(S[-1])
    print(f"sigma_max = {smax:.3e}, sigma_min = {smin:.3e}, "
          f"cond = {smax/smin:.3e}")

    # Reference defaults.
    eps = torch.finfo(J_sel.dtype).eps
    torch_default_rcond = max(J_sel.shape[0], J_sel.shape[1]) * eps
    print(f"torch.linalg.lstsq default rcond = "
          f"max(M,N)*eps = {torch_default_rcond:.3e}")
    print()
    print(f"{'rtol':>10}  {'cutoff':>10}  {'kept':>7}  {'sigma_min_kept':>14}"
          f"  {'||alpha||':>14}  {'||resid||/||g||':>14}")

    rows = []
    for rtol in sorted(args.rtols, reverse=True):
        cutoff = rtol * smax
        keep = S > cutoff
        Sinv = torch.where(keep, 1.0 / S, torch.zeros_like(S))
        alpha = Vh.T @ (Sinv * (U.T @ g))
        n_kept = int(keep.sum().item())
        smin_kept = float(S[keep][-1].item()) if keep.any() else float("nan")
        alpha_norm = float(alpha.norm())
        resid = float((J_sel @ alpha - g).norm() / (g.norm() + 1e-30))
        rows.append((rtol, cutoff, n_kept, smin_kept, alpha_norm, resid))
        print(f"{rtol:10.0e}  {cutoff:10.3e}  {n_kept:7d}"
              f"  {smin_kept:14.3e}  {alpha_norm:14.3e}  {resid:14.3e}")

    # Save for plotting / PDF.
    out = os.path.join(args.run_dir,
                       f"ngm_step1_failure_m{m}.txt")
    with open(out, "w") as fh:
        fh.write("# NGM one-step alpha vs SVD truncation\n")
        fh.write(f"# N={args.N}  m={m}  sigma_max={smax:.6e}  "
                 f"sigma_min={smin:.6e}\n")
        fh.write("# rtol  cutoff  n_kept  sigma_min_kept  alpha_norm  resid\n")
        for r in rows:
            fh.write(" ".join(f"{x:.6e}" for x in r) + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
