"""
Reproduce Figure 5 of arXiv:2509.00957: 5-D Allen-Cahn solution
snapshots on five 2-D hyperplanes.

Layout (matches the paper):
  rows    = hyperplanes (a) ... (e)
  columns = times  t = 0.0, 0.4, 0.8, 1.2, 1.6, 2.0

Hyperplane definitions from Section 3.1 (re-used by the AC figure):

  (a)  -z1 = z2,   z4 = (z2 - z5)/2,   z3 = 0     -> free (z2, z5)
  (b)  z5 = (z1 + z3)/2,   z2 = z4 = 0            -> free (z1, z3)
  (c)  z1 = z2 = 0.3,      z5 = 0.15 - z3/2       -> free (z3, z4)
  (d)  z1 = 0.4 z4 + 0.6 z5,  z2 = z3 = 0.8       -> free (z4, z5)
  (e)  z2 = 0.75 z1 + 0.25 z5,
       z4 = 0.25 z1 + 0.75 z5,  z3 = 0.5          -> free (z1, z5)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from network import PeriodicMMNN
from dtb import write_flat_into_model, device


def grid_a(n):
    """(a)  -z1 = z2, z4 = (z2 - z5)/2, z3 = 0; free (z2, z5)."""
    a = np.linspace(-1, 1, n)
    b = np.linspace(-1, 1, n)
    A, B = np.meshgrid(a, b, indexing="xy")  # A=z2, B=z5
    z = np.zeros((n * n, 5), dtype=np.float32)
    z[:, 0] = -A.reshape(-1)            # z1 = -z2
    z[:, 1] = A.reshape(-1)             # z2
    z[:, 2] = 0.0                       # z3
    z[:, 3] = ((A - B) / 2).reshape(-1)  # z4
    z[:, 4] = B.reshape(-1)             # z5
    return z, A, B, ("z2", "z5")


def grid_b(n):
    """(b)  z5 = (z1+z3)/2, z2 = z4 = 0; free (z1, z3)."""
    a = np.linspace(-1, 1, n)
    b = np.linspace(-1, 1, n)
    A, B = np.meshgrid(a, b, indexing="xy")  # A=z1, B=z3
    z = np.zeros((n * n, 5), dtype=np.float32)
    z[:, 0] = A.reshape(-1)             # z1
    z[:, 1] = 0.0                       # z2
    z[:, 2] = B.reshape(-1)             # z3
    z[:, 3] = 0.0                       # z4
    z[:, 4] = ((A + B) / 2).reshape(-1)  # z5
    return z, A, B, ("z1", "z3")


def grid_c(n):
    """(c)  z1 = z2 = 0.3, z5 = 0.15 - z3/2; free (z3, z4)."""
    a = np.linspace(-1, 1, n)
    b = np.linspace(-1, 1, n)
    A, B = np.meshgrid(a, b, indexing="xy")  # A=z3, B=z4
    z = np.zeros((n * n, 5), dtype=np.float32)
    z[:, 0] = 0.3                       # z1
    z[:, 1] = 0.3                       # z2
    z[:, 2] = A.reshape(-1)             # z3
    z[:, 3] = B.reshape(-1)             # z4
    z[:, 4] = (0.15 - A / 2).reshape(-1)  # z5
    return z, A, B, ("z3", "z4")


def grid_d(n):
    """(d)  z1 = 0.4 z4 + 0.6 z5, z2 = z3 = 0.8; free (z4, z5)."""
    a = np.linspace(-1, 1, n)
    b = np.linspace(-1, 1, n)
    A, B = np.meshgrid(a, b, indexing="xy")  # A=z4, B=z5
    z = np.zeros((n * n, 5), dtype=np.float32)
    z[:, 0] = (0.4 * A + 0.6 * B).reshape(-1)  # z1
    z[:, 1] = 0.8                              # z2
    z[:, 2] = 0.8                              # z3
    z[:, 3] = A.reshape(-1)                    # z4
    z[:, 4] = B.reshape(-1)                    # z5
    return z, A, B, ("z4", "z5")


def grid_e(n):
    """(e)  z2 = 0.75 z1 + 0.25 z5, z4 = 0.25 z1 + 0.75 z5, z3 = 0.5;
            free (z1, z5)."""
    a = np.linspace(-1, 1, n)
    b = np.linspace(-1, 1, n)
    A, B = np.meshgrid(a, b, indexing="xy")  # A=z1, B=z5
    z = np.zeros((n * n, 5), dtype=np.float32)
    z[:, 0] = A.reshape(-1)                              # z1
    z[:, 1] = (0.75 * A + 0.25 * B).reshape(-1)          # z2
    z[:, 2] = 0.5                                        # z3
    z[:, 3] = (0.25 * A + 0.75 * B).reshape(-1)          # z4
    z[:, 4] = B.reshape(-1)                              # z5
    return z, A, B, ("z1", "z5")


HYPERPLANES = [
    ("(a)", grid_a),
    ("(b)", grid_b),
    ("(c)", grid_c),
    ("(d)", grid_d),
    ("(e)", grid_e),
]


def load_model_from_meta(meta_path: str, device_):
    meta = torch.load(meta_path, weights_only=False)
    a = meta["args"]
    structure = meta["structure"]
    frozen = meta["frozen"]
    model = PeriodicMMNN(
        dim_in=5, k_per_dim=a["k_per_dim"],
        width=a["width"], rank=a["rank"], depth=a["depth"],
        activation=a["activation"], dtype=torch.float32,
    ).to(device_)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in frozen:
                p.copy_(frozen[name].to(device_))
    return model, structure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str,
                    help="snapshots/run_<tag> directory")
    ap.add_argument("--n", type=int, default=120, help="grid resolution")
    ap.add_argument("--out", type=str, default="figure5_5d_ac.png")
    ap.add_argument("--times", type=float, nargs="+",
                    default=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    ap.add_argument("--cmap", type=str, default="viridis",
                    help="matplotlib colormap (paper uses viridis)")
    args = ap.parse_args()

    dev = device()
    model, structure = load_model_from_meta(
        os.path.join(args.run_dir, "meta.pt"), dev
    )

    snap_paths = {}
    for t in args.times:
        idx = int(round(t * 100))
        path = os.path.join(args.run_dir, f"theta_t{idx:03d}.pt")
        if not os.path.exists(path):
            raise SystemExit(f"missing snapshot {path}")
        snap_paths[t] = path

    fig, axes = plt.subplots(
        nrows=len(HYPERPLANES), ncols=len(args.times),
        figsize=(2.6 * len(args.times), 2.6 * len(HYPERPLANES)),
        squeeze=False,
    )
    vmin, vmax = -1.05, 1.05

    for i, (label, grid_fn) in enumerate(HYPERPLANES):
        z_np, AA, BB, (xname, yname) = grid_fn(args.n)
        z = torch.from_numpy(z_np).to(dev)
        for j, t in enumerate(args.times):
            flat = torch.load(snap_paths[t], weights_only=False).to(dev)
            write_flat_into_model(model, flat, structure)
            with torch.no_grad():
                u = model(z).cpu().numpy().reshape(args.n, args.n)
            ax = axes[i][j]
            im = ax.pcolormesh(AA, BB, u, cmap=args.cmap,
                               vmin=vmin, vmax=vmax, shading="auto")
            if i == 0:
                ax.set_title(f"t = {t:.2f}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{label}", fontsize=11)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=7)
            ax.set_xlabel(xname, fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{label}    {yname}", fontsize=8)

    fig.suptitle("5-D Allen-Cahn via DTB (Algorithm 3 + Prop. 2.4)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.96, 0.97])
    cbar_ax = fig.add_axes([0.965, 0.08, 0.010, 0.84])
    fig.colorbar(im, cax=cbar_ax)
    out_path = os.path.join(args.run_dir, args.out)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
