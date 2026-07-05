"""
Side-by-side comparison of our DTB reproduction against the original
paper's Figure 5 panels (downloaded from arxiv.org/html/2509.00957).

Paper layout: each x{13..17}.png is one hyperplane row, six columns
left-to-right at t = 0, 0.4, 0.8, 1.2, 1.6, 2.0.

Output: one PNG per hyperplane (paper above ours, both ~3" tall) plus
one t=0 / t=2 cross-hyperplane summary.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.image import imread

from network import PeriodicMMNN
from dtb import write_flat_into_model, device
from visualize import HYPERPLANES, load_model_from_meta


def render_panel(model, structure, dev, grid_fn, theta_flat, n):
    z_np, AA, BB, _ = grid_fn(n)
    z = torch.from_numpy(z_np).to(dev)
    write_flat_into_model(model, theta_flat, structure)
    with torch.no_grad():
        u = model(z).cpu().numpy().reshape(n, n)
    return AA, BB, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--paper_dir", type=str, default="paper_figures")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--times", type=float, nargs="+",
                    default=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    args = ap.parse_args()

    paper_files = ["x13.png", "x14.png", "x15.png", "x16.png", "x17.png"]

    dev = device()
    model, structure = load_model_from_meta(
        os.path.join(args.run_dir, "meta.pt"), dev
    )
    thetas = {
        t: torch.load(os.path.join(args.run_dir,
                                   f"theta_t{int(round(t*100)):03d}.pt"),
                      weights_only=False).to(dev)
        for t in args.times
    }

    vmin, vmax = -1.05, 1.05

    # ---- per-hyperplane pair: paper above ours ----
    for i, (label, grid_fn) in enumerate(HYPERPLANES):
        paper_img = imread(os.path.join(args.paper_dir, paper_files[i]))

        fig, axes = plt.subplots(
            nrows=2, ncols=1,
            figsize=(14.0, 5.2),
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.20},
        )
        # paper
        axes[0].imshow(paper_img, aspect="auto")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].set_title(
            f"paper Fig. 5{label}   (t = 0, 0.4, 0.8, 1.2, 1.6, 2.0)",
            fontsize=11,
        )

        # ours: inset 6 panels into the bottom axis
        axes[1].set_axis_off()
        sub_gs = axes[1].get_subplotspec().subgridspec(
            1, len(args.times), wspace=0.05
        )
        for j, t in enumerate(args.times):
            AA, BB, u = render_panel(model, structure, dev, grid_fn,
                                     thetas[t], args.n)
            ax = fig.add_subplot(sub_gs[0, j])
            im = ax.pcolormesh(AA, BB, u, cmap="viridis",
                               vmin=vmin, vmax=vmax, shading="auto")
            ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=7)
            ax.set_title(f"t = {t:.2f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"ours {label}", fontsize=11)
        fig.suptitle("", y=0.99)
        cax = fig.add_axes([0.94, 0.08, 0.012, 0.84])
        fig.colorbar(im, cax=cax)
        fig.subplots_adjust(left=0.03, right=0.93, top=0.92, bottom=0.05)
        out = os.path.join(args.run_dir, f"cmp_hyperplane_{label[1]}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")

    print("done")


if __name__ == "__main__":
    main()
