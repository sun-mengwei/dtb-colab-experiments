"""
Multi-run comparison of DTB and NGM trajectories saved as log.pt
inside their `snapshots/run_*/` directories.

For each run we read the per-step log:
   t, u_max, u_rms, alpha_norm, J_norm, g_norm,
   sigma_min, sigma_max, resid, wall_s

and plot:
  - residual ||J alpha - g|| / ||g||  vs  t
  - alpha_norm                        vs  t
  - cumulative wall_s                 vs  t
  - u_max                             vs  t

Usage:
    python compare_runs.py snapshots/run_dtb_strict snapshots/run_ngm_strict
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_log(run_dir):
    p = os.path.join(run_dir, "log.pt")
    if not os.path.exists(p):
        raise SystemExit(f"missing log: {p}")
    return torch.load(p, weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+",
                    help="snapshots/run_<tag> directories to compare")
    ap.add_argument("--labels", nargs="*",
                    help="display label per run; default = directory basename")
    ap.add_argument("--out", type=str, default=None,
                    help="output PNG path; default = first run's dir")
    args = ap.parse_args()

    runs = args.runs
    labels = args.labels if args.labels else [os.path.basename(r.rstrip("/\\"))
                                              for r in runs]
    if len(labels) != len(runs):
        raise SystemExit("--labels count must match number of runs")

    logs = [load_log(r) for r in runs]
    colors = ["C0", "C3", "C2", "C1", "C4"]

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.5))

    # (top-left) residual
    ax = axs[0, 0]
    for log, lab, c in zip(logs, labels, colors):
        ax.plot(log["t"], log["resid"], "-", color=c, label=lab, lw=1.2)
    ax.set_xlabel("t"); ax.set_ylabel("||J alpha - g|| / ||g||")
    ax.set_yscale("log")
    ax.set_title("J-form residual vs time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (top-right) ||alpha||
    ax = axs[0, 1]
    for log, lab, c in zip(logs, labels, colors):
        ax.plot(log["t"], log["alpha_norm"], "-", color=c, label=lab, lw=1.2)
    ax.set_xlabel("t"); ax.set_ylabel("||alpha||")
    ax.set_yscale("log")
    ax.set_title("Galerkin update norm vs time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (bottom-left) cumulative wall time
    ax = axs[1, 0]
    for log, lab, c in zip(logs, labels, colors):
        cum = np.cumsum(log["wall_s"])
        ax.plot(log["t"], cum, "-", color=c, label=lab, lw=1.2)
    ax.set_xlabel("t (sim time)"); ax.set_ylabel("cumulative wall s")
    ax.set_title("Wall time vs simulated time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (bottom-right) |u|_max
    ax = axs[1, 1]
    for log, lab, c in zip(logs, labels, colors):
        ax.plot(log["t"], log["u_max"], "-", color=c, label=lab, lw=1.2)
    ax.axhline(1.0, color="gray", ls=":", lw=0.6, label="|u|=1")
    ax.set_xlabel("t"); ax.set_ylabel("|u|_max")
    ax.set_yscale("log")
    ax.set_title("Solution magnitude vs time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Summary printout.
    print()
    print(f"{'run':30s} {'final t':>8s} {'final |u|':>10s}"
          f" {'final alpha':>12s} {'final resid':>12s}"
          f" {'mean wall s':>12s} {'total wall s':>12s}")
    for log, lab in zip(logs, labels):
        n = len(log["t"])
        print(f"{lab:30s} {log['t'][-1]:8.3f} {log['u_max'][-1]:10.3e}"
              f" {log['alpha_norm'][-1]:12.3e} {log['resid'][-1]:12.3e}"
              f" {np.mean(log['wall_s']):12.2f}"
              f" {np.sum(log['wall_s']):12.1f}  ({n} steps)")
    print()

    fig.suptitle("DTB vs NGM -- per-step diagnostics", fontsize=12)
    fig.tight_layout()
    out = args.out or os.path.join(runs[0], "compare_runs.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
