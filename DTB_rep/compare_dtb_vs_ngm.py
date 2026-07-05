"""
Side-by-side: DTB (Algorithm 3 + Prop. 2.4) vs Neural Galerkin (forward
update, no reset) on 5-D Allen-Cahn.

Inputs
  --dtb_dir   snapshots/run_<dtb tag>
  --ngm_dir   snapshots/run_<ngm tag>

Outputs (in --ngm_dir):
  dtb_vs_ngm_traces.png   |u|_max, ||alpha||, sigma_min, residual vs t
  dtb_vs_ngm_panels.png   five-hyperplane snapshots at three
                          comparison times (start, midway, end / failure)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from network import PeriodicMMNN
from dtb import write_flat_into_model, device
from visualize import HYPERPLANES, load_model_from_meta


def list_snapshots(run_dir):
    out = {}
    for fn in sorted(os.listdir(run_dir)):
        if fn.startswith("theta_t") and fn.endswith(".pt"):
            try:
                idx = int(fn[len("theta_t"):-len(".pt")])
            except ValueError:
                continue
            out[idx / 100.0] = os.path.join(run_dir, fn)
    return out


def umax_trajectory(run_dir, dev, n_eval=4000, seed=0):
    """Evaluate |u|_max on a fresh random batch for every snapshot."""
    meta_path = os.path.join(run_dir, "meta.pt")
    model, structure = load_model_from_meta(meta_path, dev)
    snaps = list_snapshots(run_dir)
    torch.manual_seed(seed)
    z = (torch.rand(n_eval, 5, device=dev) * 2.0 - 1.0)
    ts = sorted(snaps.keys())
    umax = []
    urms = []
    for t in ts:
        flat = torch.load(snaps[t], weights_only=False).to(dev)
        write_flat_into_model(model, flat, structure)
        with torch.no_grad():
            u = model(z)
        umax.append(float(u.abs().max()))
        urms.append(float((u ** 2).mean().sqrt()))
    return np.array(ts), np.array(umax), np.array(urms), model, structure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtb_dir", type=str, default="snapshots/run_paper2")
    ap.add_argument("--ngm_dir", type=str, default="snapshots/run_ngm")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = args.ngm_dir

    dev = device()

    # ---- DTB snapshots -> u_max trajectory ----
    t_dtb, umax_dtb, urms_dtb, dtb_model, dtb_struct = \
        umax_trajectory(args.dtb_dir, dev)
    print(f"DTB: {len(t_dtb)} snapshots,  t in [{t_dtb[0]:.2f},"
          f" {t_dtb[-1]:.2f}],  |u|_max in"
          f" [{umax_dtb.min():.3f}, {umax_dtb.max():.3f}]")

    # ---- NGM log ----
    ngm_log_path = os.path.join(args.ngm_dir, "log.pt")
    if not os.path.exists(ngm_log_path):
        raise SystemExit(f"missing {ngm_log_path}; run run_ngm.py first")
    log = torch.load(ngm_log_path, weights_only=False)
    t_ngm = np.array(log["t"])
    umax_ngm = np.array(log["u_max"])
    urms_ngm = np.array(log["u_rms"])
    alpha_ngm = np.array(log["alpha_norm"])
    sigmin_ngm = np.array(log["sigma_min"])
    sigmax_ngm = np.array(log["sigma_max"])
    resid_ngm = np.array(log["resid"])
    blew_up = "blowup" in log["status"]
    if blew_up:
        t_fail = float(t_ngm[-1])
        print(f"NGM blew up at t={t_fail:.3f}")
    else:
        t_fail = None
        print(f"NGM finished without blow-up")

    # ----------- traces panel -----------
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axs[0, 0]
    ax.plot(t_dtb, umax_dtb, "o-", color="C0", label="DTB  |u|_max")
    ax.plot(t_dtb, urms_dtb, "o:", color="C0", alpha=0.6,
            label="DTB  |u|_rms")
    ax.plot(t_ngm, umax_ngm, "-", color="C3", label="NGM  |u|_max")
    ax.plot(t_ngm, urms_ngm, ":", color="C3", alpha=0.6,
            label="NGM  |u|_rms")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":",
               label="AC attractor |u|=1")
    if t_fail is not None:
        ax.axvline(t_fail, color="C3", lw=0.6, ls="--",
                   label=f"NGM blow-up t={t_fail:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("t"); ax.set_ylabel("|u|")
    ax.set_title("Solution magnitude vs time")
    ax.legend(fontsize=8, loc="best")

    ax = axs[0, 1]
    ax.plot(t_ngm, alpha_ngm, "-", color="C3", label="NGM  ||alpha||")
    ax.set_yscale("log")
    ax.set_xlabel("t"); ax.set_ylabel("||alpha||")
    ax.set_title("Parameter-update magnitude (NGM)")

    ax = axs[1, 0]
    ax.plot(t_ngm, sigmin_ngm, "-", color="C3", label="sigma_min")
    ax.plot(t_ngm, sigmax_ngm, ":", color="C3", label="sigma_max")
    ax.set_yscale("log")
    ax.set_xlabel("t"); ax.set_ylabel("singular values of J_sel")
    ax.set_title("J_sel conditioning (NGM)")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(t_ngm, resid_ngm, "-", color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("t"); ax.set_ylabel("||J alpha - g|| / ||g||")
    ax.set_title("DTB / J-form residual (NGM)")

    fig.suptitle("DTB vs NGM on 5-D Allen-Cahn (RTX 5080, fp32)")
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, "dtb_vs_ngm_traces.png")
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    print(f"saved {out1}")

    # ----------- snapshot panel comparison -----------
    # Pick three comparison times: t=0, mid, end.
    if t_fail is not None:
        cmp_times = [0.0, max(0.1, round(t_fail / 2, 1)),
                     round(t_fail, 2)]
    else:
        cmp_times = [0.0, 1.0, 2.0]

    def nearest(t, snaps):
        ts = np.array(sorted(snaps.keys()))
        i = int(np.argmin(np.abs(ts - t)))
        return ts[i]

    dtb_snaps = list_snapshots(args.dtb_dir)
    ngm_snaps = list_snapshots(args.ngm_dir)
    dtb_model2, dtb_struct2 = load_model_from_meta(
        os.path.join(args.dtb_dir, "meta.pt"), dev)
    ngm_model, ngm_struct = load_model_from_meta(
        os.path.join(args.ngm_dir, "meta.pt"), dev)

    fig, axes = plt.subplots(
        nrows=2 * len(HYPERPLANES), ncols=len(cmp_times),
        figsize=(3.0 * len(cmp_times), 2.5 * 2 * len(HYPERPLANES)),
        squeeze=False,
    )
    vmin, vmax = -1.5, 1.5  # widened in case NGM overshoots
    n = 100

    for i, (label, grid_fn) in enumerate(HYPERPLANES):
        z_np, AA, BB, _ = grid_fn(n)
        z = torch.from_numpy(z_np).to(dev)
        for j, t_target in enumerate(cmp_times):
            # DTB
            t_dtb_n = nearest(t_target, dtb_snaps)
            flat = torch.load(dtb_snaps[t_dtb_n], weights_only=False).to(dev)
            write_flat_into_model(dtb_model2, flat, dtb_struct2)
            with torch.no_grad():
                u = dtb_model2(z).cpu().numpy().reshape(n, n)
            ax = axes[2 * i][j]
            ax.pcolormesh(AA, BB, np.clip(u, vmin, vmax), cmap="viridis",
                          vmin=vmin, vmax=vmax, shading="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"DTB {label}", fontsize=10)
            if i == 0:
                ax.set_title(f"t ~ {t_target:.2f}", fontsize=10)

            # NGM
            t_ngm_n = nearest(t_target, ngm_snaps)
            flat = torch.load(ngm_snaps[t_ngm_n], weights_only=False).to(dev)
            if not torch.isnan(flat).any():
                write_flat_into_model(ngm_model, flat, ngm_struct)
                with torch.no_grad():
                    u = ngm_model(z).cpu().numpy().reshape(n, n)
                im = axes[2 * i + 1][j].pcolormesh(
                    AA, BB, np.clip(u, vmin, vmax), cmap="viridis",
                    vmin=vmin, vmax=vmax, shading="auto"
                )
                axes[2 * i + 1][j].text(
                    0.02, 0.95,
                    f"|u|_max={float(np.abs(u).max()):.2f}",
                    transform=axes[2 * i + 1][j].transAxes,
                    fontsize=7, color="white",
                    verticalalignment="top",
                )
            else:
                axes[2 * i + 1][j].text(0.5, 0.5, "NaN",
                                        ha="center", va="center",
                                        fontsize=14, color="red",
                                        transform=axes[2 * i + 1][j].transAxes)
            axes[2 * i + 1][j].set_xticks([])
            axes[2 * i + 1][j].set_yticks([])
            if j == 0:
                axes[2 * i + 1][j].set_ylabel(f"NGM {label}", fontsize=10)

    fig.suptitle("DTB (top of each pair) vs NGM (bottom) "
                 "-- five hyperplanes, three comparison times",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.96, 0.97])
    out2 = os.path.join(args.out_dir, "dtb_vs_ngm_panels.png")
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
