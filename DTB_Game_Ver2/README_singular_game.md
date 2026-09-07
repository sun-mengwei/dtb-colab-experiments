# Game Singularity example

Open [singular_game_2d_parameter_evolving_dtb.ipynb](singular_game_2d_parameter_evolving_dtb.ipynb).
It contains the mathematical derivation, configuration, visible DTB and direct-Euler
loops, detailed comments, exact-reference diagnostics, plots, and implementation notes.
The saved notebook includes outputs from its short default experiment.

## Run

Start Jupyter from this directory or the repository root, then run all cells.
Alternatively, upload the notebook to Google Colab and run all cells. In a fresh
Colab runtime, setup clones the existing `codex/game-dynamics-dtb` branch to obtain
the shared map utilities. It does not need this new notebook to have been pushed.
An existing local checkout is used without checking out branches or pulling changes.

Dependencies: Python 3.10+, PyTorch 2.x with `torch.func`, NumPy, pandas, Matplotlib,
and IPython. These are included in a standard Colab runtime. The default uses CPU
and float64; a compatible CUDA device can be selected in the setup cell.

The short configuration runs both fixed and gap-aware DTB to `t=0.45` with
512 training and 2048 fixed test particles. Regularized companion experiments use
epsilon 0.2, 0.1, 0.05 and end at `t=0.49`.

- `RUN_FULL_STUDY=True` runs three seeds, horizons 0.45/0.48/0.49, tangent
  dimensions 32/64/128, training counts 256/512/1024, and a DTB step-halving check.
  These are one-control-at-a-time comparisons, not a full Cartesian product.
- `RUN_REGULARIZED=False` skips the companion systems for a shorter first run.
- `SAVE_RUN=True` exports figures, initial clouds, parameters, selected coordinates,
  snapshots with their own physical times, diagnostics, accepted steps, and status.

## What is computed

The benchmark is the own-payoff gradient game with drift `(-2/r, -1/r)`, where
`r=x1-x2`. Its exact signed gap is `sign(r0)*sqrt(r0**2-2*t)` and the invariant is
`2*x2-x1`. The unregularized experiment never integrates to or beyond `t=0.5`.
The diagonal is not an equilibrium, and no denominator is silently clipped.

The map is the repository's identity-initialized residual tanh MLP,
`2 -> 16 -> 16 -> 2`, with 354 trainable parameters. A fixed random subset of
coordinates defines the selected tangent dimension. The Jacobian is recomputed
at each parameter state. A particle-normalized SVD ridge solve uses `eta=1e-6`.

The official state is **the updated neural map evaluated on the original labels**.
The first-order tangent update is computed separately to measure the local
nonlinear map defect. That defect is not a best-fit representation error.
Projection residuals, exact state/velocity errors, direct Euler, step halving,
and dimension changes provide complementary diagnostics; they do not yield an
additive error decomposition.

Both training and test states are monitored for crossing. Test samples never
enter the coefficient-fitting objective, but their minimum gap can influence
the adaptive step. This finite-cloud monitor does not certify the map on the
entire distribution support. The same fixed test cloud is used for all seeds,
so seed standard deviations are conditional on that cloud.

## Validation

From this directory:

```bash
python -m unittest test_singular_game -v
```

The tests load definitions directly from the notebook. They check payoff gradients,
exact invariants, flow derivatives, the normalized ridge optimality equation,
Jacobian/JVP agreement, selected-coordinate updates, the actual map state history,
matched adaptive/Euler times, collision rejection, explicit stopping, failed-SVD
bookkeeping, and regularized references.

See [singular_game_validation/README.md](singular_game_validation/README.md) for
the measured CPU results, three-seed summaries, failures, and saved comparison figures.
Shared modules and existing notebooks are unchanged by this example.
