# Singular drift: DTB and direct Euler

Open [singular_game_2d.ipynb](singular_game_2d.ipynb)
and run all cells from the repository root or `DTB_Game_Ver2`, or open in Colab.
The notebook uses the renamed GitHub path `singular_game_2d.ipynb`.

The first code cell contains:

```python
RUN_DTB = True
RUN_EULER = True
```

Both experiments run by default. Set `RUN_DTB=False` for Euler alone, or
`RUN_EULER=False` for DTB alone, and rerun from the first cell. Disabling DTB
also skips shared neural imports and any repository download. The Euler block
can run after setup and the drift cell without executing the DTB block.
Restarting setup clears old results, and plotting honors the current switches.

The notebook follows the particle/map update in
[cournot_3d_nonpotential_stochastic_mlp_dtb.ipynb](cournot_3d_nonpotential_stochastic_mlp_dtb.ipynb),
using only the deterministic singular drift `b(x)=(-2/r,-1/r)`, `r=x1-x2`:

```text
J_k(z) = partial_theta_selected f_theta0(X_k(z))
alpha_k = truncated-SVD least-squares solve of J_k * alpha ~= b(X_k)
X_{k+1}(z) = X_k(z) + h * J_k(z) * alpha_k.
```

**The network parameters theta0 remain fixed.** The Jacobian is recomputed at
current particle positions each step. The map starts at `X_0(z)=z` and evolves
by accumulating projected velocity. The independent snapshot particles use the
same coefficients evaluated through the tangent basis at their own positions.
Original particle labels are retained for coloring.

The notebook imports the existing `ResidualMLPMap` and
`game_dtb_basis_matrices` from `run_game_dtb.py`, `flat_params` and
`jform_solve` from `dtb.py`, `count_trainable` from `network.py`, and
`plot_tangent_diagnostics` from `utility.py`. The Cournot score-transport
functions are unnecessary for this zero-diffusion example. The short custom
snapshot cell retains the requested density background and fixed point colors.

Defaults: 1024 projection particles, 2048 independent snapshot particles,
`2 -> 16 -> 16 -> 2` tanh tangent network, 32 selected parameters,
SVD relative cutoff `1e-3`, step `0.001`, and requested final time `0.6`. As in Cournot,
the output layer uses ordinary random initialization; the network generates
tangents and does not encode the initial particle state. Initial states are
`(c+2*r,c+r)` with `c` uniform on `[-1,1]` and `r` uniform on `[1,2]`.
The singular denominator is used directly, and steps approaching the diagonal
are stopped before invalid states enter the history.

## Euler reference and aligned plots

The separate Euler block computes `X_next = X + h*dynamic_drift(X)` directly.
It constructs no network and uses no DTB states or coefficients. Euler and
DTB's snapshot particles start from the same seeded cloud, use the same `H`
and `T`, and request the same six snapshot times. Each run handles stopping
independently. With the current requested horizon `T=0.6`, a run may stop
earlier near the singular diagonal; later panels remain unavailable.

The displayed results are:

- Aligned DTB and Euler snapshot rows, with identical physical times in each
  column and shared axes, density bins, density scale, and point-color scale.
  Each panel has a faint density heatmap and points colored by their initial
  coordinate gap `|x1(0)-x2(0)|`. Each particle keeps its color across time,
  with one shared colorbar (purple = smaller gap, yellow = larger gap).
- Relative projection error versus time for DTB, using the shared plotting function.
- The coefficient norm `||alpha(t)||_2` versus time for DTB.

A disabled method has no plot row. If a method stops early, later requested
panels are marked unavailable; its last valid state is not reused at future
times. DTB diagnostics are shown only when DTB is enabled and has accepted
steps. Euler-only runs display only the Euler snapshot row.

For this drift, `dr/dt=-1/r`: larger positive gaps close more slowly.

Dependencies: PyTorch, NumPy, and Matplotlib. On a fresh Colab runtime, setup
clones the existing branch for shared modules. Existing local checkouts are used
without changing branches or pulling. The notebook includes executed outputs for both methods.
Checks can be run with:

```bash
python -m unittest test_singular_game -v
```
