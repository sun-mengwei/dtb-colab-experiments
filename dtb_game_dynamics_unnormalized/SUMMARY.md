# Technical Summary of the Revised Replication

## Goal

The revised project targets the supplied Figures 4.2 and 4.3 rather than a
generic game example. It runs the unnormalized Neural–DTB scheme for the
two-player non-potential Cournot game already defined in the original
repository and generates point-cloud panels at the same six times.

## Model reconstructed from the supplied material

For `b=1` and `mu=2`,

```text
b1(x) = -2 x1 + 4 x2 - 4 x2^2
b2(x) = -2 x2 + 4 x1 - 4 x1^2.
```

Both `(0,0)` and `(0.5,0.5)` are Nash equilibria. Linearization gives one
unstable direction at `(0,0)`, while `(0.5,0.5)` is stable. This matches the
target point clouds.

The two initial laws are uniform on `[0,1]^2` and Gaussian with mean
`(0.5,0.5)` and covariance `0.03 I`. The caption reports noise amplitudes
`sigma_1=sigma_2=0.1`. Under the SDE convention this gives the supplied
algorithm's Fokker–Planck matrix `D=0.01 I`.

## Neural–DTB calculation

At every step the code selects `m` scalar parameters of a smooth
`f_theta:R^2->R^2`, computes the selected Jacobians, and solves

```text
min_alpha sum_i ||J_i alpha - (b(x_i)-0.5 D q_i)||^2
```

with an explicit relative truncated SVD. The stack is unnormalized. Nested
`torch.func.jacrev` evaluates the spatial terms needed for

```text
q_next = q - h[(grad u)^T q + grad(div u)].
```

The fixed labels and evolved particles store the sampled pushforward map
values `x_i^k=X_k(z_i)`.

## Replication additions

- Exact uniform-box and Gaussian log-density/score initialization.
- Noise-amplitude-to-diffusion conversion `D=sigma sigma^T`.
- Snapshot capture at `t=0,0.2,0.4,0.6,0.8,1`.
- Figure-style 2-by-3 point-cloud panels with both equilibria marked red.
- Direct Euler–Maruyama simulation of the same SDE as an independent baseline.
- A single script that runs both Figure 4.2 and Figure 4.3 initializations.
- Reference folder containing the original TeX algorithm and supplied image.
- GitHub-first Colab tutorial and notebook.

## Reuse and isolation

The new package adapts the original DTB flat-parameter representation,
selected-coordinate Jacobian construction, and truncated-SVD pattern. It also
reuses the original repository's Cournot drift. Existing DTB files remain
unchanged; all additions live under `dtb_game_dynamics_unnormalized/`.

## Validation

Nine tests cover:

- unnormalized stacking;
- tangent-span least-squares recovery;
- exact `u`, `grad u`, `div u`, and `grad(div u)` on a closed-form field;
- Gaussian density and score formulas;
- uniform-box interior score;
- both reported Cournot equilibria;
- conversion of `sigma=0.1` to `D_ii=0.01`;
- the six requested snapshot indices;
- invariance under zero drift and zero diffusion.

An end-to-end run of both initializations produced all six finite DTB and SDE
snapshot panels. The point clouds contract toward `(0.5,0.5)` qualitatively as
in the target. This is a software smoke validation, not yet a convergence or
error analysis.

## Reproducibility caveat

The screenshot does not state the exact network, particle count, selected
basis size, time step, or SVD tolerance. The project therefore distinguishes
between a fast Colab preset and a denser `--paper-scale` preset and records all
actual values in each run's `config.json`. Quantitative replication should be
updated if the missing source parameters become available.
