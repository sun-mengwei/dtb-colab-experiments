# Technical Summary of the Revised Replication

## Goal

The revised project targets the supplied two-player Figures 4.2/4.3 and
three-player Figures 4.5/4.6. It runs the unnormalized Neural–DTB scheme and
generates point-cloud panels at the same six times.

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

For the three-player case, define `r_i=sum_{j != i}x_j`. The drift is

```text
b_i(x) = -2 b x_i + 2 b mu r_i - 2 b mu r_i^2,
```

again with `b=1` and `mu=2`. It vanishes at the five reported equilibria:
the origin, `(3/8,3/8,3/8)`, and the three permutations of `(1/2,1/2,0)`.
The source uses uniform initial samples on `[0,1]^3` and noise amplitude `0.1`
for every coordinate.

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
- A separate stable 3D driver and 3D six-panel renderer for Figures 4.5/4.6.
- Reference folder containing the original TeX algorithm and supplied image.
- GitHub-first Colab tutorial and notebook.

## Reuse and isolation

The new package adapts the original DTB flat-parameter representation,
selected-coordinate Jacobian construction, and truncated-SVD pattern. It also
reuses the original repository's Cournot drift. Existing DTB files remain
unchanged; all additions live under `dtb_game_dynamics_unnormalized/`.

## Validation

Twelve tests cover:

- unnormalized stacking;
- tangent-span least-squares recovery;
- exact `u`, `grad u`, `div u`, and `grad(div u)` on a closed-form field;
- Gaussian density and score formulas;
- uniform-box interior score;
- both reported Cournot equilibria;
- all five reported three-player equilibria and the origin's unstable
  direction;
- a finite end-to-end three-dimensional Neural–DTB step;
- conversion of `sigma=0.1` to `D_ii=0.01`;
- the six requested snapshot indices;
- invariance under zero drift and zero diffusion.

End-to-end 2D and 3D runs produced all six finite DTB and SDE snapshot panels.
The 2D clouds contract toward `(0.5,0.5)`, while the 3D uniform cube develops
the target's inward triangular geometry. This is software smoke validation,
not yet a convergence or error analysis.

## Reproducibility caveat

The screenshot does not state the exact network, particle count, selected
basis size, time step, or SVD tolerance. The project therefore distinguishes
between a fast Colab preset and a denser `--paper-scale` preset and records all
actual values in each run's `config.json`. Quantitative replication should be
updated if the missing source parameters become available.

The 3D score equation is more numerically sensitive than the 2D case. A
coarse `h=0.02` trial diverged near `t=1`; the published 3D preset uses
`h=0.005`, 200 steps, `m=128`, and `svd_rtol=1e-4`.
