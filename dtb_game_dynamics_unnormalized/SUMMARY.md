# Technical Summary of the Revised Replication

## Goal

The revised project targets the supplied two-player Figures 4.2/4.3,
three-player Figures 4.5/4.6, and five-player Section 4.7.4. It runs the
unnormalized Neural–DTB scheme and generates point-cloud panels at the same six
times.

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

For the three-player case, define `r_i=sum_{j != i}x_j`. The
equilibrium-consistent payoff and its own-action gradient are

```text
Pi_i = -d - b x_i^2 + 2 b mu x_i r_i(1-r_i),
b_i(x) = -2 b x_i + 2 b mu r_i - 2 b mu r_i^2,
```

again with `b=1` and `mu=2`. It vanishes at the five reported equilibria:
the origin, `(3/8,3/8,3/8)`, and the three permutations of `(1/2,1/2,0)`.
The printed cost uses the full total `S` where this interpretation uses `r_i`;
its literal derivative does not vanish at the reported nonzero equilibria, so
the discrepancy is documented and tested rather than silently ignored.
The source uses uniform initial samples on `[0,1]^3` and noise amplitude `0.1`
for every coordinate.

For five players, the source lists the origin, `(7/32)^5`, and the five
permutations of `(0,5/18,5/18,5/18,5/18)`. Quantities are nonnegative, so the
implemented best response is

```text
BR_i(x) = max(mu*r_i*(1-r_i), 0),
b_i(x) = 2b*(BR_i(x)-x_i).
```

The projection is mathematically consequential: without it the zero component
of a one-zero point would have a negative drift, contradicting the source's
equilibrium list. Automatic-differentiation tests confirm all seven points and
at least one unstable eigen-direction at each.

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
- Figure-style 2-by-3 point-cloud panels with stable equilibria as red circles
  and unstable equilibria as gold `X` markers.
- Direct Euler–Maruyama simulation of the same SDE as an independent baseline.
- A single script that runs both Figure 4.2 and Figure 4.3 initializations.
- A separate stable 3D driver and 3D six-panel renderer for Figures 4.5/4.6.
- A five-player driver, symmetry projection, full pairwise-coordinate plot,
  and matched depth-2/depth-4 comparison.
- A frozen-feature MMNN whose `W,b` parameters stay outside the tangent pool,
  plus a matched 500-sample comparison against the ordinary MLP.
- Reference folder containing the original TeX algorithm and supplied image.
- GitHub-first Colab tutorial and notebook.
- Source-matching periodic resetting: fixed block parameters and tangent
  coordinates, accumulated coefficients, a precomputed target on fresh
  reference-law samples, and Adam with cosine learning-rate decay.
- Refit-event, before/after RMSE, absolute projection-error, target-norm, and
  tangent-block-age diagnostics.

## Reuse and isolation

The new package adapts the original DTB flat-parameter representation,
selected-coordinate Jacobian construction, and truncated-SVD pattern. It also
reuses the original repository's Cournot drift. Existing DTB files remain
unchanged; all additions live under `dtb_game_dynamics_unnormalized/`.

## Validation

Twenty-four tests cover:

- unnormalized stacking;
- tangent-span least-squares recovery;
- exact `u`, `grad u`, `div u`, and `grad(div u)` on a closed-form field;
- Gaussian density and score formulas;
- uniform-box interior score;
- both reported Cournot equilibria;
- all five reported three-player equilibria and the origin's unstable
  direction;
- a finite end-to-end three-dimensional Neural–DTB step;
- all seven five-player constrained equilibria and their unstable directions;
- conversion of `sigma=0.1` to `D_ii=0.01`;
- the six requested snapshot indices;
- invariance under zero drift and zero diffusion.
- MMNN trainable/frozen parameter separation, output shape, and tanh
  saturation diagnostics.
- periodic block resetting, fresh reference train/test sampling, exact reset
  timing, and preservation of frozen MMNN feature parameters.

End-to-end 2D, 3D, and 5D runs produced finite six-time DTB panels; the 2D/3D
presets and the five-player depth-2 run also produced direct SDE panels.
The 2D clouds contract toward `(0.5,0.5)`, while the 3D uniform cube develops
the target's inward triangular geometry. This is software smoke validation,
not yet a convergence or error analysis.

The controlled five-player experiment used `N=2000`, seed 0, width 32,
`m=128`, `h=0.005`, 200 steps, and `svd_rtol=1e-3` for both depths. At `t=1`,
depth 2 and depth 4 had final residuals `0.919` and `0.928`; their median
distances to the nearest known equilibrium were `0.2512` and `0.2515`, and
`12.7%` and `13.0%` were within radius `0.15`. The near equality indicates
that additional depth did not materially change this run. Since the known
equilibria are all unstable, the lack of concentration at them is expected.

In the single-seed three-player `N=500` pilot, the ordinary MLP had 1,283
tangent-eligible parameters and the rank-8 MMNN had 363; each run selected
`m=128`. The MLP gave a lower mean residual over time (`0.286` versus `0.409`),
a smaller median final distance to the stable equilibria (`0.101` versus
`0.135`), and more samples within radius `0.15` (`68.6%` versus `57.8%`). The
MMNN had a slightly lower residual at the final step (`0.609` versus `0.639`).
Neither initialization contained units with `tanh'(z)<0.05`, so this comparison
did not exhibit tanh saturation. Multiple seeds and MMNN ranks are still needed.

In the simple two-player `N=1000` MLP/NODE pilot, both tangent generators used
354 parameters, width 16, depth 2, `m=64`, and identical uniform initial
particles. The NODE used four fixed RK4 steps over its internal depth interval.
The final clouds were nearly indistinguishable: both placed `93.7%` of samples
within radius `0.15` of `(0.5,0.5)`, with median distances `0.05262` (MLP) and
`0.05251` (NODE). Mean projection residuals were `0.01195` and `0.01256`.
The straightforward NODE implementation took `48.7 s` versus `5.4 s` for the
MLP because every tangent and score derivative differentiates through all RK4
stages. This one-seed result shows that the NODE basis path works, but provides
no accuracy advantage in this simple case.

## Reproducibility caveat

The screenshot does not state the exact network, particle count, selected
basis size, time step, or SVD tolerance. The project therefore distinguishes
between a fast Colab preset and a denser `--paper-scale` preset and records all
actual values in each run's `config.json`. Quantitative replication should be
updated if the missing source parameters become available.

The 3D score equation is more numerically sensitive than the 2D case. A
coarse `h=0.02` trial diverged near `t=1`; the published 3D preset uses
`h=0.005`, 200 steps, `m=128`, and `svd_rtol=1e-4`.
