# Reference Materials and Replication Target

This folder keeps the mathematical algorithm and the supplied target figure
beside the implementation so they can be read together.

## Files

- `dtb_game_dynamics_unnormalized_dimensions.tex`: the complete unnormalized
  discrete Neural--DTB algorithm supplied for this project.
- `target_figures_4_2_4_3.png`: the supplied screenshot of the desired
  uniform- and Gaussian-initialized game results.
- `three_player_game_definition.png`: the supplied three-player payoff,
  cost, and known-equilibria pages.
- `target_figures_4_5_4_6.png`: the supplied best-response and pushforward-map
  point clouds for the three-player game.
- `printed_payoff_equations.png`: the later close-up of Equations 4.61--4.62.
- `printed_payoff_derivative_check.png`: the supplied derivative check showing
  the consequence of reading the printed total literally.
- `five_player_game_definition.png`: the supplied Section 4.7.4 screenshot
  listing the five-player equilibria and their stability conclusion.

## Target experiment read from the figure

- Two-dimensional non-potential Cournot game.
- Drift parameters `b=1` and `mu=2`, using the drift already present in the
  original repository's `run_game_dtb.py`.
- Known Nash equilibria `(0,0)` and `(0.5,0.5)`; the latter is stable.
- Time interval `0 <= t <= 1` with panels at `0,0.2,0.4,0.6,0.8,1.0`.
- Figure 4.2: initial samples uniform on `[0,1]^2`.
- Figure 4.3: initial samples Gaussian with mean `(0.5,0.5)` and covariance
  `0.03 I`.
- Reported noise coefficients `sigma_1=sigma_2=0.1`.

The implementation treats `0.1` as the SDE noise amplitude in
`dX=b(X)dt+sigma dW`. Therefore the Fokker--Planck matrix in the supplied
algorithm is `D=sigma sigma^T=0.01 I`. The CLI exposes `--diffusion-entry` if
the source convention instead intends `D=0.1 I` directly.

The exact network width, tangent-basis size, particle count, SVD tolerance,
and Euler step are not visible in the supplied screenshot. The replication
script therefore offers a documented Colab-scale configuration and a denser
`--paper-scale` configuration. These are implementation choices, not claimed
values from the source document.

## Three-player target

The printed Equation 4.62 uses the total `S=x_1+x_2+x_3` in its two nonlinear
cost terms. Read literally, it gives

```text
d Pi_i / d x_i = 2 b mu [S(1-S) + x_i(1-2S)],
```

which is not zero at the four reported nonzero equilibria. We therefore use the
equilibrium-consistent interpretation confirmed for this replication: those
two aggregate terms contain the opponents' total
`r_i=sum_{j != i} x_j`. The resulting payoff and own-action gradient are

```text
Pi_i = -d - b x_i^2 + 2 b mu x_i r_i(1-r_i),
d Pi_i / d x_i = -2 b x_i + 2 b mu r_i - 2 b mu r_i^2.
```

Thus the implemented drift is

```text
b_i(x) = -2 b x_i + 2 b mu r_i - 2 b mu r_i^2,
b=1, mu=2.
```

This is also `2b` times `best_response_i-x_i`, with
`best_response_i=mu*r_i*(1-r_i)`.

The known equilibria are

```text
(0,0,0)
(3/8,3/8,3/8)
(1/2,1/2,0)
(1/2,0,1/2)
(0,1/2,1/2)
```

The source states that the origin is unstable and the other known equilibria
are asymptotically stable. Figures 4.5 and 4.6 use uniform initial samples on
`[0,1]^3`, `sigma_1=sigma_2=sigma_3=0.1`, and the same six snapshot times from
`t=0` through `t=1`.

## Five-player target

The supplied Section 4.7.4 screenshot lists seven identified equilibria:

```text
(0,0,0,0,0)
(7/32,7/32,7/32,7/32,7/32)
(0,5/18,5/18,5/18,5/18) and its five coordinate placements
```

The last line means the five distinct permutations with one zero. The source
states that none of these seven identified equilibria is stable.

For a one-zero point the opponents' total seen by the zero player is `10/9`,
so the raw best response `mu*r_i*(1-r_i)` is negative. The reported point is a
Nash equilibrium only after enforcing the Cournot quantity constraint
`x_i >= 0`. Accordingly, the five-player implementation uses
`max(mu*r_i*(1-r_i),0)` before forming the best-response drift.
