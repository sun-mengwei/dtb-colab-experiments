# Reference Materials and Replication Target

This folder keeps the mathematical algorithm and the supplied target figure
beside the implementation so they can be read together.

## Files

- `dtb_game_dynamics_unnormalized_dimensions.tex`: the complete unnormalized
  discrete Neural--DTB algorithm supplied for this project.
- `target_figures_4_2_4_3.png`: the supplied screenshot of the desired
  uniform- and Gaussian-initialized game results.

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
