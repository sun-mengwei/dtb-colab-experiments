# Verified 3D MLP versus MMNN Pilot

This is a controlled, single-seed Neural--DTB comparison with 500 uniform
initial samples on `[0,1]^3`. Both models use tanh, width 32, depth 2, the same
initial particles, seed 0, `m=128`, `h=0.005`, 200 steps, diffusion `D=0.01 I`,
and `svd_rtol=1e-3`.

The ordinary MLP has 1,283 tangent-eligible scalar parameters. The MMNN has
rank 8 and 363 tangent-eligible parameters: only its component combination
weights `A` and component biases `c`. Its random feature weights `W` and biases
`b` are registered as frozen parameters and excluded from the tangent pool.

| Metric | MLP | MMNN |
|---|---:|---:|
| Mean projection residual over time | 0.2863 | 0.4086 |
| Final projection residual | 0.6390 | 0.6086 |
| Mean retained SVD rank | 27.75 | 23.17 |
| Final retained SVD rank | 20 | 17 |
| Median distance to nearest stable equilibrium | 0.1015 | 0.1353 |
| Fraction within radius 0.15 | 68.6% | 57.8% |
| Initial tanh saturation fraction | 0% | 0% |

The MLP gives the lower time-averaged residual and greater final concentration
near the known stable equilibria. The MMNN ends with a slightly lower residual
at the final step and uses 72% fewer eligible parameters, but its retained
tangent rank is lower. The paired final particles differ by a mean absolute
coordinate value of 0.03375.

This is a small pilot rather than a general architecture ranking. Multiple
random seeds and rank/width sweeps are needed before drawing a robust
conclusion.

Files:

- `architecture_comparison.png`: shared-camera final clouds and residual curve.
- `architecture_metrics.csv`: exact metrics.
- `mlp/` and `mmnn/`: configurations, raw histories, six time slices, and
  numerical diagnostics.
