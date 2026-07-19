# Verified Five-Player Depth Comparison

These artifacts were generated from two controlled Neural--DTB runs on the
five-player Cournot game. Both runs used the same 2,000 uniform initial
samples, seed 0, width 32, tangent basis `m=128`, `h=0.005`, 200 steps,
`svd_rtol=1e-3`, and noise amplitude `0.1`. Only the number of hidden
`Linear+tanh` blocks changed.

| Depth | Final residual | Final rank | Median distance to a known equilibrium | Within 0.15 |
|---:|---:|---:|---:|---:|
| 2 | 0.9187 | 43 | 0.2512 | 12.7% |
| 4 | 0.9279 | 54 | 0.2515 | 13.0% |

All seven equilibria shown as gold `X` markers are unstable. Consequently,
distance to them is a descriptive metric, not an accuracy objective. The two
depths produce nearly the same final distribution under this controlled setup;
depth 4 retains more tangent singular directions but does not reduce the final
projection residual.

Files:

- `depth_comparison.png`: shared-axis final clouds, residual curves, and table.
- `depth_metrics.csv`: exact comparison values.
- `depth2/` and `depth4/`: configurations, raw histories, six time slices,
  diagnostics, and final pairwise-coordinate plots.
- `depth2/sde_baseline_snapshots.png`: direct Euler--Maruyama comparison; the
  baseline is independent of neural-network depth and is therefore not repeated.

The six-panel plots project five dimensions onto `x1` versus
`mean(x2,...,x5)`. The pairwise files show every coordinate pair. Equilibria
can overlap after projection even though all seven points are represented.
