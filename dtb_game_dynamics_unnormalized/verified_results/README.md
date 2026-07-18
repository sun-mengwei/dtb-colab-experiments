# Verified Three-Player Output

`three_player_n256/` is the original checked smoke-scale computation produced with:

```text
N=256 particles
dimension d=3
h=0.005
K=200 steps
m=128 selected tangent coordinates
svd_rtol=1e-4
sigma_1=sigma_2=sigma_3=0.1
D=0.01 I
initial law=Uniform([0,1]^3)
seed=0
```

Files:

- `dtb_snapshots.png`: Neural--DTB panels at `t=0,0.2,...,1`.
- `sde_baseline_snapshots.png`: direct Euler--Maruyama comparison.
- `diagnostics.png`: means, projection residual, and coefficient norm.
- `history.npz`: raw snapshots, score, log density, and diagnostics.
- `config.json`: complete recorded configuration.

The run stayed finite through `t=1`; its final particle mean was approximately
`(0.3197, 0.3162, 0.3176)` and its final relative tangent-projection residual
was approximately `0.1305`. Use `--paper-scale` for denser visualization and
perform refinement studies before quantitative interpretation.

## Fresh 512-particle computation

`three_player_n512/` contains the clean algorithm-only computation requested
after the initial verification. It uses the same configuration, except that
`N=512`, and omits the optional Euler--Maruyama comparison. The Neural--DTB
arrays are finite through `t=1`; the final mean is approximately
`(0.3144, 0.3229, 0.3178)`. The files are:

- `dtb_snapshots.png`: the six 3D Neural--DTB panels.
- `diagnostics.png`: means, tangent-projection residual, and coefficient norm.
- `history.npz`: particles, score, log density, and stepwise diagnostics.
- `config.json`: the exact reproducibility configuration.

The larger particle count was intentionally run with the same `m=128` tangent
basis so the result remains Colab-friendly. For a convergence study, increase
the particle and tangent-basis sizes together using `--paper-scale`.
