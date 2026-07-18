# Verified Three-Player Output

`three_player_n256/` is the checked smoke-scale computation produced with:

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
