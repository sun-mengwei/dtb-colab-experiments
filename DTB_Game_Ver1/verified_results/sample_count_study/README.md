# Three-player DTB sample-count study

This study compares `N=512`, `N=1024`, and `N=5000` particles at `t=1`.
Every run uses the same three-player Cournot game and numerical configuration:

```text
h=0.005
K=200
m=128
svd_rtol=1e-3
sigma_1=sigma_2=sigma_3=0.1
D=0.01 I
initial law=Uniform([0,1]^3)
seed=0
```

The conservative SVD cutoff is necessary for the large empirical system. With
the original `1e-4` cutoff, rare particle excursions eventually made the
5,000-particle matrix non-finite near `t=1`. The same `1e-3` cutoff is used at
all three sample counts so the comparison is controlled.

Run each case from the project root:

```bash
python replicate_three_player_game.py --particles 512  --svd-rtol 1e-3 --skip-sde-baseline --output-dir outputs/sample_study_stable/n512
python replicate_three_player_game.py --particles 1024 --svd-rtol 1e-3 --skip-sde-baseline --output-dir outputs/sample_study_stable/n1024
python replicate_three_player_game.py --particles 5000 --svd-rtol 1e-3 --skip-sde-baseline --output-dir outputs/sample_study_stable/n5000
python compare_sample_counts.py
```

Files:

- `sample_count_comparison.png`: shared-camera final clouds, residual histories,
  and concentration statistics.
- `sample_count_metrics.csv`: exact numeric comparison.
- `n*/history.npz`: full particle snapshots and diagnostics.
- `n*/dtb_snapshots.png`: six time panels for an individual sample count.
- `n*/diagnostics.png`: mean and projection diagnostics.
- `n*/config.json`: complete recorded configuration.

Increasing `N` makes the three branches visually denser and gives more stable
empirical proportions, but it does not by itself create three isolated modes.
The median distance to a stable equilibrium remains about `0.10`, and roughly
`69%--70%` of particles lie within radius `0.15` for all three runs. Resolving
sharper modes requires more tangent capacity or a better-conditioned projection,
not only more particles.
