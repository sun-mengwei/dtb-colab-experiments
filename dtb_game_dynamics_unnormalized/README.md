# Neural–DTB Replication of a 2D Non-Potential Game

This self-contained folder implements the supplied unnormalized Neural–DTB
algorithm and uses it to reproduce the qualitative behavior of target Figures
4.2 and 4.3: strategy samples contract toward the stable Nash equilibrium
`(0.5,0.5)` from uniform and Gaussian initial distributions.

The supplied mathematical source and target screenshot are kept beside the
code in [`references/`](references/README.md):

![Target Figures 4.2 and 4.3](references/target_figures_4_2_4_3.png)

The same engine also computes the supplied three-player Figures 4.5 and 4.6:

![Target Figures 4.5 and 4.6](references/target_figures_4_5_4_6.png)

No original Deep Tangent Bundle file is modified. The project reuses the
original flat-parameter/restricted-Jacobian pattern, truncated-SVD approach,
and Cournot drift in a separate package.

## Target experiment

The original repository's Cournot drift is

```text
dx1/dt = -2 b x1 + 2 b mu x2 - 2 b mu x2^2
dx2/dt = -2 b x2 + 2 b mu x1 - 2 b mu x1^2
```

with `b=1` and `mu=2`. Its equilibria are `(0,0)` and `(0.5,0.5)`; the second
is stable.

| Quantity | Replication value |
|---|---|
| Time interval | `0 <= t <= 1` |
| Snapshot times | `0, 0.2, 0.4, 0.6, 0.8, 1.0` |
| Figure 4.2 initial law | Uniform on `[0,1]^2` |
| Figure 4.3 initial law | `N((0.5,0.5), 0.03 I)` |
| Noise amplitude | `sigma_1=sigma_2=0.1` |
| Algorithm diffusion matrix | `D=diag(sigma_i^2)=0.01 I` |

The last line follows the SDE/Fokker–Planck convention
`dX=b(X)dt+sigma dW`. Use `--diffusion-entry 0.1` if the underlying source
instead defines the caption's `0.1` as an entry of `D` itself.

### Three-player game

Let `r_i` be the sum of the other two players' strategies. The implemented
three-player drift is

```text
b_i(x) = -2 b x_i + 2 b mu r_i - 2 b mu r_i^2,
b=1, mu=2.
```

Its five listed equilibria are the origin, `(3/8,3/8,3/8)`, and the three
permutations of `(1/2,1/2,0)`. The 3D target uses uniform samples on `[0,1]^3`,
noise amplitudes `sigma_1=sigma_2=sigma_3=0.1`, and the same six times.

## Fastest replication

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the mathematical tests:

```bash
python -m pytest
```

Run both target experiments with Colab-friendly settings:

```bash
python replicate_thesis_figures.py --device auto
```

Run the three-player case:

```bash
python replicate_three_player_game.py --device auto
```

A completed 256-particle result is included under
[`verified_results/three_player_n256/`](verified_results/README.md) so the 3D
panels and raw arrays can be inspected without rerunning Colab.

Compare the requested larger sample counts with a shared numerical setup:

```bash
python replicate_three_player_game.py --particles 512  --svd-rtol 1e-3 --skip-sde-baseline --output-dir outputs/sample_study_stable/n512
python replicate_three_player_game.py --particles 1024 --svd-rtol 1e-3 --skip-sde-baseline --output-dir outputs/sample_study_stable/n1024
python replicate_three_player_game.py --particles 5000 --svd-rtol 1e-3 --skip-sde-baseline --output-dir outputs/sample_study_stable/n5000
python compare_sample_counts.py
```

The comparison figure, metrics, individual panels, and raw histories are also
included in [`verified_results/sample_count_study/`](verified_results/sample_count_study/README.md).

This creates:

```text
outputs/thesis_replication/
  figure_4_2_uniform/
    dtb_snapshots.png
    sde_baseline_snapshots.png
    diagnostics.png
    history.npz
    config.json
  figure_4_3_gaussian/
    dtb_snapshots.png
    sde_baseline_snapshots.png
    diagnostics.png
    history.npz
    config.json
```

The red points in the snapshot panels mark `(0,0)` and `(0.5,0.5)`.

## Denser experiment

After the fast run succeeds, use the larger tangent dictionary and point cloud:

```bash
python replicate_thesis_figures.py --paper-scale --device auto
```

The paper-scale preset uses `N=5000`, `m=256`, `h=0.01`, and 100 steps. The
exact network width, particle count, tangent basis, Euler step, and SVD cutoff
are not visible in the supplied screenshot, so these values are documented
replication choices rather than claimed source parameters. Override the point
count with `--particles` when needed.

For the numerically stiffer 3D score transport, the validated preset uses
`h=0.005`, 200 steps, `m=128`, and `svd_rtol=1e-4`:

```bash
python replicate_three_player_game.py --paper-scale --device auto
```

Large empirical systems are more sensitive to rare particle excursions. The
verified `N=512,1024,5000` comparison therefore uses `svd_rtol=1e-3` for every
sample count; changing the cutoff between runs would confound the comparison.

## Run one initialization only

Uniform Figure 4.2-style run:

```bash
python run_game_dynamics.py \
  --initial-distribution uniform \
  --run-sde-baseline \
  --output-dir outputs/uniform_only
```

Gaussian Figure 4.3-style run:

```bash
python run_game_dynamics.py \
  --initial-distribution gaussian \
  --initial-mean 0.5 \
  --initial-variance 0.03 \
  --run-sde-baseline \
  --output-dir outputs/gaussian_only
```

## Algorithm implemented

For particles `x_i^k`, log density `ell_i^k`, and score `q_i^k`, the target
velocity is

```text
v_i^k = b(x_i^k) - 0.5 D q_i^k.
```

The neural tangent field is projected by the raw stacked least-squares system:

```text
J_stack = (J_1; ...; J_N)       shape (N*d, m)
V_stack = (v_1; ...; v_N)       shape (N*d)
alpha   = V_r Sigma_r^-1 U_r^T V_stack
u(x)    = J_selected(x) alpha
```

No `1/N` or `1/sqrt(N)` factor is applied. Singular values are retained when
`sigma_j > svd_rtol * sigma_1`. The explicit Euler step transports particles,
log density, and score using exact automatic differentiation of `grad u`,
`div u`, and `grad(div u)`.

## Algorithm-to-code map

| Algorithm block | Code |
|---|---|
| Initialization | `game_dtb/state.py` and Block 1 in `runner.py` |
| Select tangent coordinates | Block 2 in `algorithm.py` |
| Game plus score velocity | Block 3 in `algorithm.py` |
| Restricted Jacobians | `projection.selected_parameter_jacobian` |
| Unnormalized stack | `projection.stack_unnormalized_system` |
| Truncated SVD | `projection.truncated_svd_solve` |
| Spatial score terms | `derivatives.tangent_velocity_and_spatial_terms` |
| Euler transport | Block 9 in `algorithm.py` |
| Six-panel replication | `runner._plot_snapshots` |
| Direct SDE comparison | `runner.simulate_euler_maruyama` |

## Folder layout

```text
references/                    supplied TeX algorithm, target image, notes
verified_results/              checked 3D result, raw arrays, configuration
game_dtb/                      reusable Neural–DTB implementation
tests/                         twelve mathematical/integration tests
examples/custom_game.py        instructional custom game
run_game_dynamics.py           configurable single experiment
replicate_thesis_figures.py    one-command Figures 4.2/4.3 workflow
replicate_three_player_game.py one-command Figures 4.5/4.6 workflow
COLAB_TUTORIAL.md               GitHub-to-Colab instructions
notebooks/                      executable Colab notebook
SUMMARY.md                      technical assumptions and validation
```

## Reading the output

`history.npz` contains the six DTB particle snapshots, optional SDE baseline,
final score and log density, projection residuals, retained SVD ranks, and
coefficient norms. A low projection residual is necessary but not sufficient
for a scientifically accurate result; also compare the DTB point clouds with
the direct SDE baseline and run refinement studies.

## Important limitations

- Uniform density has zero score only in the box interior and a nonsmooth
  boundary. The implementation uses the interior score for sampled points.
- The scheme is explicit Euler and has no adaptive time step.
- The 3D problem is numerically stiffer; the rejected `h=0.02` trial diverged
  near `t=1`, so the published preset uses `h=0.005`.
- The source screenshot does not expose every training/numerical parameter.
- The neural parameters stay fixed because the supplied algorithm does not
  prescribe the original Allen–Cahn periodic reset/refit rule.
- Strong conclusions require step-size, particle-count, tangent-basis, and SVD
  tolerance refinement.

See [COLAB_TUTORIAL.md](COLAB_TUTORIAL.md) for the GitHub workflow.
