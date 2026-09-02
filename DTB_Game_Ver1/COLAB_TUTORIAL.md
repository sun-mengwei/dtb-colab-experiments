# Colab Tutorial: Reproduce the 2D, 3D, and 5D Games from GitHub

Colab **clones** source code from GitHub into temporary runtime storage. Google
Drive is mounted later only to preserve generated outputs.

## 1. Open the notebook

In Colab, choose **File → Open notebook → GitHub** and enter:

```text
https://github.com/sun-mengwei/dtb-colab-experiments
```

Choose branch `codex/game-dynamics-dtb`, then open the controlled notebook:

```text
DTB_Game_Ver1/notebooks/controlled_game_experiments_colab.ipynb
```

Select **Runtime → Change runtime type → T4 GPU** when available.

## 2. Clone the branch

```python
import pathlib, shutil, subprocess

REPO_URL = "https://github.com/sun-mengwei/dtb-colab-experiments.git"
BRANCH = "codex/game-dynamics-dtb"  # change to main after merge
REPO_DIR = pathlib.Path("/content/dtb-colab-experiments")

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", BRANCH,
     REPO_URL, str(REPO_DIR)],
    check=True,
)

%cd /content/dtb-colab-experiments/DTB_Game_Ver1
```

## 3. Read the algorithm and target together

The code folder now contains:

```text
references/dtb_game_dynamics_unnormalized_dimensions.tex
references/target_figures_4_2_4_3.png
references/three_player_game_definition.png
references/target_figures_4_5_4_6.png
references/five_player_game_definition.png
references/README.md
```

Display the target:

```python
from IPython.display import Image, display
display(Image("references/target_figures_4_2_4_3.png"))
display(Image("references/three_player_game_definition.png"))
display(Image("references/target_figures_4_5_4_6.png"))
display(Image("references/five_player_game_definition.png"))
```

Print the reference notes:

```python
print(open("references/README.md").read())
```

## 4. Install and test

```python
!python -m pip install -q -r requirements.txt
```

```python
import platform, torch
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

```python
!python -m pytest
```

Expected: `24 passed`.

## 4A. Periodic-refit controls in the controlled notebook

Each 2D, 3D, and 5D setup block exposes:

```text
REFIT_INTERVAL_*          physical steps per tangent block; 0 disables refit
REFIT_STEPS_*             Adam iterations at each block reset
REFIT_LR_*                Adam learning rate
REFIT_BATCH_*             minibatch size, sampled with replacement
REFIT_SAMPLES_*           one-shot fresh reference-law training set
REFIT_TEST_SAMPLES_*      separate fresh reference-law RMSE set
```

The controlled notebook uses the source game driver's defaults: `L=50`, 2,000
Adam steps, learning rate `1e-3`, batch size 2,048, 10,000 training samples,
and 4,000 fresh test samples. These reset fits are intentionally expensive.
Run the setup block again after changing a value, then run its `RUN_*` block.
The result block displays `refit_diagnostics.png`; purple vertical lines show
reset events.

Resetting trains only the tangent-basis network. It follows the original rule
without residual-triggered resets or rollback. The total Euler-step count must
be divisible by `L`. Set every `REFIT_INTERVAL_* = 0` to reproduce the
fixed-network behavior.

## 5. Run both replications

```python
!python replicate_thesis_figures.py \
  --device auto \
  --output-root outputs/colab_replication
```

This runs:

- Figure 4.2 with uniform initial samples on `[0,1]^2`;
- Figure 4.3 with `N((0.5,0.5), 0.03 I)` initial samples;
- the Neural–DTB scheme using `sigma_1=sigma_2=0.1`, hence `D=0.01 I`;
- an Euler–Maruyama baseline for the same SDE.

## 6. Display the results

```python
from IPython.display import Image, display

root = "outputs/colab_replication"
for relative in [
    "figure_4_2_uniform/dtb_snapshots.png",
    "figure_4_2_uniform/sde_baseline_snapshots.png",
    "figure_4_3_gaussian/dtb_snapshots.png",
    "figure_4_3_gaussian/sde_baseline_snapshots.png",
]:
    print(relative)
    display(Image(f"{root}/{relative}"))
```

Display numerical diagnostics:

```python
display(Image(f"{root}/figure_4_2_uniform/diagnostics.png"))
display(Image(f"{root}/figure_4_3_gaussian/diagnostics.png"))
```

## 7. Inspect saved arrays

```python
import numpy as np

data = np.load(f"{root}/figure_4_3_gaussian/history.npz")
print("snapshot times:", data["snapshot_times"])
print("DTB final mean:", data["snapshot_particles"][-1].mean(axis=0))
print("SDE final mean:", data["sde_baseline_particles"][-1].mean(axis=0))
print("final projection residual:", data["projection_residuals"][-1])
print("final retained rank:", data["retained_ranks"][-1])
print("final alpha norm:", data["alpha_norms"][-1])
```

## 8. Run the three-player game

```python
!python replicate_three_player_game.py \
  --device auto \
  --output-dir outputs/colab_three_player
```

Display the Neural–DTB and direct SDE panels:

```python
display(Image("outputs/colab_three_player/dtb_snapshots.png"))
display(Image("outputs/colab_three_player/sde_baseline_snapshots.png"))
display(Image("outputs/colab_three_player/diagnostics.png"))
```

The gold `X` marks the unstable origin. Red circles mark `(3/8,3/8,3/8)` and
the permutations of `(1/2,1/2,0)`. The validated 3D preset uses `h=0.005`, 200 steps, `m=128`, and
`svd_rtol=1e-4` because the coarser 2D time step is unstable for this case.

## 9. Compare 512, 1,024, and 5,000 samples

Use the same conservative SVD cutoff for all three runs. The 5,000-particle
case takes substantially longer on a CPU runtime; select a GPU runtime when
available.

```python
!python replicate_three_player_game.py \
  --particles 512 --svd-rtol 1e-3 --skip-sde-baseline \
  --device auto --output-dir outputs/sample_study_stable/n512

!python replicate_three_player_game.py \
  --particles 1024 --svd-rtol 1e-3 --skip-sde-baseline \
  --device auto --output-dir outputs/sample_study_stable/n1024

!python replicate_three_player_game.py \
  --particles 5000 --svd-rtol 1e-3 --skip-sde-baseline \
  --device auto --output-dir outputs/sample_study_stable/n5000

!python compare_sample_counts.py
```

```python
display(Image("outputs/sample_study_stable/sample_count_comparison.png"))
```

Exact metrics are saved in
`outputs/sample_study_stable/sample_count_metrics.csv`.

## 10. Run 2,000 particles with a depth-4 network

```python
!python replicate_three_player_game.py \
  --particles 2000 \
  --depth 4 \
  --svd-rtol 1e-3 \
  --skip-sde-baseline \
  --device auto \
  --output-dir outputs/three_player_n2000_depth4
```

```python
display(Image("outputs/three_player_n2000_depth4/dtb_snapshots.png"))
display(Image("outputs/three_player_n2000_depth4/diagnostics.png"))
```

The unstable origin is a gold `X`; the four stable equilibria are red circles.
Depth 4 means four hidden `Linear+tanh` blocks plus the output layer.

## 11. Run the matched five-player depth comparison

The two DTB runs below differ only in network depth. The depth-2 run also
computes the direct SDE baseline; that baseline is independent of network
depth and does not need to be repeated.

```python
!python replicate_five_player_game.py \
  --particles 2000 --depth 2 --basis-size 128 --width 32 \
  --svd-rtol 1e-3 --device auto \
  --output-dir outputs/five_player_depth_comparison/depth2

!python replicate_five_player_game.py \
  --particles 2000 --depth 4 --basis-size 128 --width 32 \
  --svd-rtol 1e-3 --skip-sde-baseline --device auto \
  --output-dir outputs/five_player_depth_comparison/depth4

!python compare_five_player_depths.py
```

```python
display(Image("outputs/five_player_depth_comparison/depth_comparison.png"))
display(Image("outputs/five_player_depth_comparison/depth2/dtb_snapshots.png"))
display(Image("outputs/five_player_depth_comparison/depth4/dtb_snapshots.png"))
display(Image("outputs/five_player_depth_comparison/depth2/pairwise_final.png"))
```

All gold `X` markers are unstable. The six-panel plot is the symmetry
projection `x1` versus `mean(x2,...,x5)`; use `pairwise_final.png` to inspect
all coordinate pairs. Exact comparison values are saved in
`outputs/five_player_depth_comparison/depth_metrics.csv`.

## 12. Compare the ordinary MLP with a frozen-feature MMNN

This small pilot uses the same 500 particles and `m=128` for both models. The
MMNN has width 32, rank 8, and two blocks. Only its `A,c` parameters can be
selected as tangents; its random-feature `W,b` parameters remain frozen.

```python
!python compare_three_player_architectures.py \
  --particles 500 --width 32 --rank 8 --depth 2 \
  --basis-size 128 --svd-rtol 1e-3 --device auto \
  --output-root outputs/three_player_mlp_mmnn_n500
```

```python
display(Image("outputs/three_player_mlp_mmnn_n500/architecture_comparison.png"))
display(Image("outputs/three_player_mlp_mmnn_n500/mmnn/dtb_snapshots.png"))
display(pd.read_csv("outputs/three_player_mlp_mmnn_n500/architecture_metrics.csv"))
```

Each `config.json` records the tangent-eligible names, frozen names, and tanh
saturation fractions. This is a one-seed pilot, not a final architecture
ranking.

## 13. Optional denser tangent basis

Only start this after the fast run succeeds:

```python
RUN_PAPER_SCALE = False
if RUN_PAPER_SCALE:
    !python replicate_thesis_figures.py \
      --paper-scale \
      --device auto \
      --output-root outputs/paper_scale
    !python replicate_three_player_game.py \
      --paper-scale \
      --device auto \
      --output-dir outputs/paper_scale_three_player
```

The 2D denser preset uses `N=5000`, `m=256`, `h=0.01`, and 100 steps. The 3D
paper-scale preset uses `m=256`, `h=0.005`, and 200 steps. These are documented
replication choices because the exact source values are not visible in the
supplied screenshot.

## 14. Save results to Drive

Colab runtime files disappear when the session ends:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!zip -qr dtb_game_replication.zip \
  outputs/colab_replication \
  outputs/colab_three_player \
  outputs/three_player_n2000_depth4 \
  outputs/five_player_depth_comparison \
  outputs/three_player_mlp_mmnn_n500
!cp dtb_game_replication.zip "/content/drive/MyDrive/"
```

## 15. Pull later changes

```python
%cd /content/dtb-colab-experiments
!git pull origin codex/game-dynamics-dtb
%cd /content/dtb-colab-experiments/DTB_Game_Ver1
```

After the branch is merged, replace the branch name with `main`.
