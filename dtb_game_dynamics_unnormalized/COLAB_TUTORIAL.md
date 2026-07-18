# Colab Tutorial: Reproduce the 2D and 3D Games from GitHub

Colab **clones** source code from GitHub into temporary runtime storage. Google
Drive is mounted later only to preserve generated outputs.

## 1. Open the notebook

In Colab, choose **File → Open notebook → GitHub** and enter:

```text
https://github.com/sun-mengwei/dtb-colab-experiments
```

Choose branch `codex/game-dynamics-dtb`, then open:

```text
dtb_game_dynamics_unnormalized/notebooks/game_dynamics_dtb_colab.ipynb
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

%cd /content/dtb-colab-experiments/dtb_game_dynamics_unnormalized
```

## 3. Read the algorithm and target together

The code folder now contains:

```text
references/dtb_game_dynamics_unnormalized_dimensions.tex
references/target_figures_4_2_4_3.png
references/three_player_game_definition.png
references/target_figures_4_5_4_6.png
references/README.md
```

Display the target:

```python
from IPython.display import Image, display
display(Image("references/target_figures_4_2_4_3.png"))
display(Image("references/three_player_game_definition.png"))
display(Image("references/target_figures_4_5_4_6.png"))
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

Expected: `12 passed`.

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

The red markers show the origin, `(3/8,3/8,3/8)`, and the permutations of
`(1/2,1/2,0)`. The validated 3D preset uses `h=0.005`, 200 steps, `m=128`, and
`svd_rtol=1e-4` because the coarser 2D time step is unstable for this case.

## 9. Optional denser runs

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

The denser preset uses `N=5000`, `m=256`, `h=0.01`, and 100 steps. These are
documented replication choices because the exact source values are not visible
in the supplied screenshot.

## 10. Save results to Drive

Colab runtime files disappear when the session ends:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!zip -qr dtb_game_replication.zip \
  outputs/colab_replication \
  outputs/colab_three_player
!cp dtb_game_replication.zip "/content/drive/MyDrive/"
```

## 11. Pull later changes

```python
%cd /content/dtb-colab-experiments
!git pull origin codex/game-dynamics-dtb
%cd /content/dtb-colab-experiments/dtb_game_dynamics_unnormalized
```

After the branch is merged, replace the branch name with `main`.
