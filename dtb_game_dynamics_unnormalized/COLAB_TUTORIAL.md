# Google Colab Tutorial Using GitHub

Colab does not mount GitHub in the same way it mounts Google Drive. Instead,
it **clones** the GitHub repository into the temporary Colab runtime. This
tutorial uses the published `codex/game-dynamics-dtb` branch. After that branch
is merged, change `BRANCH` to `main`.

The ready-made notebook is
`notebooks/game_dynamics_dtb_colab.ipynb`.

## 1. Open the notebook from GitHub

Use the notebook's GitHub page and select **Open in Colab**, or open Colab and
choose **File → Open notebook → GitHub**, then enter:

```text
https://github.com/sun-mengwei/dtb-colab-experiments
```

Select:

```text
dtb_game_dynamics_unnormalized/notebooks/game_dynamics_dtb_colab.ipynb
```

In Colab, select **Runtime → Change runtime type → T4 GPU** when available.

## 2. Clone the GitHub branch

The notebook begins with the following cell. It downloads a fresh copy of the
branch into `/content/dtb-colab-experiments`:

```python
import pathlib, shutil, subprocess

REPO_URL = "https://github.com/sun-mengwei/dtb-colab-experiments.git"
BRANCH = "codex/game-dynamics-dtb"  # use "main" after the branch is merged
REPO_DIR = pathlib.Path("/content/dtb-colab-experiments")

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

subprocess.run(
    [
        "git", "clone", "--depth", "1", "--branch", BRANCH,
        REPO_URL, str(REPO_DIR),
    ],
    check=True,
)

PROJECT_DIR = REPO_DIR / "dtb_game_dynamics_unnormalized"
assert (PROJECT_DIR / "run_game_dynamics.py").exists()
%cd /content/dtb-colab-experiments/dtb_game_dynamics_unnormalized
```

Colab storage is temporary. Rerun this cell after starting a new runtime.

## 3. Install dependencies and inspect the runtime

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

## 4. Run the mathematical tests

```python
!python -m pytest
```

The expected result is `5 passed`. Do not proceed to a long experiment if a
test fails.

## 5. Run the small linear-game example

```python
!python run_game_dynamics.py \
  --game linear \
  --particles 32 \
  --steps 5 \
  --basis-size 32 \
  --width 12 \
  --depth 2 \
  --dtype float32 \
  --device auto \
  --output-dir outputs/colab_linear
```

Display the result:

```python
from IPython.display import Image, display
display(Image("outputs/colab_linear/summary.png"))
```

Inspect numerical diagnostics:

```python
import numpy as np

data = np.load("outputs/colab_linear/history.npz")
print("final mean:", data["means"][-1])
print("projection residuals:", data["projection_residuals"])
print("retained ranks:", data["retained_ranks"])
print("alpha norms:", data["alpha_norms"])
print("final score RMS:", np.sqrt(np.mean(data["final_score"] ** 2)))
```

## 6. Run the Cournot example

Start small because the score equation requires nested automatic
differentiation:

```python
!python run_game_dynamics.py \
  --game cournot \
  --particles 64 \
  --steps 10 \
  --step-size 0.005 \
  --basis-size 64 \
  --svd-rtol 1e-5 \
  --diffusion 0.03 \
  --width 24 \
  --depth 2 \
  --dtype float32 \
  --device auto \
  --output-dir outputs/colab_cournot
```

```python
display(Image("outputs/colab_cournot/summary.png"))
```

The `1e-5` cutoff is a conservative float32 starting point. Repeat the run
with nearby tolerances and compare the retained rank, projection residual, and
`|alpha|`. Increase only one setting at a time.

## 7. Save outputs to Google Drive

GitHub stores source code, not runtime outputs. Mount Google Drive to preserve
results after the Colab runtime ends:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
!zip -qr dtb_game_outputs.zip outputs
!cp dtb_game_outputs.zip "/content/drive/MyDrive/"
```

Alternatively, download the archive directly:

```python
from google.colab import files
files.download("dtb_game_outputs.zip")
```

## 8. Pull later GitHub changes

If the repository is already cloned in the current runtime:

```python
%cd /content/dtb-colab-experiments
!git pull origin codex/game-dynamics-dtb
%cd /content/dtb-colab-experiments/dtb_game_dynamics_unnormalized
```

After the branch is merged, use `main` instead.

## 9. Modify the game drift

Open `game_dtb/games.py` or copy `examples/custom_game.py`. A valid drift maps
an `(N,d)` tensor to another `(N,d)` tensor. If the game uses a simplex or box
strategy space, derive a compatible coordinate transform or boundary model;
do not add clipping without also reconsidering the density and score equations.
