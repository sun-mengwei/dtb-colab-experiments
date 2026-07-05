# Google Colab Tutorials for This Workspace

This workspace has two runnable Python projects:

- `DTB_rep/`: the 5-D Allen-Cahn / Deep Tangent Bundle reproduction.
- `nn_approx_ladder/`: smaller neural-network approximation and PINN experiments.

Use the smaller `nn_approx_ladder` experiments first to make sure Colab is set
up correctly, then run the DTB smoke test, then try larger DTB jobs.

## Tutorial 0: Start a Colab Notebook

1. Go to https://colab.research.google.com/.
2. Create a new notebook.
3. Select `Runtime > Change runtime type`.
4. Choose `T4 GPU` or another GPU if available, then click `Save`.
5. Run this cell:

```python
import torch, os, platform

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If `CUDA available` is `False`, CPU runs will still work for the small
experiments, but the full DTB experiment will be too slow.

## Tutorial 1: First-Time GitHub Setup

GitHub is the easiest long-term way to move this project into Colab. The basic
idea is:

- GitHub stores your code online in a **repository**.
- Colab downloads that repository with `git clone`.
- Generated outputs stay in Colab or Google Drive, not usually in GitHub.

### 1A. Create a GitHub Account

1. Go to https://github.com/.
2. Click `Sign up`.
3. Create a personal account.
4. Verify your email address.

GitHub's own account setup page is here:
https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github

### 1B. Create a Repository

1. Log in to GitHub.
2. Click the `+` button in the top-right corner.
3. Click `New repository`.
4. Use a short name, for example `dtb-colab-experiments`.
5. Choose `Private` if you do not want the code visible to everyone.
6. Leave `Add a README file` unchecked if you plan to upload this existing
   project.
7. Click `Create repository`.

Official GitHub reference:
https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository

### 1C. Upload This Project Using the GitHub Website

This is the beginner-friendly route. You do not need the terminal yet.

Upload these items:

- `DTB_rep/`
- `nn_approx_ladder/`
- `COLAB_TUTORIAL.md`
- `.gitignore`

Skip these for now:

- `Tangent Bundle/`
- `*.zip`
- generated `snapshots/`
- generated `runs/`
- large PDFs unless you specifically need them in Colab

Steps:

1. Open your new GitHub repository page.
2. Click `Add file`.
3. Click `Upload files`.
4. Drag the files/folders listed above into the upload area.
5. In `Commit message`, type something like `Upload Colab project files`.
6. Click `Commit changes`.

GitHub's browser upload has limits: files uploaded through the browser are
limited to 25 MiB per file, and you can upload up to 100 files at a time. That
is why the tutorial keeps large reference material and generated outputs out of
the repository.

Official GitHub reference:
https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

### 1D. Later: Upload Using Git Commands

You can ignore this section until you are comfortable. The website upload is
fine for getting started.

If you later want to use the terminal from this folder, the flow is:

```bash
cd /Users/mengwei/Desktop/DTB_rep
git init
git add .gitignore COLAB_TUTORIAL.md DTB_rep nn_approx_ladder
git commit -m "Initial Colab project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dtb-colab-experiments.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `dtb-colab-experiments` with your actual GitHub
username and repository name.

## Tutorial 2: Get the Files into Colab

### Option A: From GitHub

Use this after your files are uploaded to GitHub.

```python
%cd /content
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git workspace
%cd /content/workspace
!ls
```

Replace `YOUR_USERNAME/YOUR_REPO` with your real repository path.

To find the repository URL:

1. Open your repository on GitHub.
2. Click the green `Code` button.
3. Copy the `HTTPS` URL.
4. Use that URL after `git clone`.

Official GitHub reference:
https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

### Option B: From Google Drive Zip Upload

Use this if the code is only on your computer.

1. Zip the top-level folder that contains `DTB_rep/` and `nn_approx_ladder/`.
2. Upload the zip to Google Drive.
3. In Colab, run:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then unzip it. Adjust the zip filename if needed:

```python
%cd /content
!cp "/content/drive/MyDrive/DTB_rep.zip" .
!unzip -q DTB_rep.zip -d workspace
!find workspace -maxdepth 2 -type f | head -40
```

If the zip contains an extra top-level folder, `cd` into that folder before
running the tutorials below.

## Tutorial 3: Run the Small Approximation Experiments

These are the best first tests because they run quickly and produce plots.

```python
%cd /content/workspace/nn_approx_ladder
!pip install -r requirements.txt
```

Run a tiny smoke test:

```python
!python train_approx.py \
  --targets smooth localized \
  --models tiny fcnn mmnn \
  --steps 500 \
  --device auto \
  --out_dir runs/colab_smoke
```

Show the generated summary plot:

```python
from IPython.display import Image, display

display(Image("runs/colab_smoke/metrics_summary.png"))
display(Image("runs/colab_smoke/smooth.png"))
```

Run a larger comparison:

```python
!python train_approx.py \
  --targets smooth oscillatory localized multiscale piecewise \
  --models fcnn mcnn mmnn \
  --width 256 \
  --rank 32 \
  --mmnn_depth 4 \
  --steps 5000 \
  --device auto \
  --out_dir runs/colab_full_compare
```

## Tutorial 4: Run the 1-D Poisson PINN

Run FCNN first:

```python
%cd /content/workspace/nn_approx_ladder

!python train_poisson_pinn.py \
  --model fcnn \
  --steps 2000 \
  --device auto \
  --out_dir runs/poisson_fcnn_colab
```

Show the result:

```python
from IPython.display import Image, display
display(Image("runs/poisson_fcnn_colab/poisson_fcnn.png"))
```

Then compare MMNN:

```python
!python train_poisson_pinn.py \
  --model mmnn \
  --steps 2000 \
  --device auto \
  --out_dir runs/poisson_mmnn_colab
```

```python
display(Image("runs/poisson_mmnn_colab/poisson_mmnn.png"))
```

## Tutorial 5: Run the DTB Smoke Test

Start with the small DTB run, not the paper-scale run.

```python
%cd /content/workspace/DTB_rep
!pip install -r requirements.txt
```

Confirm GPU visibility:

```python
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Run the smoke test:

```python
!python run_5d_ac.py \
  --T 0.2 \
  --L 10 \
  --N 2000 \
  --m 1500 \
  --width 128 \
  --rank 16 \
  --depth 4 \
  --fit_steps 800 \
  --tag colab_smoke
```

Plot the result:

```python
!python visualize.py snapshots/run_colab_smoke
```

Find and display generated images:

```python
import glob
from IPython.display import Image, display

for path in glob.glob("snapshots/run_colab_smoke/*.png"):
    print(path)
    display(Image(path))
```

## Tutorial 6: Try a Medium DTB Run

If the smoke test works, scale up gradually:

```python
!python run_5d_ac.py \
  --T 0.5 \
  --L 10 \
  --N 5000 \
  --m 2500 \
  --width 192 \
  --rank 20 \
  --depth 5 \
  --fit_steps 1500 \
  --tag colab_medium
```

Then visualize:

```python
!python visualize.py snapshots/run_colab_medium
```

## Tutorial 7: Paper-Scale DTB Run

Only do this after the smoke and medium runs succeed. It can require a strong
GPU and enough runtime.

```python
!python run_5d_ac.py \
  --tag colab_paper \
  --fit_steps 6000 \
  --fit_lr 3e-3
```

Then:

```python
!python visualize.py snapshots/run_colab_paper
```

## Tutorial 8: Save Outputs Back to Drive

Colab runtimes are temporary. Save important outputs to Drive.

```python
from google.colab import drive
drive.mount("/content/drive")
```

For `nn_approx_ladder` outputs:

```python
%cd /content/workspace/nn_approx_ladder
!zip -qr nn_approx_ladder_runs.zip runs
!cp nn_approx_ladder_runs.zip "/content/drive/MyDrive/"
```

For DTB outputs:

```python
%cd /content/workspace/DTB_rep
!zip -qr dtb_snapshots.zip snapshots
!cp dtb_snapshots.zip "/content/drive/MyDrive/"
```

## Common Fixes

### `CUDA available: False`

Go to `Runtime > Change runtime type` and select a GPU. If no GPU is available,
use the `nn_approx_ladder` tutorials or smaller DTB settings.

### Out of memory during DTB

Reduce these arguments:

- `--N`: Monte-Carlo sample count.
- `--m`: DTB basis size.
- `--width`, `--rank`, `--depth`: network size.
- `--chunk`: vmap chunk size. Smaller chunks may reduce peak memory.

Example:

```python
!python run_5d_ac.py \
  --T 0.1 \
  --L 5 \
  --N 1000 \
  --m 800 \
  --chunk 250 \
  --width 96 \
  --rank 12 \
  --depth 3 \
  --fit_steps 400 \
  --tag tiny_debug
```

### Import errors

Make sure you are in the right folder before running a script:

```python
%pwd
!ls
```

For DTB, you should see files like `run_5d_ac.py`, `dtb.py`, and `network.py`.
For the ladder project, you should see `train_approx.py`, `models.py`, and
`targets.py`.
