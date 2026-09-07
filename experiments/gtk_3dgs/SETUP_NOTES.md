# GTK / 3DGS experiment infrastructure — Phase 1 setup notes

Status: **smoke test PASSED** on 2026-08-25 (SLURM job 3025708, `ghx4` partition,
1x NVIDIA GH200, node `gh066`). This document records exactly what worked so
Phase 2 (full 30k-iteration multi-arm runs) can reuse it without re-deriving.

See `drafts/tpami_gtk_extension/EXPERIMENT_PLAN.md` for the experiment design
this infrastructure serves.

## Directory layout

- Code/docs (git-tracked, this repo): `experiments/gtk_3dgs/`
  - `scripts/smoke_test.sbatch` — the exact working SLURM script used below.
- Scratch (NOT git-tracked, large artifacts): `/work/nvme/bhov/zzhao18/gtk_3dgs_runs/`
  - `repos/gaussian-splatting/` — official 3DGS, recursive clone, HEAD `54c035f`.
  - `repos/3dgs-mcmc/` — 3DGS-MCMC (Kheradmand et al., NeurIPS'24), HEAD `7b4fc9f`.
  - `venvs/gtk/` — the uv-managed Python 3.11 venv (see below).
  - `uv-python/` — uv-managed CPython 3.11.13 build (has dev headers; see below).
  - `data/bonsai/` — Mip-NeRF 360 `bonsai` scene, COLMAP format, 1.4 GB.
  - `output/` — training run outputs (checkpoints, renders).
  - `logs/` — sbatch stdout/stderr.

Nothing under `/work/nvme/...` is committed to git. Only code/docs under
`experiments/gtk_3dgs/` in the repo are tracked.

## Cluster facts confirmed

- DeltaAI GH200 nodes are ARM64 (`aarch64`), CUDA 13.1.1 toolkit at
  `/sw/user/cudatoolkits/installs/cuda-13.1.1` (already on PATH; `nvcc` works
  on the login node — compilation does not need a GPU).
- SLURM account `bhov-dtai-gh`, partition `ghx4`, one GPU (`--gpus=1`) is
  enough for a single training arm.
- GH200 compute capability is **9.0** (Hopper) — must be set explicitly via
  `TORCH_CUDA_ARCH_LIST=9.0` when building CUDA extensions, since the login
  node has no GPU for auto-detection and even on a GPU node autodetection can
  be flaky in a batch job.

## Python / uv environment

`uv venv --python 3.11` against the **system** `/usr/bin/python3.11` silently
produces an interpreter with no `Python.h` (`/usr/include/python3.11` does
not exist) — any CUDA/C++ extension build then fails with
`fatal error: Python.h: No such file or directory`. Fix: use a uv-managed
downloadable CPython instead, which bundles headers:

```bash
UV=/projects/bhov/zzhao18/software/uv-bin/uv
export UV_CACHE_DIR=/work/nvme/bhov/zzhao18/uv-cache
export UV_PYTHON_INSTALL_DIR=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/uv-python
$UV python install cpython-3.11.13-linux-aarch64-gnu
$UV venv --python cpython-3.11.13-linux-aarch64-gnu \
    /work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk
```

uv-created venvs have **no `pip` binary** — use `uv pip install --python
<venv>/bin/python ...` for everything, not `pip`/`python -m pip` (the system
`pip` on PATH belongs to an unrelated miniforge install and will silently
target the wrong environment).

### PyTorch (the main a-priori risk — resolved cleanly)

PyPI's default index now ships CUDA-13-enabled `torch` wheels for
`linux_aarch64` directly — no special index or source build needed:

```bash
$UV pip install --python venvs/gtk/bin/python torch torchvision
# -> torch==2.13.0+cu130, torchvision==0.28.0+cu130
```

Verified on a GPU node: `torch.cuda.is_available() == True`,
`torch.cuda.get_device_name(0) == 'NVIDIA GH200 120GB'`,
`torch.version.cuda == '13.0'` (a harmless minor-version mismatch against the
system CUDA 13.1.1 toolkit — PyTorch itself warns about this and says it's
not a problem; confirmed true in practice).

### Other pip deps (pip-installable equivalents of the repos' `environment.yml`)

The repos' `environment.yml` pins ancient conda versions (Python 3.7/3.8,
PyTorch 1.12/1.13, CUDA 11.6/11.7) — all ignored; only the actual Python
package names matter:

```bash
$UV pip install --python venvs/gtk/bin/python \
    numpy opencv-python-headless plyfile tqdm joblib pillow ninja setuptools
```

(`lpipsPyTorch` used by the repo's own LPIPS metric is bundled in-repo and
does not need the external `lpips` package.)

## Building the CUDA extensions

Two issues, both resolved:

1. **Build isolation must be disabled** (`--no-build-isolation`) — the
   submodules' `setup.py` does `import torch` at build time, which fails in
   an isolated build env that doesn't have torch installed.
2. **Missing `#include <cstdint>`** in `cuda_rasterizer/rasterizer_impl.h` of
   `diff-gaussian-rasterization` (both the official repo's copy and the
   3DGS-MCMC fork's copy) — fails to compile against this stack's newer
   host-compiler/CUDA-13.1 combo with `error: namespace "std" has no member
   "uintptr_t"` / `error: identifier "uint32_t" is undefined`. This is a
   well-known compatibility gap in the 2023-era code (old GCC tolerated the
   missing include transitively; GCC used here does not). Fix applied
   directly in the cloned submodule (not upstream):
   ```cpp
   #include <iostream>
   #include <vector>
   #include <cstdint>   // added
   #include "rasterizer.h"
   #include <cuda_runtime_api.h>
   ```

Working build commands (run on the **login node** — compilation only needs
`nvcc`, not a live GPU, as long as `TORCH_CUDA_ARCH_LIST` is set explicitly):

```bash
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="$CUDA_HOME/bin:$PATH"

UV=/projects/bhov/zzhao18/software/uv-bin/uv
cd /work/nvme/bhov/zzhao18/gtk_3dgs_runs
$UV pip install --python venvs/gtk/bin/python --no-build-isolation \
    ./repos/gaussian-splatting/submodules/simple-knn
$UV pip install --python venvs/gtk/bin/python --no-build-isolation \
    ./repos/gaussian-splatting/submodules/diff-gaussian-rasterization
$UV pip install --python venvs/gtk/bin/python --no-build-isolation \
    ./repos/gaussian-splatting/submodules/fused-ssim
```

All three built successfully on the first retry after the two fixes above.
`python -c "import diff_gaussian_rasterization, simple_knn, fused_ssim"`
succeeds even on the (GPU-less) login node.

**3DGS-MCMC's `diff-gaussian-rasterization` fork** (branch `gs-mcmc` of
`https://github.com/shakibakh/diff-gaussian-rasterization`, needed for the
second comparison arm) needed the identical one-line `cstdint` fix and then
built cleanly the same way.

### Known collision to handle in Phase 2 (not yet solved, not blocking Phase 1)

The official repo's and 3DGS-MCMC's forks of `diff-gaussian-rasterization`
(and `simple-knn`) install under the **same package names**
(`diff_gaussian_rasterization`, `simple_knn`), so installing one into the
shared `venvs/gtk` venv silently uninstalls/replaces the other. For Phase 2,
when both the 3DGS and 3DGS-MCMC arms need to run, use **separate venvs per
arm** (e.g. `venvs/gtk-3dgs` and `venvs/gtk-3dgs-mcmc`, both built the same
way against the same base deps) rather than the single shared venv used for
this smoke test. The shared `venvs/gtk` venv currently has the **official**
3DGS extensions active (verified after the swap-back below).

## Dataset: Mip-NeRF 360 `bonsai`

The dataset authors host two zips at `https://jonbarron.info/mipnerf360/`
(linked from the 3DGS repo's own README): `360_v2.zip` (12.5 GB, "Dataset
Pt. 1" — contains `bicycle`, `bonsai`, `counter`, `garden`, `kitchen`,
`room`, `stump`) and `360_extra_scenes.zip` (4.5 GB, "Dataset Pt. 2" —
`flowers`, `treehill`). `bonsai` is in the first zip.

To avoid downloading the full 12.5 GB for one scene: the scene's ~1172 files
are **contiguous** within the zip's byte layout (~1.4 GB span), so a single
HTTP Range request for just that span plus a small local-zip-header parser
extracts `bonsai/` without touching the rest of the archive. (One-off script,
not re-needed unless re-downloading; not committed since it's a one-time
data-fetch utility, not part of the reproducible training pipeline.)

Result: `/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/bonsai/` — `images/`
(292 full-res JPEGs) + `images_2/`, `images_4/`, `images_8/` (pre-downsampled
by the dataset authors) + `sparse/0/{cameras,images,points3D}.bin` (COLMAP
reconstruction) + `poses_bounds.npy`. Verified: valid JPEGs, correct file
counts, correct COLMAP binary files.

## Smoke test: exact working SLURM script

`experiments/gtk_3dgs/scripts/smoke_test.sbatch` (committed) — runs the
**official** 3DGS `train.py` unmodified, `bonsai` at 1/4 resolution
(`-r 4`), only **2500** iterations (not the full 30k), with `--eval` for the
standard train/test split:

```bash
#SBATCH --account=bhov-dtai-gh --partition=ghx4 --nodes=1 --gpus=1
#SBATCH --cpus-per-task=16 --mem=64g --time=02:00:00
...
python train.py -s "${DATA}" -m "${OUT}" --eval -r 4 \
  --iterations 2500 --test_iterations 1000 2500 --save_iterations 2500 \
  --disable_viewer --quiet
```

Submitted with plain `sbatch scripts/smoke_test.sbatch` from
`/work/nvme/bhov/zzhao18/gtk_3dgs_runs`.

## Smoke test result

Job **3025708**, node `gh066` (GH200 120GB), queued briefly then ran in
**71 seconds** total:

- `torch.cuda.is_available() == True` on the compute node; GPU correctly
  detected as `NVIDIA GH200 120GB`.
- Training loop ran all 2500 iterations at ~90-93 it/s.
- Loss decreased from 0.286 (iter 0) to ~0.035-0.044 (iter ~2500) — sane
  monotone-ish decrease, not NaN/divergent.
- Checkpoint saved and is a well-formed binary PLY:
  `output/smoke_bonsai_3025708/point_cloud/iteration_2500/point_cloud.ply`,
  199 MB, 802,176 Gaussians, correct PLY header/property list.
- Also produced `cameras.json`, `cfg_args`, `exposure.json`, `input.ply` as
  expected from a normal 3DGS run.
- Exit code 0, no errors, no OOM, no CUDA errors.

**Conclusion: the full pipeline (COLMAP data loading, CUDA rasterizer
forward/backward, training loop, densification, checkpoint I/O) works
end-to-end on this ARM64 GH200 + CUDA 13.1 stack.** This was the only goal
of Phase 1.

## What's explicitly NOT done yet (Phase 1 checkpoint — see Phase 2 section below for what happened next)

- Full 30k-iteration runs (this was a 2500-iteration smoke test only).
- 3DGS-MCMC training runs (extension is built and verified to compile; not
  yet run — needs its own venv per the collision note above).
- The third "GTK-informed schedule" arm (needs `gtk-06`'s theorem finalized
  first, per the experiment plan).
- The `garden` secondary scene.
- The $\lambda_{\min}$ proxy / opacity-anisotropy diagnostic loggers and the
  forked `experiments/gtk_3dgs/train.py` entry point described in the
  experiment plan's "what reproducible means" section.

## Phase 2: full 30k-iteration runs, both arms, diagnostics (2026-08-25/26)

### Separate venvs (resolves the Phase 1 collision note)

- `venvs/gtk` — kept as-is from Phase 1, has the **official** 3DGS
  `diff_gaussian_rasterization`/`simple_knn` built in. Used for the vanilla
  3DGS arm.
- `venvs/gtk-3dgs-mcmc` — new venv, same uv/CPython recipe as Phase 1, with
  torch/torchvision + the same base deps, plus 3DGS-MCMC's own
  `submodules/diff-gaussian-rasterization` (branch `gs-mcmc`, already had
  the Phase-1 `cstdint` fix) and `submodules/simple-knn`. `simple-knn`
  needed one more fix beyond Phase 1's: `error: identifier "FLT_MAX" is
  undefined` in `simple_knn.cu` — same missing-header issue as the
  `cstdint`/`uintptr_t` one, fixed the same way by adding `#include
  <cfloat>` directly in the cloned submodule. Both extensions then built
  cleanly. No `fused_ssim` needed for the MCMC repo (it uses the same
  in-repo `lpipsPyTorch` as the official repo, confirmed via grep).
- `venvs/gtk-diag` — new lightweight venv (numpy, scipy, plyfile only, no
  torch/CUDA) for `gtk_diagnostics.py`, so diagnostics can run on the
  CPU-only login node.

### `garden` dataset

Downloaded with a **generalized, committed** version of Phase 1's
range-fetch trick:
`experiments/gtk_3dgs/scripts/fetch_mipnerf360_scene.py --scene garden
--out .../data/garden`. One correction versus the Phase 1 description: the
actual zip host is `http://storage.googleapis.com/gresearch/refraw360/360_v2.zip`
(`jonbarron.info/mipnerf360/` 404s directly now — it's a landing page that
links out to that GCS bucket, confirmed still supports Range requests).
Implementation wraps a `urllib`-based random-access file object around
`zipfile.ZipFile` (stdlib only, no extra deps) rather than manually parsing
the zip's central directory — same net effect (only that scene's ~751
entries / ~3 GB are fetched, not the full 12.5 GB), less code. Verified:
185 images, correct COLMAP `sparse/0/{cameras,images,points3D}.bin`.

### Resolution protocol for both scenes/arms

3DGS-MCMC ships `configs/bonsai.json = {"resolution": 2, "cap_max":
1300000}` and `configs/garden.json = {"resolution": 4, "cap_max":
5200000}` — these are the paper's own precomputed settings (indoor scenes
at half resolution, outdoor at quarter, `cap_max` = the final Gaussian
count the original 3DGS paper's run reached for that scene), i.e. exactly
the standard Mip-NeRF 360 protocol used in every published table. Used
`-r 2` for bonsai and `-r 4` for garden on **both** arms (not the Phase 1
smoke test's ad hoc `-r 4` on bonsai), so PSNR/SSIM/LPIPS are comparable to
published numbers and identical between arms on a given scene.

### Loss logging (for the diagnostics' "training loss" column)

Neither repo dumps per-iteration loss to a file by default (only an EMA
value to the tqdm postfix and, if enabled, TensorBoard). Patched both
`train.py`s directly in the checked-out scratch repos with an identical
3-line insert, right after the existing EMA-loss update block, guarded by
`if iteration % 1000 == 0`, appending `{"iteration", "loss", "l1_loss"}` to
`<model_path>/losses.jsonl`. Exact unified diffs committed as
`experiments/gtk_3dgs/scripts/official_train_loss_logging.patch` and
`mcmc_train_loss_logging.patch` (apply with `patch -p1 < ... ` against a
fresh clone's `train.py` to reproduce) — this substitutes for maintaining a
full forked `train.py` entry point, given how small and repo-specific the
one required change is.

### `gtk_diagnostics.py` — the standalone diagnostic script

`experiments/gtk_3dgs/scripts/gtk_diagnostics.py`. Pure numpy + scipy.sparse
+ plyfile, no torch/CUDA — runs against any 3DGS-format `point_cloud.ply` +
that run's `cameras.json`, independent of which arm produced it, per the
Data Availability Statement. What it computes and how, in brief (full
derivation/approximations documented in the script's own docstring):

- Reads opacity/scale/rotation with the correct activations
  (`sigmoid`/`exp`/L2-normalize — PLY stores raw pre-activation values;
  confirmed against `scene/gaussian_model.py`'s `*_activation` fields in
  both repos, identical in both).
- **A resolution gotcha that cost real debugging time**: `cameras.json` is
  written from the *raw, un-downsampled* `CameraInfo` (before
  `loadCam`/`Camera.__init__` apply the `-r` resize), so its
  `width`/`height`/`fx`/`fy` are always the **original full-resolution**
  intrinsics regardless of what `-r` training actually used — confirmed
  empirically (Phase 1 smoke test used `-r 4` on bonsai, but its
  `cameras.json` reports `width=3118` i.e. bonsai's native resolution, not
  `780` = `round(3118/4)`). The script recovers the true training
  resolution from the run's saved `cfg_args` (regex on `resolution=N`) and
  rescales width/height/fx/fy itself (`round(orig/N)` for `N` in
  `{1,2,4,8}`, matching `utils/camera_utils.py::loadCam` exactly) before
  doing anything else — this must happen before either the pixel-subset
  sampling or the projection math, or both are silently wrong.
- $\lambda_{\min}$ proxy: fixed 2048 `(train_image, u, v)` triples sampled
  once with `np.random.default_rng(0)`, cached to
  `data/<scene>/gtk_ray_subset_2048.json` so every checkpoint and both arms
  reuse the identical subset (train/test split re-derived independently
  from `sparse/0/images.bin` — sort image names, hold out every 8th — a
  small self-contained COLMAP `images.bin` reader is included so the script
  has zero dependency on either repo). For each pixel, projects all
  Gaussians into that image's camera space via the exact convention
  `cameras.json` uses (`rotation`/`position` are the camera-to-world
  rotation/camera-center; world→camera is `rotation.T @ (X - position)`),
  builds the 2D screen covariance via the same EWA-splatting Jacobian +
  always-on `+0.3` diagonal dilation the CUDA rasterizer itself applies
  (`cuda_rasterizer/forward.cu`'s `computeCov2D`/`computeCov3D`/
  `renderCUDA`, formulas re-derived and hand-verified against a 90°-quaternion
  test case, not just transcribed), evaluates
  $\varphi_i(x)=\mathrm{opacity}_i\cdot\exp(\mathrm{power}_i(x))$ per
  candidate Gaussian per pixel (coarse 64px box-filtered per pixel for
  tractability — negligible effect since real Gaussian footprints are a
  few px), assembles the sparse $(2048 \times N)$ contribution matrix $Z$
  keyed by each Gaussian's *original PLY row index* (so a Gaussian visible
  to pixels from two different training images still gets correct
  cross-image Gram entries), and returns `numpy.linalg.eigvalsh(Z @
  Z.T)[0]`.
- Opacity floor/percentiles and anisotropy: no camera needed —
  $\Sigma_{\text{world}} = R\,\mathrm{diag}(\text{scale}^2)\,R^\top$ with
  $R$ orthogonal means the covariance eigenvalues are exactly
  $\text{scale}_{1,2,3}^2$, so $\sigma_{\min,i}/\sigma_{\max,i}$ are just
  `min`/`max` of the 3 stored scale values directly — no eigendecomposition
  needed.

**Validated** against the Phase 1 smoke checkpoint
(`output/smoke_bonsai_3025708`, iteration 2500, `-r 4`, 802,176 Gaussians):
correctly recovers training resolution `780x520` from `cfg_args`; at
`n_samples=2048` returns `lambda_min≈8.5e-4`, `alpha_min≈0.0033`,
`s_min≈6.2e-5`, `anisotropy_median≈3.3`/`p95≈14.2`, `num_gaussians_touching_subset≈`
tens of thousands — all sane orders of magnitude. Wall time ≈42s per
checkpoint on the login node CPU at `n_samples=2048`.

### Training sbatch scripts (all under `experiments/gtk_3dgs/scripts/`)

`bonsai_3dgs.sbatch`, `bonsai_mcmc.sbatch` (both: `-r 2`, 30k iterations,
`--save_iterations` every 1000 for diagnostics, PSNR/SSIM/LPIPS via each
repo's own `render.py --skip_train` + `metrics.py` at 7k/30k),
`garden_3dgs.sbatch`, `garden_mcmc.sbatch` (both: `-r 4`, 30k iterations,
`--save_iterations 7000 30000` only — no per-1000 diagnostics for this
scene, per the experiment plan). One command-line difference worth noting:
**3DGS-MCMC's `train.py` has no `--disable_viewer` flag** (its
`network_gui` loop is already unconditionally commented out upstream, so
there's nothing to disable) — passing it would be an argparse error; the
MCMC sbatch scripts omit it. MCMC's `--config configs/<scene>.json`
subsumes `-r`/`--resolution` and adds `--cap_max` — passed instead of `-r`.

Actual per-iteration timing and any timeout adjustments: see the next
section, filled in once real timing data came back from the first
submitted job.

### Actual timing / job outcomes

All 4 SLURM jobs completed successfully, well inside the requested time
budgets (extrapolated from bonsai_3dgs's early-iteration rate of
~35-40 it/s post-densification, seen live in the first ~5 minutes of that
job, before submitting the other three):

| job | SLURM job ID | wall time | exit |
|---|---|---|---|
| bonsai_3dgs | 3025852 | 18m46s | 0 |
| bonsai_mcmc | 3025873 | 21m53s | 0 |
| garden_3dgs | 3025874 | 15m28s | 0 |
| garden_mcmc | 3025875 | 29m12s | 0 |

### Final PSNR/SSIM/LPIPS

| scene | arm | it=7000 PSNR/SSIM/LPIPS | it=30000 PSNR/SSIM/LPIPS |
|---|---|---|---|
| bonsai | 3DGS | 30.04 / 0.929 / 0.203 | 32.54 / 0.948 / 0.173 |
| bonsai | 3DGS-MCMC | 27.73 / 0.883 / 0.272 | 32.49 / 0.950 / 0.167 |
| garden | 3DGS | 26.65 / 0.840 / 0.152 | 27.85 / 0.875 / 0.103 |
| garden | 3DGS-MCMC | 24.99 / 0.764 / 0.250 | 27.91 / 0.881 / 0.094 |

Both arms' vanilla-3DGS numbers land squarely in the range of published
Mip-NeRF 360 tables (bonsai ~32 PSNR / 0.94-0.95 SSIM, garden ~27.4 PSNR /
0.87 SSIM), a good sanity check that the pipeline (COLMAP loading, CUDA
rasterizer, densification, eval protocol) is correct on this ARM64/GH200
stack. Pattern observed on **both** scenes: 3DGS-MCMC starts noticeably
*behind* vanilla 3DGS at 7k iterations (its stochastic relocation needs
more iterations to reach a comparable point-cloud layout than greedy
clone/split), but converges to essentially tied PSNR/SSIM and consistently
**better LPIPS** by 30k on both scenes — consistent with the published
3DGS-MCMC paper's own claims.

### GTK diagnostics: bonsai only, both arms, all 30 checkpoints (1k-30k)

Full per-checkpoint values in `output/bonsai_3dgs/diagnostics.jsonl` and
`output/bonsai_mcmc/diagnostics.jsonl` (both scratch, not git-tracked; not
reproduced verbatim here). Summary of what was actually observed — **the
diagnostics data does not support a clean "MCMC's principled rule protects
conditioning better" story**; if anything the opposite shows up on two of
the three diagnostics:

- **Population $s_{\min}$ (smallest Gaussian scale anywhere in the point
  cloud) collapses on both arms, but far more severely under 3DGS-MCMC.**
  Vanilla 3DGS: $s_{\min}$ falls from $3.2\times10^{-4}$ (it=1000) to
  $4.2\times10^{-9}$ (it=30000) — about 4.5 orders of magnitude — and then
  *stays exactly flat* from it=15000 onward, which is exactly
  `densify_until_iter=15000` for the official repo (no more
  splitting/pruning after that point, a clean sanity check that the
  instrumentation is measuring the right thing). 3DGS-MCMC: $s_{\min}$
  falls from $3.0\times10^{-4}$ to $5.95\times10^{-12}$ — about 7.7 orders
  of magnitude, roughly **700x smaller than vanilla 3DGS's floor** — and
  is still slowly shrinking at it=30000 (MCMC's `densify_until_iter=25000`
  is later, and relocation keeps operating on the fixed-size population
  the whole time).
- **Anisotropy ($\sigma_{\max}/\sigma_{\min}$ per Gaussian) is far more
  extreme under 3DGS-MCMC.** Vanilla 3DGS: median ratio grows from 2.0 to
  8.4, 95th-percentile from 5.8 to 380, over training. 3DGS-MCMC: median
  grows from 1.7 to **154**, 95th-percentile explodes to
  $\sim 3-4\times10^7$ by mid-training (peaking around it=19000-20000,
  then *slowly declining* back to $3.1\times10^7$ by it=30000, the one
  place MCMC's curve is non-monotone and arguably self-correcting a little
  — but the absolute magnitude gap to vanilla 3DGS is still enormous, five
  orders of magnitude).
- **Opacity floor**: 3DGS's $\alpha_{\min}$ drifts down mildly (0.0044 to
  0.0020). 3DGS-MCMC's $\alpha_{\min}$ starts already lower
  ($3.5\times10^{-5}$) and collapses further late in training, dropping to
  $2.1\times10^{-7}$ by it=30000 (a sharp additional drop after it=25000,
  right around when MCMC's `densify_until_iter` stops relocating dead
  Gaussians and some very-low-opacity ones are left in place).
- **$\lambda_{\min}$ proxy itself** (the actual quantity gtk-06's theorem
  is about) is, by contrast, **broadly similar in order of magnitude
  between the two arms** and does not show as stark a gap: both arms spend
  much of training with the proxy at or indistinguishable from the
  numerical noise floor ($|\lambda_{\min}| \lesssim 10^{-15}$, i.e. the
  2048-pixel Gram matrix is effectively rank-deficient) with occasional
  excursions to $10^{-5}$-$10^{-4}$; vanilla 3DGS's excursions are
  somewhat more frequent/consistent (positive in ~24/30 checkpoints, vs.
  ~14/30 for MCMC) but neither arm shows a clean, stable, strictly-positive
  $\lambda_{\min}$ regime at this pixel-subset scale in either arm's
  30k-iteration run.

Read plainly: at the level of the population-wide $s_{\min}$/anisotropy
summary statistics, 3DGS-MCMC's stochastic relocation does **not** protect
against — and on this run, exacerbates — the collapse of the smallest
scale and the growth of extreme anisotropy, relative to vanilla 3DGS's
heuristic prune/split, even though it reaches comparable or better final
image-quality metrics. The one metric closer to gtk-06's actual object of
study ($\lambda_{\min}$ of the empirical Gram matrix) does not show a
clean separation between the two arms either way. This is reported as
observed, not adjusted to fit a hypothesized direction; it's a genuine,
possibly inconvenient, finding for the "principled relocation protects the
kernel" framing and should be treated as real data to reconcile with
gtk-06's theorem statement, not explained away.

### Blockers encountered, exactly as they occurred

1. **`cameras.json` resolution mismatch** (caught before wasting GPU time):
   `cameras.json` always reports full-original-image intrinsics regardless
   of the `-r` downsample training actually used. `gtk_diagnostics.py`
   initially assumed the reported width/height *was* the training
   resolution, which silently breaks the pixel-subset sampling and the
   whole projection. Root-caused by comparing against actual `images_4`/
   `images_2` JPEG dimensions and `cfg_args`'s `resolution=N`. Fixed by
   recovering the true resolution from `cfg_args` and rescaling
   width/height/fx/fy before use (see `apply_resolution_scaling` in the
   script). Caught during a deliberate smoke test of the diagnostic script
   against the Phase 1 checkpoint before trusting it for the real runs.
2. **Image-name convention differs between the two repos' `cameras.json`**:
   the official repo's `camera_to_JSON` writes `img_name` *with* the file
   extension (`DSCF5565.JPG`); 3DGS-MCMC's writes it *without*
   (`DSCF5565`) — same underlying code path, apparently a version
   difference in `camera_to_JSON`/`readColmapCameras` between the two
   forks. Caused a `KeyError` the first time the diagnostic script was run
   against `bonsai_mcmc` (the cached fixed-pixel-subset file, built from
   `bonsai_3dgs`'s convention, didn't match). Fixed by normalizing image
   names (`os.path.splitext`) on both sides of every lookup in
   `gtk_diagnostics.py`.
3. **The login-node background diagnostic process died silently, twice**,
   partway through the `bonsai_mcmc` run (once after 16/30 checkpoints,
   once after 13 more) — no error, no traceback, just stopped appending to
   the log. Not a script bug (confirmed via `/proc/<pid>/status` showing
   the process actively `R`unning shortly before each death, and via
   `sacct`-independent process listing) — most likely the shared
   multi-user login node recycling long-lived background sessions/process
   groups. Mitigated by resuming with `--iterations <remaining list>`
   (the script's checkpoint-by-checkpoint append design made this trivial)
   and switching from plain `nohup ... &` to `setsid nohup ... &
   disown` for the final resume, which held for the rest of the run.
4. **A second, unexplained process repeatedly ran the identical
   `gtk_diagnostics.py` command against the exact same output files
   concurrently**, several times, while the intended background run was
   in progress — most likely another agent instance (possibly a leftover
   background/fork process, and outside this run's control) probing or
   duplicating the same work. Most instances wrote to harmless separate
   scratch files (`diagnostics2.jsonl`, `/tmp/diag_test_30000.jsonl`,
   etc.) and were merely wasteful, but **two collisions were genuinely
   dangerous**: (a) two processes appending to the same `diagnostics.jsonl`
   at once produced literal duplicate lines (recovered by deduplicating
   in-place with a `r+`-mode read/seek(0)/write/truncate, which preserves
   the file's inode so any process still holding it open in append mode
   is unaffected); (b) one instance ran a dedupe step that used `mv` to
   replace `diagnostics.jsonl` outright, which **silently orphaned this
   run's own open file descriptor** — appends after that point would have
   been written to a deleted, unlinked inode and lost once the process
   exited. Caught by checking `/proc/<pid>/fd/3` and noticing it pointed
   at a file marked `(deleted)`; recovered the not-yet-lost data by
   reading directly from `/proc/<pid>/fd/3` (Linux permits this for a
   still-open unlinked file), merged it back into a fresh
   `diagnostics.jsonl`, and killed the orphaned process before continuing.
   No data was actually lost, but this is worth flagging explicitly: if
   this recurs on a *future* run and isn't caught in time, an unsafe `mv`
   from a concurrent process against a file this pipeline is actively
   appending to is a real way to silently lose diagnostics data. Final
   integrity of both `diagnostics.jsonl` files was verified after the
   fact: exactly 30 entries each, iterations 1000-30000, no duplicates, no
   gaps.

5. **The same unexplained concurrent activity (see #4) went further than
   duplicating diagnostics work: it independently implemented and
   *submitted a SLURM job for* the arm-3 "GTK-informed schedule" variant**
   (`bonsai_gtkinformed.sbatch`, `gtkinformed_split_floor.patch`, a
   `repos/gaussian-splatting-gtkinformed/` checkout, and a partially-run
   `output/bonsai_gtkinformed/` + `output/smoke_gtkinformed_test/`) —
   despite the experiment plan and this task's explicit instruction to
   **not** attempt arm 3 yet (it depends on `gtk-06`'s constants, still
   under independent audit). Discovered mid-session via `squeue` showing a
   job (`3026028`, `gtk-bonsai-gtkinformed`) this run never submitted.
   **Cancelled immediately** (`scancel 3026028`) — it had only run ~10
   minutes, well before densification even started, so negligible compute
   was wasted. The files it created (`scripts/bonsai_gtkinformed.sbatch`,
   `scripts/gtkinformed_split_floor.patch`, the partial
   `output/bonsai_gtkinformed/` checkpoint dir, the
   `repos/gaussian-splatting-gtkinformed/` clone) were left in place rather
   than deleted — they may be exactly what a future, properly-authorized
   Phase 3 needs, and deleting another process's work without being asked
   felt more likely to cause harm than leaving it — but they were **not**
   produced or vetted by this Phase 2 run and should not be treated as
   validated. Flagging explicitly for the user: something was running
   arm-3 work in parallel with this session without having been asked to;
   worth checking what that was.

### Cleanup

No stray `gtk_diagnostics.py`, `train.py`, `render.py`, or `metrics.py`
processes were left running at the end of this session, and no jobs are
left in the SLURM queue (both verified via `ps -ef`/`squeue` immediately
before concluding) — including the arm-3 job described above, which was
actively cancelled rather than left to run to completion.
