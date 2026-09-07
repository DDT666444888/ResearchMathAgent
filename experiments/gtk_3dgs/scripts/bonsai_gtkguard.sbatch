#!/usr/bin/env bash
#SBATCH --job-name=gtk-bonsai-gtkguard
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/bonsai-gtkguard-%j.out
#
# Phase 4, "GTK-Guard": the arm-3 "GTK-Floor" checkout/patch (floored,
# non-isotropic split + densify_from_iter=1500 coarse-to-fine delay) PLUS one
# new mechanism -- a periodic, direct anisotropy correction applied to the
# ENTIRE current Gaussian population, not just newly-split children. See
# experiments/gtk_3dgs/scripts/gtkguard_anisotropy_clamp.patch for the full
# motivation/derivation (main.tex thm:exact-gaussian-gtk) and the exact
# unified diff (apply with `patch -p1` against a fresh copy of
# repos/gaussian-splatting-gtkinformed's scene/gaussian_model.py + train.py
# to reproduce repos/gaussian-splatting-gtkguard).
#
# In one line: GTK-Floor's split-time floor never revisits a Gaussian's
# scale once it exists; GTK-Guard adds GaussianModel.clamp_anisotropy(),
# called from train.py on its own independent clamp_interval=1000 cadence
# (decoupled from opt.opacity_reset_interval/opt.densify_until_iter, so it
# fires for the FULL 30k-iteration run, not just the densification window --
# see the v1->v2 fix note in gtkguard_anisotropy_clamp.patch), which raises
# (never shrinks) the smallest axis of any Gaussian whose anisotropy ratio
# exceeds tau_aniso=100 (and whose opacity exceeds tau_opacity=0.1), reset
# through replace_tensor_to_optimizer exactly like reset_opacity() does for
# opacity.
#
# This is the v2 (fixed) run. A v1 run existed with the guard mistakenly
# nested inside the densify_until_iter-gated block (same bug class as
# opacity reset's own early stop), which caused it to never actually fire
# (n_clamped=0 at every logged checkpoint) -- its output was archived to
# output/bonsai_gtkguard_v1_buggy_guard/ and excluded from the paper's
# reported comparison.
#
# Everything else held identical to arm 3 (bonsai_gtkinformed.sbatch) and
# arm 1 (bonsai_3dgs.sbatch): -r 2, 30k iterations, densify_from_iter=1500,
# same test/save iteration lists, --eval. Any PSNR/SSIM/LPIPS/lambda_min/
# anisotropy difference vs. GTK-Floor is attributable to clamp_anisotropy
# alone.
#
# This repo checkout (repos/gaussian-splatting-gtkguard) is an
# untracked-by-git copy of repos/gaussian-splatting-gtkinformed (itself a
# copy of the official repo) with gtkguard_anisotropy_clamp.patch applied on
# top of the existing gtkinformed_split_floor.patch; confirmed by diffing
# against the official baseline that only scene/gaussian_model.py and
# train.py differ, and against gaussian-splatting-gtkinformed that the diff
# is exactly this one patch.
#
# Uses the SAME venv as arm 1/3 (venvs/gtk) -- the CUDA rasterizer/simple-knn
# extensions are unmodified (only pure-Python gaussian_model.py/train.py
# changed), so the already-built extensions in venvs/gtk apply without
# rebuilding.

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/gaussian-splatting-gtkguard
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/bonsai
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/bonsai_gtkguard

export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

source "${VENV}/bin/activate"

echo "=== node / GPU info ==="
hostname
nvidia-smi -L || true

cd "${REPO}"
mkdir -p "${OUT}"

echo "=== starting bonsai / GTK-Guard, 30k iterations, -r 2, densify_from_iter=1500 ==="
python train.py \
  -s "${DATA}" \
  -m "${OUT}" \
  --eval \
  -r 2 \
  --iterations 30000 \
  --densify_from_iter 1500 \
  --test_iterations 7000 30000 \
  --save_iterations 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
                     11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 \
                     21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 \
  --disable_viewer --quiet

echo "=== training exit code: $? ==="

echo "=== rendering + metrics at 7000 and 30000 ==="
python render.py -m "${OUT}" --iteration 7000 --skip_train --quiet
python render.py -m "${OUT}" --iteration 30000 --skip_train --quiet
python metrics.py -m "${OUT}"

echo "=== output dir contents ==="
find "${OUT}" -maxdepth 2 -type f | sort
echo "=== bonsai GTK-Guard run done ==="
