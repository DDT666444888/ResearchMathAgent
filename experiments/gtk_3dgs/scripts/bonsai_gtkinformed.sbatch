#!/usr/bin/env bash
#SBATCH --job-name=gtk-bonsai-gtkinformed
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/bonsai-gtkinformed-%j.out
#
# Phase 3, arm 3/3 on the primary scene: "3DGS + GTK-informed schedule" --
# the paper's own theory-motivated intervention on top of official 3DGS,
# with everything else (opacity pruning threshold, clone logic, loss,
# iteration budget, all other hyperparameters) held identical to arm 1
# (bonsai_3dgs.sbatch) so any PSNR/SSIM/LPIPS/lambda_min difference is
# attributable to these two changes:
#
#  1. Floored, non-isotropic split (thm:density-safe clause S2). Vanilla
#     3DGS's densify_and_split shrinks a split Gaussian's scale isotropically
#     on all 3 axes by 1/(0.8*N). We instead keep that shrink for the two
#     LARGER axes but clamp the smallest axis of each child to never drop
#     below the PARENT's own pre-split sigma_min on that axis -- a local,
#     self-referential floor (s_0 read off the parent at each split) with no
#     new global hyperparameter, implemented in
#     repos/gaussian-splatting-gtkinformed/scene/gaussian_model.py
#     (patch: experiments/gtk_3dgs/scripts/gtkinformed_split_floor.patch,
#     applies cleanly with `patch -p1` against a fresh gaussian-splatting
#     clone's scene/gaussian_model.py).
#
#  2. Coarse-to-fine delay (thm:coarse-to-fine). Vanilla 3DGS's default
#     densify_from_iter is 500 (densification/splitting can start at
#     iteration 501). Here we push it back to 1500 -- 1000 extra iterations
#     (3x the default) spent at the coarser, undensified point cloud before
#     the first split fires -- via the existing --densify_from_iter CLI flag
#     (no source change needed for this half). densify_until_iter (15000),
#     densification_interval (100), densify_grad_threshold (0.0002), and the
#     opacity_reset_interval / opacity-prune threshold are all left at
#     official 3DGS defaults, same as arm 1.
#
# This repo checkout is an untracked-by-git rsync copy of
# repos/gaussian-splatting (the arm-1 checkout) with only
# scene/gaussian_model.py modified per the patch above; train.py already
# carries the same loss-logging patch as arm 1/2
# (official_train_loss_logging.patch), confirmed identical.
#
# Uses the SAME venv as arm 1 (venvs/gtk) -- the CUDA rasterizer/simple-knn
# extensions are unmodified (only pure-Python gaussian_model.py changed), so
# the already-built extensions in venvs/gtk apply without rebuilding.

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/gaussian-splatting-gtkinformed
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/bonsai
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/bonsai_gtkinformed

export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

source "${VENV}/bin/activate"

echo "=== node / GPU info ==="
hostname
nvidia-smi -L || true

cd "${REPO}"
mkdir -p "${OUT}"

echo "=== starting bonsai / GTK-informed 3DGS, 30k iterations, -r 2, densify_from_iter=1500 ==="
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
echo "=== bonsai GTK-informed run done ==="
