#!/usr/bin/env bash
#SBATCH --job-name=gtk-garden-3dgs
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=05:00:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/garden-3dgs-%j.out
#
# Phase 2, secondary scene, arm 1/2: official vanilla 3DGS on `garden`,
# full 30k iterations at the standard Mip-NeRF 360 outdoor-scene resolution
# (-r 4, matching 3DGS-MCMC's configs/garden.json). Headline PSNR/SSIM/LPIPS
# at 7k/30k only -- no per-1000-iteration diagnostic checkpoints/logging
# for this scene (per the experiment plan, to save time); only the 7000
# and 30000 checkpoints are saved.

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/gaussian-splatting
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/garden
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/garden_3dgs

export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

source "${VENV}/bin/activate"

echo "=== node / GPU info ==="
hostname
nvidia-smi -L || true

cd "${REPO}"
mkdir -p "${OUT}"

echo "=== starting garden / vanilla 3DGS, 30k iterations, -r 4 ==="
python train.py \
  -s "${DATA}" \
  -m "${OUT}" \
  --eval \
  -r 4 \
  --iterations 30000 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000 \
  --disable_viewer --quiet

echo "=== training exit code: $? ==="

echo "=== rendering + metrics at 7000 and 30000 ==="
python render.py -m "${OUT}" --iteration 7000 --skip_train --quiet
python render.py -m "${OUT}" --iteration 30000 --skip_train --quiet
python metrics.py -m "${OUT}"

echo "=== output dir contents ==="
find "${OUT}" -maxdepth 2 -type f | sort
echo "=== garden 3DGS run done ==="
