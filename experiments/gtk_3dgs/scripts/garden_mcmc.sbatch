#!/usr/bin/env bash
#SBATCH --job-name=gtk-garden-mcmc
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=05:00:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/garden-mcmc-%j.out
#
# Phase 2, secondary scene, arm 2/2: 3DGS-MCMC on `garden`, full 30k
# iterations, configs/garden.json (resolution=4, cap_max=5200000). Headline
# PSNR/SSIM/LPIPS at 7k/30k only -- no per-1000-iteration diagnostics for
# this scene; only the 7000 and 30000 checkpoints are saved.

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/3dgs-mcmc
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk-3dgs-mcmc
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/garden
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/garden_mcmc

export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

source "${VENV}/bin/activate"

echo "=== node / GPU info ==="
hostname
nvidia-smi -L || true

cd "${REPO}"
mkdir -p "${OUT}"

echo "=== starting garden / 3DGS-MCMC, 30k iterations, configs/garden.json (-r 4, cap_max=5200000) ==="
python train.py \
  -s "${DATA}" \
  -m "${OUT}" \
  --config configs/garden.json \
  --eval \
  --iterations 30000 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000 \
  --quiet

echo "=== training exit code: $? ==="

echo "=== rendering + metrics at 7000 and 30000 ==="
python render.py -m "${OUT}" --iteration 7000 --skip_train --quiet
python render.py -m "${OUT}" --iteration 30000 --skip_train --quiet
python metrics.py -m "${OUT}"

echo "=== output dir contents ==="
find "${OUT}" -maxdepth 2 -type f | sort
echo "=== garden 3DGS-MCMC run done ==="
