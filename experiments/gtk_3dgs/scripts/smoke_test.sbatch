#!/usr/bin/env bash
#SBATCH --job-name=gtk3dgs-smoke
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/smoke-%j.out
#
# Phase 1 smoke test for the GTK/3DGS TPAMI experiment plan
# (drafts/tpami_gtk_extension/EXPERIMENT_PLAN.md). Runs the OFFICIAL 3DGS
# train.py on the `bonsai` Mip-NeRF 360 scene for a short 2500-iteration run
# (not the full 30k) to confirm data loading, the CUDA rasterizer, the
# training loop and checkpoint saving all work end-to-end on this
# ARM64 (GH200) + CUDA 13.1 stack. Not a real training result.

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/gaussian-splatting
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/bonsai
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/smoke_bonsai_${SLURM_JOB_ID}

export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

source "${VENV}/bin/activate"

echo "=== node / GPU info ==="
hostname
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU VISIBLE')"

cd "${REPO}"
mkdir -p "${OUT}"

echo "=== starting short training run (2500 iterations, bonsai, downsampled x4) ==="
python train.py \
  -s "${DATA}" \
  -m "${OUT}" \
  --eval \
  -r 4 \
  --iterations 2500 \
  --test_iterations 1000 2500 \
  --save_iterations 2500 \
  --disable_viewer \
  --quiet

echo "=== training exit code: $? ==="
echo "=== output dir contents ==="
find "${OUT}" -maxdepth 3 -type f | sort

echo "=== smoke test done ==="
