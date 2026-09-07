#!/usr/bin/env bash
#SBATCH --job-name=gtk3dgs-smoke-softaniso
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/smoke-gtksoftaniso-%j.out
#
# Smoke test for "GTK-SoftAniso": GTK-Floor's split-time floor + coarse-to-fine
# delay, PLUS a differentiable anisotropy-ratio hinge loss added every
# iteration to the ordinary photometric loss (train.py, right before
# loss.backward()) -- the literature-informed alternative to GTK-Guard's
# periodic hard clamp-with-optimizer-reset, which regressed conditioning
# relative to GTK-Floor alone. See train.py's inline comment for the exact
# formula and citations (PhysGaussian CVPR 2024 arXiv:2311.12198; Effective
# Rank Regularization arXiv:2406.11672).
#
# Purpose of THIS run: sanity-check the loss term doesn't destabilize
# training and pick a reasonable lambda_aniso before committing to the full
# 30k-iteration run. Short (1200-iteration), fast (-r 8), same overrides as
# the GTK-Guard smoke test for comparability.

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/gaussian-splatting-gtksoftaniso
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/bonsai
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/smoke_gtksoftaniso_${SLURM_JOB_ID}

export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

source "${VENV}/bin/activate"

echo "=== node / GPU info ==="
hostname
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())"

cd "${REPO}"
mkdir -p "${OUT}"

echo "=== starting GTK-SoftAniso smoke test (3000 iterations, bonsai, -r 4) ==="
python train.py \
  -s "${DATA}" \
  -m "${OUT}" \
  --eval \
  -r 4 \
  --iterations 3000 \
  --densify_from_iter 200 \
  --test_iterations 1500 3000 \
  --save_iterations 3000 \
  --disable_viewer --quiet

echo "=== training exit code: $? ==="

echo "=== losses.jsonl (loss / l1 / l_aniso trajectory) ==="
cat "${OUT}/losses.jsonl" 2>&1

echo "=== output dir contents ==="
find "${OUT}" -maxdepth 2 -type f | sort
echo "=== GTK-SoftAniso smoke test done ==="
