#!/usr/bin/env bash
#SBATCH --job-name=gtk3dgs-smoke-gtkguard
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/smoke-gtkguard-%j.out
#
# Phase 4 smoke test for GTK-Guard (experiments/gtk_3dgs/scripts/
# gtkguard_anisotropy_clamp.patch), analogous to Phase 1/3's smoke tests.
# Short (1200-iteration), fast (-r 8) run against repos/gaussian-splatting-
# gtkguard, with --opacity_reset_interval overridden down to 600 (from the
# default 3000) and --densify_from_iter overridden down to 200 so
# clamp_anisotropy() actually fires (at iter 600 and 1200) within this short
# a run and its integration can be sanity-checked cheaply -- these overrides
# are ONLY for this smoke test, not used in the real 30k-iteration run
# (bonsai_gtkguard.sbatch), which keeps all defaults except
# densify_from_iter=1500 per GTK-Floor's coarse-to-fine change.

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/gaussian-splatting-gtkguard
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/bonsai
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/smoke_gtkguard_${SLURM_JOB_ID}

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

echo "=== starting GTK-Guard smoke test (1200 iterations, bonsai, -r 8) ==="
python train.py \
  -s "${DATA}" \
  -m "${OUT}" \
  --eval \
  -r 8 \
  --iterations 1200 \
  --densify_from_iter 200 \
  --opacity_reset_interval 600 \
  --test_iterations 600 1200 \
  --save_iterations 1200 \
  --disable_viewer --quiet

echo "=== training exit code: $? ==="

echo "=== output dir contents ==="
find "${OUT}" -maxdepth 2 -type f | sort
echo "=== GTK-Guard smoke test done ==="
