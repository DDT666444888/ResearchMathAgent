#!/usr/bin/env bash
#SBATCH --job-name=gtk-bonsai-3dgs
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/bonsai-3dgs-%j.out
#
# Phase 2, arm 1/2 on the primary scene: official vanilla 3DGS, full
# 30k-iteration training at the standard Mip-NeRF 360 indoor-scene
# resolution (-r 2, matching 3DGS-MCMC's own configs/bonsai.json so both
# arms train at an identical resolution), --eval for the standard
# every-8th-image test split. Checkpoints saved every 1000 iterations for
# the GTK diagnostics (experiments/gtk_3dgs/scripts/gtk_diagnostics.py).
#
# train.py in this checked-out repo already has the 3-line loss-logging
# patch applied (see official_train_loss_logging.patch in this directory;
# `git diff train.py` in repos/gaussian-splatting reproduces it from a
# fresh clone).

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/gaussian-splatting
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/bonsai
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/bonsai_3dgs

export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

source "${VENV}/bin/activate"

echo "=== node / GPU info ==="
hostname
nvidia-smi -L || true

cd "${REPO}"
mkdir -p "${OUT}"

echo "=== starting bonsai / vanilla 3DGS, 30k iterations, -r 2 ==="
python train.py \
  -s "${DATA}" \
  -m "${OUT}" \
  --eval \
  -r 2 \
  --iterations 30000 \
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
echo "=== bonsai 3DGS run done ==="
