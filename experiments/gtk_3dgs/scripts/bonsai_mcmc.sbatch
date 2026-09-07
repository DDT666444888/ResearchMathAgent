#!/usr/bin/env bash
#SBATCH --job-name=gtk-bonsai-mcmc
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/bonsai-mcmc-%j.out
#
# Phase 2, arm 2/2 on the primary scene: 3DGS-MCMC (Kheradmand et al.,
# NeurIPS'24), full 30k-iteration training. Uses the repo's own
# configs/bonsai.json (resolution=2, cap_max=1300000 -- matching the
# standard Mip-NeRF 360 indoor-scene protocol and the paper's own
# precomputed final-Gaussian-count target), so resolution is identical to
# bonsai_3dgs.sbatch's official-3DGS arm; only the densification rule
# differs, as required by the experiment plan.
#
# 3DGS-MCMC's train.py has NO --disable_viewer flag (its network_gui loop
# is already commented out unconditionally upstream) -- do not pass it here,
# argparse would reject it as unrecognized.
#
# train.py in this checked-out repo already has the 3-line loss-logging
# patch applied (see mcmc_train_loss_logging.patch in this directory).

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/3dgs-mcmc
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk-3dgs-mcmc
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/bonsai
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/bonsai_mcmc

export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

source "${VENV}/bin/activate"

echo "=== node / GPU info ==="
hostname
nvidia-smi -L || true

cd "${REPO}"
mkdir -p "${OUT}"

echo "=== starting bonsai / 3DGS-MCMC, 30k iterations, configs/bonsai.json (-r 2, cap_max=1300000) ==="
python train.py \
  -s "${DATA}" \
  -m "${OUT}" \
  --config configs/bonsai.json \
  --eval \
  --iterations 30000 \
  --test_iterations 7000 30000 \
  --save_iterations 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
                     11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 \
                     21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 \
  --quiet

echo "=== training exit code: $? ==="

echo "=== rendering + metrics at 7000 and 30000 ==="
python render.py -m "${OUT}" --iteration 7000 --skip_train --quiet
python render.py -m "${OUT}" --iteration 30000 --skip_train --quiet
python metrics.py -m "${OUT}"

echo "=== output dir contents ==="
find "${OUT}" -maxdepth 2 -type f | sort
echo "=== bonsai 3DGS-MCMC run done ==="
