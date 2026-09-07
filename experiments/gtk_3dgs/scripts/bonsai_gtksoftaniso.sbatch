#!/usr/bin/env bash
#SBATCH --job-name=gtk-bonsai-softaniso
#SBATCH --account=bhov-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/logs/bonsai-softaniso-%j.out
#
# Fifth arm, "GTK-SoftAniso": literature-informed alternative to GTK-Guard's
# periodic hard clamp. Same base as GTK-Guard/GTK-Floor (floored,
# non-isotropic split + densify_from_iter=1500 coarse-to-fine delay), but
# instead of a periodic clamp_anisotropy() hard edit + Adam-moment reset,
# adds a differentiable anisotropy-ratio hinge loss to the ordinary
# photometric loss every iteration (train.py, right before loss.backward()).
#
# Motivation: GTK-Guard's clamp fired correctly (v2, verified) but still
# regressed conditioning vs. GTK-Floor alone -- diagnosed as the periodic
# Adam-moment-buffer reset perturbing the subsequent gradient trajectory in
# a way that fights the split-time floor's parent-inherited guarantee.
# Literature research (PhysGaussian, CVPR 2024, arXiv:2311.12198; Effective
# Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting,
# arXiv:2406.11672) independently converges on a differentiable
# anisotropy/rank penalty added to the training loss as the standard way to
# discourage needle-like Gaussians -- no hard edit, no optimizer-state
# surgery, just ordinary gradient pressure every step. See
# repos/gaussian-splatting-gtksoftaniso/train.py's inline comment for the
# exact formula: mean over the FLAGGED subset (ratio > tau_aniso=100 AND
# opacity > 0.1) of clamp(ratio - tau_aniso, max=500), weighted by
# (detached) opacity, times lambda_aniso=5e-5. Normalization over the
# flagged subset (not the whole population) was fixed after an initial
# smoke test showed a population-wide mean() diluted the term to ~1e-6 and
# had no real effect.
#
# Smoke-tested (jobs 3038291, 3038297): compiles, trains without error,
# hinge correctly zero while no Gaussian is yet flagged (matches real
# trajectory data -- anisotropy only develops after roughly iteration
# 10,000 in the other four arms), confirming the term is safe-by-construction
# (self-gating) rather than validated at full magnitude, which this full run
# is what actually tests.
#
# Everything else held identical to GTK-Guard/GTK-Floor/vanilla 3DGS: -r 2,
# 30k iterations, densify_from_iter=1500, same test/save iteration lists,
# --eval. Uses the SAME venv as the other arms (venvs/gtk) -- only pure-Python
# gaussian_model.py/train.py changed, no CUDA extension rebuild needed.

set -uo pipefail

REPO=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/repos/gaussian-splatting-gtksoftaniso
VENV=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/venvs/gtk
DATA=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/bonsai
OUT=/work/nvme/bhov/zzhao18/gtk_3dgs_runs/output/bonsai_gtksoftaniso

export CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-13.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

source "${VENV}/bin/activate"

echo "=== node / GPU info ==="
hostname
nvidia-smi -L || true

cd "${REPO}"
mkdir -p "${OUT}"

echo "=== starting bonsai / GTK-SoftAniso, 30k iterations, -r 2, densify_from_iter=1500 ==="
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
echo "=== bonsai GTK-SoftAniso run done ==="
