#!/usr/bin/env python3
"""Standalone GTK (Gaussian Tangent Kernel) diagnostics for 3D Gaussian
Splatting checkpoints.

This script is completely independent of which training arm (vanilla 3DGS,
3DGS-MCMC, or any future variant) produced a checkpoint: it only reads the
standard 3DGS-format `point_cloud.ply` files and the `cameras.json` that the
official codebase (and every fork derived from it, including 3DGS-MCMC)
already writes to a run's output directory. No torch/CUDA dependency --
pure numpy + scipy.sparse, so it runs on a CPU-only login node.

It is released as part of this experiment's Data Availability Statement
(see drafts/tpami_gtk_extension/) as the exact tool used to produce the
paper's theory-diagnostic numbers; see EXPERIMENT_PLAN.md for the precise
definitions being computed here.

For each requested training iteration (a saved `point_cloud/iteration_N/`
checkpoint), it computes and appends one JSON line to an output
`diagnostics.jsonl`:

  - iteration
  - lambda_min: smallest eigenvalue of the empirical Gram matrix Z Z^T,
    where Z (subset_size x num_touching_gaussians) holds each Gaussian's
    raw per-pixel alpha-weighted contribution value
    phi_i(x) = opacity_i * exp(power_i(x))  (the exact per-pixel EWA
    splatting weight the CUDA rasterizer itself computes, minus the
    running-transmittance/compositing factor -- see below), evaluated at a
    FIXED subset of training rays/pixels (default 2048), sampled once with
    numpy.random.default_rng(0) and cached to disk so every checkpoint and
    every arm reuses the identical subset.
  - gershgorin_bound: min_a( Gram_aa - sum_{b!=a} |Gram_ab| ), the exact
    Gershgorin-circle-theorem lower bound on lambda_min from this same Gram
    matrix (main.tex thm:exact-gaussian-gtk / eq:gershgorin) -- needs no
    separation hypothesis, unlike thm:aniso-eigenvalue's private-node bound,
    and (per the same theorem) is exact rather than a bound on the entries
    themselves, since these Z values already are the true per-pixel EWA
    weights. Always <= lambda_min; how close the two are is itself a
    diagnostic of how well-conditioned the local neighborhood is.
  - num_gaussians_touching_subset: matrix rank upper bound / sanity check.
  - alpha_min, alpha_p1, alpha_p5, alpha_p50: opacity floor + percentiles
    over all currently-active Gaussians.
  - n_gaussians: population size (post prune/split at this checkpoint).
  - s_min: population minimum of each Gaussian's smallest covariance
    principal std-dev (sigma_min,i = min over the 3 axes of its `scale`
    parameters -- since Sigma_world = R diag(scale^2) R^T, R orthogonal,
    the covariance eigenvalues are exactly scale_x^2, scale_y^2, scale_z^2,
    independent of the rotation).
  - anisotropy_median, anisotropy_p95: median / 95th percentile of the
    per-Gaussian ratio sigma_max,i / sigma_min,i.
  - loss, l1_loss: merged in from the training run's losses.jsonl if that
    iteration is present there (see the *_train_loss_logging.patch files
    in this directory), else null.

Approximations, stated explicitly (this is a *proxy*, not the exact
infinite-dimensional GTK, per the experiment plan):
  - No frustum-edge clamp on the projected point before building the
    perspective Jacobian J (the CUDA kernel clamps to 1.3x the frustum
    half-angle for numerical stability at extreme edges only; skipping it
    has negligible effect for pixels that aren't at the very edge of frame,
    which a uniformly random pixel subset is very unlikely to concentrate
    on).
  - The always-on +0.3 diagonal "low-pass filter" dilation the rasterizer
    applies to the 2D covariance (independent of the `antialiasing` flag,
    which is off by default in both repos and not requested here) IS
    replicated, since it's part of what phi_i(x) actually is at render
    time.
  - phi_i(x) is the raw per-Gaussian alpha value, not multiplied by
    accumulated transmittance from front-to-back compositing -- this
    matches the task's specification of using the per-pixel alpha-weighted
    contribution "opacity * Gaussian falloff" as the GTK's Z, not a full
    differentiable-render Jacobian.

Usage:
    python gtk_diagnostics.py \\
        --run-dir /path/to/output/bonsai_3dgs \\
        --data-dir /path/to/data/bonsai \\
        --iterations 1000 2000 3000 ... 30000 \\
        --subset-cache /path/to/data/bonsai/gtk_ray_subset_2048.json \\
        --out /path/to/output/bonsai_3dgs/diagnostics.jsonl

Or omit --iterations to auto-discover every point_cloud/iteration_* found
under --run-dir.
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np
import scipy.sparse as sp
from plyfile import PlyData


# ---------------------------------------------------------------------------
# Minimal standalone COLMAP images.bin reader (name-only; we don't need the
# poses here since cameras.json already carries them in the exact convention
# training used). Self-contained so this script has zero repo dependency.
# ---------------------------------------------------------------------------
def _norm_name(name):
    """Normalize an image name for matching across repos: some forks'
    camera_to_JSON path stores the extension-stripped name (e.g. 3DGS-MCMC
    writes 'DSCF5565' where the official repo writes 'DSCF5565.JPG', both
    for the exact same image) -- confirmed by direct inspection of both
    repos' cameras.json for the same run. Strip any extension so lookups
    work regardless of which convention produced a given run's cameras.json."""
    return os.path.splitext(name)[0]


def read_colmap_image_names(images_bin_path):
    names = []
    with open(images_bin_path, "rb") as f:
        num_reg_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg_images):
            f.read(4)  # image_id
            f.read(8 * 4)  # qvec (4 doubles)
            f.read(8 * 3)  # tvec (3 doubles)
            f.read(4)  # camera_id
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            names.append(name_bytes.decode("utf-8"))
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * (8 * 2 + 8))  # (x,y double) + point3D_id int64
    return names


def read_resolution_factor(run_dir):
    """cameras.json always reports FULL original-image intrinsics (it is
    built from the raw un-downsampled CameraInfo, not the resolution-scaled
    `Camera` object -- confirmed by inspecting scene/__init__.py and
    scene/cameras.py). The actual downsampling factor training used (the
    `-r`/`--resolution` CLI flag) is only recoverable from the run's saved
    `cfg_args`. We must re-derive the true training-resolution width/height/
    fx/fy ourselves -- see apply_resolution_scaling()."""
    cfg_path = os.path.join(run_dir, "cfg_args")
    if not os.path.exists(cfg_path):
        return 1, None
    text = open(cfg_path).read()
    m = re.search(r"resolution=(-?\d+)", text)
    if not m:
        return 1, None
    return int(m.group(1)), text


def apply_resolution_scaling(cams_by_name, resolution_arg):
    """Mutates cams_by_name in place: width/height/fx/fy -> the actual
    training-resolution values, reproducing utils/camera_utils.py::loadCam's
    `round(orig_w/args.resolution)` (for resolution in {1,2,4,8}) or the
    orig_w>1600 -> 1600px heuristic (for the default resolution=-1)."""
    for c in cams_by_name.values():
        orig_w, orig_h = c["width"], c["height"]
        if resolution_arg in (1, 2, 4, 8):
            w = round(orig_w / resolution_arg)
            h = round(orig_h / resolution_arg)
        elif resolution_arg == -1:
            global_down = orig_w / 1600.0 if orig_w > 1600 else 1.0
            w = int(orig_w / global_down)
            h = int(orig_h / global_down)
        else:
            scale = orig_w / float(resolution_arg)
            w = int(orig_w / scale)
            h = int(orig_h / scale)
        c["fx"] = c["fx"] * w / orig_w
        c["fy"] = c["fy"] * h / orig_h
        c["width"] = w
        c["height"] = h


def determine_train_image_names(data_dir, llffhold=8):
    """Reproduces scene/dataset_readers.py::readColmapSceneInfo's split:
    sort all COLMAP image names alphabetically, every `llffhold`-th (by
    that sorted index) is held out as test; everything else is train."""
    images_bin = os.path.join(data_dir, "sparse", "0", "images.bin")
    names = sorted(read_colmap_image_names(images_bin))
    train = [n for i, n in enumerate(names) if i % llffhold != 0]
    return train


# ---------------------------------------------------------------------------
# PLY checkpoint loading (raw params -> activated values)
# ---------------------------------------------------------------------------
def load_gaussians(ply_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    opacity_raw = np.asarray(v["opacity"], dtype=np.float64)
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid
    scale_raw = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float64)
    scale = np.exp(scale_raw)
    rot_raw = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)
    rot = rot_raw / np.linalg.norm(rot_raw, axis=1, keepdims=True)
    return xyz, opacity, scale, rot


def quat_to_R_batch(q):
    """q: (M,4) = (r,x,y,z) normalized. Returns (M,3,3) standard active
    rotation matrices (verified against the CUDA computeCov3D formula --
    see docstring / SETUP_NOTES.md for the derivation)."""
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ---------------------------------------------------------------------------
# Fixed pixel-subset sampling / caching
# ---------------------------------------------------------------------------
def build_or_load_subset(cache_path, train_names, cams_by_name, n_samples=2048, seed=0):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    rng = np.random.default_rng(seed)
    names_present = [n for n in train_names if _norm_name(n) in cams_by_name]
    if not names_present:
        raise RuntimeError("No train image names from COLMAP match cameras.json img_name entries")
    subset = []
    for _ in range(n_samples):
        name = names_present[rng.integers(0, len(names_present))]
        cam = cams_by_name[_norm_name(name)]
        u = rng.uniform(0, cam["width"])
        v = rng.uniform(0, cam["height"])
        subset.append({"img_name": name, "u": float(u), "v": float(v)})
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(subset, f)
    return subset


# ---------------------------------------------------------------------------
# GTK lambda_min proxy
# ---------------------------------------------------------------------------
def compute_lambda_min_proxy(xyz, opacity, scale, rot, cams_by_name, subset,
                              margin_px=64.0, znear=0.2, alpha_cut=1.0 / 255.0):
    N = xyz.shape[0]
    n_pix = len(subset)
    rows, cols, vals = [], [], []

    by_image = {}
    for row_idx, s in enumerate(subset):
        by_image.setdefault(s["img_name"], []).append((row_idx, s["u"], s["v"]))

    for img_name, pix_list in by_image.items():
        cam = cams_by_name[_norm_name(img_name)]
        position = np.array(cam["position"], dtype=np.float64)
        rotation = np.array(cam["rotation"], dtype=np.float64)  # C2W rotation, rows
        fx, fy = cam["fx"], cam["fy"]
        width, height = cam["width"], cam["height"]

        # world -> camera: x_cam = rotation.T @ (X - position)   (row-vector form: (X-pos) @ rotation)
        Xc = (xyz - position[None, :]) @ rotation  # (N,3)
        z = Xc[:, 2]
        infrustum = z > znear
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * Xc[:, 0] / np.where(infrustum, z, 1.0) + width / 2.0
            v = fy * Xc[:, 1] / np.where(infrustum, z, 1.0) + height / 2.0
        inframe = infrustum & (u > -margin_px) & (u < width + margin_px) & \
                  (v > -margin_px) & (v < height + margin_px)
        idx_frustum = np.nonzero(inframe)[0]
        if idx_frustum.size == 0:
            continue
        u_f, v_f, z_f = u[idx_frustum], v[idx_frustum], z[idx_frustum]
        Xc_f = Xc[idx_frustum]

        W = rotation.T  # world-to-camera rotation (3x3), constant for this image

        for row_idx, u0, v0 in pix_list:
            box = (np.abs(u_f - u0) < margin_px) & (np.abs(v_f - v0) < margin_px)
            cand = np.nonzero(box)[0]
            if cand.size == 0:
                continue
            gidx = idx_frustum[cand]  # global gaussian indices

            R_c = quat_to_R_batch(rot[gidx])            # (M,3,3)
            s_c = scale[gidx]                             # (M,3)
            Sigma3D = np.einsum("mij,mj,mkj->mik", R_c, s_c ** 2, R_c)  # R diag(s^2) R^T, (M,3,3)

            tz = z_f[cand]
            tx = Xc_f[cand, 0]
            ty = Xc_f[cand, 1]
            M = cand.size
            J = np.zeros((M, 3, 3), dtype=np.float64)
            J[:, 0, 0] = fx / tz
            J[:, 0, 2] = -fx * tx / (tz * tz)
            J[:, 1, 1] = fy / tz
            J[:, 1, 2] = -fy * ty / (tz * tz)

            T_mat = np.einsum("ij,mjk->mik", W, J)  # W @ J, per-gaussian, (M,3,3)
            # cov2D_3x3 = T^T @ Sigma3D @ T
            tmp = np.einsum("mji,mjk->mik", T_mat, Sigma3D)   # T^T @ Sigma3D
            cov3x3 = np.einsum("mij,mjk->mik", tmp, T_mat)    # (.) @ T
            cxx = cov3x3[:, 0, 0] + 0.3
            cxy = cov3x3[:, 0, 1]
            cyy = cov3x3[:, 1, 1] + 0.3

            det = cxx * cyy - cxy * cxy
            valid = det > 1e-12
            if not np.any(valid):
                continue
            det_inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
            a = cyy * det_inv
            b = -cxy * det_inv
            c = cxx * det_inv

            dx = u_f[cand] - u0
            dy = v_f[cand] - v0
            power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
            alpha = opacity[gidx] * np.exp(np.minimum(power, 0.0))
            alpha = np.where(power > 0.0, 0.0, alpha)
            alpha = np.minimum(alpha, 0.99)
            keep = valid & (alpha >= alpha_cut)
            if not np.any(keep):
                continue
            rows.extend([row_idx] * int(keep.sum()))
            cols.extend(gidx[keep].tolist())
            vals.extend(alpha[keep].tolist())

    if not rows:
        return None, 0, None

    Z = sp.coo_matrix((vals, (rows, cols)), shape=(n_pix, N)).tocsr()
    touching_cols = Z.getnnz(axis=0) > 0
    n_touch = int(touching_cols.sum())
    Gram = (Z @ Z.T).toarray()
    eigvals = np.linalg.eigvalsh(Gram)
    # Gershgorin certificate (main.tex thm:exact-gaussian-gtk / eq:gershgorin):
    # min_a( G_aa - sum_{b!=a} |G_ab| ), a lower bound on lambda_min that needs
    # no separation hypothesis, computed directly from this same exact Gram
    # matrix (this proxy's Z entries already ARE the exact per-pixel EWA
    # splatting weights, so no approximation is introduced beyond the same
    # 2048-ray subset every other diagnostic here already uses).
    row_abs_sum = np.abs(Gram).sum(axis=1) - np.abs(np.diag(Gram))
    gershgorin = float(np.min(np.diag(Gram) - row_abs_sum))
    return float(eigvals[0]), n_touch, gershgorin


# ---------------------------------------------------------------------------
# Opacity / anisotropy diagnostics (no camera needed)
# ---------------------------------------------------------------------------
def opacity_anisotropy_stats(opacity, scale):
    alpha_min = float(opacity.min())
    p1, p5, p50 = np.percentile(opacity, [1, 5, 50])
    s_min_per_g = scale.min(axis=1)
    s_max_per_g = scale.max(axis=1)
    s_min_pop = float(s_min_per_g.min())
    ratio = s_max_per_g / np.maximum(s_min_per_g, 1e-12)
    aniso_median = float(np.median(ratio))
    aniso_p95 = float(np.percentile(ratio, 95))
    return {
        "alpha_min": alpha_min, "alpha_p1": float(p1), "alpha_p5": float(p5),
        "alpha_p50": float(p50), "s_min": s_min_pop,
        "anisotropy_median": aniso_median, "anisotropy_p95": aniso_p95,
        "n_gaussians": int(opacity.shape[0]),
    }


def load_losses(run_dir):
    path = os.path.join(run_dir, "losses.jsonl")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out[d["iteration"]] = d
    return out


def discover_iterations(run_dir):
    pc_dir = os.path.join(run_dir, "point_cloud")
    its = []
    if os.path.isdir(pc_dir):
        for name in os.listdir(pc_dir):
            if name.startswith("iteration_"):
                try:
                    its.append(int(name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
    return sorted(its)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="model output dir (contains cameras.json, point_cloud/, losses.jsonl)")
    ap.add_argument("--data-dir", required=True, help="scene COLMAP data dir (contains sparse/0/images.bin)")
    ap.add_argument("--iterations", type=int, nargs="*", default=None,
                     help="iterations to process; default = all found under run-dir/point_cloud")
    ap.add_argument("--subset-cache", required=True, help="path to cache the fixed 2048-pixel subset (shared across arms/checkpoints for one scene)")
    ap.add_argument("--out", required=True, help="output diagnostics.jsonl (appended)")
    ap.add_argument("--n-samples", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "cameras.json")) as f:
        cams = json.load(f)
    cams_by_name = {_norm_name(c["img_name"]): c for c in cams}
    resolution_arg, _ = read_resolution_factor(args.run_dir)
    apply_resolution_scaling(cams_by_name, resolution_arg)
    print(f"[info] training resolution arg = {resolution_arg}; "
          f"e.g. {next(iter(cams_by_name.values()))['width']}x"
          f"{next(iter(cams_by_name.values()))['height']} after scaling", file=sys.stderr)

    train_names = determine_train_image_names(args.data_dir)
    subset = build_or_load_subset(args.subset_cache, train_names, cams_by_name,
                                   n_samples=args.n_samples, seed=args.seed)

    iterations = args.iterations if args.iterations else discover_iterations(args.run_dir)
    if not iterations:
        print("No checkpoints found.", file=sys.stderr)
        sys.exit(1)

    losses = load_losses(args.run_dir)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as out_f:
        for it in iterations:
            ply_path = os.path.join(args.run_dir, "point_cloud", f"iteration_{it}", "point_cloud.ply")
            if not os.path.exists(ply_path):
                print(f"[skip] iteration {it}: no checkpoint at {ply_path}", file=sys.stderr)
                continue
            xyz, opacity, scale, rot = load_gaussians(ply_path)
            stats = opacity_anisotropy_stats(opacity, scale)
            lam_min, n_touch, gershgorin = compute_lambda_min_proxy(
                xyz, opacity, scale, rot, cams_by_name, subset)
            rec = {"iteration": it, "lambda_min": lam_min, "gershgorin_bound": gershgorin,
                   "num_gaussians_touching_subset": n_touch}
            rec.update(stats)
            loss_rec = losses.get(it)
            rec["loss"] = loss_rec["loss"] if loss_rec else None
            rec["l1_loss"] = loss_rec["l1_loss"] if loss_rec else None
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            print(f"[done] iter {it}: lambda_min={lam_min} gershgorin={gershgorin} "
                  f"n_gaussians={stats['n_gaussians']} "
                  f"alpha_min={stats['alpha_min']:.4g} s_min={stats['s_min']:.4g}", file=sys.stderr)


if __name__ == "__main__":
    main()
