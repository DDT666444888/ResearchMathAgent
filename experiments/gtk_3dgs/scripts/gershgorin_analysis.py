#!/usr/bin/env python3
"""Gershgorin-certificate ANALYSIS tool (not a training signal).

Companion to gtk_diagnostics.py, reusing its loaders. Where
gtk_diagnostics.py's `gershgorin_bound` field answers "is the whole
evaluation batch well-conditioned" (and, on the standard 2048-ray batch,
mostly answers "no" -- see TPAMI_COMPLIANCE_CHECKLIST.md), this script
answers a different, analysis-shaped question: on a small, deliberately
well-separated set of anchor rays, what does the EXACT closed-form GTK
(main.tex thm:exact-gaussian-gtk) say about local conditioning, and WHICH
specific Gaussians are responsible when it's poor?

Anchor selection: farthest-point sampling on camera position (greedy: start
from one image, repeatedly add the image whose distance to the nearest
already-chosen image is largest), one anchor ray per selected image (image
center pixel). This is a deliberate analysis choice, not a proxy for a large
random evaluation batch -- see the "why not just use gtk_diagnostics.py's
2048-ray batch" note in main.tex sec:gaussian-exact.

Usage:
  python gershgorin_analysis.py --run-dir <arm output dir> --data-dir <scene> \
      --n-anchors 24 --iteration 30000 [--attribute-worst-row]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtk_diagnostics import (  # noqa: E402
    _norm_name, apply_resolution_scaling, load_gaussians, quat_to_R_batch,
    read_resolution_factor,
)


def farthest_point_anchor_images(cams_by_name, n_anchors, seed=0):
    names = sorted(cams_by_name.keys())
    positions = np.array([cams_by_name[n]["position"] for n in names])
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(0, len(names)))]
    min_dist = np.linalg.norm(positions - positions[chosen[0]], axis=1)
    while len(chosen) < min(n_anchors, len(names)):
        nxt = int(np.argmax(min_dist))
        chosen.append(nxt)
        d = np.linalg.norm(positions - positions[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return [names[i] for i in chosen]


def grid_anchor_points(cams_by_name, img_name, grid_n=6, margin_frac=0.15, patch_px=None):
    """Localized-analysis anchor set: a grid_n x grid_n grid of pixels, as
    (img_name, u, v) tuples. Unlike farthest-point-across-images anchors
    (which barely interact -- see main.tex sec:gaussian-exact discussion),
    pixels close together within one image genuinely share nearby Gaussians,
    so this is where the certificate's per-node attribution has something to
    say -- but only if the grid spacing is smaller than the ~margin_px
    candidate-gathering window build_exact_gram_and_attrib uses (default 64px):
    a grid spread over the *whole* image (margin_frac-based) puts points too
    far apart to ever share a Gaussian, same failure mode as too-large a
    random batch. If `patch_px` is given, the grid instead spans a
    patch_px x patch_px square centered on the image, which is the setting
    that actually produces off-diagonal interaction."""
    cam = cams_by_name[img_name]
    w, h = cam["width"], cam["height"]
    if patch_px is not None:
        cu, cv = w / 2.0, h / 2.0
        us = np.linspace(cu - patch_px / 2.0, cu + patch_px / 2.0, grid_n)
        vs = np.linspace(cv - patch_px / 2.0, cv + patch_px / 2.0, grid_n)
        return [(img_name, float(u), float(v)) for v in vs for u in us]
    us = np.linspace(w * margin_frac, w * (1 - margin_frac), grid_n)
    vs = np.linspace(h * margin_frac, h * (1 - margin_frac), grid_n)
    return [(img_name, float(u), float(v)) for v in vs for u in us]


def build_exact_gram_and_attrib(xyz, opacity, scale, rot, cams_by_name, anchors,
                                 margin_px=64.0, znear=0.2, alpha_cut=1.0 / 255.0):
    """`anchors`: list of (img_name, u0, v0) tuples -- either whole-image
    centers (farthest_point_anchor_images) or a within-image grid
    (grid_anchor_points). Returns the exact K x K Gram matrix (K = number of
    anchors that actually hit at least one Gaussian) plus, per anchor, the
    dict of {gaussian_idx: contribution} feeding that anchor's own diagonal
    (for attribution)."""
    K_names = []
    per_anchor_contribs = []  # list of dict{gaussian_idx: alpha_i^2 * exp(power)} per anchor
    per_anchor_meta = []      # (img_name, u0, v0)

    # group by image so the per-image camera-projection work is shared
    by_image = {}
    for img_name, u0, v0 in anchors:
        by_image.setdefault(img_name, []).append((u0, v0))

    for img_name, uv_list in by_image.items():
        cam = cams_by_name[img_name]
        position = np.array(cam["position"], dtype=np.float64)
        rotation = np.array(cam["rotation"], dtype=np.float64)
        fx, fy = cam["fx"], cam["fy"]
        width, height = cam["width"], cam["height"]

        Xc = (xyz - position[None, :]) @ rotation
        z = Xc[:, 2]
        infrustum = z > znear
        with np.errstate(divide="ignore", invalid="ignore"):
            u_all = fx * Xc[:, 0] / np.where(infrustum, z, 1.0) + width / 2.0
            v_all = fy * Xc[:, 1] / np.where(infrustum, z, 1.0) + height / 2.0
        idx_frustum = np.nonzero(infrustum)[0]
        if idx_frustum.size == 0:
            continue

        # per-Gaussian projected 2D covariance (EWA splatting), computed once
        # per image and reused for every anchor pixel in this image.
        R_c = quat_to_R_batch(rot[idx_frustum])
        s_c = scale[idx_frustum]
        Sigma3D = np.einsum("mij,mj,mkj->mik", R_c, s_c ** 2, R_c)
        W = rotation.T
        tz, tx, ty = Xc[idx_frustum, 2], Xc[idx_frustum, 0], Xc[idx_frustum, 1]
        M = idx_frustum.size
        J = np.zeros((M, 3, 3), dtype=np.float64)
        J[:, 0, 0] = fx / tz
        J[:, 0, 2] = -fx * tx / (tz * tz)
        J[:, 1, 1] = fy / tz
        J[:, 1, 2] = -fy * ty / (tz * tz)
        T_mat = np.einsum("ij,mjk->mik", W, J)
        tmp = np.einsum("mji,mjk->mik", T_mat, Sigma3D)
        cov3x3 = np.einsum("mij,mjk->mik", tmp, T_mat)
        cxx, cxy, cyy = cov3x3[:, 0, 0] + 0.3, cov3x3[:, 0, 1], cov3x3[:, 1, 1] + 0.3
        det = cxx * cyy - cxy * cxy
        valid_f = det > 1e-12
        det_inv = np.where(valid_f, 1.0 / np.where(valid_f, det, 1.0), 0.0)
        a_f, b_f, c_f = cyy * det_inv, -cxy * det_inv, cxx * det_inv
        u_f, v_f = u_all[idx_frustum], v_all[idx_frustum]
        opacity_f = opacity[idx_frustum]

        for u0, v0 in uv_list:
            box = valid_f & (np.abs(u_f - u0) < margin_px) & (np.abs(v_f - v0) < margin_px)
            cand = np.nonzero(box)[0]
            if cand.size == 0:
                continue
            dx, dy = u_f[cand] - u0, v_f[cand] - v0
            power = -0.5 * (a_f[cand] * dx * dx + c_f[cand] * dy * dy) - b_f[cand] * dx * dy
            alpha = opacity_f[cand] * np.exp(np.minimum(power, 0.0))
            alpha = np.where(power > 0.0, 0.0, alpha)
            alpha = np.minimum(alpha, 0.99)
            keep = alpha >= alpha_cut
            if not np.any(keep):
                continue
            gidx = idx_frustum[cand[keep]]
            vals = alpha[keep]
            K_names.append(img_name)
            per_anchor_meta.append((img_name, u0, v0))
            per_anchor_contribs.append(dict(zip(gidx.tolist(), vals.tolist())))

    K = len(K_names)
    Gram = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            common = set(per_anchor_contribs[a]) & set(per_anchor_contribs[b])
            Gram[a, b] = sum(per_anchor_contribs[a][i] * per_anchor_contribs[b][i] for i in common)
    return Gram, K_names, per_anchor_meta, per_anchor_contribs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--mode", choices=["images", "grid"], default="grid",
                     help="'images': one anchor per far-apart camera view (global, tends to be trivially "
                          "well-conditioned -- see analysis notes). 'grid': a grid of pixels within ONE "
                          "image (local, where attribution is actually informative). Default: grid.")
    ap.add_argument("--n-anchors", type=int, default=24, help="used in --mode images")
    ap.add_argument("--grid-n", type=int, default=6, help="grid is grid-n x grid-n, used in --mode grid")
    ap.add_argument("--grid-image", default=None, help="image name for --mode grid; default: first anchor image")
    ap.add_argument("--patch-px", type=float, default=None,
                     help="if set, --mode grid spans a patch_px x patch_px square at image center "
                          "instead of the whole image -- needed for anchors close enough to interact")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attribute-worst-row", action="store_true")
    ap.add_argument("--anchor-cache", default=None,
                     help="path to save/load the chosen anchors, so all arms use the identical set")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "cameras.json")) as f:
        cams = json.load(f)
    cams_by_name = {_norm_name(c["img_name"]): c for c in cams}
    resolution_arg, _ = read_resolution_factor(args.run_dir)
    apply_resolution_scaling(cams_by_name, resolution_arg)

    if args.anchor_cache and os.path.exists(args.anchor_cache):
        anchors = [tuple(a) for a in json.load(open(args.anchor_cache))]
        print(f"[info] loaded {len(anchors)} cached anchors from {args.anchor_cache}", file=sys.stderr)
    elif args.mode == "images":
        anchor_images = farthest_point_anchor_images(cams_by_name, args.n_anchors, args.seed)
        anchors = [(nm, cams_by_name[nm]["width"] / 2.0, cams_by_name[nm]["height"] / 2.0) for nm in anchor_images]
        if args.anchor_cache:
            json.dump(anchors, open(args.anchor_cache, "w"))
            print(f"[info] saved {len(anchors)} anchors to {args.anchor_cache}", file=sys.stderr)
    else:
        img_name = args.grid_image or sorted(cams_by_name.keys())[0]
        anchors = grid_anchor_points(cams_by_name, img_name, args.grid_n, patch_px=args.patch_px)
        if args.anchor_cache:
            json.dump(anchors, open(args.anchor_cache, "w"))
            print(f"[info] saved {len(anchors)} anchors ({img_name}, {args.grid_n}x{args.grid_n} grid) to {args.anchor_cache}", file=sys.stderr)

    ply_path = os.path.join(args.run_dir, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    xyz, opacity, scale, rot = load_gaussians(ply_path)

    Gram, K_names, meta, contribs = build_exact_gram_and_attrib(xyz, opacity, scale, rot, cams_by_name, anchors)
    K = len(K_names)
    true_lmin = float(np.linalg.eigvalsh(Gram)[0]) if K > 0 else None
    row_abs_sum = np.abs(Gram).sum(axis=1) - np.abs(np.diag(Gram))
    margins = np.diag(Gram) - row_abs_sum
    gersh = float(np.min(margins)) if K > 0 else None
    worst_row = int(np.argmin(margins)) if K > 0 else None

    result = {
        "run_dir": args.run_dir, "iteration": args.iteration, "mode": args.mode,
        "n_anchors_hit": K, "true_lambda_min": true_lmin, "gershgorin_bound": gersh,
        "worst_anchor": meta[worst_row] if worst_row is not None else None,
        "worst_anchor_margin": float(margins[worst_row]) if worst_row is not None else None,
        "diag_mean": float(np.mean(np.diag(Gram))) if K > 0 else None,
        "off_diag_mean_abs": float(np.mean(row_abs_sum)) if K > 0 else None,
    }

    if args.attribute_worst_row and worst_row is not None:
        wc = contribs[worst_row]
        others = [c for i, c in enumerate(contribs) if i != worst_row]
        blame = {}
        for oc in others:
            common = set(wc) & set(oc)
            for gi in common:
                blame[gi] = blame.get(gi, 0.0) + wc[gi] * oc[gi]
        top = sorted(blame.items(), key=lambda kv: -kv[1])[:8]
        result["worst_row_top_blame_gaussians"] = [
            {"gaussian_idx": gi, "blame_contribution": val,
             "opacity": float(opacity[gi]), "scale": scale[gi].tolist(),
             "anisotropy": float(scale[gi].max() / max(scale[gi].min(), 1e-12))}
            for gi, val in top
        ]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
