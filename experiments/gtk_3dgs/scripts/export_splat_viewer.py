#!/usr/bin/env python3
"""Export a trained 3DGS checkpoint's point_cloud.ply to a compact 32-byte-
per-splat binary format for an in-browser WebGL viewer (position: 3xfloat32,
scale: 3xfloat32, rgba: 4xuint8, rotation quaternion: 4xuint8). Same packed
layout as the well-known antimatter15/splat viewer format, so the math in
the accompanying WebGL shader (2D covariance projection + EWA splatting) is
against a known-correct reference.

Applies opacity-weighted importance subsampling to a target splat count
(full checkpoints have ~1M Gaussians -- too much for a browser-embedded
demo) and SH-DC-only color (ignores higher-order spherical harmonics --
those encode view-dependent color, not needed for a static orbit viewer).

Usage:
    python export_splat_viewer.py --ply <path/to/point_cloud.ply> \
        --out <path/to/output.splat> --target-n 80000
"""
import argparse
import struct
import sys

import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-n", type=int, default=80000)
    ap.add_argument("--min-opacity", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ply = PlyData.read(args.ply)
    v = ply["vertex"]
    n = v.count
    print(f"[info] loaded {n} gaussians from {args.ply}", file=sys.stderr)

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)).astype(np.float32)
    opacity = sigmoid(np.array(v["opacity"], dtype=np.float32))
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    rot = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-8)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    color = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)

    keep = opacity > args.min_opacity
    print(f"[info] {keep.sum()}/{n} pass opacity > {args.min_opacity}", file=sys.stderr)
    xyz, scale, opacity, rot, color = xyz[keep], scale[keep], opacity[keep], rot[keep], color[keep]
    m = xyz.shape[0]

    target_n = min(args.target_n, m)
    rng = np.random.default_rng(args.seed)
    if m > target_n:
        # importance = opacity-weighted screen-footprint proxy, so visually
        # significant (opaque, larger) Gaussians are more likely retained,
        # while still sampling (not hard-thresholding) to keep fine detail
        mean_scale = scale.mean(axis=1)
        importance = opacity * np.sqrt(mean_scale)
        p = importance / importance.sum()
        idx = rng.choice(m, size=target_n, replace=False, p=p)
    else:
        idx = np.arange(m)
    xyz, scale, opacity, rot, color = xyz[idx], scale[idx], opacity[idx], rot[idx], color[idx]
    n_out = xyz.shape[0]
    print(f"[info] exporting {n_out} splats to {args.out}", file=sys.stderr)

    rgba_u8 = np.zeros((n_out, 4), dtype=np.uint8)
    rgba_u8[:, 0:3] = np.clip(color * 255.0, 0, 255).astype(np.uint8)
    rgba_u8[:, 3] = np.clip(opacity * 255.0, 0, 255).astype(np.uint8)

    rot_u8 = np.clip(rot * 128.0 + 128.0, 0, 255).astype(np.uint8)

    with open(args.out, "wb") as f:
        for i in range(n_out):
            f.write(struct.pack("<3f", *xyz[i]))
            f.write(struct.pack("<3f", *scale[i]))
            f.write(rgba_u8[i].tobytes())
            f.write(rot_u8[i].tobytes())

    print(f"[info] wrote {n_out * 32} bytes ({n_out} splats x 32 bytes)", file=sys.stderr)

    # Companion per-splat anisotropy-ratio array (log10 scale.max()/scale.min()),
    # index-aligned with the .splat file above -- lets the viewer recolor splats
    # by conditioning (this experiment's actual subject) instead of only true
    # RGB, so "where does the improvement come from" has a real, inspectable
    # answer in 3D space, not just a claim.
    s_max = scale.max(axis=1)
    s_min = np.clip(scale.min(axis=1), 1e-12, None)
    log_aniso = np.log10(s_max / s_min).astype(np.float32)
    aniso_path = args.out.rsplit(".", 1)[0] + ".aniso.f32"
    with open(aniso_path, "wb") as f:
        f.write(log_aniso.tobytes())
    print(f"[info] wrote {aniso_path}: log10(anisotropy) range "
          f"[{log_aniso.min():.2f}, {log_aniso.max():.2f}], median {np.median(log_aniso):.2f}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
