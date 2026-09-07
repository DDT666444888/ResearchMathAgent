#!/usr/bin/env python3
"""High-density splat export for a photorealistic fly-through viewer (as
opposed to export_splat_viewer.py's 80k-splat, 32-byte-per-splat analysis
viewer format). Uses a more compact 20-byte-per-splat layout so a single
model can ship at ~500k splats within a self-contained artifact's size cap:

    position:  3 x float16 (6 bytes)   -- linear, scene coords fit fp16's range
    logscale:  3 x float16 (6 bytes)   -- log(scale), NOT raw scale: raw scale
                                           spans ~7 orders of magnitude
                                           (3.8e-7 to 3.3 in this checkpoint),
                                           which would blow past fp16's dynamic
                                           range for the smallest (most
                                           improved, most interesting) Gaussians
                                           if stored linearly -- storing the log
                                           keeps full relative precision across
                                           the whole range; the viewer exponentiates
                                           on decode.
    rgba:      4 x uint8 (4 bytes)     -- SH DC color + opacity, unchanged
    rotation:  4 x uint8 (4 bytes)     -- quaternion, unchanged

No anisotropy companion file -- this export is for the realism showcase, not
the conditioning-analysis viewer (see export_splat_viewer.py for that).

Usage:
    python export_splat_hd.py --ply <path/to/point_cloud.ply> \
        --out <path/to/output.hdsplat> --target-n 500000
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
    ap.add_argument("--target-n", type=int, default=500000)
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
        mean_scale = scale.mean(axis=1)
        importance = opacity * np.sqrt(mean_scale)
        p = importance / importance.sum()
        idx = rng.choice(m, size=target_n, replace=False, p=p)
    else:
        idx = np.arange(m)
    xyz, scale, opacity, rot, color = xyz[idx], scale[idx], opacity[idx], rot[idx], color[idx]
    n_out = xyz.shape[0]
    print(f"[info] exporting {n_out} splats to {args.out}", file=sys.stderr)

    pos_f16 = xyz.astype(np.float16)
    logscale_f16 = np.log(np.clip(scale, 1e-12, None)).astype(np.float16)

    rgba_u8 = np.zeros((n_out, 4), dtype=np.uint8)
    rgba_u8[:, 0:3] = np.clip(color * 255.0, 0, 255).astype(np.uint8)
    rgba_u8[:, 3] = np.clip(opacity * 255.0, 0, 255).astype(np.uint8)
    rot_u8 = np.clip(rot * 128.0 + 128.0, 0, 255).astype(np.uint8)

    with open(args.out, "wb") as f:
        for i in range(n_out):
            f.write(pos_f16[i].tobytes())        # 6 bytes, little-endian fp16 x3
            f.write(logscale_f16[i].tobytes())    # 6 bytes
            f.write(rgba_u8[i].tobytes())         # 4 bytes
            f.write(rot_u8[i].tobytes())          # 4 bytes

    total_bytes = n_out * 20
    print(f"[info] wrote {total_bytes} bytes ({n_out} splats x 20 bytes) "
          f"= {total_bytes/1024/1024:.2f} MB raw, ~{total_bytes*4/3/1024/1024:.2f} MB base64",
          file=sys.stderr)


if __name__ == "__main__":
    main()
