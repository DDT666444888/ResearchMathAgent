#!/usr/bin/env python3
"""Fetch a single scene folder out of the Mip-NeRF 360 dataset zip without
downloading the whole 12.5 GB archive.

The dataset is hosted as one big zip (`360_v2.zip`, ~12.5 GB, containing
bicycle/bonsai/counter/garden/kitchen/room/stump) at a URL that supports
HTTP Range requests. This script wraps a `urllib`-based random-access file
object around the remote zip and hands it to the stdlib `zipfile` module,
which only reads the central directory plus the byte ranges needed for the
requested scene's entries -- so total transferred data is close to just
that scene's compressed size, not the full archive.

Usage:
    python fetch_mipnerf360_scene.py --scene garden \
        --out /work/nvme/bhov/zzhao18/gtk_3dgs_runs/data/garden

Used for both `bonsai` (Phase 1, via an earlier one-off version of this
script) and `garden` (Phase 2) -- see SETUP_NOTES.md.
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

DATASET_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"


class HttpRangeFile(io.RawIOBase):
    """A minimal seekable/readable file-like object over HTTP Range requests,
    sufficient for zipfile.ZipFile's random-access needs."""

    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0
        self._bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        b[: len(data)] = data
        self.pos += len(data)
        self._bytes_fetched += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="e.g. garden, bonsai")
    ap.add_argument("--out", required=True, help="output directory for the scene")
    ap.add_argument("--url", default=DATASET_URL)
    args = ap.parse_args()

    prefix = args.scene.rstrip("/") + "/"
    print(f"Opening remote zip (range requests): {args.url}", file=sys.stderr)
    rf = HttpRangeFile(args.url)
    bio = io.BufferedReader(rf)
    with zipfile.ZipFile(bio) as zf:
        members = [n for n in zf.namelist() if n.startswith(prefix)]
        if not members:
            print(f"ERROR: no entries found with prefix {prefix!r}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(members)} entries under {prefix!r}; extracting to {args.out}",
              file=sys.stderr)
        os.makedirs(args.out, exist_ok=True)
        for i, name in enumerate(members):
            rel = name[len(prefix):]
            if not rel:  # the directory entry itself
                continue
            dest = os.path.join(args.out, rel)
            if name.endswith("/"):
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as out_f:
                out_f.write(src.read())
            if (i + 1) % 50 == 0 or (i + 1) == len(members):
                print(f"  extracted {i+1}/{len(members)} "
                      f"({rf._bytes_fetched/1e6:.0f} MB fetched so far)",
                      file=sys.stderr)
    print(f"Done. Total bytes fetched over HTTP: {rf._bytes_fetched/1e6:.1f} MB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
