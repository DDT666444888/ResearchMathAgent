#!/usr/bin/env python3
"""Run RMA over every *filtered* dataset (the curated 10 problems per dataset)
and export a per-problem **context book** — proofs, evaluations, meetings and
issues bundled into one PDF (plus a markdown twin) — into an output folder.

For every problem this does exactly what a full RMA pass does:

    1. ``rma solve <pid> --dataset <ds>``      → generate the proof
    2. five push-forwards  (``rma push``)       → issues, meetings, evaluations
                                                  (``rma solve`` includes five
                                                  push-forwards by default; that
                                                  is the RMA_PUSHFORWARDS knob)
    3. compile the context book PDF             → proofs + evaluations
                                                  + meetings + issues
    4. copy it out named

           <date>_<time>_<language>_<dataset>_<problem>.(pdf|md)

Only datasets that carry a curated ``solve_set.json`` (the filtering system's
10 valuable-unsolved problems) are processed, so "all datasets filtered, 10
each" is exactly what runs.

Everything is configurable through environment variables (see DEFAULTS below)
so the caller (``for_yuchen_jul1.sh``) can tune it without editing this file.

    RMA_OUTPUT        output folder for the books  (default: outputs/context_books)
    RMA_LANGUAGE      language tag baked into filenames (default: en)
    RMA_PUSHFORWARDS  push-forwards per problem     (default: 5)
    RMA_ROUNDS        meeting discussion rounds / push-forward (default: 2)
    RMA_PROVIDER      LLM backend: claude-code | api (default: claude-code)
    RMA_DATASETS      space/comma list to restrict datasets (default: all filtered)
    RMA_LIMIT         cap problems per dataset      (default: all, i.e. 10)
    RMA_RESUME        1 = skip problems whose book already exists (default: 1)

Nothing is destructive: it only writes under the output folder and the repo's
own generated-artifact trees.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── locate the repo root (this file lives in <repo>/scripts/) ─────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# ── configuration ─────────────────────────────────────────────────────────────
OUTPUT       = Path(os.environ.get("RMA_OUTPUT", REPO / "outputs" / "context_books")).resolve()
LANGUAGE     = re.sub(r"[^A-Za-z0-9-]", "-", os.environ.get("RMA_LANGUAGE", "en")).strip("-") or "en"
PUSHFORWARDS = max(0, int(os.environ.get("RMA_PUSHFORWARDS", "5")))
ROUNDS       = max(1, int(os.environ.get("RMA_ROUNDS", "2")))
PROVIDER     = os.environ.get("RMA_PROVIDER", "claude-code")
LIMIT        = int(os.environ.get("RMA_LIMIT", "0"))          # 0 = no cap
RESUME       = _flag("RMA_RESUME", "1")
_DS_FILTER   = re.split(r"[\s,]+", os.environ.get("RMA_DATASETS", "").strip())
DS_FILTER    = {d for d in _DS_FILTER if d}


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def filtered_datasets() -> list[tuple[str, list[str]]]:
    """Every dataset that has a curated solve_set.json → (slug, [problem_ids])."""
    import json
    root = REPO / "data" / "datasets"
    if not root.is_dir():
        log(f"FATAL: {root} not found. On the cluster this is a symlink to the "
            f"shared data dir; make sure you have access to it.")
        sys.exit(2)
    out: list[tuple[str, list[str]]] = []
    for ds_dir in sorted(root.iterdir()):
        ss = ds_dir / "solve_set.json"
        if not ss.is_file():
            continue
        slug = ds_dir.name
        if DS_FILTER and slug not in DS_FILTER:
            continue
        try:
            ids = json.loads(ss.read_text(encoding="utf-8")).get("problem_ids", [])
        except Exception as exc:                                    # noqa: BLE001
            log(f"skip {slug}: cannot read solve_set.json ({exc})")
            continue
        if LIMIT > 0:
            ids = ids[:LIMIT]
        if ids:
            out.append((slug, ids))
    return out


def rma(*args: str) -> int:
    """Invoke the rma CLI as a module so it works without PATH tweaks."""
    cmd = [sys.executable, "-m", "rma", *args]
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO).returncode


def book_path(dataset: str, pid: str, ext: str) -> Path:
    now = datetime.now()
    safe_ds  = re.sub(r"[^A-Za-z0-9_-]", "-", dataset)
    safe_pid = re.sub(r"[^A-Za-z0-9_-]", "-", pid)
    stem = f"{now:%Y%m%d}_{now:%H%M%S}_{LANGUAGE}_{safe_ds}_{safe_pid}"
    return OUTPUT / safe_ds / f"{stem}.{ext}"


def already_done(dataset: str, pid: str) -> bool:
    safe_ds  = re.sub(r"[^A-Za-z0-9_-]", "-", dataset)
    safe_pid = re.sub(r"[^A-Za-z0-9_-]", "-", pid)
    d = OUTPUT / safe_ds
    if not d.is_dir():
        return False
    pat = f"_{LANGUAGE}_{safe_ds}_{safe_pid}.pdf"
    return any(p.name.endswith(pat) for p in d.glob("*.pdf"))


def export_book(dataset: str, pid: str) -> bool:
    """Build the context-book PDF + markdown for one problem and copy them out."""
    import shutil
    from webapp.context_report import compile_report_pdf, build_problem_report

    # 1. context book PDF — proofs + evaluations + meetings + issues, one file.
    res = compile_report_pdf(REPO, pid, dataset, force=True)
    if not res.get("ok"):
        log(f"  PDF build FAILED for {dataset}/{pid}: {res.get('log')}")
        return False
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{pid}_{dataset}")
    src = REPO / "documents" / "pdf" / f"report_{safe}.pdf"
    dest = book_path(dataset, pid, "pdf")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copyfile(src, dest)
        log(f"  ✅ book → {dest.relative_to(OUTPUT.parent)}")
    else:
        log(f"  WARN: expected {src} missing")
        return False

    # 2. markdown twin (readable / greppable context book).
    try:
        report = build_problem_report(REPO, pid, dataset, full=True)
        md_dest = book_path(dataset, pid, "md")
        md_dest.write_text(report.get("markdown", ""), encoding="utf-8")
    except Exception as exc:                                        # noqa: BLE001
        log(f"  WARN: markdown twin failed for {dataset}/{pid}: {exc}")
    return True


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    datasets = filtered_datasets()
    if not datasets:
        log("No filtered datasets found (need data/datasets/<slug>/solve_set.json).")
        return 1

    total = sum(len(p) for _, p in datasets)
    log(f"repo:        {REPO}")
    log(f"output:      {OUTPUT}")
    log(f"language:    {LANGUAGE}")
    log(f"provider:    {PROVIDER}")
    log(f"pushforwards:{PUSHFORWARDS}  rounds/pf:{ROUNDS}  resume:{RESUME}")
    log(f"datasets:    {len(datasets)}  problems:{total}")
    for slug, pids in datasets:
        log(f"    {slug}: {len(pids)} problems")

    ok = fail = skipped = 0
    for slug, pids in datasets:
        for pid in pids:
            log(f"══ {slug} / {pid} ══")
            if RESUME and already_done(slug, pid):
                log("  resume: book already exists, skipping")
                skipped += 1
                continue

            # 1. solve → initial proof
            rma("solve", pid, "--dataset", slug, "--model-provider", PROVIDER)

            # 2. five push-forwards → issues, meetings, evaluations
            for i in range(1, PUSHFORWARDS + 1):
                log(f"  push-forward {i}/{PUSHFORWARDS}")
                rma("push", "--dataset", slug, "--problems", pid,
                    "--provider", PROVIDER, "--rounds", str(ROUNDS))

            # 3+4. compile & export the context book
            if export_book(slug, pid):
                ok += 1
            else:
                fail += 1

    log("─" * 60)
    log(f"DONE  exported:{ok}  failed:{fail}  skipped:{skipped}  → {OUTPUT}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
