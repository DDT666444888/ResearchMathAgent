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
    3. export four artifacts, all sharing one timestamped stem
       <date>_<time>_<language>_<dataset>_<problem>_… :

           _report.tex   the full context report LaTeX (all parts)
           _report.pdf   its PDF rendering (for humans)
           _proof.tex    the best proof LaTeX
           _proof.pdf    its PDF rendering

Only datasets that carry a curated ``solve_set.json`` (the filtering system's
10 valuable-unsolved problems) are processed, so "all datasets filtered, 10
each" is exactly what runs.

Everything is configurable through environment variables (see DEFAULTS below)
so the caller (``for_yuchen_jul1.sh``) can tune it without editing this file.

    RMA_OUTPUT        output folder for the books  (default: outputs/context_books)
    RMA_LANGUAGE      report language(s): both | en | cn  (default: both = EN+CN)
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

def _languages() -> list[str]:
    """RMA_LANGUAGE=both|en|cn|zh → canonical list. Default both (EN + CN)."""
    raw = os.environ.get("RMA_LANGUAGE", "both").strip().lower()
    if raw in ("both", "en+cn", "all", ""):
        return ["en", "cn"]
    codes = []
    for tok in re.split(r"[\s,+]+", raw):
        if not tok:
            continue
        c = "cn" if tok in ("cn", "zh", "zh-cn", "chinese", "中文") else "en"
        if c not in codes:
            codes.append(c)
    return codes or ["en"]

LANGUAGES    = _languages()
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


def _stem(dataset: str, pid: str, when: datetime, lang: str) -> str:
    safe_ds  = re.sub(r"[^A-Za-z0-9_-]", "-", dataset)
    safe_pid = re.sub(r"[^A-Za-z0-9_-]", "-", pid)
    return f"{when:%Y%m%d}_{when:%H%M%S}_{lang}_{safe_ds}_{safe_pid}"


def out_dir(dataset: str) -> Path:
    return OUTPUT / re.sub(r"[^A-Za-z0-9_-]", "-", dataset)


def already_done(dataset: str, pid: str) -> bool:
    """Done once the four artifacts exist for EVERY requested language."""
    d = out_dir(dataset)
    if not d.is_dir():
        return False
    safe_ds  = re.sub(r"[^A-Za-z0-9_-]", "-", dataset)
    safe_pid = re.sub(r"[^A-Za-z0-9_-]", "-", pid)
    def _has(lang: str, part: str, ext: str) -> bool:
        tail = f"_{lang}_{safe_ds}_{safe_pid}_"
        return any(p.name.endswith(f"{part}.{ext}") and tail in p.name for p in d.glob("*"))
    return all(_has(l, "report", "pdf") and _has(l, "report", "tex")
               and _has(l, "proof", "pdf") and _has(l, "proof", "tex")
               for l in LANGUAGES)


def export_book(dataset: str, pid: str) -> bool:
    """For EACH requested language, emit four per-problem artifacts sharing one
    timestamped, language-tagged stem
    <date>_<time>_<language>_<dataset>_<problem>_… :

        _report.tex  (1) full context report LaTeX (English or Chinese)
        _report.pdf  (3) its PDF rendering (for humans)
        _proof.tex   (2) best proof LaTeX
        _proof.pdf   (4) its PDF rendering

    (The proof is language-neutral LaTeX; it is copied into each language's
    bundle so every <language> set is self-contained.)
    """
    import shutil
    from webapp.context_report import compile_report_pdf
    from webapp.proofs import get_best_proof, compile_best_pdf, _best_dir

    d = out_dir(dataset); d.mkdir(parents=True, exist_ok=True)
    pdf_dir = REPO / "documents" / "pdf"
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{pid}_{dataset}")

    # Best proof (shared across languages) — build once.
    bp = get_best_proof(pid, dataset)
    proof_tex = (bp or {}).get("solution_tex") or ""
    proof_pdf = None
    try:
        if compile_best_pdf(pid, dataset):
            cand = _best_dir(dataset) / pid / "solution.pdf"
            proof_pdf = cand if cand.is_file() else None
    except Exception as exc:                                        # noqa: BLE001
        log(f"  WARN: best-proof PDF failed for {dataset}/{pid}: {exc}")

    all_ok = True
    for lang in LANGUAGES:
        when = datetime.now()
        stem = _stem(dataset, pid, when, lang)
        got: list[str] = []

        def _emit(src, part, ext):
            if src and Path(src).is_file() and Path(src).stat().st_size > 0:
                shutil.copyfile(src, d / f"{stem}_{part}.{ext}"); got.append(f"{part}.{ext}")

        def _write(text, part, ext):
            if text and text.strip():
                (d / f"{stem}_{part}.{ext}").write_text(text, encoding="utf-8"); got.append(f"{part}.{ext}")

        # (1)+(3) report LaTeX + PDF in this language
        prefix = "cn_report" if lang == "cn" else "report"
        res = compile_report_pdf(REPO, pid, dataset, force=True, language=lang)
        if res.get("ok"):
            _emit(pdf_dir / f"{prefix}_{safe}.pdf", "report", "pdf")
            _emit(pdf_dir / f"{prefix}_{safe}.tex", "report", "tex")
        else:
            log(f"  WARN: {lang} report compile failed for {dataset}/{pid}: {res.get('log')}")

        # (2)+(4) best proof LaTeX + PDF (shared content)
        _write(proof_tex, "proof", "tex")
        _emit(proof_pdf, "proof", "pdf")

        log(f"  [{lang}] wrote {len(got)}/4: {', '.join(got) or 'NONE'}  (stem={stem})")
        all_ok = all_ok and ("report.tex" in got) and ("proof.tex" in got)
    return all_ok


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    datasets = filtered_datasets()
    if not datasets:
        log("No filtered datasets found (need data/datasets/<slug>/solve_set.json).")
        return 1

    total = sum(len(p) for _, p in datasets)
    log(f"repo:        {REPO}")
    log(f"output:      {OUTPUT}")
    log(f"languages:   {', '.join(LANGUAGES)}")
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
