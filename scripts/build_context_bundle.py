#!/usr/bin/env python3
"""Assemble the full inspection bundle for one Algorithm-1 run.

Given a problem and the run's output directory, this produces, in one dated
folder under documents/context_books/:

  * MODULE_EVIDENCE.md   — per-module usefulness table (module_evidence.py)
  * <pid>_context_report_en.pdf / _cn.pdf  — the context book (all chapters)
  * <pid>_proof.tex / .pdf                 — the delivered best-of-rounds proof
  * orchestrator_summary.json              — per-round stop/metrics
  * README.md                              — what to look at and why

Usage:
    build_context_bundle.py <problem_id> <run_output_dir> <dest_dir>
                            [--dataset first_proof_1]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PY = sys.executable


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(REPO), text=True, capture_output=True)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem_id")
    ap.add_argument("run_output_dir")
    ap.add_argument("dest_dir")
    ap.add_argument("--dataset", default="first_proof_1")
    args = ap.parse_args()

    pid = args.problem_id
    run_dir = pathlib.Path(args.run_output_dir)
    dest = pathlib.Path(args.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"problem": pid, "dest": str(dest)}

    # 1) per-module evidence, scoped to THIS run's telemetry
    tel = run_dir / pid / "artifacts" / "orchestration_log.jsonl"
    ev_cmd = [PY, "scripts/module_evidence.py", pid, "--dataset", args.dataset,
              "--out", str(dest / "MODULE_EVIDENCE.md")]
    if tel.is_file():
        ev_cmd += ["--telemetry", str(tel)]
    rc, out = _run(ev_cmd)
    report["evidence"] = {"ok": rc == 0, "log": out if rc else ""}

    # 2) context report PDFs (EN + CN) — compile_report_pdf reads the store
    from webapp.context_report import compile_report_pdf
    for lang, src_name in (("en", f"report_{pid}_{args.dataset}.pdf"),
                           ("cn", f"cn_report_{pid}_{args.dataset}.pdf")):
        try:
            res = compile_report_pdf(REPO, pid, args.dataset, force=True, language=lang)
            src = REPO / "documents" / "pdf" / src_name
            ok = bool(res.get("ok")) and src.is_file()
            if ok:
                shutil.copy2(src, dest / f"{pid}_context_report_{lang}.pdf")
            report[f"report_{lang}"] = {
                "ok": ok,
                "bytes": src.stat().st_size if src.is_file() else 0,
                "log": ("" if ok else str(res.get("log", ""))[-600:]),
            }
        except Exception as e:  # noqa: BLE001
            report[f"report_{lang}"] = {"ok": False, "error": repr(e)}

    # 3) delivered proof (tex + rendered pdf)
    sol_tex = run_dir / f"{pid}_solution.tex"
    sol_pdf = run_dir / f"{pid}_solution.pdf"
    if sol_tex.is_file():
        shutil.copy2(sol_tex, dest / f"{pid}_proof.tex")
    if sol_pdf.is_file():
        shutil.copy2(sol_pdf, dest / f"{pid}_proof.pdf")
    report["proof"] = {"tex": sol_tex.is_file(), "pdf": sol_pdf.is_file(),
                       "tex_bytes": sol_tex.stat().st_size if sol_tex.is_file() else 0}

    # 4) orchestrator summary
    summ = run_dir / pid / "artifacts" / "orchestrator_summary.json"
    if summ.is_file():
        shutil.copy2(summ, dest / "orchestrator_summary.json")
        report["summary"] = json.loads(summ.read_text())

    # 5) README
    (dest / "README.md").write_text(_readme(pid, report), encoding="utf-8")

    print(json.dumps(report, indent=2, default=str))
    return 0


def _readme(pid: str, report: dict) -> str:
    s = report.get("summary", {}) or {}
    per = s.get("per_round", []) or []
    rows = "\n".join(
        f"| {r.get('round')} | {', '.join(r.get('units', []))} | "
        f"{r.get('completeness')} | {r.get('open_critical')} | {r.get('stop')} |"
        for r in per) or "| — | — | — | — | — |"
    return f"""# Context inspection bundle — {pid}

This bundle lets you check, by hand, that **every module in Algorithm 1 did
useful work** on {pid}, and read the proof it delivered.

## Files
- **MODULE_EVIDENCE.md** — the one-page answer: each of the seven operations
  (critic, solver, literature, meeting, revise, concepts, evaluator), how many
  times it ran, what it wrote into the persistent store S, and a sample.
- **{pid}_context_report_en.pdf / _cn.pdf** — the full context book: Open/Resolved
  Issues, Meetings, Concepts, Insights, Literature, Best Proof, Evaluation.
- **{pid}_proof.tex / .pdf** — the delivered best-of-rounds proof.
- **orchestrator_summary.json** — machine-readable per-round record.

## Run summary
- stop_reason: **{s.get('stop_reason', '?')}**  ·  rounds: {s.get('rounds', '?')}
  ·  delivered_round: {s.get('delivered_round', '?')}  ·  solved: {s.get('solved', '?')}

| round | units run | completeness | open_critical | stop |
|---|---|---|---|---|
{rows}

## How to read it
1. Open **MODULE_EVIDENCE.md** first — confirm each module has Runs > 0 and
   records written (✅). The solver row's repair-mode split (patch vs
   full_rewrite) shows the localized-patch module working, not just falling back.
2. Open the **context report PDF** to see the actual content each module produced.
3. Open **{pid}_proof.pdf** to read the delivered proof.
"""


if __name__ == "__main__":
    raise SystemExit(main())
