#!/usr/bin/env python3
"""Per-module usefulness evidence for one Algorithm-1 run.

The context book (compile_report_pdf) narrates the persistent store chapter by
chapter, but it does not answer the blunt question a reviewer actually asks:
"did every module in Algorithm 1 do something useful on this problem?" This
script answers exactly that. For each of the six operations it reports, from the
run's own telemetry and the persistent store S=(Pi,I,M,L,K,H,E):

  * how many times the operation ran (Run(u,q,S,B) calls in orchestration_log),
  * what it wrote back into S (records, by component),
  * a concrete sample of that output,

plus the two global claims the paper makes about the loop: context stayed within
budget B on every call, and delivery was best-of-rounds.

Usage:
    module_evidence.py <problem_id> [--dataset first_proof_1]
                       [--since ISO8601]  # scope to records at/after this time
                       [--out PATH.md]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rma.orchestrator import read_telemetry, telemetry_path_for  # noqa: E402
from rma.store import ResearchStore  # noqa: E402

# The seven operations the round loop actually runs (round_loop.py), each mapped
# to the store component(s) it writes. `meeting` populates BOTH M (its plan) and
# H (durable insights) — there is no separate insight operation; H is a meeting
# by-product, which is why it is listed under meeting.
MODULES = [
    ("critic", ["I"], "Critique: enumerate issues (semantic + structural) with severity"),
    ("solver", ["Pi"], "Solver: localized repair of one issue, preserving correct text"),
    ("literature", ["L"], "Literature: retrieve external theorems/techniques with limits"),
    ("meeting", ["M", "H"], "Meeting: synthesize a plan (M) + record durable insights (H)"),
    ("revise", ["Pi"], "Revise: coordinated multi-issue proof revision"),
    ("concepts", ["K"], "Concepts: distill reusable notation/definitions"),
    ("evaluator", ["E"], "Evaluator: score the current proof (rubric)"),
]


def _records(store: ResearchStore, component: str) -> list:
    return list(store.records(component=component))


def _fmt_sample(rec) -> str:
    body = (getattr(rec, "body", "") or "").strip().replace("\n", " ")
    meta = getattr(rec, "meta", {}) or {}
    title = meta.get("title") or meta.get("name") or meta.get("theorem_or_technique") or ""
    head = f"[{title}] " if title else ""
    return (head + body)[:200]


def build(problem_id: str, dataset: str, since: str | None,
          telemetry: str | None = None) -> str:
    store = ResearchStore.open(REPO, problem_id, dataset)
    # Telemetry lives in the RUN's output tree (artifacts/orchestration_log.jsonl),
    # not the persistent store dir — accept an explicit path so the Runs column
    # reflects the run being reported, not a stale store-dir log.
    tel_path = pathlib.Path(telemetry) if telemetry else telemetry_path_for(store)
    tel = read_telemetry(store, telemetry_path=tel_path)

    # per-unit Run counts from telemetry
    run_counts = collections.Counter(e.get("unit") for e in tel)
    rounds = sorted({e.get("round") for e in tel if e.get("round") is not None})

    def _keep(rec) -> bool:
        if not since:
            return True
        return (getattr(rec, "created_at", "") or "") >= since

    out = []
    w = out.append
    w(f"# Module usefulness — {problem_id} ({dataset})\n")
    w(f"Telemetry: `{tel_path}` — {len(tel)} Run calls"
      + (f", rounds {rounds[0]}..{rounds[-1]}" if rounds else "") + ".\n")
    if since:
        w(f"Store records scoped to created_at >= `{since}`.\n")

    # ---- global loop claims -------------------------------------------------
    over = [e for e in tel if e.get("over_budget")]
    budgets = {e.get("budget") for e in tel if e.get("budget") is not None}
    w("## Loop-level guarantees\n")
    w(f"- **Context budget respected**: {len(tel) - len(over)}/{len(tel)} Run calls "
      f"were within budget B" + (f" (B in {sorted(budgets)})" if budgets else "")
      + (f"; {len(over)} exceeded." if over else "; none exceeded."))
    proofs = store.proofs
    w(f"- **Proof revisions accumulated**: {len(proofs)} in Π (persistent).")
    w(f"- **Open issues remaining**: {len(store.open_issues())}.\n")

    # ---- per-module evidence -----------------------------------------------
    w("## Per-module evidence\n")
    w("| Module | Runs | Wrote → | New records | Useful? |")
    w("|---|---|---|---|---|")
    details = []
    for unit, comps, role in MODULES:
        runs = run_counts.get(unit, 0)
        recs = [r for comp in comps for r in _records(store, comp)
                if (getattr(r, "meta", {}) or {}).get("unit") == unit and _keep(r)]
        # A module is useful if it ran AND wrote back. When telemetry is absent
        # (Runs unknown, an older store), fall back to writeback as the signal.
        if runs > 0:
            useful = "✅" if recs else "⚠️ ran, no writeback"
        else:
            useful = "✅ (writeback present)" if recs else "—"
        w(f"| **{unit}** | {runs or '—'} | {'+'.join(comps)} | {len(recs)} | {useful} |")
        details.append((unit, comps, role, runs, recs))

    # ---- samples ------------------------------------------------------------
    w("\n## What each module actually produced\n")
    for unit, comps, role, runs, recs in details:
        w(f"### {unit} — {role}")
        w(f"Runs: {runs}. Records written to {'+'.join(comps)}: {len(recs)}.")
        if unit == "critic" and recs:
            sev = collections.Counter((r.meta or {}).get("severity") for r in recs)
            w(f"Severity distribution: {dict(sev)}.")
        if unit == "solver" and recs:
            modes = collections.Counter((r.meta or {}).get("mode") for r in recs)
            w(f"Repair modes: {dict(modes)} "
              f"(patch = localized; full_rewrite = fallback — lower is better).")
        if unit == "evaluator" and recs:
            last = recs[-1]
            w(f"Latest score: total={ (last.meta or {}).get('total') }, "
              f"scores={ (last.meta or {}).get('scores') }.")
        for r in recs[:2]:
            w(f"- `{r.id}` (round {getattr(r,'round','?')}): {_fmt_sample(r)}")
        if not recs:
            w("- _(no records — this module did not contribute on this run)_")
        w("")

    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem_id")
    ap.add_argument("--dataset", default="first_proof_1")
    ap.add_argument("--since", default=None)
    ap.add_argument("--telemetry", default=None,
                    help="Path to the run's orchestration_log.jsonl (defaults to store dir).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    md = build(args.problem_id, args.dataset, args.since, telemetry=args.telemetry)
    if args.out:
        pathlib.Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
