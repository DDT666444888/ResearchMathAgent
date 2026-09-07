#!/usr/bin/env python3
"""Curate the *jul5* deliverable: ~60 research-level problems spanning EVERY
rm14k domain, written as one ``solve_set.json`` per domain dataset.

Design (mirrors the jul1 filtering-system curation, informed by
outputs/jul5_rma_results_six_dataset/RMA_analytics.html):

  * Cover ALL 11 rm14k domains (algebra, analysis, appliedmath, crossdisc,
    discretemath, geomtopology, logic, mathphysics, numbertheory, probstat,
    tcs) so the set exposes "different aspects in all domains".
  * Within each domain, rank by AI-solvability (data/datasets/<d>/
    solvability_cache.json). The analytics run showed *completeness* is the
    scarce dimension — the agent makes the most genuine headway on the
    higher-solvability, more self-contained statements — so solvability is the
    right ranking key, exactly as the jul1 solve_sets used.
  * Allocation: BASE_PER_DOMAIN from every domain guarantees breadth; the
    remaining slots go to the globally highest-scoring leftovers so the
    standout "closest-to-solved" outliers (e.g. mathphysics rm-01516 @0.82,
    crossdisc rm-13348 @0.80, numbertheory rm-07152 @0.74) are always in.
  * NON-OVERLAP: excludes every id already shipped in the jul1 deliverable
    and anything from the first_proof_1 / first_proof_2 batches. (rm14k ids are
    rm-XXXXX, disjoint from the jul1 erdos_/aim_/q*/prob-/fc*/um* ids anyway,
    but the guard is explicit.)

Run from the repo root:  python scripts/curate_rm14k_solve_sets.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "data" / "datasets"

TARGET_TOTAL    = 60
BASE_PER_DOMAIN = 5          # 11 domains * 5 = 55 guaranteed (breadth)
DOMAINS = [
    "algebra", "analysis", "appliedmath", "crossdisc", "discretemath",
    "geomtopology", "logic", "mathphysics", "numbertheory", "probstat", "tcs",
]
DOMAIN_SLUGS = [f"{d}_rm14k" for d in DOMAINS]

CRITERIA = ("all rm14k domains, ranked by AI-solvability (filtering system); "
            "non-overlap with jul1 deliverable; no first_proof batches")


def _prev_jul1_ids() -> set[str]:
    """Every problem id already shipped in the jul1 solve_sets (to exclude)."""
    prev: set[str] = set()
    for ds in DATASETS.iterdir():
        ss = ds / "solve_set.json"
        if ds.name in DOMAIN_SLUGS or not ss.is_file():
            continue
        try:
            prev.update(json.loads(ss.read_text()).get("problem_ids", []))
        except Exception:
            pass
    return prev


def _ranked(slug: str, exclude: set[str]) -> list[tuple[str, float]]:
    cache = DATASETS / slug / "solvability_cache.json"
    scores = json.loads(cache.read_text())
    avail = {p for p in (DATASETS / slug / "problems").glob("*.json")}
    have = {p.stem for p in avail}
    items = [(pid, float(sc)) for pid, sc in scores.items()
             if pid in have and pid not in exclude]
    # stable: score desc, then id for determinism
    return sorted(items, key=lambda kv: (-kv[1], kv[0]))


def main() -> int:
    exclude = _prev_jul1_ids()
    print(f"excluding {len(exclude)} jul1 ids (none expected to be rm14k)")

    ranked = {slug: _ranked(slug, exclude) for slug in DOMAIN_SLUGS}

    chosen: dict[str, list[str]] = {slug: [r[0] for r in ranked[slug][:BASE_PER_DOMAIN]]
                                    for slug in DOMAIN_SLUGS}
    picked = sum(len(v) for v in chosen.values())

    # global top-ups from the leftovers, highest score first
    leftovers: list[tuple[float, str, str]] = []
    for slug in DOMAIN_SLUGS:
        for pid, sc in ranked[slug][BASE_PER_DOMAIN:]:
            leftovers.append((sc, slug, pid))
    leftovers.sort(key=lambda t: (-t[0], t[1], t[2]))
    for sc, slug, pid in leftovers:
        if picked >= TARGET_TOTAL:
            break
        chosen[slug].append(pid)
        picked += 1

    now = datetime.now(timezone.utc).isoformat()
    total = 0
    for slug in DOMAIN_SLUGS:
        ids = chosen[slug]
        total += len(ids)
        score_map = dict(ranked[slug])
        payload = {
            "problem_ids": ids,
            "sampled_at": now,
            "criteria": CRITERIA,
            "scores": {pid: round(score_map[pid], 3) for pid in ids},
        }
        out = DATASETS / slug / "solve_set.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  {slug:22s} {len(ids):2d} problems  "
              f"(scores {min(payload['scores'].values()):.2f}"
              f"–{max(payload['scores'].values()):.2f})  -> {out.name}")

    print(f"\nTOTAL selected: {total} problems across {len(DOMAIN_SLUGS)} domains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
