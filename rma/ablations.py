"""Ablation matrix — every configuration the paper reports, as runnable runs.

main.tex:880-940 reports two ablation panels: Figure 3(c) leave-one-out component
ablations, and Figure 3(d) graded ladders. Per the paper's own protocol
(app-ablation-protocol), each ablation must be a *runnable* configuration, not a
deleted code path — "removing a component" means reducing it to its simplest
functioning variant. Here every entry is a RunConfig that the round loop
executes to completion.

The matrix is `full` (the baseline, no ablations) plus one run per ablation
name. `rma ablate-matrix --list` enumerates them; `rma report-ablations`
executes them offline and emits per-config metrics that regenerate the figure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import ABLATIONS, RunConfig

# The figure's configurations, in reporting order: the full system first, then
# each single-component ablation. Each maps to the set of ablation flags it
# switches on.
MATRIX: list[tuple[str, frozenset[str]]] = [("full", frozenset())] + [
    (name, frozenset({name})) for name in ABLATIONS
]


def matrix_names() -> list[str]:
    return [name for name, _ in MATRIX]


@dataclass
class AblationRun:
    config: str
    rounds: int
    stop_reason: str | None
    delivered_completeness: float | None
    proved_terminal_fraction: float | None
    open_critical: int | None
    issues_opened: int
    context_tokens_mean: float
    records: dict           # per-component final counts
    observable: dict        # the signal vector used to prove "not a no-op"


def _run_one(name: str, flags: frozenset[str], *, n_rounds: int, backend, analyses,
             initial_proof: str, problem: dict) -> AblationRun:
    """Execute one configuration on a fresh in-memory store, offline."""
    import tempfile

    from .orchestrator import read_telemetry
    from .round_loop import solve_problem
    from .store import memory_model_for, open_store

    cfg = RunConfig(n_rounds=n_rounds, ablations=flags)
    with tempfile.TemporaryDirectory() as tmp:
        store = open_store(tmp, "q6", "first_proof_1", memory=memory_model_for(cfg))
        store.add_proof_revision(initial_proof, produced_by="proposer", round=0)
        telem = Path(tmp) / "telem.jsonl"
        result = solve_problem(store, problem, cfg, backend=backend(),
                               critic_analyses=analyses(), telemetry_path=telem)

        log = read_telemetry(store, telem)
        ctx_tokens = [e["context_tokens"] for e in log] or [0]
        last = result.rounds[-1].metrics if result.rounds else None
        counts = store.counts()
        # The observable signal vector: if two configs produce identical vectors,
        # the ablation did nothing to what the operations actually did. It mixes
        # OUTCOME signals (records produced, which units ran, stop) with the
        # APPLIED-configuration signals recorded per call in telemetry
        # (context mode, order mode, excluded components) — so an ablation whose
        # effect is a different context/ordering/exclusion is caught even when a
        # trivial fixture makes the final outcome coincide. An UNWIRED flag
        # would leave all of these unchanged and correctly register as a no-op.
        observable = {
            "rounds": len(result.rounds),
            "stop_reason": result.stop_reason,
            "ctx_tokens_max": max(ctx_tokens),
            "unit_calls": len(log),
            "units": tuple(sorted({e["unit"] for e in log})),
            "records": tuple(sorted(counts.items())),
            "issue_order": tuple(
                (r.meta.get("severity"), r.id) for r in store.issues[:8]),
            "context_modes": tuple(sorted({e.get("mode") for e in log})),
            "order_modes": tuple(sorted({e.get("order_mode") for e in log})),
            "excluded": tuple(sorted({tuple(e.get("excluded_components") or []) for e in log})),
        }
        return AblationRun(
            config=name, rounds=len(result.rounds), stop_reason=result.stop_reason,
            delivered_completeness=(last.completeness if last else None),
            proved_terminal_fraction=(last.proved_terminal_fraction if last else None),
            open_critical=(last.open_critical if last else None),
            issues_opened=sum(r.issues_opened for r in result.rounds),
            context_tokens_mean=round(sum(ctx_tokens) / len(ctx_tokens), 1),
            records=counts, observable=observable,
        )


def run_matrix(*, n_rounds: int = 3, backend=None, analyses=None,
               initial_proof: str | None = None, problem: dict | None = None) -> list[AblationRun]:
    """Run every configuration in the matrix and return their metrics."""
    from .solve import _fake_backend, _fake_critic_analyses

    backend = backend or _fake_backend
    analyses = analyses or _fake_critic_analyses
    problem = problem or {"title": "T", "normalized_statement": "prove it"}
    initial_proof = initial_proof or _DEFAULT_PROOF
    return [_run_one(name, flags, n_rounds=n_rounds, backend=backend, analyses=analyses,
                     initial_proof=initial_proof, problem=problem)
            for name, flags in MATRIX]


_DEFAULT_PROOF = (
    r"\documentclass{article}\begin{document}"
    r"\begin{lemma}\label{crux}The crux bound holds.\end{lemma}"
    r"\begin{theorem}\label{main}The result follows.\end{theorem}"
    r"\begin{proof}By \ref{crux}.\end{proof}"
    r"\end{document}"
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def run_ablate_matrix(args) -> int:
    """`rma ablate-matrix --list` — enumerate the runnable configurations."""
    if getattr(args, "json", False):
        print(json.dumps(matrix_names(), indent=2))
    else:
        for name in matrix_names():
            print(name)
    return 0


def run_report_ablations(args) -> int:
    """`rma report-ablations --out X.json` — run every config, emit metrics."""
    backend_mode = getattr(args, "backend", "fake")
    if backend_mode != "fake":
        print("report-ablations currently supports --backend fake only "
              "(offline, deterministic).")
        return 1
    n_rounds = int(getattr(args, "rounds", None) or 3)
    runs = run_matrix(n_rounds=n_rounds)

    full = next(r for r in runs if r.config == "full")
    configs = []
    for r in runs:
        configs.append({
            "config": r.config,
            "rounds": r.rounds,
            "stop_reason": r.stop_reason,
            "completeness": r.delivered_completeness,
            "proved_terminal_fraction": r.proved_terminal_fraction,
            "open_critical": r.open_critical,
            "issues_opened": r.issues_opened,
            "context_tokens_mean": r.context_tokens_mean,
            "records": r.records,
            "differs_from_full": r.config != "full" and r.observable != full.observable,
        })
    out = {"n_rounds": n_rounds, "backend": "fake", "n_configs": len(configs),
           "configs": configs}

    dest = getattr(args, "out", None)
    if dest:
        Path(dest).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(configs)} configs -> {dest}")
    else:
        print(json.dumps(out, indent=2))
    return 0
