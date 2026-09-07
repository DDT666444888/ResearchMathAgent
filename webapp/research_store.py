"""One door to the research store S, shared by the website and rma solve.

The persistence was already shared — the store's adapters read and write the
same webapp/issues, webapp/proof_history, webapp/meets, … that push-forward
uses. This module makes that convergence explicit and by-design: both the
push-forward round and the Algorithm 1 orchestrator open the store here, so
"each question has one research store S" (main.tex:357) is enforced at a single
call site rather than left to two code paths happening to touch the same files.
"""
from __future__ import annotations

from pathlib import Path


def for_problem(repo_root, problem_id: str, dataset: str = "first_proof_1", *,
                memory: str = "full"):
    """Open the research store for one problem. The one entry point both the
    website's push-forward and `rma solve`'s orchestrator use."""
    from rma.store import open_store

    return open_store(Path(repo_root), problem_id, dataset, memory=memory)


def record_push_forward_proof(repo_root, problem_id: str, tex: str, *,
                              dataset: str = "first_proof_1",
                              produced_by: str = "push-forward",
                              round: int | None = None):
    """Record a proof the push-forward produced as a Pi revision in the shared
    store, so website work and orchestrator work land on one append-only chain
    with provenance — instead of push-forward writing proof_history directly and
    the orchestrator never learning who produced it."""
    store = for_problem(repo_root, problem_id, dataset)
    return store.add_proof_revision(tex, produced_by=produced_by, round=round)


def snapshot(repo_root, problem_id: str, dataset: str = "first_proof_1") -> dict:
    """Per-component counts + current proof id — the shared-state view a
    dashboard or a push-forward summary can render for either system."""
    store = for_problem(repo_root, problem_id, dataset)
    proof = store.current_proof()
    return {
        **store.counts(),
        "problem_id": problem_id,
        "dataset": dataset,
        "current_proof_id": proof.id if proof else None,
        "open_issues": len(store.open_issues()),
    }
