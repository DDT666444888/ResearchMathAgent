"""Regression tests for the adversarial-review findings (all CONFIRMED)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.store import open_store


class EmptyBodyIssueTest(unittest.TestCase):
    """store.py:160 — an empty-body, no-title issue must not IndexError."""

    def test_empty_body_issue_uses_untitled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = open_store(tmp, "q6", "first_proof_1")
            rec = s.add("I", "issue", "")          # empty body, no meta title
            self.assertIsNotNone(rec)
            from webapp.issues import get_issue
            issue = get_issue(Path(tmp), "q6", rec.meta["issue_id"], "first_proof_1")
            self.assertEqual(issue["title"], "Untitled")

    def test_writeback_empty_issue_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = open_store(tmp, "q6", "first_proof_1")
            written = s.write_back("critic", None, [{"component": "I", "kind": "issue"}])
            self.assertEqual(len(written), 1)


class ConceptsNoDoubleCountTest(unittest.TestCase):
    """store.py:319 — a concept added without meta['name'] must not double-count."""

    def test_body_named_concept_not_double_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = open_store(tmp, "q6", "first_proof_1")
            s.add("K", "concept", "a spectral gap is the second eigenvalue")  # no meta name
            self.assertEqual(len(s.reopen().concepts), 1)


class MemoryAblationReopenTest(unittest.TestCase):
    """store.py:539 — reopen() must preserve the memory-ablation subclass."""

    def test_stateless_store_reopen_stays_stateless(self) -> None:
        from rma.store import StatelessStore
        with tempfile.TemporaryDirectory() as tmp:
            s = open_store(tmp, "q6", "first_proof_1", memory="stateless")
            self.assertIsInstance(s.reopen(), StatelessStore)


class BestOfRoundsDeliveryTest(unittest.TestCase):
    """round_loop — the orchestrator must deliver the best round, not the last."""

    def test_solve_result_records_best_delivered_round(self) -> None:
        from rma.ops.base import FakeBackend
        from rma.ranking import Issue
        from rma.round_loop import solve_problem

        INITIAL = (r"\documentclass{article}\begin{document}"
                   r"\begin{lemma}\label{a}A.\end{lemma}"
                   r"\begin{theorem}\label{t}T.\end{theorem}\begin{proof}By \ref{a}.\end{proof}"
                   r"\end{document}")
        with tempfile.TemporaryDirectory() as tmp:
            store = open_store(tmp, "q6", "first_proof_1")
            store.add_proof_revision(INITIAL, produced_by="proposer")

            # completeness: round 0 best (9), then regress (3,3) — non-monotone.
            scores = iter([9, 3, 3])

            def _eval(_obs):
                try:
                    c = next(scores)
                except StopIteration:
                    c = 3
                return {"answer_accuracy": 0, "logical_correctness": c,
                        "proof_completeness": c, "proof_clarity": c, "verdict": "x"}

            backend = FakeBackend({
                "solver": {"replaced_text": "", "replacement": ""},
                "literature": [], "meeting": {"record": "m", "action_plan": {"summary": "", "steps": []}, "insights": []},
                "revise": "", "concepts": [], "evaluator": _eval,
            })

            def _an():
                return {"lm": lambda p, a: [], "semantic": lambda pr, p, a: (None, [])}

            res = solve_problem(store, {"title": "T"}, RunConfig(n_rounds=3),
                                backend=backend, critic_analyses=_an())
            # round 0 had completeness 9; later rounds 3 -> best delivered is round 0
            self.assertIsNotNone(res.delivered_proof_id)
            self.assertEqual(res.delivered_round, 0,
                             f"delivered {res.delivered_round}, expected the peak round 0")


class SolverProductionFallbackTest(unittest.TestCase):
    """solver.py:210 — the full-rewrite fallback must not be inert when no
    backend is injected (it should build the real one)."""

    def test_fallback_builds_a_real_invoker_when_none(self) -> None:
        from unittest.mock import patch
        from rma.ops import get_operation
        from rma.ops.base import OpContext

        with tempfile.TemporaryDirectory() as tmp:
            store = open_store(tmp, "q6", "first_proof_1")
            store.add_proof_revision(r"\documentclass{article}\begin{document}orig\end{document}",
                                     produced_by="p")
            ctx = OpContext(store=store, config=RunConfig(), problem={"title": "T"}, round=1)

            class R:
                text = r"\documentclass{article}\begin{document}REWRITTEN\end{document}"
                provider = "claude-code"; model = "claude-opus-4-8"

            # A patch that always fails to apply -> forces the fallback path with
            # invoke=None (production), which must build a real CLI call.
            calls = {"cc": 0}
            def fake_cc(**kw):
                calls["cc"] += 1; return R()
            def fake_patch(**kw):
                return R()  # any non-applying reply; solver will reject + retry

            solver = get_operation("solver")
            with patch("rma.models.call_claude_code", side_effect=fake_cc), \
                 patch("rma.ops.solver.apply_patch") as ap:
                from rma.ops.solver import PatchResult
                ap.return_value = PatchResult(store.current_proof().body, False, "reject", 1, "no")
                solver.run(ctx, issue={"id": "i1", "message": "m"}, invoke=None)
            # the fallback made at least one real (subscription) model call
            self.assertGreaterEqual(calls["cc"], 1, "production fallback was inert")


if __name__ == "__main__":
    unittest.main()


class FakeStoreIsolationTest(unittest.TestCase):
    """`--backend fake` must not write into the real research store (keyed by
    repo_root). Offline runs at the real repo were polluting webapp/issues."""

    def test_fake_backend_uses_isolated_store(self) -> None:
        import inspect
        from rma import solve
        src = inspect.getsource(solve._run_orchestrator)
        self.assertIn("_fake_store", src)
        self.assertIn('backend_mode == "fake"', src)
