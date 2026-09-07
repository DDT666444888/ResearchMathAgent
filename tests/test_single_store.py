"""T7.1 — push-forward and rma solve read/write ONE store per question.

The paper says "each question has one research store S" (main.tex:357). The
website's push-forward round uses webapp.issues / webapp.proof_history / ...;
the orchestrator uses rma.store. This test proves those are two views of the
same state: work done through either entry point is visible to the other.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.store import open_store


class SharedStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_push_forward_issue_visible_to_the_orchestrator(self) -> None:
        """An issue opened the webapp/push-forward way (webapp.issues) must be
        an open issue in the store the critic dedups against."""
        from webapp.issues import create_issue

        created = create_issue(self.root, "q6", title="Crux lemma unproved",
                               body="opened by the push-forward critic agent",
                               dataset="first_proof_1", priority="P0", issue_type="proof-gap")

        store = open_store(self.root, "q6", "first_proof_1")
        open_ids = {r.meta.get("issue_id") for r in store.open_issues()}
        self.assertIn(created["id"], open_ids, "webapp issue not visible as an open store issue")
        self.assertEqual(store.issues[0].body, "Crux lemma unproved")   # title surfaces as body
        self.assertEqual(store.issues[0].meta["severity"], "P0")

    def test_orchestrator_issue_visible_to_push_forward(self) -> None:
        """And the reverse: an issue the critic writes via the store is a normal
        webapp issue that push-forward's list_issues sees."""
        from webapp.issues import list_issues

        store = open_store(self.root, "q6", "first_proof_1")
        store.add("I", "issue", "orchestrator-found gap",
                  meta={"title": "gap", "severity": "P1"})

        issues = list_issues(self.root, "q6", "first_proof_1")
        self.assertEqual([i["title"] for i in issues], ["gap"])
        self.assertEqual(issues[0]["priority"], "P1")

    def test_proof_revision_from_solve_visible_in_webapp(self) -> None:
        """A Pi revision the orchestrator writes is in the same proof_history
        the website and push-forward read."""
        from webapp.proof_history import list_proof_history

        store = open_store(self.root, "q6", "first_proof_1")
        store.add_proof_revision(r"\begin{document}orchestrator proof\end{document}",
                                 produced_by="revise")

        history = list_proof_history(self.root, "q6")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["agent"], "revise")

    def test_working_proof_from_webapp_visible_to_the_store(self) -> None:
        """A proof recorded the push-forward way (save_working_proof ->
        proof_history) is the store's current proof."""
        from webapp.issue_agents import save_working_proof

        save_working_proof(self.root, "q6", r"\begin{document}website proof\end{document}",
                           agent="solver-agent", dataset="first_proof_1")
        store = open_store(self.root, "q6", "first_proof_1")
        self.assertIsNotNone(store.current_proof())
        self.assertIn("website proof", store.current_proof().body)

    def test_solver_resolution_visible_to_webapp(self) -> None:
        """When the orchestrator's solver closes an issue, the website sees it
        resolved (this is what lets a shared dashboard reflect solve progress)."""
        from webapp.issues import get_issue

        store = open_store(self.root, "q6", "first_proof_1")
        rec = store.add("I", "issue", "a gap", meta={"title": "a gap", "severity": "P1"})
        issue_id = rec.meta["issue_id"]

        store.set_issue_status(rec.id, "resolved")
        self.assertEqual(get_issue(self.root, "q6", issue_id, "first_proof_1")["status"],
                         "resolved")

    def test_bridge_helper_opens_the_same_store(self) -> None:
        """webapp.research_store.for_problem is the one entry point both sides
        use, so convergence is by design rather than by coincidence."""
        from webapp.research_store import for_problem

        store_a = for_problem(self.root, "q6", "first_proof_1")
        store_a.add("H", "insight", "shared insight")
        store_b = open_store(self.root, "q6", "first_proof_1")
        self.assertIn("shared insight", [r.body for r in store_b.insights])


if __name__ == "__main__":
    unittest.main()
