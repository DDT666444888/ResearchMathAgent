"""T1.2 — the store is a facade over existing persistence, not a fork.

Writes made through the store must be visible to the webapp modules, and work
done in the website must be visible to the orchestrator. If these fail, the
paper's "each question has one research store S" is false for the system.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.store import ResearchStore


class AdapterTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")

    def tearDown(self) -> None:
        self._tmp.cleanup()


class IssuesAdapterTest(AdapterTestBase):
    def test_issue_written_via_store_visible_to_webapp(self) -> None:
        from webapp.issues import list_issues

        self.store.add("I", "issue", "The crux lemma is unproved.",
                       meta={"title": "Crux lemma unproved", "severity": "P0",
                             "issue_type": "proof-gap"})
        issues = list_issues(self.root, "q6", "first_proof_1")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["title"], "Crux lemma unproved")
        self.assertEqual(issues[0]["priority"], "P0")
        self.assertEqual(issues[0]["issue_type"], "proof-gap")

    def test_issue_written_via_webapp_visible_to_store(self) -> None:
        from webapp.issues import create_issue

        create_issue(self.root, "q6", title="Opened in the website",
                     body="A human spotted this.", dataset="first_proof_1",
                     priority="P1", issue_type="missing-case")
        records = self.store.issues
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].body, "Opened in the website")
        self.assertEqual(records[0].meta["severity"], "P1")
        self.assertEqual(records[0].meta["origin"], "webapp")

    def test_no_double_counting_across_both_paths(self) -> None:
        from webapp.issues import create_issue, list_issues

        self.store.add("I", "issue", "from store", meta={"title": "from store"})
        create_issue(self.root, "q6", title="from webapp", dataset="first_proof_1")
        self.assertEqual(len(list_issues(self.root, "q6", "first_proof_1")), 2)
        self.assertEqual(len(self.store.issues), 2,
                         "a store-written issue was counted twice")

    def test_status_change_in_webapp_is_reflected_in_store(self) -> None:
        """Canonical fields are owned by the issue tracker, not the sidecar."""
        from webapp.issues import update_issue

        rec = self.store.add("I", "issue", "gap", meta={"title": "gap"})
        self.assertEqual(self.store.issues[0].meta["status"], "open")
        update_issue(self.root, "q6", rec.meta["issue_id"],
                     dataset="first_proof_1", status="resolved")
        self.assertEqual(self.store.issues[0].meta["status"], "resolved")


class ProofsAdapterTest(AdapterTestBase):
    def test_proof_written_via_store_visible_to_webapp(self) -> None:
        from webapp.proof_history import get_proof_version_tex, list_proof_history

        self.store.add_proof_revision("\\begin{proof}x\\end{proof}", produced_by="solver")
        history = list_proof_history(self.root, "q6")
        self.assertEqual(len(history), 1)
        self.assertEqual(get_proof_version_tex(self.root, "q6", 1),
                         "\\begin{proof}x\\end{proof}")

    def test_proof_written_via_webapp_visible_to_store(self) -> None:
        from webapp.proof_history import record_proof_version

        record_proof_version(self.root, "q6", "website proof", agent="human")
        proofs = self.store.proofs
        self.assertEqual(len(proofs), 1)
        self.assertEqual(proofs[0].body, "website proof")
        self.assertEqual(self.store.current_proof().body, "website proof")


class LiteratureAdapterTest(AdapterTestBase):
    def test_literature_written_via_store_visible_to_webapp(self) -> None:
        from webapp.literature import load_index

        self.store.add("L", "literature_note", "Gives the paving bound.",
                       meta={"url": "https://arxiv.org/abs/1408.4421",
                             "title": "Interlacing Families II"})
        papers = load_index(self.root, "q6")
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "Interlacing Families II")
        self.assertEqual(papers[0]["notes"], "Gives the paving bound.")

    def test_literature_written_via_webapp_visible_to_store(self) -> None:
        from webapp.literature import add_paper

        add_paper(self.root, "q6", url="https://example.org/p", title="A paper",
                  notes="useful lemma")
        records = self.store.literature
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].body, "useful lemma")


class ConceptsAdapterTest(AdapterTestBase):
    def test_concept_written_via_store_visible_to_webapp(self) -> None:
        from webapp.concepts import load_concepts

        self.store.add("K", "concept", "L_S <= eps L", meta={"name": "eps-light"})
        entries = load_concepts(self.root, "q6")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "eps-light")

    def test_concept_written_via_webapp_visible_to_store(self) -> None:
        from webapp.concepts import save_concepts

        save_concepts(self.root, "q6", [{"name": "Laplacian", "definition": "D - A"}])
        self.assertEqual(len(self.store.concepts), 1)
        self.assertEqual(self.store.concepts[0].meta["name"], "Laplacian")

    def test_duplicate_concept_name_not_appended_twice(self) -> None:
        from webapp.concepts import load_concepts

        self.store.add("K", "concept", "first", meta={"name": "Laplacian"})
        self.store.add("K", "concept", "second", meta={"name": "laplacian"})
        self.assertEqual(len(load_concepts(self.root, "q6")), 1)


class InsightsAdapterTest(AdapterTestBase):
    def test_insight_written_via_store_visible_to_webapp(self) -> None:
        from webapp.insights import get_question_insight

        self.store.add("H", "insight", "Try a random partition.")
        data = get_question_insight(self.root, "q6", "first_proof_1")
        self.assertIsNotNone(data)
        self.assertEqual(data["insights"][0]["text"], "Try a random partition.")


class EvaluationsAdapterTest(AdapterTestBase):
    def test_evaluation_series_is_append_only_and_refreshes_canonical(self) -> None:
        from webapp.proof_eval import load_proof_eval

        self.store.begin_round(0)
        self.store.add("E", "evaluation", "round 0",
                       meta={"scores": {"proof_completeness": 4}})
        self.store.begin_round(1)
        self.store.add("E", "evaluation", "round 1",
                       meta={"scores": {"proof_completeness": 7}})

        # The store keeps every evaluation...
        self.assertEqual(len(self.store.evaluations), 2)
        # ...while the website's single-slot file shows the latest.
        self.assertEqual(load_proof_eval(self.root, "q6")["proof_completeness"], 7)


class MeetingsAdapterTest(AdapterTestBase):
    def test_room_created_in_webapp_visible_to_store(self) -> None:
        from webapp.meet import create_room

        create_room(self.root, "q6", topic="Strategy review", goal="pick an approach")
        records = self.store.meetings
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].kind, "meeting_record")
        self.assertEqual(records[0].meta["topic"], "Strategy review")

    def test_action_plan_is_a_separate_higher_priority_record(self) -> None:
        from webapp.meet import create_room, set_plan

        room = create_room(self.root, "q6", topic="t")
        set_plan(self.root, "q6", room["id"],
                 summary="Bound the spectral gap first.",
                 steps=[{"title": "prove Lemma 1"}])
        kinds = {r.kind: r for r in self.store.meetings}
        self.assertIn("action_plan", kinds)
        self.assertEqual(kinds["action_plan"].body, "Bound the spectral gap first.")
        self.assertGreater(kinds["action_plan"].priority, kinds["meeting_record"].priority)


class InsightShapeTest(AdapterTestBase):
    """Insight files written before this store use the generator's shape
    (summary / highlights / suggested_todos), not a flat "insights" list."""

    def test_legacy_generator_shape_is_read(self) -> None:
        from webapp.insights import save_question_insight

        save_question_insight(self.root, "q6", "first_proof_1", {
            "summary": "The paving route looks strongest.",
            "highlights": ["c = 1/42 is achievable", "the dense case is the blocker"],
            "suggested_todos": ["check the disconnected case"],
        })
        bodies = {r.body for r in self.store.insights}
        self.assertIn("The paving route looks strongest.", bodies)
        self.assertIn("c = 1/42 is achievable", bodies)
        self.assertIn("check the disconnected case", bodies)

    def test_store_written_insight_still_round_trips(self) -> None:
        rec = self.store.add("H", "insight", "Try averaging.")
        self.assertIn("Try averaging.", {r.body for r in self.store.reopen().insights})
        self.assertEqual(self.store.reopen().get(rec.id).body, "Try averaging.")


class EvaluationPullTest(AdapterTestBase):
    def test_existing_proof_eval_is_surfaced_as_E(self) -> None:
        from webapp.proof_eval import _save_proof_eval

        _save_proof_eval(self.root, "q6", {
            "answer_accuracy": 1, "logical_correctness": 8,
            "proof_completeness": 6, "proof_clarity": 9,
            "verdict": "Solid but incomplete.",
        })
        records = self.store.evaluations
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].body, "Solid but incomplete.")
        self.assertEqual(records[0].meta["scores"]["proof_completeness"], 6)

    def test_canonical_eval_not_duplicated_once_store_writes_its_own(self) -> None:
        from webapp.proof_eval import _save_proof_eval

        _save_proof_eval(self.root, "q6", {"proof_completeness": 6, "verdict": "v"})
        self.assertEqual(len(self.store.evaluations), 1)
        self.store.add("E", "evaluation", "round 0", meta={"scores": {"proof_completeness": 7}})
        self.assertEqual(len(self.store.evaluations), 1,
                         "canonical single-slot eval double-counted alongside the series")


class AllComponentsAdaptedTest(AdapterTestBase):
    """Every component the module docstring claims to adapt must really pull."""

    def test_no_component_is_a_silent_noop(self) -> None:
        from rma.store import ADAPTERS, COMPONENTS

        for comp in COMPONENTS:
            adapter = ADAPTERS[comp]
            self.assertIs(type(adapter).pull is not None, True)
            self.assertNotEqual(
                type(adapter).__name__, "ComponentAdapter",
                f"component {comp} still uses the no-op base adapter",
            )


class DatasetScopeTest(AdapterTestBase):
    def test_two_datasets_do_not_share_issue_records(self) -> None:
        other = ResearchStore.open(self.root, "q6", "first_proof_2")
        self.store.add("I", "issue", "fp1 issue", meta={"title": "fp1 issue"})
        other.add("I", "issue", "fp2 issue", meta={"title": "fp2 issue"})
        self.assertEqual([r.body for r in self.store.issues], ["fp1 issue"])
        self.assertEqual([r.body for r in other.issues], ["fp2 issue"])


if __name__ == "__main__":
    unittest.main()
