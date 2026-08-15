"""T1.1 / T1.3 — the research store S = (Pi, I, M, L, K, H, E)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.store import COMPONENTS, Record, ResearchStore, StoreError


class StoreTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")

    def tearDown(self) -> None:
        self._tmp.cleanup()


# Bodies that exercise each component's adapter path.
_SAMPLES = {
    "Pi": ("proof_revision", "\\begin{proof}Full argument.\\end{proof}", {}),
    "I": ("issue", "Lemma 2 is never proved", {"title": "Lemma 2 unproved", "severity": "P0"}),
    "M": ("meeting_record", "Coordinator: we should bound the spectral gap.", {}),
    "L": ("literature_note", "Paving theorem gives the needed bound.",
          {"url": "https://arxiv.org/abs/1408.4421", "title": "Interlacing Families II"}),
    "K": ("concept", "A subset S is eps-light when L_S <= eps L.", {"name": "eps-light subset"}),
    "H": ("insight", "Try averaging over a random partition instead.", {}),
    "E": ("evaluation", "completeness 6/10",
          {"scores": {"proof_completeness": 6, "logical_correctness": 7}}),
}


class RoundtripTest(StoreTestBase):
    """Every component must survive a write -> reopen -> read cycle."""

    def _roundtrip(self, component: str) -> None:
        kind, body, meta = _SAMPLES[component]
        written = self.store.add(component, kind, body, meta=meta)
        fetched = self.store.reopen().get(written.id)
        self.assertIsNotNone(fetched, f"{component}: record vanished after reopen")
        self.assertEqual(fetched.id, written.id)
        self.assertEqual(fetched.component, component)
        self.assertEqual(fetched.body, body, f"{component}: body not preserved")
        self.assertEqual(fetched.kind, kind)
        self.assertEqual(fetched.round, written.round)

    def test_roundtrip_Pi(self) -> None:
        self._roundtrip("Pi")

    def test_roundtrip_I(self) -> None:
        self._roundtrip("I")

    def test_roundtrip_M(self) -> None:
        self._roundtrip("M")

    def test_roundtrip_L(self) -> None:
        self._roundtrip("L")

    def test_roundtrip_K(self) -> None:
        self._roundtrip("K")

    def test_roundtrip_H(self) -> None:
        self._roundtrip("H")

    def test_roundtrip_E(self) -> None:
        self._roundtrip("E")

    def test_roundtrip_all_seven_components_covered(self) -> None:
        self.assertEqual(set(_SAMPLES), set(COMPONENTS))


class RecordTest(unittest.TestCase):
    def test_unknown_component_rejected(self) -> None:
        with self.assertRaises(StoreError):
            Record(id="x", component="Z", round=0, kind="k", body="b")

    def test_json_round_trip(self) -> None:
        rec = Record(id="q6-I-1", component="I", round=2, kind="issue", body="b",
                     links=["q6-Pi-1"], meta={"severity": "P0"})
        self.assertEqual(Record.from_json(rec.to_json()), rec)


class ComponentQueryTest(StoreTestBase):
    def test_records_filtered_by_component(self) -> None:
        self.store.add("I", "issue", "one", meta={"title": "one"})
        self.store.add("H", "insight", "two")
        self.assertEqual(len(self.store.records("I")), 1)
        self.assertEqual(len(self.store.records("H")), 1)

    def test_records_filtered_by_round(self) -> None:
        self.store.begin_round(0)
        self.store.add("H", "insight", "r0")
        self.store.begin_round(1)
        self.store.add("H", "insight", "r1")
        self.assertEqual(len(self.store.records("H", round=0)), 1)
        self.assertEqual(len(self.store.records("H", round=1)), 1)
        self.assertEqual(len(self.store.records("H")), 2)

    def test_counts_covers_every_component(self) -> None:
        self.assertEqual(set(self.store.counts()), set(COMPONENTS))

    def test_named_views_match_components(self) -> None:
        self.store.add("H", "insight", "x")
        self.assertEqual(len(self.store.insights), 1)
        self.assertEqual(self.store.insights[0].body, "x")

    def test_unknown_component_query_rejected(self) -> None:
        with self.assertRaises(StoreError):
            self.store.records("nope")

    def test_open_issues_excludes_resolved(self) -> None:
        self.store.add("I", "issue", "a", meta={"title": "a", "status": "open"})
        self.store.add("I", "issue", "b", meta={"title": "b"})
        from webapp.issues import update_issue

        resolved = self.store.issues[0]
        update_issue(self.root, "q6", resolved.meta["issue_id"],
                     dataset="first_proof_1", status="resolved")
        self.assertEqual(len(self.store.open_issues()), 1)


class RevisionTest(StoreTestBase):
    """T1.3 — Pi is an append-only chain, not an overwritten file."""

    def test_three_revisions_chain(self) -> None:
        v1 = self.store.add_proof_revision("proof one", produced_by="proposer")
        v2 = self.store.add_proof_revision("proof two", produced_by="solver")
        v3 = self.store.add_proof_revision("proof three", produced_by="revise")

        history = self.store.proof_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(self.store.current_proof().body, "proof three")
        self.assertEqual(v2.meta["parent_id"], v1.id)
        self.assertEqual(v3.meta["parent_id"], v2.id)
        self.assertIn(v2.id, [r.id for r in history])

    def test_revision_records_producing_unit(self) -> None:
        rec = self.store.add_proof_revision("p", produced_by="solver")
        self.assertEqual(rec.meta["produced_by"], "solver")
        self.assertEqual(self.store.reopen().current_proof().meta["produced_by"], "solver")

    def test_current_proof_none_when_empty(self) -> None:
        self.assertIsNone(self.store.current_proof())

    def test_revision_survives_reopen(self) -> None:
        self.store.add_proof_revision("first", produced_by="proposer")
        self.store.add_proof_revision("second", produced_by="solver")
        self.assertEqual(self.store.reopen().current_proof().body, "second")

    def test_revision_visible_in_webapp_proof_history(self) -> None:
        from webapp.proof_history import list_proof_history

        self.store.add_proof_revision("shared proof", produced_by="solver")
        entries = list_proof_history(self.root, "q6")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["agent"], "solver")


class WriteBackTest(StoreTestBase):
    """S <- WriteBack(S, u, q, y) is the single entry point for op output."""

    def test_write_back_attaches_provenance(self) -> None:
        written = self.store.write_back(
            "critic", {"id": "q6-Pi-1"},
            [{"component": "I", "kind": "issue", "body": "gap A", "meta": {"title": "gap A"}},
             {"component": "I", "kind": "issue", "body": "gap B", "meta": {"title": "gap B"}}],
        )
        self.assertEqual(len(written), 2)
        for rec in written:
            self.assertEqual(rec.meta["unit"], "critic")
            self.assertEqual(rec.meta["query_id"], "q6-Pi-1")

    def test_write_back_ignores_unknown_components(self) -> None:
        self.assertEqual(self.store.write_back("critic", None, [{"component": "Z", "body": "x"}]), [])

    def test_write_back_accepts_single_dict(self) -> None:
        written = self.store.write_back("meeting", None,
                                        {"component": "M", "kind": "record", "body": "notes"})
        self.assertEqual(len(written), 1)

    def test_write_back_none_is_noop(self) -> None:
        self.assertEqual(self.store.write_back("critic", None, None), [])


if __name__ == "__main__":
    unittest.main()
