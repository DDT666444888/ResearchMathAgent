"""T4.4 — Literature keyed by Q, with the paper's four fields."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.ops import get_operation
from rma.ops.base import OpContext
from rma.ops.literature import validate_source
from rma.ranking import Issue
from rma.store import ResearchStore

GOOD_SOURCE = {
    "theorem_or_technique": "Marcus-Spielman-Srivastava paving",
    "assumptions": "finite Hermitian matrices with bounded diagonal",
    "claims_supported": ["q6-I-1"],
    "applicability_limits": "requires the interlacing family to exist",
    "source": "https://arxiv.org/abs/1408.4421",
}


class ValidateTest(unittest.TestCase):
    def test_all_four_fields_required(self) -> None:
        self.assertTrue(validate_source(GOOD_SOURCE))
        for missing in ("theorem_or_technique", "assumptions", "applicability_limits"):
            bad = dict(GOOD_SOURCE); bad[missing] = ""
            self.assertFalse(validate_source(bad), f"missing {missing} should be rejected")

    def test_claims_supported_may_be_empty(self) -> None:
        bg = dict(GOOD_SOURCE); bg["claims_supported"] = []
        self.assertTrue(validate_source(bg))

    def test_claims_supported_key_required(self) -> None:
        bad = {k: v for k, v in GOOD_SOURCE.items() if k != "claims_supported"}
        self.assertFalse(validate_source(bad))


class LiteratureOpTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(r"\begin{document}proof\end{document}", produced_by="p")
        # a ranked queue with a real issue id
        issue = self.store.add("I", "issue", "crux lemma unproved",
                               meta={"title": "crux", "severity": "P0"})
        self.q = [Issue(id=issue.id, code="unproved_lemma", message="crux lemma unproved")]
        self.ctx = OpContext(store=self.store, config=RunConfig(),
                             problem={"title": "T"}, round=0,
                             extra={"queue": self.q})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_query_derives_from_ranked_queue(self) -> None:
        op = get_operation("literature")
        q = op.query(self.ctx)
        self.assertIn(self.q[0].id, q.links)
        self.assertIn("crux lemma unproved", q.text)

    def test_all_four_fields_required_end_to_end(self) -> None:
        op = get_operation("literature")
        incomplete = {k: v for k, v in GOOD_SOURCE.items() if k != "assumptions"}
        result = op.run(self.ctx, invoke=lambda u, o: [GOOD_SOURCE, incomplete])
        # only the complete source survives
        self.assertEqual(len(result.written), 1)
        rec = self.store.literature[0]
        for field in ("theorem_or_technique", "assumptions", "claims_supported",
                      "applicability_limits"):
            self.assertIn(field, rec.meta)

    def test_record_links_to_issue_ids(self) -> None:
        op = get_operation("literature")
        op.run(self.ctx, invoke=lambda u, o: [GOOD_SOURCE])
        rec = self.store.literature[0]
        self.assertIn(self.q[0].id, rec.links)

    def test_records_retrievable_as_linked_records_next_round(self) -> None:
        """The regression that matters: L must reach a later operation's context.
        A solver querying the same issue should pull the literature note via
        LinkedRecords."""
        from rma.orchestrator import Query, linked_records

        op = get_operation("literature")
        op.run(self.ctx, invoke=lambda u, o: [GOOD_SOURCE])
        note = self.store.literature[0]
        # the solver's query links to the issue; the note links to the issue too,
        # so one hop from the issue reaches the note.
        issue_id = self.q[0].id
        linked = linked_records(self.store, Query(text="fix it", links=[issue_id]))
        self.assertIn(note.id, {r.id for r in linked},
                      "literature note is not retrievable as a linked record")

    def test_empty_reply_writes_nothing(self) -> None:
        op = get_operation("literature")
        result = op.run(self.ctx, invoke=lambda u, o: [])
        self.assertEqual(result.written, [])


if __name__ == "__main__":
    unittest.main()
