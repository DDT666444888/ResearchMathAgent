"""T4.7 — Evaluator scores the current pi and appends to E."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.ops import get_operation
from rma.ops.base import OpContext
from rma.store import ResearchStore

SCORES = {"answer_accuracy": 1, "logical_correctness": 8, "proof_completeness": 6,
          "proof_clarity": 9, "verdict": "Solid but one lemma is thin."}


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(r"\begin{document}proof one\end{document}", produced_by="p")
        self.ctx = OpContext(store=self.store, config=RunConfig(),
                             problem={"title": "T", "normalized_statement": "prove it"}, round=0)

    def tearDown(self):
        self._tmp.cleanup()

    def test_evaluation_appended_to_E(self):
        op = get_operation("evaluator")
        op.run(self.ctx, invoke=lambda u, o: SCORES)
        self.assertEqual(len(self.store.evaluations), 1)
        rec = self.store.evaluations[0]
        self.assertEqual(rec.meta["scores"]["proof_completeness"], 6)
        self.assertEqual(rec.meta["total"], 1 + 8 + 6 + 9)

    def test_evaluation_records_proof_hash_and_links_the_proof(self):
        op = get_operation("evaluator")
        op.run(self.ctx, invoke=lambda u, o: SCORES)
        rec = self.store.evaluations[0]
        self.assertIn("proof_hash", rec.meta)
        self.assertIn(self.store.current_proof().id, rec.links)

    def test_append_only_series_across_rounds(self):
        op = get_operation("evaluator")
        self.ctx.round = 0
        op.run(self.ctx, invoke=lambda u, o: SCORES)
        # a genuinely different proof next round -> a second evaluation
        self.store.add_proof_revision(r"\begin{document}proof two, better\end{document}",
                                      produced_by="solver")
        self.ctx.round = 1
        op.run(self.ctx, invoke=lambda u, o: {**SCORES, "proof_completeness": 9})
        self.assertEqual(len(self.store.evaluations), 2)
        self.assertEqual(self.store.evaluations[-1].meta["scores"]["proof_completeness"], 9)

    def test_unchanged_proof_reuses_cached_score(self):
        op = get_operation("evaluator")
        op.run(self.ctx, invoke=lambda u, o: SCORES)
        # same proof, evaluate again -> no duplicate appended
        op.run(self.ctx, invoke=lambda u, o: {**SCORES, "proof_completeness": 3})
        self.assertEqual(len(self.store.evaluations), 1,
                         "identical proof re-rolled the judge / duplicated E")

    def test_scores_exposed_on_context(self):
        op = get_operation("evaluator")
        op.run(self.ctx, invoke=lambda u, o: SCORES)
        self.assertEqual(self.ctx.extra["evaluation"]["proof_completeness"], 6)


if __name__ == "__main__":
    unittest.main()
