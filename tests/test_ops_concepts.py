"""T4.8 — UpdateConcepts folds proof + DeltaH into the glossary K."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.ops import get_operation
from rma.ops.base import OpContext
from rma.store import ResearchStore


class ConceptsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(r"\begin{document}uses an eps-light subset\end{document}",
                                      produced_by="p")
        self.ctx = OpContext(store=self.store, config=RunConfig(), problem={"title": "T"}, round=1)

    def tearDown(self):
        self._tmp.cleanup()

    def test_new_symbol_in_revision_appears_in_glossary(self):
        op = get_operation("concepts")
        op.run(self.ctx, invoke=lambda u, o: [
            {"name": "eps-light subset", "definition": "S with L_S <= eps L"}])
        self.assertEqual(len(self.store.concepts), 1)
        self.assertEqual(self.store.concepts[0].meta["name"], "eps-light subset")

    def test_existing_entries_not_duplicated(self):
        self.store.add("K", "concept", "already here", meta={"name": "Laplacian"})
        op = get_operation("concepts")
        op.run(self.ctx, invoke=lambda u, o: [
            {"name": "Laplacian", "definition": "D - A"},        # dup
            {"name": "eps-light subset", "definition": "..."}])  # new
        names = sorted(r.meta["name"] for r in self.store.concepts)
        self.assertEqual(names, ["Laplacian", "eps-light subset"])

    def test_insight_delta_reaches_the_prompt(self):
        op = get_operation("concepts")
        self.ctx.extra["insights"] = ["paving is the key technique"]
        q = op.query(self.ctx)
        self.assertIn("paving is the key technique", q.text)

    def test_existing_names_seeded_into_instructions(self):
        self.store.add("K", "concept", "x", meta={"name": "Laplacian"})
        op = get_operation("concepts")
        op.run(self.ctx, invoke=lambda u, o: [])
        # after run, the existing-names hint was populated
        self.assertIn("Laplacian", self.ctx.extra["existing_concept_names"])

    def test_nameless_entries_ignored(self):
        op = get_operation("concepts")
        op.run(self.ctx, invoke=lambda u, o: [{"definition": "no name"}])
        self.assertEqual(len(self.store.concepts), 0)


if __name__ == "__main__":
    unittest.main()
