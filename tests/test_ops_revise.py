"""T4.6 — Revise consumes the action plan and emits a new revision."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.ops import get_operation
from rma.ops.base import OpContext
from rma.store import ResearchStore

ORIG = r"\documentclass{article}\begin{document}" + ("x " * 200) + r"\end{document}"
REVISED = r"\documentclass{article}\begin{document}" + ("y " * 400) + r"\end{document}"


class ReviseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(ORIG, produced_by="proposer")
        plan = self.store.add("M", "action_plan", "restructure around paving",
                              meta={"steps": ["s1", "s2"], "is_action_plan": True})
        self.ctx = OpContext(store=self.store, config=RunConfig(), problem={"title": "T"},
                             round=3, extra={"action_plan": {"summary": "restructure", "steps": ["s1"]},
                                             "action_plan_id": plan.id})

    def tearDown(self):
        self._tmp.cleanup()

    def test_revise_consumes_action_plan_and_emits_new_revision(self):
        op = get_operation("revise")
        op.run(self.ctx, invoke=lambda u, o: REVISED)
        self.assertEqual(len(self.store.proofs), 2)
        self.assertEqual(self.store.current_proof().body, REVISED)
        self.assertEqual(self.store.current_proof().meta["mode"], "coordinated_revision")

    def test_instructions_carry_the_plan_steps(self):
        op = get_operation("revise")
        instr = op.instructions(self.ctx)
        self.assertIn("restructure", instr)
        self.assertIn("s1", instr)

    def test_revision_records_plan_id_as_link(self):
        op = get_operation("revise")
        op.run(self.ctx, invoke=lambda u, o: REVISED)
        latest = self.store.current_proof()
        self.assertEqual(latest.meta["action_plan_id"], self.ctx.extra["action_plan_id"])
        self.assertIn(self.ctx.extra["action_plan_id"], latest.links)

    def test_parent_is_the_prior_proof(self):
        op = get_operation("revise")
        op.run(self.ctx, invoke=lambda u, o: REVISED)
        self.assertEqual(self.store.current_proof().meta["parent_id"], self.store.proofs[0].id)

    def test_empty_reply_writes_no_revision(self):
        op = get_operation("revise")
        op.run(self.ctx, invoke=lambda u, o: "")
        self.assertEqual(len(self.store.proofs), 1)

    def test_revise_diff_is_larger_than_solver_diff(self):
        """The coordinated/localized distinction is real, not nominal: a revise
        replaces the whole proof; a solver patch touches one span."""
        from rma.ops.solver import apply_patch
        op = get_operation("revise")
        op.run(self.ctx, invoke=lambda u, o: REVISED)
        revise_changed = sum(1 for a, b in zip(ORIG, REVISED) if a != b) + abs(len(ORIG) - len(REVISED))
        patch = apply_patch(ORIG, {"replaced_text": "x x", "replacement": "x z"})
        solver_changed = sum(1 for a, b in zip(ORIG, patch.tex) if a != b)
        self.assertGreater(revise_changed, solver_changed)

    def test_revise_strips_narration_and_fences_before_storing(self):
        """The model prepends narration and wraps the body in a ```latex fence;
        stored verbatim that pollutes Pi and every downstream PDF. The revise op
        must store pure LaTeX."""
        fence = "```"
        reply = ("I'll execute the plan. Here is the document.\n\n"
                 + fence + "latex\n" + REVISED + "\n" + fence + "\n\nThat is done.")
        op = get_operation("revise")
        op.run(self.ctx, invoke=lambda u, o: reply)
        body = self.store.current_proof().body
        self.assertEqual(body, REVISED)
        self.assertNotIn(fence, body)
        self.assertNotIn("I'll execute", body)


class CleanLatexReplyTest(unittest.TestCase):
    def test_premature_close_and_stray_fence_keep_all_content(self):
        """The worst real case: a fence closed early and a stray ```latex left
        mid-document. Extracting 'the fenced block' would truncate the proof —
        the cleaner must keep every part and drop only the markers/narration."""
        from rma.models import clean_latex_reply
        fence = "```"
        raw = ("Here it is.\n\n" + fence + "latex\n\\clearpage\n\\section*{X}\n"
               "part1 <1" + fence + "latex\n$ part2\n\\end{thebibliography}\n"
               + fence + "\n\nThat is the complete document.")
        out = clean_latex_reply(raw)
        self.assertTrue(out.startswith("\\clearpage"))
        self.assertIn("part1", out)
        self.assertIn("part2", out)          # not truncated at the early close
        self.assertNotIn(fence, out)
        self.assertNotIn("Here it is", out)
        self.assertNotIn("complete document", out)

    def test_clean_document_passes_through_unchanged(self):
        from rma.models import clean_latex_reply
        d = "\\documentclass{article}\n\\begin{document}\nhi\n\\end{document}"
        self.assertEqual(clean_latex_reply(d), d)

    def test_leading_macro_is_not_eaten(self):
        from rma.models import clean_latex_reply
        self.assertTrue(clean_latex_reply("\\usepackage{amsmath}\n\\section{a}")
                        .startswith("\\usepackage"))


if __name__ == "__main__":
    unittest.main()
