"""T4.5 — Meeting emits (rho, a, DeltaH)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.ops import get_operation
from rma.ops.base import OpContext
from rma.store import ResearchStore

SYNTH = {
    "record": "Coordinator: the spectral gap is the crux. Persona A suggests paving.",
    "action_plan": {"summary": "Bound the spectral gap via paving, then average.",
                    "steps": ["prove the paving lemma", "average over parts"]},
    "insights": ["The dense case is the real obstacle.",
                 "A random partition may beat the greedy one."],
}


class MeetingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(r"\begin{document}p\end{document}", produced_by="p")
        self.ctx = OpContext(store=self.store, config=RunConfig(), problem={"title": "T"}, round=2)

    def tearDown(self):
        self._tmp.cleanup()

    def test_synthesis_returns_rho_a_and_delta_h(self):
        op = get_operation("meeting")
        op.run(self.ctx, invoke=lambda u, o: SYNTH)
        kinds = {r.kind for r in self.store.meetings}
        self.assertIn("meeting_record", kinds)   # rho
        self.assertIn("action_plan", kinds)      # a
        self.assertEqual(len(self.store.insights), 2)  # DeltaH

    def test_rho_written_to_M_and_delta_h_to_H(self):
        op = get_operation("meeting")
        op.run(self.ctx, invoke=lambda u, o: SYNTH)
        rho = [r for r in self.store.meetings if r.kind == "meeting_record"]
        self.assertEqual(len(rho), 1)
        self.assertIn("spectral gap", rho[0].body)
        self.assertEqual({r.body for r in self.store.insights},
                         set(SYNTH["insights"]))

    def test_action_plan_id_captured_for_revise(self):
        op = get_operation("meeting")
        op.run(self.ctx, invoke=lambda u, o: SYNTH)
        self.assertIn("action_plan_id", self.ctx.extra)
        plan = self.store.get(self.ctx.extra["action_plan_id"])
        self.assertEqual(plan.kind, "action_plan")
        self.assertEqual(plan.meta["steps"], SYNTH["action_plan"]["steps"])

    def test_action_plan_outranks_transcript(self):
        op = get_operation("meeting")
        op.run(self.ctx, invoke=lambda u, o: SYNTH)
        by_kind = {r.kind: r for r in self.store.meetings}
        self.assertGreater(by_kind["action_plan"].priority, by_kind["meeting_record"].priority)

    def test_string_action_plan_tolerated(self):
        op = get_operation("meeting")
        op.run(self.ctx, invoke=lambda u, o: {"record": "r", "action_plan": "just do X",
                                              "insights": []})
        plan = [r for r in self.store.meetings if r.kind == "action_plan"]
        self.assertEqual(plan[0].body, "just do X")

    def test_ablate_meeting_is_a_config_flag(self):
        # the round loop honours cfg.ablated("meeting"); here we assert the flag exists
        cfg = RunConfig(ablations=frozenset({"meeting"}))
        self.assertTrue(cfg.ablated("meeting"))


if __name__ == "__main__":
    unittest.main()
