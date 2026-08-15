"""T2.3 / T2.4 — Run(u, q, S, B): section order, write-back, telemetry."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.orchestrator import (
    SECTION_ORDER,
    Query,
    build_context,
    linked_records,
    read_telemetry,
    recent_outputs,
    run_unit,
)
from rma.store import ResearchStore


def _issues_artifact(n: int = 2):
    return [{"component": "I", "kind": "issue", "body": f"gap {i}",
             "meta": {"title": f"gap {i}"}} for i in range(n)]


class OrchestratorTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.calls: list[tuple[str, str]] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _invoke(self, artifact):
        def _fn(unit: str, observation: str):
            self.calls.append((unit, observation))
            return artifact
        return _fn


class SectionOrderTest(OrchestratorTestBase):
    def test_context_sections_in_paper_order(self) -> None:
        """C = Instructions || q || CurrentProof || LinkedRecords || RecentOutputs."""
        self.store.add_proof_revision("THE PROOF BODY", produced_by="proposer")
        issue = self.store.add("I", "issue", "LINKED ISSUE", meta={"title": "linked"})
        self.store.add("M", "meeting_record", "PRIOR CRITIC OUTPUT",
                       meta={"unit": "critic"})

        sections = build_context(
            self.store, "critic",
            Query(text="THE QUERY", id="q1", links=[issue.id]),
            instructions="THE INSTRUCTIONS",
        )
        self.assertEqual([s.name for s in sections], list(SECTION_ORDER))

        text = "\n".join(c.text for s in sections for c in s.chunks)
        positions = [text.index(m) for m in
                     ("THE INSTRUCTIONS", "THE QUERY", "THE PROOF BODY",
                      "LINKED ISSUE", "PRIOR CRITIC OUTPUT")]
        self.assertEqual(positions, sorted(positions),
                         "context parts are not in the paper's order")

    def test_instructions_and_query_are_not_droppable(self) -> None:
        sections = build_context(self.store, "critic", Query(text="q"),
                                 instructions="i")
        for section in sections[:2]:
            for chunk in section.chunks:
                self.assertFalse(chunk.droppable)

    def test_linked_records_follow_one_hop(self) -> None:
        a = self.store.add("M", "note", "A")
        b = self.store.add("M", "note", "B", links=[a.id])
        c = self.store.add("M", "note", "C", links=[b.id])
        # Query names c: c and b are reachable within one hop, a is two away.
        found = {r.id for r in linked_records(self.store, Query(text="q", links=[c.id]))}
        self.assertIn(c.id, found)
        self.assertIn(b.id, found)
        self.assertNotIn(a.id, found)

    def test_recent_outputs_are_newest_first(self) -> None:
        self.store.begin_round(0)
        self.store.add("M", "note", "oldest", meta={"unit": "critic"})
        self.store.begin_round(1)
        self.store.add("M", "note", "newest", meta={"unit": "critic"})
        bodies = [r.body for r in recent_outputs(self.store, "critic")]
        self.assertEqual(bodies[0], "newest")

    def test_recent_outputs_only_include_that_unit(self) -> None:
        self.store.add("M", "note", "mine", meta={"unit": "critic"})
        self.store.add("M", "note", "theirs", meta={"unit": "solver"})
        self.assertEqual([r.body for r in recent_outputs(self.store, "critic")], ["mine"])

    def test_proof_not_duplicated_into_linked_records(self) -> None:
        proof = self.store.add_proof_revision("P", produced_by="proposer")
        sections = build_context(self.store, "critic",
                                 Query(text="q", links=[proof.id]), instructions="i")
        by_name = {s.name: s for s in sections}
        linked_ids = [c.record_id for c in by_name["linked_records"].chunks]
        self.assertNotIn(proof.id, linked_ids)


class WriteBackTest(OrchestratorTestBase):
    def test_writeback_called_once_per_run(self) -> None:
        result = run_unit("critic", Query(text="find gaps"), self.store,
                          instructions="i", invoke=self._invoke(_issues_artifact(2)))
        self.assertEqual(len(result.written), 2)
        self.assertEqual(len(self.store.issues), 2)
        for rec in result.written:
            self.assertEqual(rec.meta["unit"], "critic")

    def test_writeback_is_not_optional(self) -> None:
        """Output that names no component writes nothing — and says so."""
        result = run_unit("critic", Query(text="q"), self.store,
                          instructions="i", invoke=self._invoke("just prose"))
        self.assertEqual(result.written, [])

    def test_run_returns_the_raw_output(self) -> None:
        artifact = _issues_artifact(1)
        result = run_unit("critic", "find gaps", self.store,
                          instructions="i", invoke=self._invoke(artifact))
        self.assertEqual(result.output, artifact)

    def test_round_is_recorded_on_written_records(self) -> None:
        self.store.begin_round(3)
        result = run_unit("critic", Query(text="q"), self.store,
                          instructions="i", invoke=self._invoke(_issues_artifact(1)))
        self.assertEqual(result.round, 3)
        self.assertEqual(result.written[0].round, 3)


class TelemetryTest(OrchestratorTestBase):
    def test_every_run_logs_one_line(self) -> None:
        for i in range(3):
            run_unit("critic", Query(text=f"q{i}"), self.store,
                     instructions="i", invoke=self._invoke(_issues_artifact(1)))
        log = read_telemetry(self.store)
        self.assertEqual(len(log), 3)
        for entry in log:
            for key in ("unit", "round", "context_tokens", "records_included",
                        "records_dropped", "mode", "budget", "output_tokens"):
                self.assertIn(key, entry)

    def test_no_call_exceeds_budget(self) -> None:
        """Across a synthetic 5-round run, no observation may exceed B."""
        budget = 400
        cfg = RunConfig(context_budget=budget)
        for r in range(5):
            self.store.begin_round(r)
            self.store.add_proof_revision("PROOF " + "p" * 2000 * (r + 1),
                                          produced_by="solver")
            for unit in ("critic", "solver", "literature", "meeting", "evaluator"):
                run_unit(unit, Query(text="task " + "t" * 200), self.store,
                         config=cfg, instructions="inst",
                         invoke=self._invoke(_issues_artifact(1)))

        log = read_telemetry(self.store)
        self.assertEqual(len(log), 25)
        over = [e for e in log if e["context_tokens"] > budget and not e["over_budget"]]
        self.assertEqual(over, [], "an over-budget call was not flagged")
        # Only the mandatory head may push a call over B, and it must be marked.
        for entry in log:
            if entry["context_tokens"] > budget:
                self.assertTrue(entry["over_budget"])
                self.assertGreater(entry["mandatory_floor_tokens"], budget)

    def test_telemetry_records_dropped_ids(self) -> None:
        self.store.add_proof_revision("P" * 20000, produced_by="proposer")
        for i in range(6):
            self.store.add("M", "note", "N" * 4000, meta={"unit": "critic"})
        run_unit("critic", Query(text="q"), self.store,
                 config=RunConfig(context_budget=300),
                 instructions="i", invoke=self._invoke(None))
        entry = read_telemetry(self.store)[-1]
        self.assertGreater(entry["records_dropped"], 0)
        self.assertEqual(len(entry["dropped_ids"]), entry["records_dropped"])

    def test_telemetry_path_override(self) -> None:
        path = self.root / "artifacts" / "orchestration_log.jsonl"
        run_unit("critic", Query(text="q"), self.store, telemetry_path=path,
                 instructions="i", invoke=self._invoke(None))
        self.assertTrue(path.is_file())
        self.assertEqual(len(read_telemetry(self.store, path)), 1)


class ContextModeTest(OrchestratorTestBase):
    def test_ablation_selects_the_context_mode(self) -> None:
        self.store.add_proof_revision("P" * 8000, produced_by="proposer")
        for mode_ablation, expected in (("context.dump", "dump"),
                                        ("context.truncate", "truncate")):
            cfg = RunConfig(context_budget=100, ablations=frozenset({mode_ablation}))
            result = run_unit("critic", Query(text="q"), self.store, config=cfg,
                              instructions="i", invoke=self._invoke(None))
            self.assertEqual(result.observation.mode, expected)

    def test_default_mode_is_budget(self) -> None:
        result = run_unit("critic", Query(text="q"), self.store,
                          instructions="i", invoke=self._invoke(None))
        self.assertEqual(result.observation.mode, "budget")


if __name__ == "__main__":
    unittest.main()
