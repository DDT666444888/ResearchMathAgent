"""T4.2 — Critic produces the ranked queue Q from three analyses.

All model-backed analyses are injected, so the whole test runs offline.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.ops import get_operation
from rma.ops.base import OpContext
from rma.ops.critic import structural_analysis
from rma.ranking import Issue
from rma.store import ResearchStore

# A proof with a real structural defect: lemma-1 is a terminal claim with no proof.
PROOF = r"""\documentclass{article}\begin{document}
\begin{lemma}\label{crux}The crux estimate holds.\end{lemma}

\begin{lemma}\label{easy}The easy bound holds.\end{lemma}
\begin{proof}Cauchy--Schwarz.\end{proof}

\begin{theorem}\label{main}The main result follows.\end{theorem}
\begin{proof}Combine \ref{crux} and \ref{easy}.\end{proof}
\end{document}"""


def _mk_issues(raw, analysis):
    """Wrap raw finding dicts into Issue objects — the contract the real
    analyses return (list[Issue]), so the fakes honour the same interface."""
    out = []
    for i, item in enumerate(raw or []):
        out.append(Issue(id=f"{analysis}-{i}", code=item["code"], message=item["message"],
                         detail=item.get("detail", ""), meta={"analysis": analysis}))
    return out


def _fake_analyses(lm=None, semantic=None, comp=6.0):
    """Injectable stand-ins for the two model-backed analyses. They return
    list[Issue], exactly like rma.ops.critic.lm_gap_analysis / semantic_analysis."""
    def _lm(proof_text, args):
        return _mk_issues(lm, "lm")

    def _sem(problem_text, proof_text, args):
        return comp, _mk_issues(semantic, "semantic")

    return {"lm": _lm, "semantic": _sem}


class CriticTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(PROOF, produced_by="proposer")
        self.ctx = OpContext(store=self.store, config=RunConfig(context_budget=60000),
                             problem={"title": "Main result", "normalized_statement": "prove it"},
                             round=0)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class StructuralAnalysisTest(unittest.TestCase):
    def test_unproved_terminal_claim_flagged(self) -> None:
        issues = structural_analysis(PROOF)
        codes = {(i.code, i.claim_id) for i in issues}
        self.assertIn(("unproved_lemma", "lemma-1"), codes)
        # lemma-2 IS proved, so it must not be flagged.
        self.assertNotIn(("unproved_lemma", "lemma-2"), codes)

    def test_cycle_flagged_as_circular_reasoning(self) -> None:
        cyc = r"""\begin{lemma}\label{a}A.\end{lemma}\begin{proof}By \ref{b}.\end{proof}
        \begin{lemma}\label{b}B.\end{lemma}\begin{proof}By \ref{a}.\end{proof}"""
        codes = {i.code for i in structural_analysis(cyc)}
        self.assertIn("circular_reasoning", codes)

    def test_deterministic_no_model_needed(self) -> None:
        # Runs with no args / no backend at all.
        self.assertTrue(structural_analysis(PROOF))


class CriticQueueTest(CriticTestBase):
    def test_three_analyses_all_contribute(self) -> None:
        lm = [{"code": "logical_gap", "message": "step 3 unsupported", "detail": "s3"}]
        sem = [{"code": "incomplete_step", "message": "case n=0 missing", "detail": "n=0"}]
        result = get_operation("critic").run(
            self.ctx, analyses=_fake_analyses(lm=lm, semantic=sem))
        analyses = {i.meta.get("analysis") for i in result.artifact}
        self.assertEqual(analyses, {"structural", "lm", "semantic"},
                         "an analysis contributed nothing to Q")

    def test_queue_written_to_the_issue_store(self) -> None:
        result = get_operation("critic").run(self.ctx, analyses=_fake_analyses())
        self.assertGreater(len(result.written), 0)
        self.assertEqual(len(self.store.issues), len(result.artifact))
        # every written issue carries a severity
        for rec in self.store.issues:
            self.assertIn(rec.meta["severity"], ("P0", "P1", "P2", "P3"))

    def test_queue_is_severity_then_impact_ordered(self) -> None:
        lm = [{"code": "logical_gap", "message": "main step wrong", "detail": ""}]
        result = get_operation("critic").run(self.ctx, analyses=_fake_analyses(lm=lm))
        ranks = [{"P0": 0, "P1": 1, "P2": 2, "P3": 3}[i.severity] for i in result.artifact]
        self.assertEqual(ranks, sorted(ranks), "Q is not severity-ordered")

    def test_disable_lm_analysis_drops_only_its_issues(self) -> None:
        lm = [{"code": "logical_gap", "message": "gap from LM", "detail": ""}]
        sem = [{"code": "incomplete_step", "message": "gap from semantic", "detail": ""}]
        cfg = RunConfig(ablations=frozenset({"critic.lm"}))
        ctx = OpContext(store=self.store, config=cfg, problem=self.ctx.problem)
        result = get_operation("critic").run(ctx, analyses=_fake_analyses(lm=lm, semantic=sem))
        analyses = {i.meta.get("analysis") for i in result.artifact}
        self.assertNotIn("lm", analyses)
        self.assertIn("semantic", analyses)
        self.assertIn("structural", analyses)

    def test_disable_structural(self) -> None:
        cfg = RunConfig(ablations=frozenset({"critic.structural"}))
        ctx = OpContext(store=self.store, config=cfg, problem=self.ctx.problem)
        result = get_operation("critic").run(ctx, analyses=_fake_analyses())
        self.assertNotIn("structural", {i.meta.get("analysis") for i in result.artifact})

    def test_rediscovered_issue_is_deduped_not_duplicated(self) -> None:
        # First round opens the structural issue.
        r1 = get_operation("critic").run(self.ctx, analyses=_fake_analyses())
        n_after_first = len(self.store.issues)
        self.assertGreater(len(r1.written), 0)
        # Second round, same proof -> the same gap must NOT be written again...
        self.ctx.round = 1
        result2 = get_operation("critic").run(self.ctx, analyses=_fake_analyses())
        self.assertEqual(len(self.store.issues), n_after_first,
                         "the same gap was reopened on the next round")
        self.assertEqual(result2.written, [], "a duplicate issue was written")
        # ...but it must still be IN the queue, so the solver can re-attempt it.
        self.assertGreater(len(result2.artifact), 0,
                           "persistent open issue vanished from Q — it can never be re-repaired")

    def test_persistent_open_issue_stays_in_the_queue(self) -> None:
        """An issue found in round 0 and left open must appear in round 1's Q."""
        r1 = get_operation("critic").run(self.ctx, analyses=_fake_analyses())
        open_id = r1.written[0].id
        self.ctx.round = 1
        r2 = get_operation("critic").run(self.ctx, analyses=_fake_analyses())
        self.assertIn(open_id, [i.id for i in r2.artifact],
                      "the still-open issue is not in the next round's queue")

    def test_completeness_score_captured(self) -> None:
        get_operation("critic").run(self.ctx, analyses=_fake_analyses(comp=4.0))
        self.assertEqual(self.ctx.extra["completeness_score"], 4.0)

    def test_critic_logged_with_a_telemetry_line(self) -> None:
        from rma.orchestrator import read_telemetry

        get_operation("critic").run(self.ctx, analyses=_fake_analyses())
        log = read_telemetry(self.store)
        self.assertTrue(any(e["unit"] == "critic" for e in log))

    def test_no_fresh_issues_writes_nothing(self) -> None:
        # A complete proof with proofs on every terminal claim, no injected gaps.
        complete = (r"\begin{lemma}\label{a}A.\end{lemma}\begin{proof}p\end{proof}"
                    r"\begin{theorem}\label{t}T.\end{theorem}\begin{proof}By \ref{a}.\end{proof}")
        self.store.add_proof_revision(complete, produced_by="solver")
        result = get_operation("critic").run(self.ctx, analyses=_fake_analyses())
        self.assertEqual(result.artifact, [])


if __name__ == "__main__":
    unittest.main()
