"""T5.1 / T5.2 / e2e — the Algorithm 1 round loop, fully offline."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.ops.base import FakeBackend
from rma.ranking import Issue
from rma.round_loop import solve_problem
from rma.store import ResearchStore

INITIAL = r"""\documentclass{article}\begin{document}
\begin{lemma}\label{crux}The crux bound holds.\end{lemma}

\begin{theorem}\label{main}The result follows.\end{theorem}
\begin{proof}By \ref{crux}.\end{proof}
\end{document}"""

# A revised proof that closes the crux (has a proof for every terminal claim).
COMPLETE = r"""\documentclass{article}\begin{document}
\begin{lemma}\label{crux}The crux bound holds.\end{lemma}
\begin{proof}Full derivation.\end{proof}

\begin{theorem}\label{main}The result follows.\end{theorem}
\begin{proof}By \ref{crux}.\end{proof}
\end{document}"""

PATCH = {"replaced_text": "The crux bound holds.\\end{lemma}",
         "replacement": "The crux bound holds.\\end{lemma}\\begin{proof}Now proved.\\end{proof}"}
MEETING = {"record": "discussed the crux", "action_plan": {"summary": "prove the crux",
           "steps": ["derive the bound"]}, "insights": ["the dense case is the obstacle"]}
LIT = [{"theorem_or_technique": "paving", "assumptions": "finite",
        "claims_supported": [], "applicability_limits": "none", "source": "ref"}]
CONCEPTS = [{"name": "crux bound", "definition": "the key estimate"}]


def _critic_analyses(lm=None, comp=6.0):
    def _lm(proof, args):
        return [Issue(id=f"lm-{i}", code=d["code"], message=d["message"],
                      claim_id=d.get("claim_id"))
                for i, d in enumerate(lm or [])]

    def _sem(problem, proof, args):
        return comp, []
    return {"lm": _lm, "semantic": _sem}


def _backend(revise_tex=COMPLETE, scores=None):
    scores = scores or {"answer_accuracy": 1, "logical_correctness": 7,
                        "proof_completeness": 6, "proof_clarity": 8, "verdict": "ok"}
    return FakeBackend({
        "solver": PATCH,
        "literature": LIT,
        "meeting": MEETING,
        "revise": revise_tex,
        "concepts": CONCEPTS,
        "evaluator": scores,
    })


class RoundOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(INITIAL, produced_by="proposer")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_round_call_sequence_matches_algorithm_1(self) -> None:
        cfg = RunConfig(n_rounds=1, issue_budget=5)
        res = solve_problem(self.store, {"title": "T", "normalized_statement": "prove"},
                            cfg, backend=_backend(),
                            critic_analyses=_critic_analyses(
                                lm=[{"code": "logical_gap", "message": "gap"}]))
        units = res.rounds[0].units
        # critic first, then solver(s), then literature, meeting, revise,
        # update_concepts, evaluator, finalize — the paper's order.
        self.assertEqual(units[0], "critic")
        self.assertEqual(units[-1], "finalize")
        self.assertEqual(units[-2], "evaluator")
        # the non-solver phases appear once, in order
        phases = [u for u in units if u != "solver"]
        self.assertEqual(phases,
                         ["critic", "literature", "meeting", "revise",
                          "update_concepts", "evaluator", "finalize"])

    def test_solver_runs_at_most_b_times(self) -> None:
        cfg = RunConfig(n_rounds=1, issue_budget=2)
        res = solve_problem(self.store, {"title": "T"}, cfg, backend=_backend(),
                            critic_analyses=_critic_analyses(lm=[
                                {"code": "logical_gap", "message": f"gap {i}"} for i in range(6)]))
        n_solver = res.rounds[0].units.count("solver")
        self.assertLessEqual(n_solver, 2, "solver exceeded the issue budget b")

    def test_ablate_meeting_removes_meeting_and_revise(self) -> None:
        cfg = RunConfig(n_rounds=1, ablations=frozenset({"meeting"}))
        res = solve_problem(self.store, {"title": "T"}, cfg, backend=_backend(),
                            critic_analyses=_critic_analyses())
        units = res.rounds[0].units
        self.assertNotIn("meeting", units)
        self.assertNotIn("revise", units)
        self.assertIn("evaluator", units)


class FinalizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(INITIAL, produced_by="proposer")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_one_summary_per_round(self) -> None:
        cfg = RunConfig(n_rounds=3)
        res = solve_problem(self.store, {"title": "T"}, cfg, backend=_backend(),
                            critic_analyses=_critic_analyses())
        summaries = [e for e in self.store.evaluations if e.kind == "round_summary"]
        self.assertEqual(len(summaries), len(res.rounds))

    def test_round_metrics_recorded(self) -> None:
        cfg = RunConfig(n_rounds=1)
        res = solve_problem(self.store, {"title": "T"}, cfg, backend=_backend(),
                            critic_analyses=_critic_analyses())
        m = res.rounds[0].metrics
        self.assertIsNotNone(m)
        self.assertIn(m.proved_terminal_fraction, (0.0, 0.5, 1.0))


class TerminationInLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(INITIAL, produced_by="proposer")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_solved_breaks_loop_early(self) -> None:
        # revise closes the proof; a high completeness score + no P0/P1 open +
        # terminal fraction 1.0 should stop before the round budget.
        cfg = RunConfig(n_rounds=5)
        scores = {"answer_accuracy": 1, "logical_correctness": 9,
                  "proof_completeness": 9, "proof_clarity": 9, "verdict": "complete"}
        res = solve_problem(self.store, {"title": "T"}, cfg,
                            backend=_backend(revise_tex=COMPLETE, scores=scores),
                            critic_analyses=_critic_analyses(comp=9.0))
        self.assertEqual(res.stop_reason, "solved")
        self.assertLess(len(res.rounds), 5, "loop did not stop early on Solved")

    def test_budget_exhausted_when_never_solved(self) -> None:
        cfg = RunConfig(n_rounds=3)
        # low completeness, crux never closed -> runs the full budget
        res = solve_problem(self.store, {"title": "T"},
                            cfg, backend=_backend(revise_tex=INITIAL),
                            critic_analyses=_critic_analyses(comp=2.0))
        self.assertIn(res.stop_reason, ("budget_exhausted", "stalled"))


class DriverTest(unittest.TestCase):
    """T5.4 — run_solve drives the orchestrator via the CLI, offline."""

    def test_cli_orchestrator_end_to_end(self) -> None:
        from argparse import Namespace

        from rma.solve import run_solve

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = Namespace(
                problem="q6", all=False, tier="standard", output=str(out),
                exp_name="alg1_test", model_name="rma-skeleton", model_provider="auto",
                no_render=True, max_rounds=None, rounds=4, issue_budget=3,
                context_budget=None, context_mode=None, effort=None, ablate=None,
                skill_path="skills/math-research/SKILL.md", repo_root=".",
                resume=False, fast=False, strategies=1, parent_run=None,
                dataset="first_proof_1", orchestrator=True, legacy_pipeline=False,
                backend="fake",
            )
            rc = run_solve(args)
            self.assertIn(rc, (0, 1))

            log_path = out / "q6" / "artifacts" / "orchestration_log.jsonl"
            self.assertTrue(log_path.is_file(), "orchestration_log.jsonl not written")
            import json
            log = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
            rounds = sorted({e["round"] for e in log})
            # The loop may stop early on Stalled/Solved before the round budget
            # (flat metrics => no progress => stall). It must run a contiguous
            # prefix starting at 0, at least 2 rounds (multi-round works), and
            # never exceed the requested budget of 4.
            self.assertEqual(rounds, list(range(len(rounds))),
                             "rounds are not a contiguous prefix starting at 0")
            self.assertGreaterEqual(len(rounds), 2, "loop did not run multiple rounds")
            self.assertLessEqual(len(rounds), 4, "loop exceeded the round budget")
            self.assertLessEqual(max(e["context_tokens"] for e in log), 60000)
            units_r0 = {e["unit"] for e in log if e["round"] == 0}
            for u in ("critic", "solver", "literature", "meeting", "revise", "evaluator"):
                self.assertIn(u, units_r0, f"{u} did not run in round 0")

            summary = json.loads((out / "q6" / "artifacts" / "orchestrator_summary.json").read_text())
            self.assertEqual(summary["rounds"], len(rounds),
                             "summary round count disagrees with telemetry")
            self.assertIn(summary["stop_reason"], ("solved", "stalled", "budget_exhausted"))

    def test_legacy_pipeline_flag_bypasses_orchestrator(self) -> None:
        from argparse import Namespace

        from rma.solve import run_solve

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = Namespace(
                problem="q6", all=False, tier="standard", output=str(out),
                exp_name="legacy_test", model_name="rma-skeleton", model_provider="offline",
                no_render=True, max_rounds=2, rounds=None, issue_budget=None,
                context_budget=None, context_mode=None, effort=None, ablate=None,
                skill_path="skills/math-research/SKILL.md", repo_root=".",
                resume=False, fast=False, strategies=1, parent_run=None,
                dataset="first_proof_1", orchestrator=False, legacy_pipeline=True,
                backend="auto",
            )
            run_solve(args)
            # legacy path writes verifications, not an orchestration log
            self.assertFalse((out / "q6" / "artifacts" / "orchestration_log.jsonl").is_file())
            self.assertTrue((out / "q6" / "artifacts" / "verifications").is_dir())


class PushForwardTest(unittest.TestCase):
    """Multi-round push-forward must make progress: each round refines the BEST
    proof so far, so a regressing round cannot compound into the next and the
    best proof is monotone non-decreasing across rounds."""

    GOOD = COMPLETE                    # closes the crux -> high score
    BAD = INITIAL                      # crux unproved  -> low score

    def _run(self, revise_by_round, scores_by_round, *, push_forward_best=True,
             n_rounds=3):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ResearchStore.open(Path(tmp.name), "q6", "first_proof_1")
        store.add_proof_revision(INITIAL, produced_by="proposer")
        rev = iter(revise_by_round)
        sc = iter(scores_by_round)

        def _revise(_obs):
            try:
                return next(rev)
            except StopIteration:
                return self.BAD

        def _eval(_obs):
            try:
                c = next(sc)
            except StopIteration:
                c = 3
            return {"answer_accuracy": 0, "logical_correctness": c,
                    "proof_completeness": c, "proof_clarity": c, "verdict": "x"}

        backend = FakeBackend({
            "solver": {"replaced_text": "", "replacement": ""},
            "literature": LIT, "meeting": MEETING, "revise": _revise,
            "concepts": CONCEPTS, "evaluator": _eval,
        })
        cfg = RunConfig(n_rounds=n_rounds, push_forward_best=push_forward_best)
        res = solve_problem(store, {"title": "T"}, cfg, backend=backend,
                            critic_analyses=_critic_analyses(comp=None))
        return store, res

    def test_regressing_round_is_carried_forward_not_compounded(self) -> None:
        # round 0 produces the GOOD proof (score 9); round 1 regresses to BAD (3).
        store, res = self._run([self.GOOD, self.BAD, self.BAD],
                               [9, 3, 3], push_forward_best=True)
        # The best round (0) is delivered, not the last.
        self.assertEqual(res.delivered_round, 0)
        # Round 2 must have STARTED from the best (GOOD) proof, not round 1's BAD
        # one: a carry_forward revision equal to the best body was inserted.
        carried = [p for p in store.proofs
                   if (p.meta or {}).get("produced_by") == "carry_forward"]
        self.assertTrue(carried, "no carry-forward happened; a regression compounded")
        self.assertEqual(carried[-1].body, self.GOOD)

    def test_no_carry_forward_when_disabled(self) -> None:
        store, res = self._run([self.GOOD, self.BAD, self.BAD],
                               [9, 3, 3], push_forward_best=False)
        carried = [p for p in store.proofs
                   if (p.meta or {}).get("produced_by") == "carry_forward"]
        self.assertEqual(carried, [], "carry-forward ran despite being disabled")

    def test_a_later_improving_round_is_recognized_as_best(self) -> None:
        # round 0 mediocre (5), round 1 improves to GOOD (9): progress must be
        # visible — the later round is delivered.
        _store, res = self._run([self.BAD, self.GOOD],
                                [5, 9], push_forward_best=True, n_rounds=2)
        self.assertEqual(res.delivered_round, 1,
                         "an improving later round was not recognized -> no progress")

    def test_best_completeness_is_monotone_across_rounds(self) -> None:
        # oscillating scores; the best-so-far completeness must never decrease.
        store, _res = self._run([self.GOOD, self.BAD, self.GOOD],
                                [7, 2, 9], push_forward_best=True)
        summaries = [e for e in store.evaluations if e.kind == "round_summary"]
        comps = [(e.meta or {}).get("completeness") for e in summaries]
        best_so_far = []
        cur = -1.0
        for c in comps:
            if c is not None:
                cur = max(cur, c)
            best_so_far.append(cur)
        self.assertEqual(best_so_far, sorted(best_so_far),
                         f"best completeness regressed across rounds: {comps}")


class ResilienceTest(unittest.TestCase):
    """A single operation failing (a transient model error, a narration-only
    reply) must not crash the whole multi-round run and discard its progress."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ResearchStore.open(Path(self._tmp.name), "q6", "first_proof_1")
        self.store.add_proof_revision(INITIAL, produced_by="proposer")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_raising_op_does_not_crash_the_loop(self) -> None:
        from rma.models import ModelRequestError

        def _revise(_obs):
            raise ModelRequestError("Claude Code produced no LaTeX document")

        backend = FakeBackend({
            "solver": PATCH, "literature": LIT, "meeting": MEETING,
            "revise": _revise,           # blows up every round
            "concepts": CONCEPTS,
            "evaluator": {"answer_accuracy": 1, "logical_correctness": 6,
                          "proof_completeness": 6, "proof_clarity": 6, "verdict": "ok"},
        })
        # The run must complete all rounds despite revise raising each time.
        res = solve_problem(self.store, {"title": "T"}, RunConfig(n_rounds=2),
                            backend=backend, critic_analyses=_critic_analyses())
        self.assertEqual(len(res.rounds), 2, "loop aborted instead of surviving the failure")
        self.assertIsNotNone(res.final_proof_id)
        # the failure was recorded (not silently swallowed)
        errs = [r for r in self.store.records("H")
                if (r.meta or {}).get("failed_op") == "revise"]
        self.assertTrue(errs, "op failure was not recorded")


class PaperFaithfulTest(unittest.TestCase):
    """paper_faithful=True reproduces Algorithm 1 literally: pi = CurrentProof(S)
    = last revision each round, and the run outputs the FINAL pi — no
    carry-forward, no best-of-rounds selection."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ResearchStore.open(Path(self._tmp.name), "q6", "first_proof_1")
        self.store.add_proof_revision(INITIAL, produced_by="proposer")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, paper_faithful):
        rev = iter([COMPLETE, INITIAL, INITIAL])   # round 0 best, then regress
        sc = iter([9, 3, 3])
        backend = FakeBackend({
            "solver": {"replaced_text": "", "replacement": ""},
            "literature": LIT, "meeting": MEETING,
            "revise": lambda o: next(rev, INITIAL),
            "concepts": CONCEPTS,
            "evaluator": lambda o: (lambda c: {"answer_accuracy": 0,
                "logical_correctness": c, "proof_completeness": c,
                "proof_clarity": c, "verdict": "x"})(next(sc, 3)),
        })
        cfg = RunConfig(n_rounds=3, paper_faithful=paper_faithful)
        self.assertEqual(cfg.push_forward_best, not paper_faithful,
                         "paper_faithful must disable the carry-forward enhancement")
        return solve_problem(self.store, {"title": "T"}, cfg, backend=backend,
                             critic_analyses=_critic_analyses())

    def test_delivers_the_final_round_not_the_best(self) -> None:
        res = self._run(paper_faithful=True)
        # Algorithm 1 returns the final pi; round 0 was better but the paper does
        # not select it.
        self.assertEqual(res.delivered_round, res.rounds[-1].round)
        self.assertEqual(res.delivered_proof_id, res.final_proof_id)
        # and no carry-forward revision was inserted
        carried = [p for p in self.store.proofs
                   if (p.meta or {}).get("produced_by") == "carry_forward"]
        self.assertEqual(carried, [])

    def test_default_mode_still_delivers_best_of_rounds(self) -> None:
        res = self._run(paper_faithful=False)
        self.assertEqual(res.delivered_round, 0, "default must keep best-of-rounds")


class EndToEndTest(unittest.TestCase):
    def test_e2e_five_rounds_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ResearchStore.open(root, "q6", "first_proof_1")
            store.add_proof_revision(INITIAL, produced_by="proposer")
            cfg = RunConfig(n_rounds=5, issue_budget=3, context_budget=8000)
            backend = _backend(revise_tex=INITIAL,   # never fully closes -> runs longer
                               scores={"answer_accuracy": 0, "logical_correctness": 4,
                                       "proof_completeness": 3, "proof_clarity": 5, "verdict": "wip"})
            res = solve_problem(store, {"title": "T", "normalized_statement": "prove it"},
                                cfg, backend=backend,
                                critic_analyses=_critic_analyses(
                                    lm=[{"code": "logical_gap", "message": "step 2 unproved"}],
                                    comp=3.0))

            # every component saw activity
            counts = store.counts()
            for comp in ("Pi", "I", "M", "L", "K", "E"):
                self.assertGreater(counts[comp], 0, f"component {comp} stayed empty")

            # telemetry: every logged call is within budget
            from rma.orchestrator import read_telemetry
            log = read_telemetry(store)
            self.assertTrue(log)
            over = [e for e in log if e["context_tokens"] > cfg.context_budget and not e["over_budget"]]
            self.assertEqual(over, [], "a call exceeded the budget without being flagged")

            # the run produced a final proof
            self.assertIsNotNone(res.final_proof_id)


if __name__ == "__main__":
    unittest.main()
