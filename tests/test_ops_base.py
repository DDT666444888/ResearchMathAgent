"""T4.1 — Operation base, registry, and the offline FakeBackend.

The design invariant under test: an operation declares instructions, builds a
query, and parses a reply. It never calls a model and never reads the store for
context. If it could do either, the context budget and the telemetry would both
be bypassable, and Run(u,q,S,B) would stop being the single execution path the
paper describes.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.ops.base import (
    REGISTRY,
    FakeBackend,
    OpContext,
    Operation,
    get_operation,
    register,
)
from rma.orchestrator import Query, read_telemetry
from rma.store import ResearchStore

REPO = Path(__file__).resolve().parents[1]


@register
class _Echo(Operation):
    name = "_test_echo"

    def instructions(self, ctx):
        return "ECHO INSTRUCTIONS"

    def query(self, ctx):
        return Query(text="ECHO QUERY", id="q-echo")

    def parse(self, raw, ctx):
        return {"seen": raw}

    def writeback(self, artifact, ctx):
        return [{"component": "M", "kind": "note", "body": str(artifact["seen"])}]


class RegistryTest(unittest.TestCase):
    def test_register_and_lookup(self) -> None:
        self.assertIn("_test_echo", REGISTRY)
        self.assertIsInstance(get_operation("_test_echo"), _Echo)

    def test_unknown_operation_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_operation("nope")

    def test_nameless_operation_rejected(self) -> None:
        with self.assertRaises(ValueError):
            register(type("Anon", (Operation,), {"name": ""}))


class ExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.ctx = OpContext(store=self.store, config=RunConfig(context_budget=4000),
                             problem={"title": "T"}, round=0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_operation_runs_through_Run_and_writes_back(self) -> None:
        backend = FakeBackend({"_test_echo": "REPLY"})
        result = get_operation("_test_echo").run(self.ctx, invoke=backend)
        self.assertEqual(result.artifact, {"seen": "REPLY"})
        self.assertEqual(len(result.written), 1)
        self.assertEqual(self.store.meetings[0].body, "REPLY")

    def test_observation_contains_the_declared_instructions_and_query(self) -> None:
        backend = FakeBackend(default="x")
        get_operation("_test_echo").run(self.ctx, invoke=backend)
        obs = backend.observation_for("_test_echo")
        self.assertIn("ECHO INSTRUCTIONS", obs)
        self.assertIn("ECHO QUERY", obs)

    def test_run_is_logged_with_the_budget(self) -> None:
        get_operation("_test_echo").run(self.ctx, invoke=FakeBackend(default="x"))
        log = read_telemetry(self.store)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["unit"], "_test_echo")
        self.assertLessEqual(log[0]["context_tokens"], 4000)

    def test_budget_is_enforced_for_operations(self) -> None:
        self.store.add_proof_revision("P" * 200_000, produced_by="proposer")
        backend = FakeBackend(default="x")
        result = get_operation("_test_echo").run(self.ctx, invoke=backend)
        self.assertLessEqual(result.context_tokens, 4000)

    def test_round_is_propagated(self) -> None:
        self.ctx.round = 4
        result = get_operation("_test_echo").run(self.ctx, invoke=FakeBackend(default="x"))
        self.assertEqual(result.round, 4)
        self.assertEqual(result.written[0].round, 4)


class FakeBackendTest(unittest.TestCase):
    def test_records_calls_and_observations(self) -> None:
        backend = FakeBackend({"a": 1}, default=2)
        self.assertEqual(backend("a", "obs-a"), 1)
        self.assertEqual(backend("b", "obs-b"), 2)
        self.assertEqual(backend.units(), ["a", "b"])
        self.assertEqual(backend.observation_for("b"), "obs-b")

    def test_callable_reply_sees_the_observation(self) -> None:
        backend = FakeBackend({"a": lambda obs: f"len={len(obs)}"})
        self.assertEqual(backend("a", "1234"), "len=4")

    def test_missing_unit_is_a_clear_assertion(self) -> None:
        backend = FakeBackend(default=None)
        backend("a", "x")
        with self.assertRaises(AssertionError):
            backend.observation_for("never-called")


class NoBypassTest(unittest.TestCase):
    """The invariant, enforced by inspection of the package source."""

    def _op_sources(self) -> list[Path]:
        return [p for p in (REPO / "rma" / "ops").glob("*.py")
                if p.name not in ("base.py", "__init__.py")]

    def test_no_operation_calls_a_model_directly(self) -> None:
        pattern = re.compile(r"call_anthropic|call_claude_code|call_json|llm\.complete")
        offenders = [p.name for p in self._op_sources()
                     if pattern.search(p.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [],
                         f"operations must go through Run(), not the model: {offenders}")

    def test_no_operation_trawls_the_store_for_context(self) -> None:
        """`store.records()` returns the WHOLE store; calling it inside an
        operation would smuggle unbounded context past PrefixToBudget. That is
        the real budget-escape, and it is forbidden.

        Reading the operation's SUBJECT is not that escape and is allowed:
        `current_proof()` (the proof the critic analyses / the solver patches;
        already the top observation section) and `open_issues()` (the critic's
        write-back dedup target). Those are the thing being acted on, not extra
        ambient context."""
        pattern = re.compile(r"\bstore\.records\(")
        offenders = []
        for path in self._op_sources():
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{i}")
        self.assertEqual(offenders, [], f"an operation trawled the whole store: {offenders}")

    def test_grep_check_used_by_the_runner_agrees(self) -> None:
        """Mirrors the shell check in scripts/check_alg1.sh so the two cannot drift."""
        proc = subprocess.run(
            ["bash", "-c",
             'grep -rn --include=\\*.py "call_anthropic\\|call_claude_code\\|llm.complete" '
             'rma/ops/ | grep -v "/base.py:" | grep -q .'],
            cwd=REPO, capture_output=True)
        self.assertNotEqual(proc.returncode, 0, "the runner's bypass grep would fail")


if __name__ == "__main__":
    unittest.main()
