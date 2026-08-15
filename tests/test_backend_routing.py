"""The orchestrator must run on the Pro/Max SUBSCRIPTION by default, never the
paid per-token API. This is a hard requirement: a real run should bill to the
user's plan via the `claude` CLI (which strips ANTHROPIC_API_KEY), not to a
metered Messages-API account."""
from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from rma.config import RunConfig
from rma.ops import get_operation
from rma.ops.base import OpContext, default_invoker
from rma.store import ResearchStore


class _Reply:
    text = '{"answer_accuracy": 1, "logical_correctness": 7, "proof_completeness": 6, "proof_clarity": 8, "verdict": "ok"}'
    provider = "claude-code"
    model = "claude-opus-4-8"


class BackendRoutingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ResearchStore.open(Path(self._tmp.name), "q6", "first_proof_1")
        self.store.add_proof_revision(r"\begin{document}p\end{document}", produced_by="p")

    def tearDown(self):
        self._tmp.cleanup()

    def _count(self, cfg, unit):
        ctx = OpContext(store=self.store, config=cfg, problem={"title": "T"}, round=0)
        calls = {"cc": 0, "api": 0}
        with patch("rma.models.call_claude_code", side_effect=lambda **k: (calls.__setitem__("cc", calls["cc"] + 1) or _Reply())), \
             patch("rma.models.call_anthropic", side_effect=lambda **k: (calls.__setitem__("api", calls["api"] + 1) or _Reply())):
            default_invoker(get_operation(unit), ctx)(unit, "obs")
        return calls

    def test_default_is_subscription(self) -> None:
        self.assertEqual(RunConfig().provider, "claude-code")
        self.assertTrue(RunConfig().uses_subscription)

    def test_json_op_routes_to_subscription_not_api(self) -> None:
        calls = self._count(RunConfig(), "evaluator")
        self.assertEqual(calls["api"], 0, "a JSON op billed the paid API")
        self.assertEqual(calls["cc"], 1)

    def test_latex_op_routes_to_subscription(self) -> None:
        calls = self._count(RunConfig(), "revise")   # revise expects latex
        self.assertEqual(calls["api"], 0)
        self.assertEqual(calls["cc"], 1)

    def test_explicit_api_provider_uses_the_api(self) -> None:
        calls = self._count(RunConfig(provider="anthropic"), "evaluator")
        self.assertEqual(calls["cc"], 0)
        self.assertEqual(calls["api"], 1)

    def test_subscription_alias(self) -> None:
        self.assertTrue(RunConfig(provider="subscription").uses_subscription)
        calls = self._count(RunConfig(provider="subscription"), "evaluator")
        self.assertEqual(calls["api"], 0)


if __name__ == "__main__":
    unittest.main()
