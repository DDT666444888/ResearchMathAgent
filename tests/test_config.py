"""T0.4 — one object owns every knob the paper specifies.

Paper defaults (main.tex:1251-1268): N_R=5, b=5, B=60,000 tokens, backbone
claude-opus-4-8, adaptive thinking at high effort, 64,000 max output tokens.
"""
from __future__ import annotations

import os
import unittest
from argparse import Namespace
from unittest.mock import patch

from rma.config import (
    ABLATIONS,
    DEFAULT_CONTEXT_BUDGET,
    DEFAULT_EFFORT,
    DEFAULT_ISSUE_BUDGET,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_N_ROUNDS,
    ConfigError,
    RunConfig,
    parse_ablations,
)


class PaperDefaultsTest(unittest.TestCase):
    def test_defaults_match_the_paper(self) -> None:
        cfg = RunConfig()
        self.assertEqual(cfg.n_rounds, 5)
        self.assertEqual(cfg.issue_budget, 5)
        self.assertEqual(cfg.context_budget, 60_000)
        self.assertEqual(cfg.model, "claude-opus-4-8")
        self.assertEqual(cfg.effort, "high")
        self.assertEqual(cfg.max_output_tokens, 64_000)

    def test_module_constants_agree_with_dataclass(self) -> None:
        cfg = RunConfig()
        self.assertEqual(cfg.n_rounds, DEFAULT_N_ROUNDS)
        self.assertEqual(cfg.issue_budget, DEFAULT_ISSUE_BUDGET)
        self.assertEqual(cfg.context_budget, DEFAULT_CONTEXT_BUDGET)
        self.assertEqual(cfg.model, DEFAULT_MODEL)
        self.assertEqual(cfg.effort, DEFAULT_EFFORT)
        self.assertEqual(cfg.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS)

    def test_describe_is_the_documented_one_liner(self) -> None:
        self.assertEqual(
            RunConfig().describe(),
            "n_rounds=5 issue_budget=5 context_budget=60000 "
            "model=claude-opus-4-8 effort=high max_output_tokens=64000",
        )


class BackboneConsistencyTest(unittest.TestCase):
    """Every entry point must resolve to the paper's backbone, not fable."""

    def test_webapp_llm_default_is_the_paper_backbone(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RMA_MODEL", None)
            import importlib

            from webapp import llm

            importlib.reload(llm)
            self.assertEqual(llm.DEFAULT_MODEL, DEFAULT_MODEL)

    def test_webapp_agent_output_cap_is_64k(self) -> None:
        from webapp import agent

        self.assertEqual(agent.MAX_TOKENS, 64_000)
        self.assertEqual(agent.EFFORT, "high")

    def test_rma_cli_model_arg_pins_opus(self) -> None:
        from rma.models import _claude_code_model_arg

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RMA_CLAUDE_CODE_MODEL", None)
            self.assertEqual(_claude_code_model_arg("claude-code"), DEFAULT_MODEL)


class FromArgsTest(unittest.TestCase):
    def _args(self, **kw) -> Namespace:
        base = dict(rounds=None, issue_budget=None, context_budget=None,
                    context_mode=None, effort=None, ablate=None,
                    max_rounds=None, model_name=None)
        base.update(kw)
        return Namespace(**base)

    def test_cli_overrides_defaults(self) -> None:
        cfg = RunConfig.from_args(self._args(rounds=2, issue_budget=1, context_budget=1000))
        self.assertEqual((cfg.n_rounds, cfg.issue_budget, cfg.context_budget), (2, 1, 1000))

    def test_legacy_max_rounds_maps_to_n_rounds(self) -> None:
        self.assertEqual(RunConfig.from_args(self._args(max_rounds=3)).n_rounds, 3)

    def test_backend_names_do_not_override_the_backbone(self) -> None:
        """`--model-name claude-code` picks a backend, not a model."""
        for backend in ("claude-code", "rma-skeleton"):
            self.assertEqual(RunConfig.from_args(self._args(model_name=backend)).model, DEFAULT_MODEL)

    def test_real_model_name_does_override(self) -> None:
        cfg = RunConfig.from_args(self._args(model_name="claude-sonnet-5"))
        self.assertEqual(cfg.model, "claude-sonnet-5")

    def test_env_is_honoured_when_flag_absent(self) -> None:
        with patch.dict(os.environ, {"RMA_N_ROUNDS": "7"}):
            self.assertEqual(RunConfig.from_args(self._args()).n_rounds, 7)

    def test_flag_beats_env(self) -> None:
        with patch.dict(os.environ, {"RMA_N_ROUNDS": "7"}):
            self.assertEqual(RunConfig.from_args(self._args(rounds=2)).n_rounds, 2)


class AblationTest(unittest.TestCase):
    def test_thirteen_plus_named_ablations_registered(self) -> None:
        """Every configuration the paper reports must be a valid name."""
        for name in ("store.stateless", "context.dump", "context.truncate",
                     "order.fifo", "order.severity", "critic.lm",
                     "critic.structural", "critic.semantic", "meeting",
                     "literature", "concepts", "insights", "evaluator-feedback"):
            self.assertIn(name, ABLATIONS)

    def test_unknown_ablation_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            parse_ablations("store.telepathy")
        with self.assertRaises(ConfigError):
            RunConfig(ablations=frozenset({"nope"}))

    def test_parse_accepts_comma_and_space(self) -> None:
        self.assertEqual(parse_ablations("meeting, literature"), frozenset({"meeting", "literature"}))
        self.assertEqual(parse_ablations("meeting literature"), frozenset({"meeting", "literature"}))
        self.assertEqual(parse_ablations(None), frozenset())

    def test_context_ablation_changes_effective_mode(self) -> None:
        self.assertEqual(RunConfig().effective_context_mode, "budget")
        self.assertEqual(RunConfig(ablations=frozenset({"context.dump"})).effective_context_mode, "dump")
        self.assertEqual(
            RunConfig(ablations=frozenset({"context.truncate"})).effective_context_mode, "truncate")

    def test_order_ablation_gives_the_papers_ladder(self) -> None:
        self.assertEqual(RunConfig().order_mode, "severity+impact")
        self.assertEqual(RunConfig(ablations=frozenset({"order.severity"})).order_mode, "severity")
        self.assertEqual(RunConfig(ablations=frozenset({"order.fifo"})).order_mode, "fifo")

    def test_ablated_query(self) -> None:
        cfg = RunConfig(ablations=frozenset({"meeting"}))
        self.assertTrue(cfg.ablated("meeting"))
        self.assertFalse(cfg.ablated("literature"))
        with self.assertRaises(ConfigError):
            cfg.ablated("not-a-thing")


class ValidationTest(unittest.TestCase):
    def test_budgets_must_be_positive(self) -> None:
        for kwargs in ({"n_rounds": 0}, {"issue_budget": 0},
                       {"context_budget": 0}, {"max_output_tokens": -1}):
            with self.assertRaises(ConfigError):
                RunConfig(**kwargs)

    def test_bad_context_mode_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            RunConfig(context_mode="vibes")

    def test_non_integer_env_is_a_clear_error(self) -> None:
        with patch.dict(os.environ, {"RMA_N_ROUNDS": "five"}):
            with self.assertRaises(ConfigError):
                RunConfig.from_env()


if __name__ == "__main__":
    unittest.main()
