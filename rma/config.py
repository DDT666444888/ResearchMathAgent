"""Run configuration for the Algorithm 1 orchestrator.

One object owns every knob the paper specifies, so a run's parameters are
stated in a single place instead of being scattered across defaults in
``rma/cli.py``, ``rma/models.py``, ``webapp/agent.py`` and ``webapp/llm.py``.

Paper defaults (main.tex:1251-1268, "Agent configuration" / "Models and
inference"): N_R = 5 research rounds, per-round issue budget b = 5, per-call
context budget B = 60,000 tokens, backbone ``claude-opus-4-8`` with adaptive
thinking at high reasoning effort and a 64,000-token maximum output length.
"""
from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass, field, replace


# The paper's numbers, in one place. Every other module should read these
# rather than hard-coding its own default.
DEFAULT_N_ROUNDS = 5           # N_R
DEFAULT_ISSUE_BUDGET = 5       # b
DEFAULT_CONTEXT_BUDGET = 60_000  # B, in tokens
DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_OUTPUT_TOKENS = 64_000
# Default backend: the Claude Pro/Max SUBSCRIPTION via the local `claude` CLI
# (no API key, no per-token billing). It runs the same claude-opus-4-8 backbone.
DEFAULT_PROVIDER = "claude-code"

# How a candidate context is reduced to fit B. "budget" is the paper's
# PrefixToBudget; the other two exist so the context ablation is a runnable
# configuration rather than a separate code path (main.tex:920-925).
CONTEXT_MODES = ("budget", "truncate", "dump")

# Every ablation the paper reports (main.tex:880-940). Wired in T6.1; declared
# here so the names are validated from the start and cannot drift.
ABLATIONS = (
    "store.stateless",
    "store.last-round-only",
    "context.dump",
    "context.truncate",
    "order.fifo",
    "order.severity",
    "critic.lm",
    "critic.structural",
    "critic.semantic",
    "critic.fidelity",
    "meeting",
    "literature",
    "concepts",
    "insights",
    "evaluator-feedback",
)


class ConfigError(ValueError):
    """Raised for an unknown ablation name or an out-of-range budget."""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class RunConfig:
    """Parameters for one Algorithm 1 run."""

    n_rounds: int = DEFAULT_N_ROUNDS
    issue_budget: int = DEFAULT_ISSUE_BUDGET
    context_budget: int = DEFAULT_CONTEXT_BUDGET
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    context_mode: str = "budget"
    ablations: frozenset[str] = field(default_factory=frozenset)
    # Backend the orchestrator's operations call. "claude-code" (default) is the
    # local `claude` CLI on the user's Pro/Max SUBSCRIPTION — no API key, no
    # per-token charge; it still runs the paper's claude-opus-4-8 backbone.
    # "anthropic" uses the pay-per-token Messages API (needs a key); "offline"
    # is the deterministic no-model path.
    provider: str = DEFAULT_PROVIDER
    # Push-forward-from-best: each round refines the BEST proof discovered so far,
    # not whatever the last round happened to produce. Refinement is not monotone
    # (a round can add repetition or reintroduce a gap), so without this a bad
    # round compounds into the next and the multi-round loop regresses instead of
    # progressing. With it, the best proof is monotone non-decreasing across
    # rounds — later rounds can only help. Set False for the paper's literal
    # CurrentProof = last-revision semantics.
    push_forward_best: bool = True
    # Paper-faithful mode: implement Algorithm 1 (main.tex) LITERALLY — each round
    # uses pi = CurrentProof(S) = the last revision, and the run outputs that
    # final pi (\KwOut{proof pi}), with NO best-of-rounds selection and NO
    # carry-forward. Our default adds those two enhancements (push_forward_best +
    # best-of-rounds delivery); set this True to reproduce the paper's exact
    # procedure. The seven operations, their order, PrefixToBudget, and the
    # Solved/Stalled/budget termination match the paper in both modes.
    paper_faithful: bool = False

    def __post_init__(self) -> None:
        # In paper-faithful mode CurrentProof is the last revision (Alg. 1 line
        # "pi <- CurrentProof(S)"), so the carry-forward enhancement is off.
        # (Frozen dataclass -> object.__setattr__ to normalize the field.)
        if self.paper_faithful and self.push_forward_best:
            object.__setattr__(self, "push_forward_best", False)
        unknown = sorted(self.ablations - set(ABLATIONS))
        if unknown:
            raise ConfigError(
                f"unknown ablation(s): {', '.join(unknown)}. "
                f"Valid names: {', '.join(ABLATIONS)}"
            )
        if self.context_mode not in CONTEXT_MODES:
            raise ConfigError(
                f"context_mode must be one of {CONTEXT_MODES}, got {self.context_mode!r}"
            )
        if self.provider not in ("claude-code", "subscription", "auto", "anthropic", "api", "offline"):
            raise ConfigError(
                f"provider must be a known backend, got {self.provider!r}"
            )
        for name in ("n_rounds", "issue_budget", "context_budget", "max_output_tokens"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> RunConfig:
        return cls(
            n_rounds=_env_int("RMA_N_ROUNDS", DEFAULT_N_ROUNDS),
            issue_budget=_env_int("RMA_ISSUE_BUDGET", DEFAULT_ISSUE_BUDGET),
            context_budget=_env_int("RMA_CONTEXT_BUDGET", DEFAULT_CONTEXT_BUDGET),
            model=os.environ.get("RMA_MODEL") or DEFAULT_MODEL,
            effort=os.environ.get("RMA_EFFORT") or DEFAULT_EFFORT,
            max_output_tokens=_env_int("RMA_MAX_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
            provider=os.environ.get("RMA_MODEL_PROVIDER") or DEFAULT_PROVIDER,
            ablations=parse_ablations(os.environ.get("RMA_ABLATE")),
        )

    @classmethod
    def from_args(cls, args: Namespace) -> RunConfig:
        """Build from parsed CLI args, falling back to env then paper defaults.

        A CLI flag left unset (None) does not override the environment; this
        keeps `RMA_ABLATE=... rma solve q6` working.
        """
        cfg = cls.from_env()
        overrides: dict = {}
        for attr, arg_name in (
            ("n_rounds", "rounds"),
            ("issue_budget", "issue_budget"),
            ("context_budget", "context_budget"),
            ("context_mode", "context_mode"),
        ):
            value = getattr(args, arg_name, None)
            if value is not None:
                overrides[attr] = value
        # `rma solve --max-rounds` predates this config; honour it as n_rounds.
        max_rounds = getattr(args, "max_rounds", None)
        if max_rounds is not None and "n_rounds" not in overrides:
            overrides["n_rounds"] = int(max_rounds)
        model = getattr(args, "model_name", None)
        # "claude-code" names a backend, not a model; the backbone stays pinned.
        if model and model not in ("claude-code", "rma-skeleton"):
            overrides["model"] = model
        effort = getattr(args, "effort", None)
        if effort:
            overrides["effort"] = effort
        # Honour --model-provider: without this, `rma solve --orchestrator
        # --model-provider anthropic` silently kept the subscription default.
        # An explicit "auto" still means subscription (the paper's backbone on
        # the user's plan); only "anthropic"/"api" switches to the paid API.
        model_provider = getattr(args, "model_provider", None)
        if model_provider and model_provider != "auto":
            overrides["provider"] = model_provider
        ablate = getattr(args, "ablate", None)
        if ablate:
            overrides["ablations"] = parse_ablations(ablate)
        # --paper-faithful: reproduce Algorithm 1 literally (last-revision
        # CurrentProof + final-pi delivery, no enhancements).
        if getattr(args, "paper_faithful", False):
            overrides["paper_faithful"] = True
        return replace(cfg, **overrides) if overrides else cfg

    # ── queries ─────────────────────────────────────────────────────────────

    def ablated(self, name: str) -> bool:
        """True when `name` is switched off for this run."""
        if name not in ABLATIONS:
            raise ConfigError(f"unknown ablation: {name}")
        return name in self.ablations

    @property
    def effective_context_mode(self) -> str:
        """Context ablations are expressed as ablation names but act on the mode."""
        if "context.dump" in self.ablations:
            return "dump"
        if "context.truncate" in self.ablations:
            return "truncate"
        return self.context_mode

    @property
    def order_mode(self) -> str:
        """Issue ordering: the paper's fifo -> severity -> severity+impact ladder."""
        if "order.fifo" in self.ablations:
            return "fifo"
        if "order.severity" in self.ablations:
            return "severity"
        return "severity+impact"

    @property
    def uses_subscription(self) -> bool:
        """True when the run bills to the Pro/Max subscription (no API tokens).
        "auto" resolves to the subscription here, matching how the orchestrator's
        default_invoker actually routes it (rma/ops/base.py)."""
        return self.provider in ("claude-code", "subscription", "auto")

    def describe(self) -> str:
        """One-line summary; the shape `rma config --print` emits."""
        return (
            f"n_rounds={self.n_rounds} "
            f"issue_budget={self.issue_budget} "
            f"context_budget={self.context_budget} "
            f"model={self.model} "
            f"effort={self.effort} "
            f"max_output_tokens={self.max_output_tokens}"
        )

    def as_dict(self) -> dict:
        return {
            "n_rounds": self.n_rounds,
            "issue_budget": self.issue_budget,
            "context_budget": self.context_budget,
            "model": self.model,
            "effort": self.effort,
            "max_output_tokens": self.max_output_tokens,
            "provider": self.provider,
            "billing": "subscription" if self.uses_subscription else "api-tokens",
            "context_mode": self.effective_context_mode,
            "order_mode": self.order_mode,
            "ablations": sorted(self.ablations),
        }


def parse_ablations(spec: str | None) -> frozenset[str]:
    """Parse a comma/space separated ablation spec into a validated set."""
    if not spec:
        return frozenset()
    if isinstance(spec, (list, tuple, set, frozenset)):
        names = {str(s).strip() for s in spec}
    else:
        names = {part.strip() for part in str(spec).replace(",", " ").split()}
    names.discard("")
    unknown = sorted(names - set(ABLATIONS))
    if unknown:
        raise ConfigError(
            f"unknown ablation(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(ABLATIONS)}"
        )
    return frozenset(names)


def run_config(args: Namespace) -> int:
    """`rma config --print` — show the resolved configuration for this run."""
    import json as _json

    cfg = RunConfig.from_args(args)
    if getattr(args, "json", False):
        print(_json.dumps(cfg.as_dict(), indent=2))
    else:
        print(cfg.describe())
        if cfg.ablations:
            print(f"ablations={','.join(sorted(cfg.ablations))}")
            print(f"context_mode={cfg.effective_context_mode} order_mode={cfg.order_mode}")
    return 0
