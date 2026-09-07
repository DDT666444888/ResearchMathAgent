"""Locally scoped research operations.

The paper (main.tex:394-400):

    "the orchestration layer is implemented through a collection of locally
     scoped research operations... Each operation is responsible for one
     explicit research task and receives only the bounded, operation-specific
     observation prepared by the orchestrator, rather than direct access to the
     complete research store S. It returns a structured artifact... that the
     orchestrator validates and writes back."

Two invariants make that true rather than aspirational:

  * An operation never touches a model. It declares its instructions, builds
    its query, and parses a reply; ``Run`` does the compiling, invoking and
    writing back. ``tests/test_ops_base.py`` greps ``rma/ops/`` to enforce it.
  * An operation never reads the store directly for context. What it sees is
    the observation, so the context budget cannot be bypassed by a helpful
    ``store.records()`` call inside an operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..budget import count_tokens
from ..config import RunConfig
from ..orchestrator import Query, RunResult, run_unit
from ..store import ResearchStore

# name -> Operation subclass, populated by @register.
REGISTRY: dict[str, type["Operation"]] = {}


def register(cls: type["Operation"]) -> type["Operation"]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a name")
    REGISTRY[cls.name] = cls
    return cls


def get_operation(name: str) -> "Operation":
    if name not in REGISTRY:
        raise KeyError(f"unknown operation {name!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]()


@dataclass
class OpContext:
    """Everything an operation needs that is not the observation itself."""

    store: ResearchStore
    config: RunConfig
    problem: dict = field(default_factory=dict)
    round: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def problem_text(self) -> str:
        for key in ("normalized_statement", "statement_excerpt", "title"):
            value = self.problem.get(key)
            if value:
                return str(value)
        return ""

    def current_proof_text(self) -> str:
        """The proof an operation acts ON — its subject, not ambient context.

        The critic analyses this, the solver patches it, the evaluator scores
        it. Reading the subject is not the budget-escape the no-bypass rule
        guards against: what that rule forbids is trawling the whole store
        (``store.records()``) to smuggle in extra context past PrefixToBudget.
        The proof is already the highest-priority section of the observation.
        """
        proof = self.store.current_proof()
        return proof.body if proof else ""

    def args_namespace(self):
        """A Namespace shaped for rma.completeness / rma.solve helpers, which
        predate the store and read model settings off argparse args."""
        from argparse import Namespace

        return Namespace(
            model_name=self.extra.get("model_name", self.config.model),
            model_provider=self.extra.get("model_provider", "auto"),
        )


class Operation:
    """One research operation: u in Run(u, q, S, B)."""

    name: str = ""
    #: what the operation returns; "json" goes through call_json, "latex"
    #: through the proof-document path.
    expects: str = "json"
    #: include CurrentProof(S) in the compiled context
    include_proof: bool = True

    # ── the four things a subclass defines ──────────────────────────────────

    def instructions(self, ctx: OpContext) -> str:
        raise NotImplementedError

    def query(self, ctx: OpContext) -> Query:
        raise NotImplementedError

    def parse(self, raw, ctx: OpContext):
        """Model reply -> the operation's own artifact type."""
        return raw

    def writeback(self, artifact, ctx: OpContext) -> list[dict]:
        """Artifact -> store records, as {component, kind, body, meta} dicts."""
        return []

    # ── execution ───────────────────────────────────────────────────────────

    def run(self, ctx: OpContext, *, invoke: Callable[[str, str], object] | None = None,
            telemetry_path=None) -> RunResult:
        """Compile, invoke, write back — all of it via Run(u,q,S,B)."""
        invoker = invoke or default_invoker(self, ctx)

        def _invoke(unit: str, observation: str):
            raw = invoker(unit, observation)
            artifact = self.parse(raw, ctx)
            ctx.extra["artifact"] = artifact
            return self.writeback(artifact, ctx)

        result = run_unit(
            self.name,
            self.query(ctx),
            ctx.store,
            ctx.config.context_budget,
            instructions=self.instructions(ctx),
            invoke=_invoke,
            config=ctx.config,
            telemetry_path=telemetry_path,
            round=ctx.round,
            include_proof=self.include_proof,
        )
        result.artifact = ctx.extra.get("artifact")
        return result


def default_invoker(op: "Operation", ctx: OpContext) -> Callable[[str, str], object]:
    """The real backend. Imported lazily so tests never touch a model.

    Routing honours ``cfg.provider`` (default "claude-code" = the Pro/Max
    SUBSCRIPTION via the local `claude` CLI, no API key, no per-token billing).
    When the provider is the subscription, both the LaTeX and JSON paths go
    through ``call_claude_code`` / ``call_json`` with provider forced, so the
    operation runs on the user's plan rather than the paid Messages API. The
    model that CLI actually runs is still the paper's claude-opus-4-8.
    """
    provider = getattr(ctx.config, "provider", "claude-code")
    # On the subscription CLI, the model argument the CLI understands is a
    # backend selector; pass "claude-code" so it maps to the opus-4-8 backbone,
    # while the API path passes the concrete model name.
    from .. import models

    subscription = provider in ("claude-code", "subscription", "auto")
    model = "claude-code" if subscription else ctx.config.model

    def _invoke(unit: str, observation: str):
        system = f"You are the {unit} operation for Research Math Agent."
        if op.expects == "latex":
            if subscription:
                # Benchmark hook: route LaTeX ops through the meter too, or the
                # cost axis would count only the JSON ops and undercount RMA.
                _shim = __import__("os").environ.get("RMA_VIA_SHIM")
                if _shim:
                    _t = models._shim_complete(_shim, system, observation)
                    if _t is not None:
                        return _t
                response = models.call_claude_code(
                    model=model, system=system, prompt=observation,
                    cwd=ctx.store.repo_root, expect="latex")
            else:
                # Explicit API provider: a LaTeX op runs on the paid Messages API
                # rather than the subscription CLI, honouring cfg.provider.
                response = models.call_anthropic(
                    model=ctx.config.model, system=system, prompt=observation,
                    max_tokens=ctx.config.max_output_tokens)
            return response.text
        return models.call_json(
            model=model, system=system, prompt=observation,
            provider=("claude-code" if subscription else provider),
            max_tokens=ctx.config.max_output_tokens, cwd=ctx.store.repo_root)

    return _invoke


# ─────────────────────────────────────────────────────────────────────────────
# offline backend for tests
# ─────────────────────────────────────────────────────────────────────────────
class FakeBackend:
    """Canned replies keyed by operation name, so the whole round loop runs
    offline with no network and no token spend.

    Records every observation it was given, which is what lets a test assert
    what an operation was actually shown rather than what it was meant to be.
    """

    def __init__(self, replies: dict | None = None, default=None) -> None:
        self.replies = dict(replies or {})
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def __call__(self, unit: str, observation: str):
        self.calls.append((unit, observation))
        reply = self.replies.get(unit, self.default)
        return reply(observation) if callable(reply) else reply

    # ── assertions tests find useful ────────────────────────────────────────

    def observation_for(self, unit: str) -> str:
        for name, obs in self.calls:
            if name == unit:
                return obs
        raise AssertionError(f"{unit} was never invoked; called: {[c[0] for c in self.calls]}")

    def units(self) -> list[str]:
        return [name for name, _ in self.calls]

    def max_observation_tokens(self) -> int:
        return max((count_tokens(obs) for _, obs in self.calls), default=0)
