"""The Research Context Orchestrator — Run(u, q, S, B).

Algorithm 1's basic unit of execution (main.tex:400):

    C = Instructions(u) || q || CurrentProof(S) || LinkedRecords(S,q) || RecentOutputs(S,u)
    O = PrefixToBudget(C, B)
    y = Invoke(u, O)
    S = WriteBack(S, u, q, y)
    return (y, S)

Every operation goes through this. The point is not tidiness: it is that the
five context parts are assembled in one place in a known order, the budget is
enforced in one place, and the write-back is not optional — so what a given
call was shown, and what it changed, are both recoverable after the fact from
``orchestration_log.jsonl``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .budget import Chunk, Observation, Section, mandatory_floor_tokens, prefix_to_budget
from .config import RunConfig
from .store import Record, ResearchStore

# The five parts of C, in the paper's order. Earlier sections outrank later
# ones: PrefixToBudget drops from the end.
SECTION_ORDER = ("instructions", "query", "current_proof", "linked_records", "recent_outputs")

# How many recent outputs of the same unit to consider before budgeting.
RECENT_OUTPUT_LIMIT = int(os.environ.get("RMA_RECENT_OUTPUTS", "8"))


@dataclass
class Query:
    """q — an operation's local task.

    ``links`` names the store records the task is about (the issue being
    repaired, the proof slice, the action plan). Those become LinkedRecords.
    """

    text: str
    id: str | None = None
    links: list[str] = field(default_factory=list)
    payload: dict = field(default_factory=dict)

    @classmethod
    def coerce(cls, value) -> Query:
        if isinstance(value, Query):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, dict):
            return cls(text=str(value.get("text", "")), id=value.get("id"),
                       links=list(value.get("links", [])), payload=value)
        return cls(text=str(value))

    def as_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "links": list(self.links)}


@dataclass
class RunResult:
    """(y, S) plus what it cost and what it saw."""

    unit: str
    output: object
    observation: Observation
    written: list[Record]
    round: int
    #: the operation's own parsed artifact (issue queue, action plan, ...);
    #: `output` is the raw write-back payload.
    artifact: object = None

    @property
    def context_tokens(self) -> int:
        return self.observation.tokens


# ─────────────────────────────────────────────────────────────────────────────
# context assembly
# ─────────────────────────────────────────────────────────────────────────────
def linked_records(store: ResearchStore, query: Query, *, hops: int = 1) -> list[Record]:
    """LinkedRecords(S,q) — records related to what q names, one hop out.

    "Related" is bidirectional on the first hop, which matters: a literature
    note or a proof revision points TO the issue it addresses (a backward
    edge), so retrieving "everything relevant to this issue" must follow edges
    in both directions. Following only forward links would leave the solver
    unable to see the literature that was gathered for its own issue — the one
    connection the paper's Run relies on.

    One hop, not transitive closure: the note's neighbour's neighbour is noise.
    """
    if not query.links:
        return []
    all_records = store.records()
    by_id = {r.id: r for r in all_records}
    # reverse adjacency: id -> records that link TO it
    incoming: dict[str, list[Record]] = {}
    for record in all_records:
        for target in record.links:
            incoming.setdefault(target, []).append(record)

    seen: list[Record] = []
    visited: set[str] = set()
    frontier = list(query.links)
    for depth in range(hops + 1):
        nxt: list[str] = []
        for rid in frontier:
            if rid in visited:
                continue
            visited.add(rid)
            record = by_id.get(rid)
            if record is not None:
                seen.append(record)
                nxt.extend(record.links)                       # forward
            nxt.extend(r.id for r in incoming.get(rid, []))    # backward
        frontier = nxt
        if not frontier:
            break
    # Highest context priority first, so the budget drops the weakest links.
    seen.sort(key=lambda r: (-r.priority, r.created_at))
    return seen


def recent_outputs(store: ResearchStore, unit: str,
                   limit: int = RECENT_OUTPUT_LIMIT) -> list[Record]:
    """RecentOutputs(S,u) — this unit's own prior artifacts, newest first."""
    mine = [r for r in store.records() if r.meta.get("unit") == unit]
    mine.sort(key=lambda r: (r.round, r.created_at), reverse=True)
    return mine[:limit]


def build_context(store: ResearchStore, unit: str, query: Query, *,
                  instructions: str, include_proof: bool = True,
                  exclude_components: set[str] | None = None) -> list[Section]:
    """Assemble C in the paper's fixed order.

    ``exclude_components`` drops whole store components from the compiled
    context — the mechanism behind the paper's "drop K / H / E from each
    operation's bounded context" ablations (e.g. evaluator-feedback removes E).
    """
    query_links = set(query.links)
    exclude = exclude_components or set()

    sections = [
        Section("instructions", [Chunk(instructions, droppable=False)]),
        Section("query", [Chunk(query.text, droppable=False)]),
    ]

    proof = store.current_proof() if include_proof else None
    sections.append(Section("current_proof", [
        Chunk(proof.body, record_id=proof.id, priority=proof.priority)
    ] if proof else []))

    linked = [r for r in linked_records(store, query)
              if r.id != (proof.id if proof else None) and r.component not in exclude]
    sections.append(Section("linked_records", [
        Chunk(_render_record(r), record_id=r.id, priority=r.priority) for r in linked
    ]))

    linked_ids = {r.id for r in linked}
    recent = [r for r in recent_outputs(store, unit)
              if r.id not in linked_ids and r.id != (proof.id if proof else None)
              and r.id not in query_links and r.component not in exclude]
    sections.append(Section("recent_outputs", [
        # Recency is the ordering here, so priority decreases down the list and
        # PrefixToBudget drops the oldest first.
        Chunk(_render_record(r), record_id=r.id, priority=max(1, r.priority - 10 * i))
        for i, r in enumerate(recent)
    ]))
    return sections


def _render_record(record: Record) -> str:
    head = f"[{record.id} · {record.component}/{record.kind} · round {record.round}]"
    extras = []
    for key in ("severity", "status", "title", "url"):
        if record.meta.get(key):
            extras.append(f"{key}={record.meta[key]}")
    if extras:
        head += " " + " ".join(extras)
    return f"{head}\n{record.body}"


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
def run_unit(
    unit: str,
    query,
    store: ResearchStore,
    budget: int | None = None,
    *,
    instructions: str,
    invoke: Callable[[str, str], object],
    config: RunConfig | None = None,
    telemetry_path: Path | None = None,
    round: int | None = None,
    include_proof: bool = True,
) -> RunResult:
    """Run(u, q, S, B) — compile, invoke, write back, log.

    ``invoke(unit, observation_text) -> y`` is the only part that talks to a
    model, so an offline test swaps it for a canned artifact.
    """
    cfg = config or RunConfig()
    budget = cfg.context_budget if budget is None else budget
    q = Query.coerce(query)
    r = store.round if round is None else int(round)

    sections = build_context(store, unit, q, instructions=instructions,
                             include_proof=include_proof,
                             exclude_components=_excluded_components(cfg))
    observation = prefix_to_budget(sections, budget, mode=cfg.effective_context_mode)

    output = invoke(unit, observation.text)
    written = store.write_back(unit, q.as_dict(), output, round=r)

    _log_call(store, unit, q, r, observation, written, output,
              telemetry_path=telemetry_path,
              floor=mandatory_floor_tokens(sections),
              order_mode=cfg.order_mode,
              excluded=sorted(_excluded_components(cfg)))

    return RunResult(unit=unit, output=output, observation=observation,
                     written=written, round=r)


# The paper's name, for call sites that read alongside Algorithm 1.
Run = run_unit


def _excluded_components(cfg) -> set[str]:
    """Store components an ablation removes from every operation's context.
    Maps the paper's 'drop K/H/E' ablations onto component letters."""
    exclude: set[str] = set()
    ablations = getattr(cfg, "ablations", frozenset())
    if "evaluator-feedback" in ablations:
        exclude.add("E")
    if "insights" in ablations:
        exclude.add("H")
    if "concepts" in ablations:
        exclude.add("K")   # the paper's "drop K from each operation's context"
    return exclude


def telemetry_path_for(store: ResearchStore) -> Path:
    return store.store_dir / "orchestration_log.jsonl"


def _log_call(store: ResearchStore, unit: str, query: Query, round_idx: int,
              observation: Observation, written: list[Record], output: object,
              *, telemetry_path: Path | None, floor: int,
              order_mode: str | None = None, excluded: list[str] | None = None) -> None:
    """One line per Run. This is the evidence behind the budget claim, and — via
    context_mode / order_mode / excluded_components — the record of which
    *applied* configuration each call ran under, so an ablation can be verified
    to actually change what the operations did."""
    path = telemetry_path or telemetry_path_for(store)
    entry = {
        "unit": unit,
        "round": round_idx,
        "problem_id": store.problem_id,
        "dataset": store.dataset,
        "query_id": query.id,
        "output_tokens": _estimate_output_tokens(output),
        "records_written": [r.id for r in written],
        "mandatory_floor_tokens": floor,
        "over_budget": observation.tokens > observation.budget,
        "order_mode": order_mode,
        "excluded_components": excluded or [],
        **observation.as_dict(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # telemetry must never break a run


def _estimate_output_tokens(output: object) -> int:
    from .budget import count_tokens

    if output is None:
        return 0
    if isinstance(output, str):
        return count_tokens(output)
    try:
        return count_tokens(json.dumps(output, ensure_ascii=False, default=str))
    except Exception:
        return 0


def read_telemetry(store: ResearchStore, telemetry_path: Path | None = None) -> list[dict]:
    """Every logged Run for this problem, oldest first."""
    path = telemetry_path or telemetry_path_for(store)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# rma inspect-context
# ─────────────────────────────────────────────────────────────────────────────
def run_inspect_context(args) -> int:
    """Show exactly what an operation would be sent, without calling a model."""
    from .doctor import _resolve_repo_root
    from .store import open_store

    repo_root = _resolve_repo_root(getattr(args, "repo_root", None))
    if repo_root is None:
        print("RMA inspect-context")
        print("FAIL repo root: could not find README.md and data/first_proof_1/problems")
        return 1

    cfg = RunConfig.from_args(args)
    dataset = getattr(args, "dataset", None) or "first_proof_1"
    problem_id = getattr(args, "problem", None)
    unit = getattr(args, "unit", None) or "critic"
    budget = getattr(args, "budget", None) or cfg.context_budget

    store = open_store(repo_root, problem_id, dataset)
    query = Query(text=f"[{unit}] inspect context for {problem_id}",
                  id=f"{problem_id}-inspect",
                  links=[r.id for r in store.open_issues()[:3]])
    sections = build_context(store, unit, query,
                             instructions=f"Instructions for the {unit} operation.")
    observation = prefix_to_budget(sections, budget, mode=cfg.effective_context_mode)

    if getattr(args, "json", False):
        payload = observation.as_dict()
        # `sections` is what survived (a part can be legitimately empty — a
        # unit with no prior output has no recent_outputs); `section_order` is
        # the five-part order C is always built in.
        payload["sections"] = list(observation.sections)
        payload["section_order"] = list(SECTION_ORDER)
        payload["problem_id"] = problem_id
        payload["unit"] = unit
        payload["mandatory_floor_tokens"] = mandatory_floor_tokens(sections)
        if getattr(args, "show_text", False):
            payload["text"] = observation.text
        print(json.dumps(payload, indent=2))
        return 0

    print("RMA inspect-context")
    print(f"problem: {problem_id}  unit: {unit}  budget: {budget}  mode: {observation.mode}")
    for name in observation.sections:
        print(f"  section: {name}")
    print(f"context_tokens: {observation.tokens}")
    print(f"records included: {len(observation.included)}  dropped: {observation.n_dropped}")
    if observation.dropped:
        print(f"  dropped: {', '.join(observation.dropped[:10])}")
    if getattr(args, "show_text", False):
        print("-" * 60)
        print(observation.text)
    return 0
