"""The persistent research store S = (Pi, I, M, L, K, H, E).

Algorithm 1 keeps one store per problem (main.tex:357) and every operation
reads and writes it through ``Run(u,q,S,B)``. This module is that store.

    Pi  proof artifacts and their revision history, including the current proof
    I   open and resolved proof issues and obligations
    M   structured research meeting records
    L   retrieved literature and associated notes
    K   mathematical concepts and background knowledge
    H   strategic insights
    E   evaluation records

Adapt, do not fork
------------------
Each component already has a home in the webapp. The store is a typed facade
over those homes, so the website and ``rma solve`` converge on one state
instead of maintaining two. Canonical content stays where it is:

    Pi  webapp/proof_history/<pid>/          via webapp.proof_history
    I   webapp/issues/<dataset>/<pid>/       via webapp.issues
    M   webapp/meets/<pid>/                  via webapp.meet
    L   documents/questions/<pid>/literature via webapp.literature
    K   documents/questions/<pid>/concepts.json via webapp.concepts
    H   webapp/insights/questions/...        via webapp.insights
    E   documents/questions/<pid>/proof_eval.json via webapp.proof_eval
        (proof_eval.json is a single slot; the paper needs an append-only
        series, so E additionally keeps one record per evaluation and still
        refreshes the canonical file for the website.)

Alongside those, the store keeps a sidecar index at

    documents/questions/<pid>/store/<dataset>/<component>.jsonl

holding the fields the canonical formats have nowhere to put: the round that
produced a record, its context priority, and its links to other records. The
sidecar is an index, not a second source of truth — canonical fields (an
issue's status, say) are refreshed from the owning module on read.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Store components, in the paper's order.
COMPONENTS = ("Pi", "I", "M", "L", "K", "H", "E")

COMPONENT_NAMES = {
    "Pi": "proofs",
    "I": "issues",
    "M": "meetings",
    "L": "literature",
    "K": "concepts",
    "H": "insights",
    "E": "evaluations",
}

# Default context priority per component: higher survives PrefixToBudget
# longer. The current proof and open issues matter most to every operation;
# background knowledge is the first thing to drop. Run() overrides these
# per call, since what is "linked" to the task outranks a blanket default.
DEFAULT_PRIORITY = {
    "Pi": 100,
    "I": 90,
    "E": 60,
    "M": 50,
    "L": 40,
    "K": 30,
    "H": 20,
}


class StoreError(RuntimeError):
    """Raised for an unknown component or a malformed record."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Record:
    """One addressable artifact in the store.

    ``priority`` is *context* priority (what PrefixToBudget drops first), not
    issue severity — an issue's P0-P3 lives in ``meta["severity"]``.
    """

    id: str
    component: str
    round: int
    kind: str
    body: str
    priority: int = 50
    created_at: str = field(default_factory=_now)
    links: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.component not in COMPONENTS:
            raise StoreError(
                f"unknown component {self.component!r}; expected one of {', '.join(COMPONENTS)}"
            )

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> Record:
        return cls(
            id=data["id"],
            component=data["component"],
            round=int(data.get("round", 0)),
            kind=data.get("kind", ""),
            body=data.get("body", ""),
            priority=int(data.get("priority", 50)),
            created_at=data.get("created_at", ""),
            links=list(data.get("links", [])),
            meta=dict(data.get("meta", {})),
        )


# ─────────────────────────────────────────────────────────────────────────────
# adapters onto the existing per-component persistence
# ─────────────────────────────────────────────────────────────────────────────
class ComponentAdapter:
    """Projects one store component onto its canonical webapp home.

    ``push`` writes a new record's canonical form; ``pull`` returns records
    that exist canonically but were not created through the store (so work
    done in the website is visible to the orchestrator), and ``refresh``
    re-reads mutable canonical fields for a record the store already knows.
    """

    component = ""

    def push(self, store: ResearchStore, record: Record) -> None:  # pragma: no cover - default
        return None

    def pull(self, store: ResearchStore, known_ids: set[str]) -> list[Record]:
        return []

    def refresh(self, store: ResearchStore, record: Record) -> Record:
        return record


class IssuesAdapter(ComponentAdapter):
    component = "I"

    def push(self, store: ResearchStore, record: Record) -> None:
        from webapp.issues import create_issue

        # Compute the first line safely: an empty body makes "".splitlines()
        # return [], so [0] would IndexError before the "Untitled" fallback.
        first_line = (record.body.splitlines() or [""])[0][:120]
        issue = create_issue(
            store.repo_root,
            store.problem_id,
            title=record.meta.get("title") or first_line or "Untitled",
            body=record.body,
            author=record.meta.get("author", "rma"),
            dataset=store.dataset,
            issue_type=record.meta.get("issue_type"),
            priority=record.meta.get("severity"),
        )
        record.meta["issue_id"] = issue["id"]
        record.meta.setdefault("status", issue.get("status", "open"))
        record.meta.setdefault("severity", issue.get("priority", "P2"))

    def pull(self, store: ResearchStore, known_ids: set[str]) -> list[Record]:
        from webapp.issues import list_issues

        known_issue_ids = {
            r.meta.get("issue_id")
            for r in store._sidecar_records("I")
            if r.meta.get("issue_id")
        }
        out = []
        for issue in list_issues(store.repo_root, store.problem_id, store.dataset):
            if issue["id"] in known_issue_ids:
                continue
            out.append(Record(
                id=f"{store.problem_id}-I-{issue['id']}",
                component="I",
                round=int(issue.get("round", 0)),
                kind="issue",
                body=issue.get("title", ""),
                priority=DEFAULT_PRIORITY["I"],
                created_at=issue.get("created_at", ""),
                meta={
                    "issue_id": issue["id"],
                    "status": issue.get("status", "open"),
                    "severity": issue.get("priority", "P2"),
                    "issue_type": issue.get("issue_type"),
                    "origin": "webapp",
                },
            ))
        return out

    def refresh(self, store: ResearchStore, record: Record) -> Record:
        """An issue's status and severity are owned by the issue tracker."""
        issue_id = record.meta.get("issue_id")
        if not issue_id:
            return record
        from webapp.issues import get_issue

        issue = get_issue(store.repo_root, store.problem_id, issue_id, store.dataset)
        if issue:
            record.meta["status"] = issue.get("status", record.meta.get("status"))
            record.meta["severity"] = issue.get("priority", record.meta.get("severity"))
        return record


class ProofsAdapter(ComponentAdapter):
    component = "Pi"

    def push(self, store: ResearchStore, record: Record) -> None:
        from webapp.proof_history import record_proof_version

        previous = store.current_proof()
        entry = record_proof_version(
            store.repo_root,
            store.problem_id,
            record.body,
            old_tex=previous.body if previous else None,
            issue_id=record.meta.get("issue_id"),
            issue_title=record.meta.get("issue_title"),
            agent=record.meta.get("produced_by"),
        )
        if entry:
            record.meta["version"] = entry.get("version")
            record.meta["sha"] = entry.get("sha")

    def pull(self, store: ResearchStore, known_ids: set[str]) -> list[Record]:
        from webapp.proof_history import get_proof_version_tex, list_proof_history

        known_versions = {
            r.meta.get("version") for r in store._sidecar_records("Pi")
        }
        out = []
        for entry in list_proof_history(store.repo_root, store.problem_id):
            version = entry.get("version")
            if version in known_versions:
                continue
            tex = get_proof_version_tex(store.repo_root, store.problem_id, version) or ""
            out.append(Record(
                id=f"{store.problem_id}-Pi-v{version:04d}",
                component="Pi",
                round=0,
                kind="proof_revision",
                body=tex,
                priority=DEFAULT_PRIORITY["Pi"],
                created_at=entry.get("timestamp", ""),
                meta={"version": version, "sha": entry.get("sha"),
                      "produced_by": entry.get("agent"), "origin": "webapp"},
            ))
        return out


class LiteratureAdapter(ComponentAdapter):
    component = "L"

    def push(self, store: ResearchStore, record: Record) -> None:
        url = record.meta.get("url")
        if not url:
            return  # a note with no source stays store-native
        from webapp.literature import add_paper

        paper = add_paper(
            store.repo_root, store.problem_id,
            url=url,
            title=record.meta.get("title", record.body[:80]),
            notes=record.body,
            tags=list(record.meta.get("tags", [])),
            added_by=record.meta.get("author", "rma"),
        )
        record.meta["paper_id"] = paper["id"]

    def pull(self, store: ResearchStore, known_ids: set[str]) -> list[Record]:
        from webapp.literature import load_index

        known = {r.meta.get("paper_id") for r in store._sidecar_records("L")}
        out = []
        for paper in load_index(store.repo_root, store.problem_id):
            if paper.get("id") in known:
                continue
            out.append(Record(
                id=f"{store.problem_id}-L-{paper['id']}",
                component="L",
                round=0,
                kind="literature_note",
                body=paper.get("notes") or paper.get("abstract", ""),
                priority=DEFAULT_PRIORITY["L"],
                created_at=paper.get("added_at", ""),
                meta={"paper_id": paper.get("id"), "url": paper.get("url"),
                      "title": paper.get("title"), "origin": "webapp"},
            ))
        return out


class ConceptsAdapter(ComponentAdapter):
    component = "K"

    def push(self, store: ResearchStore, record: Record) -> None:
        from webapp.concepts import load_concepts, save_concepts

        entries = load_concepts(store.repo_root, store.problem_id)
        name = record.meta.get("name") or record.body[:60]
        # Persist the resolved name back onto the record, so the sidecar carries
        # it and pull()'s known-set can match it. Without this a name derived
        # from the body is lost, and the concept is pulled again from
        # concepts.json on reopen — a double count.
        record.meta["name"] = name
        if any((c.get("name") or "").strip().lower() == name.strip().lower() for c in entries):
            return  # glossary entries are unique by name
        entries.append({
            "name": name,
            "definition": record.body,
            "assumptions": record.meta.get("assumptions", ""),
            "notation": record.meta.get("notation", ""),
            "source": record.meta.get("source", ""),
        })
        save_concepts(store.repo_root, store.problem_id, entries)

    def pull(self, store: ResearchStore, known_ids: set[str]) -> list[Record]:
        from webapp.concepts import load_concepts

        known = {(r.meta.get("name") or "").lower() for r in store._sidecar_records("K")}
        out = []
        for i, concept in enumerate(load_concepts(store.repo_root, store.problem_id)):
            name = concept.get("name", f"concept-{i}")
            if name.lower() in known:
                continue
            out.append(Record(
                id=f"{store.problem_id}-K-{i}",
                component="K",
                round=0,
                kind="concept",
                body=concept.get("definition", ""),
                priority=DEFAULT_PRIORITY["K"],
                meta={"name": name, "origin": "webapp"},
            ))
        return out


class InsightsAdapter(ComponentAdapter):
    component = "H"

    def push(self, store: ResearchStore, record: Record) -> None:
        from webapp.insights import get_question_insight, save_question_insight

        current = get_question_insight(store.repo_root, store.problem_id, store.dataset) or {}
        items = list(current.get("insights") or [])
        items.append({"text": record.body, "round": record.round, "at": record.created_at})
        current["insights"] = items
        save_question_insight(store.repo_root, store.problem_id, store.dataset, current)

    def pull(self, store: ResearchStore, known_ids: set[str]) -> list[Record]:
        from webapp.insights import get_question_insight

        data = get_question_insight(store.repo_root, store.problem_id, store.dataset) or {}
        known = {r.body for r in store._sidecar_records("H")}
        out = []
        for i, item in enumerate(data.get("insights") or []):
            text = item.get("text") if isinstance(item, dict) else str(item)
            if not text or text in known:
                continue
            out.append(Record(
                id=f"{store.problem_id}-H-{i}", component="H", round=0,
                kind="insight", body=text, priority=DEFAULT_PRIORITY["H"],
                meta={"origin": "webapp"},
            ))
        return out


class MeetingsAdapter(ComponentAdapter):
    """Meeting rooms live in webapp.meet; the store projects each room (and its
    action plan, when synthesised) as records, and — crucially — a meeting
    written THROUGH the store is materialised as a real webapp room, so the
    context-book report (which reads webapp.meet.list_rooms) shows it. Without
    push() the orchestrator's meetings landed only in the sidecar and never
    appeared in the report."""

    component = "M"

    def push(self, store: ResearchStore, record: Record) -> None:
        from webapp.meet import create_room, post_message, set_plan

        if record.kind == "meeting_record":
            room = create_room(store.repo_root, store.problem_id,
                               topic=record.meta.get("topic") or "Orchestrator meeting",
                               goal="Coordinated proof-strategy review",
                               participants=list(record.meta.get("participants") or []) or None)
            record.meta["room_id"] = room["id"]
            if record.body:
                post_message(store.repo_root, store.problem_id, room["id"],
                             author="coordinator", body=record.body, role="agent")
        elif record.kind == "action_plan":
            # Attach the plan to the most recent room for this problem, so plan
            # and transcript render together in the Meetings chapter.
            from webapp.meet import list_rooms
            rooms = list_rooms(store.repo_root, store.problem_id)
            if rooms:
                room_id = rooms[-1]["id"]
                record.meta["room_id"] = room_id
                steps = record.meta.get("steps") or []
                set_plan(store.repo_root, store.problem_id, room_id,
                         steps=[s if isinstance(s, dict) else {"title": str(s)} for s in steps],
                         summary=record.body or "")

    def pull(self, store: ResearchStore, known_ids: set[str]) -> list[Record]:
        from webapp.meet import list_rooms, transcript_text

        known = {r.meta.get("room_id") for r in store._sidecar_records("M")}
        out = []
        for room in list_rooms(store.repo_root, store.problem_id):
            if room.get("id") in known:
                continue
            out.append(Record(
                id=f"{store.problem_id}-M-{room['id']}",
                component="M",
                round=0,
                kind="meeting_record",
                body=transcript_text(room),
                priority=DEFAULT_PRIORITY["M"],
                created_at=room.get("created_at", ""),
                meta={"room_id": room.get("id"), "topic": room.get("topic"),
                      "participants": room.get("participants", []),
                      "origin": "webapp"},
            ))
            plan = room.get("plan")
            if isinstance(plan, dict) and (plan.get("summary") or plan.get("steps")):
                out.append(Record(
                    id=f"{store.problem_id}-M-{room['id']}-plan",
                    component="M",
                    round=0,
                    kind="action_plan",
                    body=plan.get("summary", ""),
                    priority=DEFAULT_PRIORITY["M"] + 5,  # a plan outranks a transcript
                    created_at=room.get("created_at", ""),
                    meta={"room_id": room.get("id"), "steps": plan.get("steps", []),
                          "origin": "webapp"},
                ))
        return out


class InsightsAdapter(ComponentAdapter):
    component = "H"

    def push(self, store: ResearchStore, record: Record) -> None:
        from webapp.insights import get_question_insight, save_question_insight

        current = get_question_insight(store.repo_root, store.problem_id, store.dataset) or {}
        items = list(current.get("insights") or [])
        items.append({"text": record.body, "round": record.round, "at": record.created_at})
        current["insights"] = items
        save_question_insight(store.repo_root, store.problem_id, store.dataset, current)

    def pull(self, store: ResearchStore, known_ids: set[str]) -> list[Record]:
        """Existing insight files predate this store and use the generator's
        shape (summary / highlights / suggested_todos) rather than a flat
        "insights" list, so read both."""
        from webapp.insights import get_question_insight

        data = get_question_insight(store.repo_root, store.problem_id, store.dataset) or {}
        known = {r.body for r in store._sidecar_records("H")}
        out: list[Record] = []

        def _emit(text: str, kind: str, idx: int) -> None:
            if not text or text in known or any(r.body == text for r in out):
                return
            out.append(Record(
                id=f"{store.problem_id}-H-{kind}-{idx}", component="H", round=0,
                kind=kind, body=text, priority=DEFAULT_PRIORITY["H"],
                created_at=data.get("generated_at", ""),
                meta={"origin": "webapp"},
            ))

        for i, item in enumerate(data.get("insights") or []):
            _emit(item.get("text") if isinstance(item, dict) else str(item), "insight", i)
        for i, item in enumerate(data.get("highlights") or []):
            _emit(item if isinstance(item, str) else json.dumps(item), "highlight", i)
        for i, item in enumerate(data.get("suggested_todos") or []):
            _emit(item if isinstance(item, str) else json.dumps(item), "suggested_todo", i)
        if data.get("summary"):
            _emit(str(data["summary"]), "insight_summary", 0)
        return out


class EvaluationsAdapter(ComponentAdapter):
    component = "E"

    def push(self, store: ResearchStore, record: Record) -> None:
        """Keep the append-only series in the store and refresh the single-slot
        canonical file the website reads."""
        scores = record.meta.get("scores")
        if not isinstance(scores, dict):
            return
        from webapp.proof_eval import _save_proof_eval

        payload = dict(scores)
        payload.setdefault("round", record.round)
        payload.setdefault("proof_hash", record.meta.get("proof_hash"))
        try:
            _save_proof_eval(store.repo_root, store.problem_id, payload)
        except Exception:
            pass

    def pull(self, store: ResearchStore, known_ids: set[str]) -> list[Record]:
        """Surface a pre-existing proof_eval.json as the first element of E.

        proof_eval.json is a single slot with no round field, so an evaluation
        already on disk is only projected when the store has none of its own —
        otherwise it would duplicate the newest record the store just wrote.
        Round-summary records (kind == 'round_summary') are NOT evaluations, so
        they do not suppress the canonical eval.
        """
        if any(r.kind == "evaluation" for r in store._sidecar_records("E")):
            return []
        from webapp.proof_eval import load_proof_eval

        data = load_proof_eval(store.repo_root, store.problem_id)
        if not data or "error" in data:
            return []
        body = data.get("verdict") or data.get("notes") or "evaluation"
        return [Record(
            id=f"{store.problem_id}-E-canonical",
            component="E",
            round=int(data.get("round", 0)),
            kind="evaluation",
            body=str(body),
            priority=DEFAULT_PRIORITY["E"],
            created_at=data.get("evaluated_at", ""),
            meta={"scores": {k: data.get(k) for k in
                             ("answer_accuracy", "logical_correctness",
                              "proof_completeness", "proof_clarity")},
                  "proof_hash": data.get("proof_hash"), "origin": "webapp"},
        )]


ADAPTERS: dict[str, ComponentAdapter] = {
    "Pi": ProofsAdapter(),
    "I": IssuesAdapter(),
    "M": MeetingsAdapter(),
    "L": LiteratureAdapter(),
    "K": ConceptsAdapter(),
    "H": InsightsAdapter(),
    "E": EvaluationsAdapter(),
}


# ─────────────────────────────────────────────────────────────────────────────
# the store
# ─────────────────────────────────────────────────────────────────────────────
class ResearchStore:
    """S for one problem. Open with :meth:`open`."""

    def __init__(self, repo_root: Path, dataset: str, problem_id: str) -> None:
        self.repo_root = Path(repo_root)
        self.dataset = dataset
        self.problem_id = problem_id
        self.round = 0
        self._cache: dict[str, list[Record]] | None = None

    @classmethod
    def open(cls, repo_root: Path | str, problem_id: str,
             dataset: str = "first_proof_1") -> ResearchStore:
        return cls(Path(repo_root), dataset, problem_id)

    def reopen(self) -> ResearchStore:
        """A fresh handle on the same problem — nothing cached in memory.
        Preserves the concrete class, so reopening a memory-ablation store
        (StatelessStore / LastRoundOnlyStore) does not silently widen it to a
        full-memory store."""
        store = type(self)(self.repo_root, self.dataset, self.problem_id)
        store.round = self.round
        return store

    # ── paths ───────────────────────────────────────────────────────────────

    @property
    def store_dir(self) -> Path:
        return (self.repo_root / "documents" / "questions" / self.problem_id
                / "store" / self.dataset)

    def _sidecar_path(self, component: str) -> Path:
        return self.store_dir / f"{component}.jsonl"

    # ── round cursor ────────────────────────────────────────────────────────

    def begin_round(self, r: int) -> None:
        self.round = int(r)
        self._cache = None

    # ── reading ─────────────────────────────────────────────────────────────

    def _sidecar_records(self, component: str) -> list[Record]:
        path = self._sidecar_path(component)
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Record.from_json(json.loads(line)))
            except Exception:
                continue
        return out

    def _all_records(self, component: str) -> list[Record]:
        records = [ADAPTERS[component].refresh(self, r)
                   for r in self._sidecar_records(component)]
        known = {r.id for r in records}
        records.extend(ADAPTERS[component].pull(self, known))
        records.sort(key=lambda r: (r.round, r.created_at, r.id))
        return records

    def records(self, component: str | None = None,
                round: int | None = None) -> list[Record]:
        """Every visible record, optionally filtered by component and round."""
        if component is not None and component not in COMPONENTS:
            raise StoreError(f"unknown component: {component}")
        components = (component,) if component else COMPONENTS
        out: list[Record] = []
        for comp in components:
            out.extend(self._all_records(comp))
        out = [r for r in out if self._visible(r)]
        if round is not None:
            out = [r for r in out if r.round == round]
        return out

    def _visible(self, record: Record) -> bool:
        """Overridden by the memory ablations."""
        return True

    def get(self, record_id: str) -> Record | None:
        for record in self.records():
            if record.id == record_id:
                return record
        return None

    def counts(self) -> dict[str, int]:
        return {comp: len(self.records(comp)) for comp in COMPONENTS}

    # component views, named as the paper names them
    @property
    def proofs(self) -> list[Record]:
        return self.records("Pi")

    @property
    def issues(self) -> list[Record]:
        return self.records("I")

    @property
    def meetings(self) -> list[Record]:
        return self.records("M")

    @property
    def literature(self) -> list[Record]:
        return self.records("L")

    @property
    def concepts(self) -> list[Record]:
        return self.records("K")

    @property
    def insights(self) -> list[Record]:
        return self.records("H")

    @property
    def evaluations(self) -> list[Record]:
        return self.records("E")

    def open_issues(self) -> list[Record]:
        return [r for r in self.issues if r.meta.get("status", "open") != "resolved"]

    def set_issue_status(self, record_id: str, status: str) -> bool:
        """Mark an issue record's status (e.g. 'resolved') in its canonical
        home so open_issues() and the termination predicates see it.

        Returns True if an issue was updated. Used by the solver to close the
        issue it repaired — without this, Solved() can never fire, because the
        gap the critic opened stays 'open' forever even after it is fixed.
        """
        record = self.get(record_id)
        if record is None or record.component != "I":
            return False
        issue_id = record.meta.get("issue_id")
        if not issue_id:
            return False
        from webapp.issues import update_issue

        with self._lock():
            update_issue(self.repo_root, self.problem_id, issue_id,
                         dataset=self.dataset, status=status)
        self._cache = None
        return True

    def _lock(self):
        from webapp.locks import file_lock

        return file_lock(self.repo_root, f"store_{self.dataset}_{self.problem_id}")

    # ── writing ─────────────────────────────────────────────────────────────

    def _next_id(self, component: str) -> str:
        n = len(self._sidecar_records(component)) + 1
        return f"{self.problem_id}-{component}-{n}"

    def add(self, component: str, kind: str, body: str, *,
            priority: int | None = None, links: Iterable[str] | None = None,
            meta: dict | None = None, round: int | None = None) -> Record:
        """Create a record, push it to its canonical home, and index it."""
        if component not in COMPONENTS:
            raise StoreError(f"unknown component: {component}")
        from webapp.locks import file_lock

        with file_lock(self.repo_root, f"store_{self.dataset}_{self.problem_id}"):
            record = Record(
                id=self._next_id(component),
                component=component,
                round=self.round if round is None else int(round),
                kind=kind,
                body=body,
                priority=DEFAULT_PRIORITY.get(component, 50) if priority is None else priority,
                links=list(links or []),
                meta=dict(meta or {}),
            )
            ADAPTERS[component].push(self, record)
            self._append_sidecar(record)
        self._cache = None
        return record

    def _append_sidecar(self, record: Record) -> None:
        path = self._sidecar_path(record.component)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic append: rewrite via a temp file so a crash cannot leave a
        # half-written line that makes the whole component unreadable.
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(existing + json.dumps(record.to_json(), ensure_ascii=False) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)

    # ── Pi helpers ──────────────────────────────────────────────────────────

    def current_proof(self) -> Record | None:
        """pi <- CurrentProof(S): the latest proof revision that is still a proof.

        Taking the last revision unconditionally makes progress non-monotone: any
        op that writes a degenerate revision -- an empty string, a backend refusal,
        a stub that discards most of the argument -- instantly becomes "the current
        proof" and the real one is gone. Measured on the improvement task, that is
        exactly what happened: a run seeded with a 10,361-character proof reported
        "nothing substantive to work on yet" one round later and delivered nothing,
        three times out of five problems.

        A revision is skipped as degenerate when it is empty, when it is a backend
        refusal rather than mathematics, or when it has thrown away three quarters
        of the longest revision written so far. If every revision is degenerate the
        newest is still returned, so this can only ever preserve information.
        """
        revisions = self.records("Pi")
        if not revisions:
            return None

        def body_of(rec) -> str:
            return (getattr(rec, "body", None) or getattr(rec, "text", "") or "")

        longest = max((len(body_of(r)) for r in revisions), default=0)

        def degenerate(rec) -> bool:
            text = body_of(rec).strip()
            if not text:
                return True
            try:
                from .call_budget import is_refusal
                if is_refusal(text):
                    return True
            except Exception:
                pass
            return longest > 0 and len(text) < 0.25 * longest

        for rec in reversed(revisions):
            if not degenerate(rec):
                return rec
        return revisions[-1]

    def add_proof_revision(self, tex: str, *, produced_by: str,
                           round: int | None = None, parent_id: str | None = None,
                           meta: dict | None = None) -> Record:
        payload = dict(meta or {})
        payload["produced_by"] = produced_by
        parent = parent_id
        if parent is None:
            previous = self.current_proof()
            parent = previous.id if previous else None
        payload["parent_id"] = parent
        return self.add("Pi", "proof_revision", tex,
                        links=[parent] if parent else [], meta=payload, round=round)

    def proof_history(self) -> list[Record]:
        return self.records("Pi")

    # ── write-back ──────────────────────────────────────────────────────────

    def write_back(self, unit: str, query: object, artifact: object,
                   round: int | None = None) -> list[Record]:
        """S <- WriteBack(S, u, q, y).

        One entry point so every operation's output lands in the store with
        provenance attached, instead of each call site inventing its own.
        ``artifact`` is a list of ``{component, kind, body, ...}`` dicts.
        """
        if artifact is None:
            return []
        items = artifact if isinstance(artifact, list) else [artifact]
        written = []
        for item in items:
            if not isinstance(item, dict):
                continue
            component = item.get("component")
            if component not in COMPONENTS:
                continue
            meta = dict(item.get("meta", {}))
            meta.setdefault("unit", unit)
            if isinstance(query, dict) and query.get("id"):
                meta.setdefault("query_id", query["id"])
            written.append(self.add(
                component,
                item.get("kind", unit),
                str(item.get("body", "")),
                priority=item.get("priority"),
                links=item.get("links"),
                meta=meta,
                round=round,
            ))
        return written


# ─────────────────────────────────────────────────────────────────────────────
# memory ablations (paper: 1.8 stateless, 3.4 last-round-only, 6.0 full)
# ─────────────────────────────────────────────────────────────────────────────
class StatelessStore(ResearchStore):
    """Nothing from a previous round is visible.

    Records are still written, so telemetry and the final proof survive; the
    orchestrator simply cannot see them, which is what "stateless" means for
    the ablation.
    """

    def _visible(self, record: Record) -> bool:
        return record.round >= self.round


class LastRoundOnlyStore(ResearchStore):
    """Only the current and immediately preceding round are visible."""

    def _visible(self, record: Record) -> bool:
        return record.round >= self.round - 1


def open_store(repo_root: Path | str, problem_id: str,
               dataset: str = "first_proof_1", *, memory: str = "full") -> ResearchStore:
    """Open S with the requested memory model.

    memory: "full" (default) | "last-round-only" | "stateless"
    """
    cls = {"full": ResearchStore,
           "last-round-only": LastRoundOnlyStore,
           "stateless": StatelessStore}.get(memory)
    if cls is None:
        raise StoreError(f"unknown memory model: {memory}")
    return cls(Path(repo_root), dataset, problem_id)


def run_inspect_store(args) -> int:
    """`rma inspect-store` — per-component counts and the current proof id."""
    from .doctor import _resolve_repo_root

    repo_root = _resolve_repo_root(getattr(args, "repo_root", None))
    if repo_root is None:
        print("RMA inspect-store")
        print("FAIL repo root: could not find README.md and data/first_proof_1/problems")
        return 1

    dataset = getattr(args, "dataset", None) or "first_proof_1"
    problem_id = getattr(args, "problem", None)
    if not problem_id:
        print("RMA inspect-store")
        print("FAIL problem: --problem is required")
        return 1

    store = open_store(repo_root, problem_id, dataset,
                       memory=getattr(args, "memory", None) or "full")
    counts = store.counts()
    current = store.current_proof()
    payload = {
        **counts,
        "problem_id": problem_id,
        "dataset": dataset,
        "current_proof_id": current.id if current else None,
        "current_proof_chars": len(current.body) if current else 0,
        "open_issues": len(store.open_issues()),
        "store_dir": str(store.store_dir),
    }

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return 0

    print("RMA inspect-store")
    print(f"problem: {problem_id}  dataset: {dataset}")
    for comp in COMPONENTS:
        print(f"  {comp:<3} {COMPONENT_NAMES[comp]:<12} {counts[comp]}")
    print(f"current proof: {payload['current_proof_id'] or '(none)'} "
          f"({payload['current_proof_chars']} chars)")
    print(f"open issues:   {payload['open_issues']}")
    return 0


def memory_model_for(config) -> str:
    """Map RunConfig ablations onto a memory model."""
    ablations = getattr(config, "ablations", frozenset())
    if "store.stateless" in ablations:
        return "stateless"
    if "store.last-round-only" in ablations:
        return "last-round-only"
    return "full"
