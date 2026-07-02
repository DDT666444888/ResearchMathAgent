"""Global daily push-forward across all benchmark problems.

Runs once per day (gated by date stamp, overridable with force=True):
  1. For every active problem: discover new proof gaps (critic agent)
  2. Resolve up to N open issues per problem (solver agent)
  3. Create a meeting room with field-appropriate mathematician personas
  4. Run multiple rounds of discussion (each participant responds to the others)
  5. Synthesize the discussion into a concrete action plan
  6. Save meeting notes to documents/questions/{pid}/meets/{room_id}-notes.md

All agents use the Claude subscription (claude CLI).
State is persisted in webapp/push_forward_state.json.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_STATE_FILE = "push_forward_state.json"
_METRICS_FILE = "push_forward_metrics.json"

# job_id → progress dict (lives in memory for the server lifetime)
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


# ── State persistence ─────────────────────────────────────────────────────────

def _state_path(repo_root: Path) -> Path:
    return repo_root / "webapp" / _STATE_FILE


# ── Metrics persistence (survives restarts, stored in data/) ──────────────────

def _metrics_path(repo_root: Path) -> Path:
    return repo_root / "data" / _METRICS_FILE


def load_metrics(repo_root: Path) -> dict:
    p = _metrics_path(repo_root)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"snapshots": []}


def _save_metrics(repo_root: Path, metrics: dict) -> None:
    p = _metrics_path(repo_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def snapshot_metrics(repo_root: Path, job_id: str, problems: list[str], dataset: str = "first_proof_1") -> dict:
    """Compute document-size and issue-resolve metrics for a set of problems."""
    doc_root = repo_root / "documents" / "questions"
    issues_root = repo_root / "webapp" / "issues" / dataset

    _DOC_EXTS = {".tex", ".md", ".txt"}

    total_bytes = 0
    per_problem_bytes: dict[str, int] = {}
    total_issues = 0
    resolved_issues = 0
    per_problem_issues: dict[str, dict] = {}

    for pid in problems:
        # Document size: all .tex/.md/.txt under documents/questions/<pid>/
        size = 0
        pid_doc = doc_root / pid
        if pid_doc.is_dir():
            for f in pid_doc.rglob("*"):
                if f.is_file() and f.suffix in _DOC_EXTS:
                    try:
                        size += f.stat().st_size
                    except OSError:
                        pass
        per_problem_bytes[pid] = size
        total_bytes += size

        # Issue counts
        open_c = resolved_c = 0
        pid_issues = issues_root / pid
        if pid_issues.is_dir():
            for f in pid_issues.glob("*.json"):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    if d.get("status") == "resolved":
                        resolved_c += 1
                    else:
                        open_c += 1
                except Exception:
                    pass
        per_problem_issues[pid] = {"open": open_c, "resolved": resolved_c}
        total_issues += open_c + resolved_c
        resolved_issues += resolved_c

    n = len(problems)
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "job_id": job_id,
        "dataset": dataset,
        "total_doc_bytes": total_bytes,
        "avg_doc_bytes": total_bytes // n if n else 0,
        "total_issues": total_issues,
        "resolved_issues": resolved_issues,
        "open_issues": total_issues - resolved_issues,
        "solve_rate": round(resolved_issues / total_issues, 4) if total_issues else 0.0,
        "per_problem_doc_bytes": per_problem_bytes,
        "per_problem_issues": per_problem_issues,
    }


def append_metrics_snapshot(repo_root: Path, snap: dict) -> None:
    metrics = load_metrics(repo_root)
    snapshots = metrics.setdefault("snapshots", [])
    snap["round"] = len(snapshots) + 1
    snapshots.append(snap)
    _save_metrics(repo_root, metrics)


# ── Per-problem push-forward score history ────────────────────────────────────
# Every push-forward for a problem appends one entry recording how many
# push-forwards have run and the proof-evaluation score at that point, so the
# Evaluation chapter can show the score trajectory.  Stored next to the proof
# evaluation itself so it travels with the problem's documents.

def _history_path(repo_root: Path, pid: str) -> Path:
    return repo_root / "documents" / "questions" / pid / "push_forward_history.json"


def load_push_forward_history(repo_root: Path, pid: str) -> list[dict]:
    """Return the ordered list of push-forward score entries for a problem."""
    p = _history_path(repo_root, pid)
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            hist = data.get("history", data if isinstance(data, list) else [])
            return hist if isinstance(hist, list) else []
        except Exception:
            pass
    return []


def _save_push_forward_history(repo_root: Path, pid: str, history: list[dict]) -> None:
    p = _history_path(repo_root, pid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"history": history}, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def _score_from_proof_eval(pe: dict | None) -> dict:
    """Extract a compact score record (with total/max) from a proof_eval dict."""
    if not pe or "error" in pe:
        return {"answer_accuracy": None, "logical_correctness": None,
                "proof_completeness": None, "proof_clarity": None,
                "scale": 10, "total": None, "max": None, "recorded": False}
    scale = int(pe.get("scale") or 10)
    aa = pe.get("answer_accuracy")
    lc = pe.get("logical_correctness")
    pc = pe.get("proof_completeness")
    cl = pe.get("proof_clarity")
    vals = [(1 if aa else 0) if aa is not None else None,
            int(lc) if lc is not None else None,
            int(pc) if pc is not None else None,
            int(cl) if cl is not None else None]
    maxes = [1 if aa is not None else None,
             scale if lc is not None else None,
             scale if pc is not None else None,
             scale if cl is not None else None]
    total = sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None
    mx = sum(v for v in maxes if v is not None) if any(v is not None for v in maxes) else None
    return {"answer_accuracy": aa, "logical_correctness": lc,
            "proof_completeness": pc, "proof_clarity": cl,
            "scale": scale, "total": total, "max": mx, "recorded": total is not None}


def append_push_forward_score(repo_root: Path, pid: str, dataset: str = "first_proof_1",
                              room_id: str | None = None, job_id: str | None = None,
                              date: str | None = None) -> dict:
    """Record the current proof-evaluation score as the next push-forward entry.

    Reads documents/questions/<pid>/proof_eval.json, computes the total, and
    appends a numbered entry to the problem's push-forward history.  Returns the
    entry that was appended.
    """
    try:
        from .proof_eval import load_proof_eval
        pe = load_proof_eval(repo_root, pid)
    except Exception:
        pe = None
    entry = {
        "round": 0,  # filled below
        "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "dataset": dataset,
        "job_id": job_id,
        "room_id": room_id,
        **_score_from_proof_eval(pe),
    }
    history = load_push_forward_history(repo_root, pid)
    entry["round"] = len(history) + 1
    history.append(entry)
    _save_push_forward_history(repo_root, pid, history)
    return entry


def backfill_push_forward_history(repo_root: Path, pid: str,
                                  dataset: str = "first_proof_1") -> int:
    """Seed history with past push-forwards recorded in push_forward_state.json.

    Historical proof scores were not captured at the time, so those entries carry
    null scores (recorded=False) but give an accurate push-forward count and
    date/room lineage.  Runs already present in the history (matched by job_id)
    are skipped, so this is idempotent.  Returns the number of entries added.
    """
    history = load_push_forward_history(repo_root, pid)
    known_jobs = {h.get("job_id") for h in history if h.get("job_id")}
    added = 0
    for run in load_state(repo_root).get("runs", []):
        if run.get("dataset") != dataset or pid not in (run.get("problems") or []):
            continue
        jid = run.get("job_id")
        if jid and jid in known_jobs:
            continue
        room_id = None
        for res in run.get("results", []):
            if res.get("problem") == pid:
                room_id = res.get("room_id")
                break
        entry = {
            "round": len(history) + 1,
            "date": run.get("date"),
            "dataset": dataset,
            "job_id": jid,
            "room_id": room_id,
            **_score_from_proof_eval(None),  # historical scores unknown
        }
        history.append(entry)
        known_jobs.add(jid)
        added += 1
    if added:
        # keep chronological order, then renumber rounds
        history.sort(key=lambda h: (h.get("date") or "", h.get("round") or 0))
        for i, h in enumerate(history, 1):
            h["round"] = i
        _save_push_forward_history(repo_root, pid, history)
    return added


def load_state(repo_root: Path) -> dict:
    p = _state_path(repo_root)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_run_date": None, "runs": []}


def _save_state(repo_root: Path, state: dict) -> None:
    _state_path(repo_root).write_text(json.dumps(state, indent=2), encoding="utf-8")


def already_ran_today(repo_root: Path) -> bool:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return load_state(repo_root).get("last_run_date") == today


# ── Job tracking ──────────────────────────────────────────────────────────────

def get_job(job_id: str) -> dict | None:
    return _JOBS.get(job_id)


def list_jobs() -> list[dict]:
    with _LOCK:
        return [{"job_id": jid, **info} for jid, info in _JOBS.items()]


def running_job() -> dict | None:
    with _LOCK:
        for jid, info in _JOBS.items():
            if info.get("status") == "running":
                return {"job_id": jid, **info}
    return None


# ── Meeting documentation ─────────────────────────────────────────────────────

def _save_meeting_notes(repo_root: Path, pid: str, room_id: str) -> Path | None:
    """Write a markdown notes file for a completed meeting room.

    Saved to documents/questions/{pid}/meets/{room_id}-notes.md so it is
    surfaced in the Documents tab and survives server restarts.
    """
    from .meet import get_room, transcript_text
    room = get_room(repo_root, pid, room_id)
    if not room:
        return None

    transcript = transcript_text(room)
    plan = room.get("plan")

    # Skip saving if the meeting produced no substantive content
    # (only "Meeting opened" type events, no real discussion)
    substantive_lines = [
        ln for ln in transcript.splitlines()
        if ln.strip() and "Meeting opened" not in ln and not ln.startswith("[coordinator] Meeting")
    ]
    if len(substantive_lines) < 2 and not plan:
        return None

    lines: list[str] = [
        f"# Meeting Notes: {room.get('topic', room_id)}",
        f"**Participants:** {', '.join(room.get('participants', []))}",
        f"**Date:** {room.get('created_at', '')[:10]}",
        "",
    ]

    if plan and plan.get("summary"):
        lines += ["## Action Plan", "", plan.get("summary", "")]
        for i, step in enumerate(plan.get("steps", []), 1):
            title = step.get("title", f"Step {i}")
            body = step.get("body", step.get("description", ""))
            agent = step.get("agent", "")
            lines.append(f"\n### {i}. {title}" + (f" *(agent: {agent})*" if agent else ""))
            if body:
                lines.append(body)
        lines.append("")

    if substantive_lines:
        lines += ["## Discussion Transcript", "", transcript]

    doc_dir = repo_root / "documents" / "questions" / pid / "meets"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / f"{room_id}-notes.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    return doc_path


# ── Main runner ───────────────────────────────────────────────────────────────

_DEFAULT_MEETING_ROUNDS = 2  # reduced to limit shared quota pressure


def run_push_forward(
    repo_root: Path,
    job_id: str,
    problems: list[str] | None = None,
    max_resolve: int = 2,
    n_meeting_rounds: int = _DEFAULT_MEETING_ROUNDS,
    dataset: str = "first_proof_1",
) -> None:
    """Execute the global push-forward. Blocking — call from a daemon thread.

    For each problem:
      1. Run a full issue cycle (discover + resolve)
      2. Create a meeting room with field-appropriate mathematician personas
      3. Run n_meeting_rounds of interleaved discussion
      4. Synthesize the discussion into a concrete action plan
      5. Save meeting notes to documents/questions/{pid}/meets/{room_id}-notes.md
    """
    from .issue_agents import run_issue_cycle
    from .meet import create_room, get_personas_for_problem
    from .meet_agents import run_round_offline, run_synthesis

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if dataset == "first_proof_2":
        default_problems = [f"prob-{i:02d}" for i in range(1, 11)]
        active_check = lambda p: True  # problems are in dataset store, not .tex files
    else:
        default_problems = [f"q{i}" for i in range(1, 11)]
        active_check = lambda p: (repo_root / "problems" / f"{p}.tex").is_file()
    candidate = problems or default_problems
    active = [p for p in candidate if active_check(p)]

    with _LOCK:
        _JOBS[job_id] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total": len(active),
            "done": 0,
            "current": None,
            "log": [],
            "results": [],
        }

    def _log(msg: str) -> None:
        log.info("[push-forward %s] %s", job_id[:8], msg)
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["log"].append(msg)

    try:
        from .issues import list_issues as _list_issues

        for pid in active:
            with _LOCK:
                job = _JOBS.get(job_id, {})
                job["current"] = pid

            # Snapshot issue counts + IDs before cycle
            try:
                _before = _list_issues(repo_root, pid, dataset)
                open_before = sum(1 for i in _before if i.get("status") in ("open", "in_progress"))
                total_before = len(_before)
                _before_ids = {i["id"] for i in _before}
                _before_open_ids = {i["id"] for i in _before if i.get("status") in ("open", "in_progress")}
            except Exception:
                open_before = total_before = 0
                _before_ids = set()
                _before_open_ids = set()

            _log(f"{pid}: starting issue cycle (discover + resolve up to {max_resolve})")
            cycle_error = None
            try:
                cycle_log = run_issue_cycle(repo_root, pid, max_resolve=max_resolve, dataset=dataset)
                _log(f"{pid}: issue cycle done ({len(cycle_log)} log lines)")
            except Exception as exc:
                cycle_log = []
                cycle_error = str(exc)
                _log(f"{pid}: issue cycle error: {exc}")

            # Snapshot after cycle — capture titles of new/resolved issues
            new_issue_titles: list[str] = []
            resolved_issue_titles: list[str] = []
            try:
                _after = _list_issues(repo_root, pid, dataset)
                open_after = sum(1 for i in _after if i.get("status") in ("open", "in_progress"))
                total_after = len(_after)
                issues_discovered = max(0, total_after - total_before)
                issues_resolved = max(0, open_before - open_after)
                # Collect titles of truly new issues
                new_issue_titles = [
                    i.get("title", i["id"])
                    for i in _after
                    if i["id"] not in _before_ids
                ][:6]
                # Collect titles of issues that were open before but are now resolved
                resolved_issue_titles = [
                    i.get("title", i["id"])
                    for i in _after
                    if i["id"] in _before_open_ids and i.get("status") == "resolved"
                ][:6]
            except Exception:
                open_after = total_after = issues_discovered = issues_resolved = 0

            # Full meeting: multi-round discussion → synthesis → documented notes
            room_id = None
            notes_path = None
            meeting_participants: list[str] = []
            try:
                personas = get_personas_for_problem(pid)
                if personas:
                    meeting_participants = ["coordinator"] + [p["id"] for p in personas[:3]]
                    topic = f"Push-forward {today} — {pid} proof review"
                    goal = (
                        f"Review today's issue discovery and resolution results for {pid}. "
                        "Agree on the highest-priority remaining gaps and produce a concrete action plan."
                    )
                    room = create_room(repo_root, pid, topic=topic, goal=goal,
                                      participants=meeting_participants)
                    room_id = room["id"]
                    _log(f"{pid}: created meeting room {room_id} ({len(meeting_participants)} participants, "
                         f"{n_meeting_rounds} rounds planned)")

                    # Multi-round interleaved discussion
                    run_round_offline(
                        repo_root, pid, room_id,
                        f"{job_id}-{pid}-meet",
                        n_rounds=n_meeting_rounds,
                    )
                    _log(f"{pid}: {n_meeting_rounds} discussion rounds complete")

                    # Synthesis: coordinator reads transcript and produces an action plan
                    action_plan_summary = ""
                    try:
                        for _ in run_synthesis(repo_root, pid, room_id):
                            pass
                        _log(f"{pid}: action plan synthesized")
                        # Pull the summary from the synthesised room plan
                        try:
                            from .meet import get_room as _get_room
                            _room = _get_room(repo_root, pid, room_id)
                            if _room:
                                plan = _room.get("plan") or {}
                                action_plan_summary = plan.get("summary", "")
                                if not action_plan_summary:
                                    steps = plan.get("steps", [])
                                    if steps:
                                        action_plan_summary = steps[0].get("title", "")
                        except Exception:
                            pass
                    except Exception as exc:
                        _log(f"{pid}: synthesis error: {exc}")

                    # Persist meeting notes to disk
                    try:
                        notes_path = _save_meeting_notes(repo_root, pid, room_id)
                        if notes_path:
                            _log(f"{pid}: meeting notes saved → {notes_path.name}")
                    except Exception as exc:
                        _log(f"{pid}: notes save error: {exc}")

                    # If the meeting produced no real discussion and no plan
                    # (e.g. Vertex was quota-blocked), drop the empty shell so
                    # it never appears as a "null meeting" in the UI.
                    try:
                        from .meet_pdf import room_is_substantive
                        from .meet import get_room as _gr, delete_room as _dr
                        _chk = _gr(repo_root, pid, room_id)
                        if not room_is_substantive(_chk):
                            _dr(repo_root, pid, room_id)
                            room_id = None
                            _log(f"{pid}: meeting produced no content — empty room removed")
                    except Exception as exc:
                        _log(f"{pid}: empty-room cleanup error: {exc}")

            except Exception as exc:
                _log(f"{pid}: meeting error: {exc}")

            # Update all four documents for this problem (progress, timeline, strategies)
            try:
                from .rich_documents import update_question_document
                update_question_document(repo_root, pid)
                _log(f"{pid}: documents updated (progress, timeline, strategies)")
            except Exception as de:
                log.warning("[push-forward %s] document update for %s failed: %s", job_id[:8], pid, de)

            # Re-evaluate the (possibly improved) proof and record this
            # push-forward's score so the Evaluation chapter can track the
            # per-push-forward trajectory. Both steps are best-effort.
            try:
                from .proof_eval import evaluate_proof
                evaluate_proof(repo_root, pid, dataset=dataset, force=True)
            except Exception as ee:
                log.warning("[push-forward %s] proof re-eval for %s failed: %s", job_id[:8], pid, ee)
            try:
                rec = append_push_forward_score(repo_root, pid, dataset=dataset,
                                                room_id=room_id, job_id=job_id, date=today)
                _log(f"{pid}: push-forward #{rec['round']} score "
                     f"{rec.get('total')}/{rec.get('max')} recorded")
            except Exception as se:
                log.warning("[push-forward %s] score record for %s failed: %s", job_id[:8], pid, se)

            with _LOCK:
                job = _JOBS.get(job_id, {})
                job["done"] = job.get("done", 0) + 1
                job["results"].append({
                    "problem": pid,
                    "issues_open_before": open_before,
                    "issues_open_after": open_after,
                    "issues_discovered": issues_discovered,
                    "issues_resolved": issues_resolved,
                    "new_issue_titles": new_issue_titles,
                    "resolved_issue_titles": resolved_issue_titles,
                    "room_id": room_id,
                    "meeting_participants": meeting_participants,
                    "action_plan_summary": action_plan_summary,
                    "notes_path": str(notes_path) if notes_path else None,
                    "error": cycle_error,
                })

        # Persist run record
        state = load_state(repo_root)
        state["last_run_date"] = today
        with _LOCK:
            saved_results = list(_JOBS.get(job_id, {}).get("results", []))
        state.setdefault("runs", []).append({
            "date": today,
            "job_id": job_id,
            "dataset": dataset,
            "problems": active,
            "n_meeting_rounds": n_meeting_rounds,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "results": saved_results,
        })
        # Keep at most 30 run records to avoid unbounded growth
        state["runs"] = state["runs"][-30:]
        _save_state(repo_root, state)

        # Snapshot metrics into data/push_forward_metrics.json
        try:
            snap = snapshot_metrics(repo_root, job_id, active, dataset=dataset)
            append_metrics_snapshot(repo_root, snap)
            _log(f"metrics snapshot round {snap['round']} saved")
        except Exception as me:
            log.warning("metrics snapshot failed: %s", me)

        # Update cross-problem discussion index
        try:
            from .rich_documents import update_discussion_index
            update_discussion_index(repo_root)
            _log("discussion index updated")
        except Exception as di:
            log.warning("discussion index update failed: %s", di)

        # Update system-level literature survey
        try:
            from .literature import discover_system_literature
            _log("updating system literature survey…")
            added = 0
            for event in discover_system_literature(repo_root):
                if event.get("type") == "done":
                    added = event.get("added", 0)
            _log(f"system literature updated ({added} papers added/refreshed)")
        except Exception as le:
            log.warning("system literature update failed: %s", le)

        with _LOCK:
            job = _JOBS.get(job_id, {})
            if job:
                job.update({
                    "status": "done",
                    "current": None,
                    "done_at": datetime.now(timezone.utc).isoformat(),
                })
        _log("push-forward complete")

    except Exception as exc:
        log.exception("push-forward job %s failed", job_id)
        with _LOCK:
            job = _JOBS.get(job_id, {})
            if job:
                job.update({"status": "error", "error": str(exc), "current": None})
