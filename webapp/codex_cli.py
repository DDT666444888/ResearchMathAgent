"""Codex CLI provider authenticated with the user's ChatGPT subscription.

This is deliberately a CLI integration, rather than an undocumented ChatGPT
web-session integration: ``codex login`` owns the browser OAuth flow and
``codex exec`` owns the stored credential.  No ChatGPT cookies or passwords are
read by RMA.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator

from rma import config as _cfg

from .agent import AgentConfig, AgentEvent
from .claude_code import _seed_workspace
from .runs import RunHandle

_WALL_CLOCK_SECONDS = 1800

_SYSTEM = """You are the Research Math Agent. Work on exactly one research-level
mathematics problem. Read problem.tex and preamble.tex in the working directory,
reason rigorously, and write your final self-contained proof to solution.tex.
Stay inside the working directory and do not read unrelated files. Report only
what you have established."""


def codex_available() -> str | None:
    """Return the configured Codex CLI path, if executable."""
    override = os.environ.get("RMA_CODEX_BIN", "").strip()
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    return shutil.which("codex")


def run_codex_agent(cfg: AgentConfig, handle: RunHandle | None = None) -> Iterator[AgentEvent]:
    """Run the officially authenticated ``codex exec`` subscription workflow."""
    binary = codex_available()
    if not binary:
        yield AgentEvent("error", {"message": "Codex CLI is unavailable. Install it, then run `codex login` and choose ChatGPT sign-in."})
        yield AgentEvent("done", {"reason": "error"})
        return

    repo_root = cfg.repo_root or Path(__file__).resolve().parents[1]
    workspace = cfg.workspace or _seed_workspace(cfg, repo_root)
    if workspace is None:
        yield AgentEvent("error", {"message": f"Problem '{cfg.problem_id}' not found."})
        yield AgentEvent("done", {"reason": "error"})
        return
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    last_message = workspace / ".codex-last-message.txt"
    prompt = (cfg.initial_message or "").strip() or (
        f"{cfg.system_prompt or _SYSTEM}\n\nSolve problem.tex (benchmark id {cfg.problem_id}). "
        "Write the complete final LaTeX document to solution.tex. Your final message "
        "must briefly state what was established and confirm that solution.tex was written."
    )
    cmd = [
        binary, "exec", prompt, "--cd", str(workspace), "--sandbox", "workspace-write",
        "--skip-git-repo-check", "--ephemeral", "--json",
        "--output-last-message", str(last_message),
    ]
    if cfg.model:
        cmd += ["--model", cfg.model]

    yield AgentEvent("status", {"state": "running", "model": cfg.model or "Codex default",
                                "provider": "codex", "workspace": str(workspace)})
    if handle is not None and handle.cancelled:
        yield AgentEvent("done", {"reason": "stopped"})
        return
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(workspace), text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
    except OSError as exc:
        yield AgentEvent("error", {"message": f"Failed to start Codex CLI: {exc}"})
        yield AgentEvent("done", {"reason": "error"})
        return
    if handle is not None:
        handle.attach_proc(proc)
    stderr_chunks: list[str] = []
    stderr_drain = threading.Thread(target=_drain, args=(proc.stderr, stderr_chunks), daemon=True)
    stderr_drain.start()
    stream: queue.Queue[str | None] = queue.Queue()
    stdout_drain = threading.Thread(target=_queue_lines, args=(proc.stdout, stream), daemon=True)
    stdout_drain.start()

    deadline = time.time() + (cfg.max_wall_seconds or _WALL_CLOCK_SECONDS)
    final_message = ""
    cli_error = ""
    cancelled = False
    timed_out = False
    while True:
        if handle is not None and handle.cancelled:
            handle.kill_proc()
            cancelled = True
            break
        if time.time() > deadline:
            if handle is not None:
                handle.kill_proc()
            else:
                proc.terminate()
            timed_out = True
            break
        try:
            line = stream.get(timeout=0.2)
        except queue.Empty:
            if proc.poll() is not None:
                break
            continue
        if line is None:
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for translated in _translate_event(event):
            if translated.type == "text_delta":
                final_message += translated.data.get("text", "")
            elif translated.type == "error":
                cli_error = translated.data.get("message", "")
            yield translated

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if handle is not None:
            handle.force_kill_proc()
        else:
            proc.kill()
        proc.wait(timeout=10)
    stdout_drain.join(timeout=1)
    stderr_drain.join(timeout=1)
    for stream_handle in (proc.stdout, proc.stderr):
        if stream_handle is not None:
            try:
                stream_handle.close()
            except OSError:
                pass
    if cancelled:
        yield AgentEvent("done", {"reason": "stopped"})
        return
    if timed_out:
        yield AgentEvent("error", {"message": "Codex run exceeded the time limit; stopped."})
        yield AgentEvent("done", {"reason": "timeout"})
        return
    stderr = "".join(stderr_chunks).strip()
    if proc.returncode:
        yield AgentEvent("error", {"message": (cli_error or stderr or "codex exec failed")[-1500:]})
        yield AgentEvent("done", {"reason": "error"})
        return

    final_text = ""
    if last_message.is_file():
        final_text = last_message.read_text(encoding="utf-8", errors="replace").strip()
    if not final_text:
        final_text = final_message.strip()
    solution = workspace / "solution.tex"
    if not solution.is_file():
        latex = _latex_document(final_text)
        if latex:
            solution.write_text(latex, encoding="utf-8")
            yield AgentEvent("status", {"state": "finalizing", "provider": "codex",
                                        "model": cfg.model or "Codex default",
                                        "message": "Recovered final LaTeX from Codex output."})
    if solution.is_file():
        text = solution.read_text(encoding="utf-8", errors="replace")
        artifact = {"name": "solution.tex", "content": text, "compile_pdf": True}
        if len(text) > 60_000:
            artifact["oversized_chars"] = len(text)
        yield AgentEvent("artifact", artifact)
    yield AgentEvent("done", {"reason": "end_turn"})


def _drain(stream, sink: list[str]) -> None:
    if stream is None:
        return
    try:
        for line in stream:
            sink.append(line)
    except OSError:
        pass


def _queue_lines(stream, sink: queue.Queue[str | None]) -> None:
    if stream is None:
        sink.put(None)
        return
    try:
        for line in stream:
            sink.put(line)
    finally:
        sink.put(None)


def _translate_event(event: dict) -> Iterator[AgentEvent]:
    """Translate Codex ``exec --json`` JSONL to the shared web UI events."""
    event_type = event.get("type", "")
    if event_type == "error":
        yield AgentEvent("error", {"message": str(event.get("message") or event.get("error") or "Codex CLI error")})
        return
    if event_type == "turn.completed":
        usage = event.get("usage") or {}
        yield AgentEvent("usage", {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_input_tokens": usage.get("cached_input_tokens", 0),
            "num_turns": 1,
        })
        return
    if event_type == "turn.failed":
        yield AgentEvent("error", {"message": str(event.get("error") or "Codex turn failed")})
        return
    if not event_type.startswith("item."):
        return

    item = event.get("item") or {}
    item_type = item.get("type", "event")
    item_id = item.get("id", "")
    if item_type == "agent_message" and event_type == "item.completed":
        text = item.get("text", "")
        if text:
            yield AgentEvent("text_delta", {"text": text})
        return
    if item_type == "reasoning":
        yield AgentEvent("status", {"state": "thinking", "provider": "codex",
                                    "message": "Codex is reasoning…"})
        return

    label = {
        "command_execution": "Shell",
        "web_search": "Web search",
        "mcp_tool_call": "MCP tool",
        "file_change": "File change",
        "plan": "Plan",
    }.get(item_type, item_type.replace("_", " ").title())
    if event_type == "item.started":
        detail = item.get("command") or item.get("query") or item.get("path") or item.get("text") or item
        yield AgentEvent("tool_use", {"id": item_id, "name": label, "input": detail})
    elif event_type == "item.completed":
        detail = (item.get("aggregated_output") or item.get("output") or item.get("result")
                  or item.get("text") or item.get("command") or "completed")
        yield AgentEvent("tool_result", {"id": item_id, "name": label,
                                          "output": str(detail), "is_error": False})


def _latex_document(text: str) -> str | None:
    """Accept only an actual complete LaTeX document as a safe artifact fallback."""
    start = text.find("\\documentclass")
    if start < 0 or "\\begin{document}" not in text or "\\end{document}" not in text:
        return None
    return text[start:].strip() + "\n"
