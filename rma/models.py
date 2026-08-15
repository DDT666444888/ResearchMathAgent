from __future__ import annotations

import json
import os
import re
import select
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_EFFORT, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_KEYCHAIN_SERVICE = "rma_anthropic_api_key"


class ModelConfigurationError(RuntimeError):
    """Raised when a requested model backend is not configured."""


class ModelRequestError(RuntimeError):
    """Raised when a configured model backend rejects or fails a request."""


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider: str
    model: str


def should_use_anthropic(model_name: str, provider: str | None = None) -> bool:
    provider = _model_provider(provider)
    if provider == "anthropic":
        return True
    if provider in {"offline", "claude-code"}:
        return False
    return model_name.lower().startswith("claude-") and not should_use_claude_code(model_name, provider)


def should_use_claude_code(model_name: str, provider: str | None = None) -> bool:
    provider = _model_provider(provider)
    name = model_name.lower()
    return provider == "claude-code" or name in {"claude-code", "claude-code-fable", "claude-code-sonnet", "claude-code-opus", "claude-code-haiku"}


def _model_provider(provider: str | None) -> str:
    return (provider or os.environ.get("RMA_MODEL_PROVIDER", "auto")).lower()


def call_anthropic(
    *,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 8192,
    temperature: float = 0.2,
) -> ModelResponse:
    api_key = _load_anthropic_api_key()
    if not api_key:
        raise ModelConfigurationError(
            "Claude model requested, but no Anthropic API key was found. "
            "Set ANTHROPIC_API_KEY or store a key in the macOS Keychain service "
            f"`{ANTHROPIC_KEYCHAIN_SERVICE}` before running RMA."
        )

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Fable 5 / Opus 4.7+ / Sonnet 5 reject sampling params (HTTP 400); only
    # attach temperature for models that still accept it.
    if not any(k in model for k in ("fable", "opus-4-7", "opus-4-8", "sonnet-5")):
        payload["temperature"] = temperature
    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300, context=_ssl_context()) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ModelRequestError(f"Anthropic API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ModelRequestError(f"Anthropic API request failed: {exc.reason}") from exc

    data = json.loads(raw)
    parts = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    text = "\n".join(part for part in parts if part).strip()
    if not text:
        raise ModelRequestError("Anthropic API returned no text content.")
    return ModelResponse(text=text, provider="anthropic", model=model)


def call_json(
    *,
    model: str,
    system: str,
    prompt: str,
    provider: str | None = None,
    max_tokens: int = 4096,
    cwd: Path | None = None,
):
    """Ask the model for a structured artifact and return the parsed JSON.

    Returns a ``dict``/``list`` on success and ``None`` when no JSON could be
    obtained. Never routes through the LaTeX quality gate in
    ``call_claude_code`` — that gate exists to stop prose masquerading as a
    proof, and applying it to a JSON reply rejected every structured call.

    Every operation that returns structured data (issue lists, rubric scores,
    action plans) must use this rather than ``call_claude_code`` directly.
    """
    if should_use_claude_code(model, provider):
        with tempfile.TemporaryDirectory() as tmp:
            response = call_claude_code(
                model=model,
                system=system,
                prompt=prompt,
                cwd=Path(cwd) if cwd is not None else Path(tmp),
                expect="json",
            )
        return parse_json_response(response.text)
    if should_use_anthropic(model, provider):
        response = call_anthropic(
            model=model,
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return parse_json_response(response.text)
    return None


def parse_json_response(raw: str):
    """Decode the first complete JSON value in a model reply.

    Tolerates ```json fences and trailing prose. Returns None if there is no
    decodable value. Mirrors rma.completeness._parse_json_block so both the
    orchestrator and the completeness passes accept the same replies.
    """
    raw = (raw or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Decode the FIRST complete value starting at the earliest '{' or '[' so a
    # wrapper object is not mistaken for one of its inner arrays.
    decoder = json.JSONDecoder()
    for start in sorted(i for i in (raw.find("{"), raw.find("[")) if i >= 0):
        try:
            value, _ = decoder.raw_decode(raw[start:])
            return value
        except Exception:
            continue
    return None


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _load_anthropic_api_key() -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    return _load_anthropic_api_key_from_keychain()


def _load_anthropic_api_key_from_keychain() -> str | None:
    security_bin = shutil.which("security")
    if security_bin is None:
        return None

    service = os.environ.get("RMA_ANTHROPIC_KEYCHAIN_SERVICE", ANTHROPIC_KEYCHAIN_SERVICE)
    command = [security_bin, "find-generic-password", "-s", service, "-w"]
    account = os.environ.get("RMA_ANTHROPIC_KEYCHAIN_ACCOUNT") or os.environ.get("USER")
    if account:
        command[2:2] = ["-a", account]

    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10,
                                encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def call_claude_code(
    *,
    model: str,
    system: str,
    prompt: str,
    cwd: Path,
    # <=0 means NO time limit — deep-thinking runs are monitored externally
    # rather than killed (set RMA_CLAUDE_CODE_TIMEOUT to restore a ceiling).
    timeout: int = 0,
    partial_output_dir: Path | None = None,
    fallback_file: Path | None = None,
    effort: str | None = None,
    # "latex": the reply must be a proof document; narration-only output is
    # refused (the quality gate that keeps prose from masquerading as a proof).
    # "json":  the reply is a structured artifact (issue list, rubric scores,
    # action plan). Applying the LaTeX gate here rejected every such reply and
    # the caller's `except Exception` turned it into an empty result — which is
    # why the LM gap critic silently contributed nothing. See call_json below.
    expect: str = "latex",
) -> ModelResponse:
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        raise ModelConfigurationError(
            "Claude Code backend requested, but the `claude` command is not installed or not on PATH. "
            "Install Claude Code and log in before running RMA with --model-provider claude-code."
        )

    want_latex = expect == "latex"
    timeout = int(os.environ.get("RMA_CLAUDE_CODE_TIMEOUT", timeout))
    # Structured calls are single-shot judgements, not research turns.
    max_turns = int(os.environ.get("RMA_CLAUDE_CODE_MAX_TURNS", "12" if want_latex else "3"))
    # Headless permission model: tools on this allowlist are auto-approved,
    # everything else is auto-denied (no human present to answer prompts).
    # Literature search plus read-only inspection and a few safe commands —
    # deliberately NOT --dangerously-skip-permissions.
    allowed_tools = os.environ.get(
        "RMA_CLAUDE_CODE_ALLOWED_TOOLS",
        "WebSearch,WebFetch,Read,Glob,Grep,"
        "Bash(curl:*),Bash(latexmk:*),Bash(pdflatex:*),Bash(bibtex:*),"
        "Bash(ls:*),Bash(cat:*),Bash(head:*),Bash(tail:*),Bash(grep:*),Bash(wc:*)",
    )
    # Benchmark-fairness hardening: even with read access allowed, prior
    # solutions must stay unreadable (deny wins over allow).
    disallowed_tools = os.environ.get(
        "RMA_CLAUDE_CODE_DISALLOWED_TOOLS",
        "Read(**/output_solutions/**),Read(**/final_solutions/**),"
        "Read(**/baselines/**),Read(**/skill_solutions/**)",
    )
    # Paper: adaptive thinking at "high" reasoning effort (main.tex:1251-1262).
    # Interactive-session effort (often "max") also makes deep-thinking turns
    # exceed the 30-minute request budget, so high is both faithful and safer.
    effort = effort or os.environ.get("RMA_CLAUDE_CODE_EFFORT") or DEFAULT_EFFORT

    task_line = (
        "Generate the requested Research Math Agent proof artifact from the prompt on stdin."
        if want_latex
        else "Answer the Research Math Agent request on stdin. Reply with the requested "
             "structured data only — no prose, no Markdown fences."
    )
    command = [
        claude_bin,
        "-p",
        task_line,
        "--output-format",
        "stream-json",
        "--verbose",
        "--append-system-prompt",
        system,
        "--max-turns",
        str(max_turns),
        "--allowedTools",
        allowed_tools,
        "--disallowedTools",
        disallowed_tools,
        "--effort",
        effort,
        "--no-session-persistence",
    ]
    model_arg = _claude_code_model_arg(model)
    if model_arg is not None:
        command.extend(["--model", model_arg])

    # Inherit env; ensure CLAUDE_CODE_MAX_OUTPUT_TOKENS passes through (avoids silent 16K cutoff)
    env = os.environ.copy()
    # Force the user's OWN Claude subscription (the `claude login` OAuth credential).
    # Strip any API key/token in the environment so a run is billed to the
    # subscription, never to a developer's pay-per-token Anthropic API account.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    if "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in env:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(DEFAULT_MAX_OUTPUT_TOKENS)

    _partial_dir = partial_output_dir if partial_output_dir is not None else cwd
    _partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = _partial_dir / "partial_output.tex"

    # stderr goes to a spooled file, NOT a pipe: with --verbose the CLI logs
    # enough to fill a 16KB pipe that nobody drains, which blocks the child on
    # write() and deadlocks the whole call (observed as 40+ min of 0% CPU on
    # both sides while the 30-min deadline never fired because we were stuck
    # in readline()).
    stderr_spool = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_spool,
        text=True,
        # Never let a stray/partial byte in the model stream raise mid-read and
        # abort the solve; replace undecodable bytes instead.
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=env,
    )

    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except BrokenPipeError:
        pass

    accumulated: list[str] = []
    infinite = timeout <= 0
    deadline = None if infinite else time.monotonic() + timeout
    last_partial_write = 0.0

    # Unbypassable timeout: the in-loop deadline check can be skated past when
    # readline() blocks on a partial line (observed: worker alive at 35+ min
    # with a 30-min budget). A watchdog kills the child no matter where the
    # reader is stuck; the reader then sees EOF and unwinds normally.
    # With no time limit (timeout <= 0) neither mechanism is armed.
    watchdog_fired = threading.Event()

    def _watchdog() -> None:
        watchdog_fired.set()
        try:
            proc.kill()
        except OSError:
            pass

    watchdog: threading.Timer | None = None
    if not infinite:
        watchdog = threading.Timer(timeout + 5, _watchdog)
        watchdog.daemon = True
        watchdog.start()

    try:
        while True:
            if deadline is None:
                remaining = 60.0
            else:
                remaining = deadline - time.monotonic()
            if deadline is not None and remaining <= 0:
                proc.kill()
                proc.wait()
                text = "".join(accumulated).strip()
                if not text and fallback_file is not None and fallback_file.is_file():
                    text = fallback_file.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    # Same quality gate as the normal path: salvage a real LaTeX
                    # document if one made it out, refuse narration-only output.
                    if want_latex:
                        doc = _extract_latex_document(text)
                        if doc:
                            text = doc
                        elif not _looks_like_latex(text):
                            raise ModelRequestError(
                                "Claude Code timed out with narration-only output: "
                                + text[:300].replace("\n", " ")
                            )
                    try:
                        partial_path.write_text(text)
                        if want_latex:
                            _try_compile_latex(partial_path)
                    except OSError:
                        pass
                    return ModelResponse(text=text, provider="claude-code", model=model_arg or "claude-code")
                raise ModelRequestError("Claude Code request timed out and produced no output.")

            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 60.0))
            if not ready:
                continue

            line = proc.stdout.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            # Token-level streaming deltas (--verbose exposes these)
            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunk = delta.get("text", "")
                    if chunk:
                        accumulated.append(chunk)

            # Complete assistant turn
            elif event_type == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        chunk = block.get("text", "")
                        if chunk:
                            accumulated.append(chunk)
                # Always flush on turn completion
                text_so_far = "".join(accumulated)
                if text_so_far.strip():
                    try:
                        partial_path.write_text(text_so_far)
                        _try_compile_latex(partial_path)
                        last_partial_write = time.monotonic()
                    except OSError:
                        pass

            elif event_type == "result" and not accumulated:
                result_text = event.get("result", "")
                if result_text:
                    accumulated.append(result_text)

            # Periodic flush during token streaming (every 30 s)
            now = time.monotonic()
            if accumulated and now - last_partial_write >= 30:
                text_so_far = "".join(accumulated)
                if text_so_far.strip():
                    try:
                        partial_path.write_text(text_so_far)
                        _try_compile_latex(partial_path)
                        last_partial_write = now
                    except OSError:
                        pass

    finally:
        if watchdog is not None:
            watchdog.cancel()
        try:
            proc.stdout.close()
        except OSError:
            pass

    proc.wait()

    try:
        stderr_spool.seek(0)
        stderr_output = stderr_spool.read()
    except OSError:
        stderr_output = ""
    finally:
        stderr_spool.close()

    text = "".join(accumulated).strip()
    if not text and fallback_file is not None and fallback_file.is_file():
        text = fallback_file.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        if watchdog_fired.is_set():
            raise ModelRequestError(
                f"Claude Code request timed out after {timeout}s (watchdog kill) and produced no output."
            )
        if proc.returncode != 0:
            raise ModelRequestError(f"Claude Code returned exit code {proc.returncode}: {stderr_output.strip()[-4000:]}")
        # Process succeeded but streamed no text — model wrote output via file tools.
        return ModelResponse(text="", provider="claude-code", model=model_arg or "claude-code")

    # The stream concatenates EVERY assistant text block, so tool-chatter or
    # progress narration can precede (or entirely replace) the document. Keep
    # only the LaTeX document when one is present; refuse narration-only output
    # instead of letting it masquerade as a proof downstream.
    if want_latex:
        doc = _extract_latex_document(text)
        if doc:
            text = doc
        elif not _looks_like_latex(text):
            # The CLI is an agent: it often WRITES the document with file tools
            # and then narrates ("Writing the complete document now..."). The
            # artifact exists, just not in the reply. Look on disk before
            # discarding the whole turn — refusing here cost prob-05 four of
            # its five refinement rounds, with the finished proof sitting in
            # the output file the whole time.
            salvaged = _salvage_latex_document(fallback_file, partial_path)
            if salvaged:
                text = salvaged
            else:
                raise ModelRequestError(
                    "Claude Code produced no LaTeX document (narration-only output "
                    "and no document on disk): " + text[:300].replace("\n", " ")
                )

    for suffix in (".tex", ".aux", ".log"):
        try:
            partial_path.with_suffix(suffix).unlink(missing_ok=True)
        except OSError:
            pass

    return ModelResponse(text=text, provider="claude-code", model=model_arg or "claude-code")


def _salvage_latex_document(*candidates: Path | None) -> str | None:
    """Return the first real LaTeX document found among these files.

    Used when the model narrates instead of pasting the document. Only a
    genuine document counts — a file holding the narration itself, or a stub,
    is rejected, so this cannot smuggle prose through the quality gate.
    """
    for path in candidates:
        if path is None:
            continue
        try:
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not content:
            continue
        doc = _extract_latex_document(content)
        if doc:
            return doc
        if _looks_like_latex(content):
            return content
    return None


def _extract_latex_document(text: str) -> str | None:
    """Return the last complete \\documentclass … \\end{document} span, if any."""
    start = text.rfind("\\documentclass")
    if start == -1:
        return None
    end_marker = "\\end{document}"
    end = text.find(end_marker, start)
    if end == -1:
        return None
    return text[start : end + len(end_marker)].strip()


# A Markdown fence marker ON ITS OWN LINE (a real code fence), as opposed to a
# ```latex the model typed inside a sentence when describing the bug it fixed.
_FENCE_LINE_RE = re.compile(r"^[ \t]*```(?:latex|tex)?[ \t]*$", re.MULTILINE)
# Any triple-backtick run (used to scrub stray inline markers from the body).
_INLINE_FENCE_RE = re.compile(r"```(?:latex|tex)?")
# The first token at which real LaTeX content plausibly begins.
_LATEX_START_RE = re.compile(r"\\(documentclass|clearpage|section|chapter|"
                             r"begin\{document\}|begin\{)")


def clean_latex_reply(raw: str) -> str:
    """Strip a model's narration and Markdown fences from a LaTeX reply.

    A latex-returning operation (revise, the solver's full-rewrite fallback) is
    told to return "ONLY the complete LaTeX document", but models routinely
    prepend narration ("I'll execute the action plan…"), wrap the body in a
    ```latex fence, add a closing note, and — worse — sometimes emit a stray
    ```latex mid-document or describe one in prose with literal backticks.
    Stored verbatim, that chatter pollutes Pi and every downstream artifact (the
    report's Best-Proof chapter, the standalone proof PDF, the master PDF). This
    returns just the LaTeX.

    Strategy: find the fence markers that sit on their OWN line — the real code
    fences, not the ```latex a model types inside a sentence. Keep the text
    between the first and last such line (dropping leading/trailing narration),
    then scrub any remaining inline ```latex runs (a stray fence injected into
    the middle of the proof). Keeping *between the fence lines* — rather than
    "the fenced block" — preserves the whole document even when the model closed
    the fence prematurely. Then, if a full \\documentclass…\\end{document} span is
    present, return it; otherwise drop any remaining leading prose.
    Whitespace-only input and already-clean LaTeX pass through unchanged.
    """
    if not raw or not raw.strip():
        return raw or ""
    text = raw.strip()

    fence_lines = list(_FENCE_LINE_RE.finditer(text))
    if len(fence_lines) >= 2:
        text = text[fence_lines[0].end():fence_lines[-1].start()]
    elif len(fence_lines) == 1:
        fl = fence_lines[0]
        text = text[:fl.start()] + text[fl.end():]
    # Scrub any stray inline fence markers left in the body (valid LaTeX never
    # contains a triple backtick).
    text = _INLINE_FENCE_RE.sub("", text).replace("```", "").strip()

    doc = _extract_latex_document(text)
    if doc is not None:
        return doc

    m = _LATEX_START_RE.search(text)
    if m and m.start() > 0:
        # Only trim if what precedes the first token looks like prose (no
        # backslash command), so we never eat a legitimate leading macro.
        if "\\" not in text[:m.start()]:
            text = text[m.start():]
    return text.strip()


def _looks_like_latex(text: str) -> bool:
    """Heuristic: does this read as LaTeX mathematics rather than narration?"""
    if "\\documentclass" in text or "\\begin{" in text:
        return True
    return len(text) > 1000 and text.count("$") >= 4


def _try_compile_latex(tex_path: Path) -> None:
    """Render LaTeX to HTML (pandoc+MathJax) or PDF if a compiler is available."""
    text = tex_path.read_text(errors="replace")
    if r"\begin{document}" not in text or r"\end{document}" not in text:
        return
    # Try PDF compilers first
    tectonic = shutil.which("tectonic") or _find_tectonic()
    if tectonic is not None:
        try:
            subprocess.run(
                [tectonic, str(tex_path), "--outdir", str(tex_path.parent)],
                capture_output=True, timeout=120, cwd=tex_path.parent,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    for compiler in ("pdflatex", "xelatex", "lualatex", "latexmk"):
        binary = shutil.which(compiler)
        if binary is None:
            continue
        cmd = [binary, "-interaction=nonstopmode", "-output-directory", str(tex_path.parent), str(tex_path)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=90, cwd=tex_path.parent)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    # Fallback: pandoc → HTML with MathJax
    pandoc = shutil.which("pandoc") or _find_pandoc()
    if pandoc is None:
        return
    html_path = tex_path.with_suffix(".html")
    cmd = [pandoc, str(tex_path), "--from", "latex", "--to", "html",
           "--mathjax", "--standalone", "-o", str(html_path)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60, cwd=tex_path.parent)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _find_tectonic() -> str | None:
    for candidate in (
        "/projects/bhov/zzhao18/software/bin/tectonic",
        "/usr/local/bin/tectonic",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def _find_pandoc() -> str | None:
    for prefix in (
        "/sw/user/python/miniforge3-tensorflow-cpu/bin",
        "/sw/user/python/miniforge3-datascience/bin",
        "/sw/user/python/miniforge3-pytorch-2.5.0/bin",
    ):
        p = Path(prefix) / "pandoc"
        if p.is_file():
            return str(p)
    return None


def _claude_code_model_arg(model: str) -> str | None:
    name = model.lower()
    # Default: run everything on Claude Opus 4.8 explicitly (rather than
    # inheriting whatever the user's interactive `claude` default happens to be).
    # Override the default with RMA_CLAUDE_CODE_MODEL.
    if name in {"claude-code", "claude-code-default", "claude-code-opus48"}:
        return os.environ.get("RMA_CLAUDE_CODE_MODEL", DEFAULT_MODEL)
    if name == "claude-code-fable":
        return "claude-fable-5"
    if name == "claude-code-sonnet":
        return "sonnet"
    if name == "claude-code-opus":
        return "opus"
    if name == "claude-code-haiku":
        return "haiku"
    return model
