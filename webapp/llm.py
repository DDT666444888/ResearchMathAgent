"""One-shot LLM completions on the Claude Pro/Max subscription.

Replaces the former Vertex AI helpers. Everything routes through the local
``claude`` CLI (subscription billing, no API key, no Google Cloud), so there is
no per-token API spend and nothing can fan out onto a paid backend.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Paper backbone (main.tex:1251-1262); override with RMA_MODEL for one-offs.
# Was "claude-fable-5", which quietly put every one-shot judge call — the
# completeness gate, the gap enumerator, the rubric evaluator — on a different
# model than the one the paper reports.
from rma.config import DEFAULT_MODEL as _PAPER_MODEL

DEFAULT_MODEL = os.environ.get("RMA_MODEL", _PAPER_MODEL)


def complete(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 8192,  # noqa: ARG001 - accepted for call-site compatibility
    thinking_budget: int = 0,  # noqa: ARG001 - accepted for call-site compatibility
) -> str | None:
    """Run a single-turn completion on the Claude subscription via the `claude`
    CLI. Returns text, or None on failure.

    ``max_tokens`` / ``thinking_budget`` are accepted for backwards compatibility
    with the previous Vertex helper but are managed by the CLI itself.
    """
    # Benchmark hook: with RMA_VIA_SHIM set, every completion goes through the
    # local OpenAI-compatible shim instead of straight to the CLI. Same backend
    # model either way -- the point is that RMA's calls then land on the SAME
    # meter as the other systems being compared, so "how much compute did this
    # method use" is one number measured one way for everybody.
    import os as _os
    base = _os.environ.get("RMA_VIA_SHIM")
    if base:
        import json as _json
        import urllib.request as _u
        body = _json.dumps({"model": "rma", "messages": (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}])}).encode()
        req = _u.Request(base.rstrip("/") + "/chat/completions", data=body,
                         headers={"Content-Type": "application/json"})
        try:
            with _u.urlopen(req, timeout=1800) as r:
                return _json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as exc:          # a metering outage must not look like a model refusal
            logging.getLogger(__name__).warning("shim call failed: %r", exc)
            return None

    from .claude_code import complete_via_cli

    return complete_via_cli(prompt, system=system, model=model or DEFAULT_MODEL)


def estimate_cost_usd(*_args, **_kwargs) -> float:
    """The subscription has no marginal per-call cost, so estimated spend is 0.

    Kept as a no-op shim so historical token-log accounting keeps working.
    """
    return 0.0
