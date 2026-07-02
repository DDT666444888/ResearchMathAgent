"""Ablation switches for context-report sections.

A single place to declare *which* parts of a per-problem context report get
emitted, so any part can be turned off for ablation studies without touching the
report-building code.  Every flag defaults to ``True`` — the all-on config
reproduces the current report byte-for-byte, so existing pipelines are
unaffected until a flag is explicitly flipped.

Three ways to select a config (later overrides earlier):

    1. Nothing            → everything on (current behaviour).
    2. Env ``RMA_REPORT_SECTIONS`` → applies to every build in the process,
       e.g.  ``RMA_REPORT_SECTIONS="concepts=0,meetings=off"``.
    3. Explicit ``sections=`` argument to a build/compile call.

Spec grammar (comma/space separated tokens), case-insensitive:

    concepts=0   concepts=off   concepts=false   concepts=no   → OFF
    concepts=1   concepts=on    concepts        (bare name)     → ON
    no_concepts  no-concepts  !concepts  -concepts              → OFF

Unknown names are ignored, so a typo can never silently drop a real section.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, fields, replace

_ENV_VAR = "RMA_REPORT_SECTIONS"


@dataclass(frozen=True)
class ReportSections:
    """One boolean per ablatable section of a context report."""

    # ── executive-summary blocks (front matter, before the table of contents) ──
    evaluation: bool = True            # proof-eval score table + "Evaluation" chapter
    research_status: bool = True       # "Research Status" block + open-issue teaser
    push_forward_history: bool = True  # push-forward score-history table(s)

    # ── chapters ──────────────────────────────────────────────────────────────
    problem_statement: bool = True     # Chapter: Problem Statement
    best_proof: bool = True            # Chapter: Best Proof (full proof body inline)
    concepts: bool = True              # Chapter: Key Concepts (core + background)
    meetings: bool = True              # Chapter: Meetings (plans + transcripts)
    open_issues: bool = True           # Chapter: Open Issues
    resolved_issues: bool = True       # Chapter: Resolved Issues
    insights: bool = True              # Chapter: Insights & lessons
    human_comparison: bool = True      # Eval subsection + Chapter: Comparison to Human Solution
                                       # (first_proof_2 only; needs a human ref solution)

    # ── markdown-report extras (system/UI markdown path only) ─────────────────
    candidate_answer: bool = True      # "Candidate Answer" + "Core Approach"
    strategy: bool = True              # "Strategy & Difficulty" + attempt history

    # ── helpers ───────────────────────────────────────────────────────────────
    def is_default(self) -> bool:
        """True when every section is on (identical to legacy behaviour)."""
        return all(getattr(self, f.name) for f in fields(self))

    def disabled(self) -> list[str]:
        """Names of the sections currently turned off, sorted."""
        return sorted(f.name for f in fields(self) if not getattr(self, f.name))

    def signature(self) -> str:
        """Short deterministic tag of the OFF set — ``""`` when all-on.

        Used to give ablated PDFs distinct filenames / cache keys so a variant
        never overwrites the canonical (all-on) report.
        """
        off = self.disabled()
        return hashlib.md5(",".join(off).encode()).hexdigest()[:6] if off else ""


ALL_ON = ReportSections()

_VALID = {f.name for f in fields(ReportSections)}
_FALSY = {"0", "off", "false", "no", "n", ""}


def parse_spec(spec: str) -> dict[str, bool]:
    """Turn a spec string into a ``{name: bool}`` override dict (see module doc)."""
    out: dict[str, bool] = {}
    for tok in re.split(r"[,\s]+", (spec or "").strip()):
        if not tok:
            continue
        if "=" in tok:
            key, val = tok.split("=", 1)
            on = val.strip().lower() not in _FALSY
        else:
            key = re.sub(r"^(!|-{1,2}|no[_-])", "", tok)  # !x / -x / --x / no_x / no-x
            on = key == tok  # a prefix was stripped ⇒ this is a negation
        key = key.strip()
        if key in _VALID:
            out[key] = on
    return out


def resolve_sections(overrides=None) -> ReportSections:
    """Resolve the effective config: all-on → env ``RMA_REPORT_SECTIONS`` →
    explicit ``overrides`` (a ``ReportSections``, a dict, or a spec string).
    Later sources win; a passed ``ReportSections`` is returned as-is.
    """
    if isinstance(overrides, ReportSections):
        return overrides

    cfg = ALL_ON
    env = os.environ.get(_ENV_VAR, "")
    if env:
        cfg = replace(cfg, **parse_spec(env))

    if overrides:
        if isinstance(overrides, str):
            patch = parse_spec(overrides)
        else:
            patch = {k: bool(v) for k, v in dict(overrides).items() if k in _VALID}
        cfg = replace(cfg, **patch)
    return cfg
