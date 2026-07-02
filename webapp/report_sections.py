"""Report section toggles for context reports.

Restores the module referenced by ``context_report.py`` (``resolve_sections``)
that was missing from the repo; the interface is reconstructed from its usage
there: a ``ReportSections`` value exposes one boolean per report section, all
defaulting to on, so omitting ``sections`` reproduces the full report.

``resolve_sections(spec)`` accepts:
  * ``None``            — read ``RMA_REPORT_SECTIONS`` from the environment if
                          set (parsed as a spec string), else everything on;
  * ``ReportSections``  — returned as-is;
  * ``dict``            — per-section overrides applied on top of all-on,
                          e.g. ``{"meetings": False}``;
  * spec string         — comma-separated section names. Bare names select
                          exactly those sections (everything else off);
                          ``+name``/``-name`` tokens instead toggle on top of
                          all-on, e.g. ``"-meetings,-concepts"``. ``"all"``
                          means everything on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

ENV_VAR = "RMA_REPORT_SECTIONS"


@dataclass
class ReportSections:
    research_status: bool = True
    problem_statement: bool = True
    evaluation: bool = True
    push_forward_history: bool = True
    best_proof: bool = True
    concepts: bool = True
    meetings: bool = True
    open_issues: bool = True
    resolved_issues: bool = True
    insights: bool = True

    @classmethod
    def all_on(cls) -> "ReportSections":
        return cls()

    @classmethod
    def all_off(cls) -> "ReportSections":
        return cls(**{f.name: False for f in fields(cls)})


_SECTION_NAMES = tuple(f.name for f in fields(ReportSections))

# Friendly aliases → canonical field name(s).
_ALIASES: dict[str, tuple[str, ...]] = {
    "status": ("research_status",),
    "problem": ("problem_statement",),
    "statement": ("problem_statement",),
    "eval": ("evaluation",),
    "history": ("push_forward_history",),
    "pushforward": ("push_forward_history",),
    "pushforwards": ("push_forward_history",),
    "proof": ("best_proof",),
    "proofs": ("best_proof",),
    "issues": ("open_issues", "resolved_issues"),
    "insight": ("insights",),
}


def _canonical(name: str) -> tuple[str, ...]:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _SECTION_NAMES:
        return (key,)
    return _ALIASES.get(key, ())


def _from_spec_string(spec: str) -> ReportSections:
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if not tokens or any(t.lower() == "all" for t in tokens):
        return ReportSections.all_on()

    signed = [t for t in tokens if t[0] in "+-"]
    if signed and len(signed) == len(tokens):
        # Toggle mode: start from all-on, apply +/- adjustments.
        s = ReportSections.all_on()
        for tok in tokens:
            value = tok[0] == "+"
            for field_name in _canonical(tok[1:]):
                setattr(s, field_name, value)
        return s

    # Selection mode: only the listed sections are enabled.
    s = ReportSections.all_off()
    for tok in tokens:
        for field_name in _canonical(tok.lstrip("+")):
            setattr(s, field_name, True)
    return s


def resolve_sections(spec=None) -> ReportSections:
    """Normalize any accepted ``sections`` value into a ReportSections."""
    if isinstance(spec, ReportSections):
        return spec
    if spec is None:
        env = os.environ.get(ENV_VAR, "").strip()
        return _from_spec_string(env) if env else ReportSections.all_on()
    if isinstance(spec, dict):
        s = ReportSections.all_on()
        for name, value in spec.items():
            for field_name in _canonical(str(name)):
                setattr(s, field_name, bool(value))
        return s
    if isinstance(spec, str):
        return _from_spec_string(spec)
    raise TypeError(f"unsupported sections spec: {type(spec).__name__}")
