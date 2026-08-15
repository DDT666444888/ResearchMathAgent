"""Token counting and PrefixToBudget — fitting a candidate context into B.

Algorithm 1's Run unit builds a candidate context in a fixed priority order and
then compiles it (main.tex:400):

    C = Instructions(u) || q || CurrentProof(S) || LinkedRecords(S,q) || RecentOutputs(S,u)
    O = PrefixToBudget(C, B)

"It serializes these components in that order and removes the lowest-priority
records from the end until the resulting observation O fits within the budget B."

Two properties matter and neither holds of the ~60 hard-coded character slices
this replaces:

  * whole RECORDS are dropped, never a slice through the middle of one. Half a
    lemma statement is worse than no lemma statement, because the model cannot
    tell it is reading a fragment.
  * the drop order is defined and observable, so a run can report what the
    model was not shown.

Three modes exist so the paper's context ablation (dump 2.4 -> naive truncation
4.0 -> orchestrated 6.0, main.tex:920-925) is a configuration rather than a
separate code path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Approximate characters per token, used only when no tokenizer is installed.
_CHARS_PER_TOKEN = 4

_ENCODER = None
_ENCODER_TRIED = False


def _encoder():
    """cl100k_base via tiktoken, loaded once.

    It is an approximation of Claude's tokenizer, chosen because it is
    deterministic, free, and works offline — which matters because the default
    backend is the subscription CLI and there is no API key to call a hosted
    counter with.
    """
    global _ENCODER, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    if os.environ.get("RMA_DISABLE_TIKTOKEN"):
        _ENCODER = None
        return None
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    """Token count for `text`. Falls back to a character estimate."""
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // _CHARS_PER_TOKEN)
    return len(enc.encode(text, disallowed_special=()))


def tokenizer_name() -> str:
    return "cl100k_base" if _encoder() is not None else f"chars/{_CHARS_PER_TOKEN}"


@dataclass
class Chunk:
    """One droppable unit of context — normally one store record."""

    text: str
    record_id: str | None = None
    priority: int = 50
    droppable: bool = True

    def tokens(self) -> int:
        return count_tokens(self.text)


@dataclass
class Section:
    """One of the five ordered parts of the candidate context C."""

    name: str
    chunks: list[Chunk] = field(default_factory=list)

    def tokens(self) -> int:
        return sum(c.tokens() for c in self.chunks)


@dataclass
class Observation:
    """O — what the operation actually receives."""

    text: str
    tokens: int
    sections: list[str]
    included: list[str]
    dropped: list[str]
    mode: str
    budget: int

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    def as_dict(self) -> dict:
        return {
            "context_tokens": self.tokens,
            "sections": self.sections,
            "records_included": len(self.included),
            "records_dropped": len(self.dropped),
            "dropped_ids": self.dropped,
            "mode": self.mode,
            "budget": self.budget,
        }


def _render(sections: list[Section], keep: set[int]) -> tuple[str, list[str]]:
    """Serialize the surviving chunks, section by section, in order."""
    parts: list[str] = []
    included: list[str] = []
    index = 0
    for section in sections:
        body: list[str] = []
        for chunk in section.chunks:
            if index in keep:
                body.append(chunk.text)
                if chunk.record_id:
                    included.append(chunk.record_id)
            index += 1
        if body:
            parts.append(f"## {section.name}\n" + "\n\n".join(body))
    return "\n\n".join(parts), included


def mandatory_floor_tokens(sections: list[Section]) -> int:
    """Tokens that cannot be dropped: the non-droppable chunks plus the
    rendering overhead (section headers and separators).

    A caller should compare this against B before dispatching: if the floor
    already exceeds the budget, no amount of dropping will fit and the run
    should say so rather than silently sending an over-budget request.
    """
    flat = [c for s in sections for c in s.chunks]
    keep = {i for i, c in enumerate(flat) if not c.droppable}
    text, _ = _render(sections, keep)
    return count_tokens(text)


def prefix_to_budget(sections: list[Section], budget: int,
                     mode: str = "budget") -> Observation:
    """Compile C into an observation of at most `budget` tokens.

    mode:
      "budget"   PrefixToBudget — drop whole lowest-priority records from the
                 end until it fits. The paper's mechanism.
      "truncate" naive character truncation of the serialized context, which is
                 what the codebase did before: cuts mid-record.
      "dump"     no reduction at all; the whole state goes in.
    """
    flat: list[Chunk] = [c for s in sections for c in s.chunks]
    all_ids = [c.record_id for c in flat if c.record_id]
    section_names = [s.name for s in sections if s.chunks]

    if mode == "dump":
        text, included = _render(sections, set(range(len(flat))))
        return Observation(text, count_tokens(text), section_names, included, [],
                           "dump", budget)

    if mode == "truncate":
        text, included = _render(sections, set(range(len(flat))))
        max_chars = max(0, budget) * _CHARS_PER_TOKEN
        cut = text[:max_chars]
        # When the text is cut, we cannot cleanly attribute which records
        # survived (truncation slices mid-record on purpose), so report the
        # records whose full body still appears intact. Uncut -> all included.
        if len(cut) == len(text):
            kept = included
        else:
            kept = [c.record_id for c in flat
                    if c.record_id and c.text and c.text in cut]
        return Observation(cut, count_tokens(cut), section_names, kept, [],
                           "truncate", budget)

    if mode != "budget":
        raise ValueError(f"unknown context mode: {mode}")

    keep = set(range(len(flat)))
    # Drop candidates: lowest priority first; among equals, the one latest in
    # the serialization order goes first ("from the end"). Section order and
    # within-section order already encode priority, so for a well-formed C this
    # is exactly "remove from the end".
    candidates = sorted(
        (i for i, c in enumerate(flat) if c.droppable),
        key=lambda i: (flat[i].priority, -i),
    )
    dropped: list[str] = []

    text, included = _render(sections, keep)
    total = count_tokens(text)
    for i in candidates:
        if total <= budget:
            break
        keep.discard(i)
        if flat[i].record_id:
            dropped.append(flat[i].record_id)
        text, included = _render(sections, keep)
        total = count_tokens(text)

    # Note: if everything droppable is gone and the mandatory head (instructions
    # + query) still overflows, we return over budget rather than truncating it.
    # A sliced instruction block produces a malformed request; the caller sees
    # context_tokens > budget in the telemetry and can react.
    return Observation(text, total, _surviving_sections(sections, keep),
                       included, dropped, "budget", budget)


def _surviving_sections(sections: list[Section], keep: set[int]) -> list[str]:
    """Names of the sections that still contribute at least one chunk."""
    names: list[str] = []
    index = 0
    for section in sections:
        alive = False
        for _ in section.chunks:
            if index in keep:
                alive = True
            index += 1
        if alive:
            names.append(section.name)
    return names
