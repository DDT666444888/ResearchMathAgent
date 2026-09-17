"""The Arena's public scoring rule, computed locally.

    problem score = mean( length reduction %, heartbeat reduction %, zero-shot compatibility % )
    run score     = mean of problem scores over ALL benchmark problems

A problem that is not submitted, does not compile, changes the statement or
trips the forbidden-pattern filter scores 0 on every axis and still counts in
the mean. Reductions are not clamped: a proof that elaborates more slowly than
the reference has a negative heartbeat axis.

Heartbeats measured locally with the Arena harness match the server's counts
for an unchanged proof and toolchain. The Arena's length tokenizer runs on its
evaluation worker and is not published; `arena_tokens` is a Lean-aware lexer
fitted to 41 official (proof, length) pairs from this benchmark: 21 exact, mean
absolute error 1.3 tokens, worst 11. On a several-hundred-token proof that is
well under half a point on the length axis, so candidate selection does not
need the server.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from .lean import strip_comments

_SUBSCRIPTS = '₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₔₕₖₗₘₙₚₛₜᵢᵣᵤᵥⱼ'
# Identifiers (optionally dot-prefixed, with dotted/primed/subscripted parts),
# numerals, bracket-like single characters, and maximal runs of other symbols.
_LEXER = re.compile(
    r"\.?[^\W\d][\w.'!?" + _SUBSCRIPTS + r"\-={⁻]*"
    r"|\d+(?:\.\d+)?"
    r"|[()\[\]{}⟨⟩,+|⋈]"
    r"|[^\w\s()\[\]{}⟨⟩,+|⋈]+")


def arena_tokens(body: str) -> int:
    """Estimated Arena length of a proof body (the text after `:=`)."""
    try:
        text = strip_comments(body)
    except ValueError:
        text = re.sub(r'--[^\n]*', '', body)
    return len(_LEXER.findall(text))


def listed_versions(entry: dict) -> list[str]:
    return [v for pair in entry.get('version_info', []) for v in pair]


def _count(text: str) -> int:
    return int(str(text).replace(',', ''))


def parse_official(row: dict) -> dict:
    """One server-result row: '277 of 407', '510 of 1,383', '4/4', '65.02%'."""
    length, ref_length = map(_count, row['length'].split(' of '))
    heartbeats, ref_heartbeats = map(_count, row['heartbeats'].split(' of '))
    passed, listed = map(int, row['zero_shot'].split('/'))
    return {'length': length, 'ref_length': ref_length, 'heartbeats': heartbeats,
            'ref_heartbeats': ref_heartbeats, 'zero_passed': passed, 'zero_listed': listed,
            'score': float(str(row['score']).rstrip('%'))}


@dataclass
class Axes:
    length_pct: float
    heartbeat_pct: float
    zero_shot_pct: float

    @property
    def score(self) -> float:
        # Always the three-axis mean: an unknown axis is an error, never a smaller mean.
        return (self.length_pct + self.heartbeat_pct + self.zero_shot_pct) / 3

    def as_dict(self) -> dict:
        return {'length_reduction_pct': self.length_pct, 'heartbeat_reduction_pct': self.heartbeat_pct,
                'zero_shot_pct': self.zero_shot_pct, 'score': self.score}


def problem_score(*, tokens: int, heartbeats: int | None, ref_tokens: int,
                  ref_heartbeats: int | None, passed: int, listed: int) -> Axes:
    """The Arena's per-problem score for a proof that compiled on `passed` of `listed` versions.

    A missing heartbeat count or reference makes the score undefined; it is never
    read as zero heartbeats (a perfect axis) or dropped to a two-axis mean."""
    if ref_tokens <= 0 or listed <= 0:
        raise ValueError('Reference length and listed versions must be positive')
    if heartbeats is None or not ref_heartbeats or ref_heartbeats <= 0:
        raise ValueError('Heartbeat count or reference unknown: problem score undefined')
    return Axes(100 * (1 - tokens / ref_tokens), 100 * (1 - heartbeats / ref_heartbeats), 100 * passed / listed)


def run_score(problem_scores: dict, names) -> float:
    """Mean over every benchmark problem; anything missing scores zero."""
    names = list(names)
    return sum(problem_scores.get(n, 0.) for n in names) / len(names)


def objective(*, axes: Axes, tokens: int, heartbeats: int | None, ref_tokens: int,
              ref_heartbeats: int | None, versions: list[str]) -> str:
    """The scoring rule and this problem's live numbers, for the solver's prompt."""
    lines = ['Objective (the public Arena rule; L from a lexer fitted to the Arena tokenizer, H measured):',
             '  problem score = mean(length reduction %, heartbeat reduction %, zero-shot compatibility %)',
             f'  length reduction = 1 - L/{ref_tokens} (L = tokens after `:=`, comments ignored)']
    if ref_heartbeats:
        lines.append(f'  heartbeat reduction = 1 - H/{ref_heartbeats} (H = #count_heartbeats of this proof; '
                     'negative when slower than the reference)')
    lines += [f'  zero-shot = share of listed Lean versions {", ".join(versions)} on which the proof compiles unchanged',
              f'Current proof: L={tokens}, H={heartbeats}, score {axes.score:.2f} '
              f'(length {axes.length_pct:.2f}, heartbeats {axes.heartbeat_pct:.2f}, zero-shot {axes.zero_shot_pct:.0f}).',
              f'Exchange rate: 1 token = {100/(3*ref_tokens):.3f} points'
              + (f'; 100 heartbeats = {10000/(3*ref_heartbeats):.3f} points.' if ref_heartbeats else '.'),
              'A shorter proof that elaborates more slowly can LOSE score: check both axes.',
              f'Failing one of the {len(versions)} listed versions costs {100/(3*len(versions)):.2f} points, '
              'more than most edits gain: avoid tactics, lemma names or syntax that differ across them.']
    return '\n'.join(lines)
