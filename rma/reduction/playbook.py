"""Cross-problem reduction memory for one run.

RMA's strategy memory (rma/memory.py) records what an attempt did and how it
went; a reduction run needs the same signal in a form the solver can act on.
Only *kernel-verified* promotions are recorded, and only at the level of tactic
vocabulary and measured deltas, never whole proofs: a lesson tells the next
problem which moves paid off, without pasting another theorem's proof into its
context, which would waste budget and invite irrelevant copying.
"""
from __future__ import annotations
from collections import Counter
import json
import re
from pathlib import Path
from webapp.locks import file_lock
from .lean import strip_comments

# Lean identifiers and tactic names: a non-digit word character, then word
# characters, dots and primes. Unicode names (greek, subscripts) are included.
IDENTIFIER = re.compile(r"[^\W\d][\w.']*")


def vocabulary(body: str) -> Counter:
    return Counter(IDENTIFIER.findall(strip_comments(body)))


def transformation(before: str, after: str, top: int = 6) -> dict:
    """What the promotion actually changed, as tactic-vocabulary deltas."""
    b, a = vocabulary(before), vocabulary(after)
    return {'dropped': [w for w, _ in (b - a).most_common(top)],
            'introduced': [w for w, _ in (a - b).most_common(top)]}


class Playbook:
    """Append-only JSONL, shared across the run's concurrent problem workers."""

    def __init__(self, path: Path, limit: int = 6):
        self.path, self.limit = Path(path), limit

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # A truncated lesson is skipped; memory is never load-bearing.
        return rows

    def record(self, pid: str, strategy: str, before: str, after: str, *,
               tokens_gain: float, heartbeat_gain: float, utility_gain: float) -> None:
        record = dict(transformation(before, after), id=pid, strategy=strategy,
                      tokens_gain=round(tokens_gain, 4), heartbeat_gain=round(heartbeat_gain, 4),
                      utility_gain=round(utility_gain, 6))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path.parent, self.path.name + '-playbook'):
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def lessons(self, exclude: str | None = None, limit: int | None = None) -> str:
        rows = [r for r in self.entries() if r.get('id') != exclude]
        rows.sort(key=lambda r: r.get('utility_gain', 0), reverse=True)
        lines = []
        for row in rows[:limit or self.limit]:
            lines.append(
                f"- [{row.get('id')}] {row.get('strategy','')} | length {row.get('tokens_gain',0):+.0%}, "
                f"heartbeats {row.get('heartbeat_gain',0):+.0%} | dropped: "
                f"{', '.join(row.get('dropped') or []) or 'none'} | introduced: "
                f"{', '.join(row.get('introduced') or []) or 'none'}")
        if not lines:
            return ''
        return ('Verified reduction moves from other problems in this run (vocabulary only, '
                'not proofs; apply only where they are sound here):\n' + '\n'.join(lines))
