"""Bounded, deterministic, API-free proof reduction with kernel-gated promotion.

Candidates are syntactic proposals, not trusted transformations. Callers supply
an objective (higher is better); compatibility must be tested on every supplied
repository. No imports, statements, source files, or external ledgers are edited.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from .backend import atomic_json
from .lean import extract_body, declaration

@dataclass(frozen=True)
class Edit:
    label: str
    body: str


def candidates(body: str):
    """Conservative proposal enumeration; the Lean kernel decides validity."""
    seen = {body}
    # An empty simp list can remain after argument deletion in a tactic chain.
    # Removing this stage is still only a proposal (simp may do definitional work).
    for match in re.finditer(r'(?m)^[ \t]*simp(?:\s+only)?\s*\[\s*\]\s*<;>[ \t]*\n', body):
        proposed = body[:match.start()] + body[match.end():]
        if proposed not in seen:
            seen.add(proposed); yield Edit('drop empty simp stage', proposed)
    # Only simple identifier arguments: never split nested terms on commas.
    pattern = re.compile(r'\b(?:simp|simp_all|simpa)(?:\s+only)?\s*\[([^\[\]\n]*)\]')
    for match in pattern.finditer(body):
        args = [a.strip() for a in match[1].split(',')]
        if not all(re.fullmatch(r'-?\s*[\w.\'!?]+', a) for a in args):
            continue
        for i, arg in enumerate(args):
            replacement = ', '.join(args[:i] + args[i+1:])
            proposed = body[:match.start(1)] + replacement + body[match.end(1):]
            if proposed not in seen:
                seen.add(proposed); yield Edit(f'drop simp argument {arg}', proposed)
    lines = body.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Avoid deleting branch delimiters, declarations or multiline tactic heads.
        if not re.match(r'^(?:classical|simp(?:_all)?|simp_rw|dsimp|rw|unfold|clear|try|exact|assumption|rfl|trivial|omega|aesop)\b', stripped):
            continue
        if stripped.endswith(('=>', 'by', '<;>', ',', ':=', 'try')):
            continue
        if any(stripped.count(a) != stripped.count(b) for a,b in [('(',')'),('[',']'),('{','}')]):
            continue
        proposed = ''.join(lines[:i] + lines[i+1:])
        if proposed not in seen:
            seen.add(proposed); yield Edit(f'drop line {i+1}: {stripped[:80]}', proposed)


class LocalSearch:
    def __init__(self, verifier, objective, *, max_attempts=30):
        if max_attempts < 0:
            raise ValueError('max_attempts must be nonnegative')
        self.verifier, self.objective, self.max_attempts = verifier, objective, max_attempts

    def run(self, entry, body: str, out: Path):
        """New run only. Saves every attempted body and measurement, even failures.

        Greedy restart after promotion can expose further deletions. The global
        attempt cap bounds search, and the verifier bounds each compiler call.
        An independent pass must match the exact proof digest before promotion.
        """
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        if (out/'result.json').exists():
            raise ValueError('Use a new output directory; existing run is retained')
        body = extract_body(body)
        best = self.verifier.verify(entry, body)
        if not best.valid or best.heartbeats is None:
            raise ValueError('Baseline lacks valid measured compilation')
        initial = best; attempts = []; seen = {body}; bestbody = body
        versions = len(self.verifier.repos[entry['source']])
        def checkpoint():
            result = {'name':entry['name'], 'proof':declaration(entry,bestbody),
                      'baseline':initial.summary(), 'measurement':best.summary(),
                      'attempts':attempts, 'improved':bestbody!=body, 'api_calls':0,
                      'configured_versions':versions, 'objective_gain':self.objective(best)-self.objective(initial)}
            atomic_json(out/'result.json', result)
            (out/'best.lean').write_text(result['proof'])
            return result
        checkpoint()
        while len(attempts) < self.max_attempts:
            promoted = False
            for edit in candidates(bestbody):
                if edit.body in seen: continue
                if len(attempts) >= self.max_attempts: break
                seen.add(edit.body); tag = f'{len(attempts)+1:04d}'
                (out/f'{tag}.lean').write_text(declaration(entry,edit.body))
                row = {'tag':tag,'edit':edit.label,'status':'checking'}; attempts.append(row)
                checkpoint()
                try:
                    candidate = extract_body(edit.body)
                    measured = self.verifier.verify(entry,candidate)
                    atomic_json(out/f'{tag}.measurement.json',measured.summary())
                    (out/f'{tag}.log').write_text(measured.log)
                    row['status'] = 'invalid'
                    if measured.valid and measured.heartbeats is not None:
                        row['gain'] = self.objective(measured)-self.objective(best)
                        row['status'] = 'not_better'
                        if row['gain'] > 1e-9:
                            row['status'] = 'independent_or_version_failed'
                            valid = True
                            expected = hashlib.sha256(declaration(entry,candidate).encode()).hexdigest()
                            for version in range(versions):
                                check = self.verifier.verify(entry,candidate,measure=False,version=version)
                                atomic_json(out/f'{tag}.v{version}.json',check.summary())
                                (out/f'{tag}.v{version}.log').write_text(check.log)
                                if not check.valid or check.proof_sha256 != expected:
                                    valid = False; break
                            if valid:
                                bestbody,best = candidate,measured
                                row['status'] = 'promoted'; promoted = True
                except ValueError as exc:
                    row.update(status='rejected',error=str(exc))
                checkpoint()
                if promoted: break
            if not promoted: break
        return checkpoint()
