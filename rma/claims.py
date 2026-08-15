"""Claim-dependency graph — the paper's deterministic structural checker.

Algorithm 1's critic runs three analyses; this is the second (main.tex:368):

    "a deterministic structural checker parses the proof into claims and
     dependencies, measures the fraction of terminal claims supported by
     completed arguments, and flags finite or numerical claims that lack
     executable verification"

``rma.completeness.lemma_dag`` is called a DAG but has no edges at all — it is
a flat list of claim environments with a proved/unproved flag. Without edges
there is no notion of a terminal claim and no dependency-impact signal, so the
paper's severity-plus-dependency ordering has nothing to rank on. This module
supplies the missing structure.

Definitions used here
---------------------
edge A -> B          A's proof uses B ("A depends on B")
terminal claim       a claim reachable from a root that itself depends on no
                     other in-document claim: the leaves the argument rests on.
                     An unproved terminal claim is where a proof actually
                     bottoms out in nothing.
dependency impact    how many unresolved claims transitively depend on a claim;
                     the paper's tiebreak within a severity level.
external edge        a \\cite to a result not proved here — a black box. Tracked
                     separately because "cited" is not the same as "proved".

Known limitations (measured, not hidden)
----------------------------------------
* Proof attribution is positional: a claim counts as proved if a \\begin{proof}
  falls between it and the NEXT claim environment. This matches interleaved
  "claim then proof" documents (the realistic case — the repo's own proofs
  parse at 100%%), but a "state all lemmas, then prove all" layout would
  mis-attribute the proofs to the last claim. Such documents are rare here; a
  label-matched attribution would be the fix if they appear.
* Natural-language numeric references ("by Lemma 3") are resolved by per-kind
  appearance order, which can disagree with LaTeX's \\label numbering when the
  document renumbers. \\ref/\\cref targets (label-based) are exact; the recall
  gate on tests/fixtures/claims/ measures the combined edge recall (>= 0.80).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CLAIM_ENVS = ("theorem", "lemma", "claim", "proposition", "corollary")
_CLAIM_ENV_RE = "|".join(CLAIM_ENVS)

# Environments whose presence marks a claim as argued.
_PROOF_OPEN = re.compile(r"\\begin\{proof\}")

# \ref / \eqref / \cref / \Cref / \autoref, possibly comma-separated targets.
_REF_RE = re.compile(r"\\(?:eq|c|C|auto)?ref\{([^}]*)\}")
_CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")

# Natural-language references: "by Lemma 3", "Lemma 3 gives", "(Proposition 2)".
# Deliberately broad on the connective and narrow on the noun+number, because
# the failure mode that matters is missing a real edge, not adding a spurious
# one between two claims that genuinely appear together.
_NL_REF_RE = re.compile(
    r"\b(Theorem|Lemma|Claim|Proposition|Corollary)s?~?\s*\\?ref\{[^}]*\}|"
    r"\b(Theorem|Lemma|Claim|Proposition|Corollary)s?~?\s*(\d+)",
    re.IGNORECASE,
)

# Finite / numeric assertions that ought to be discharged by a runnable check.
_FINITE_RE = re.compile(
    r"(by (?:a |direct )?computation|numerical(?:ly)? (?:check|verif)\w*|"
    r"one (?:can )?(?:verif|check)\w*|it is (?:easy|straightforward) to (?:check|verify)|"
    r"exhaustive(?:ly)? (?:check|search|verif\w*)|finite (?:check|verification)|"
    r"a (?:short|quick) (?:computation|calculation)|direct calculation)",
    re.IGNORECASE)
_CODE_EVIDENCE_RE = re.compile(
    r"\\begin\{(?:verbatim|lstlisting|minted|algorithm)\}|sympy|numpy|"
    r"verified (?:by|via) (?:code|computation)|computational appendix|"
    r"the following (?:script|code)", re.IGNORECASE)


@dataclass
class Claim:
    """One node: a theorem/lemma/claim/proposition/corollary and its proof."""

    id: str
    kind: str
    index: int                      # 1-based, per kind, in order of appearance
    title: str
    statement: str
    proved: bool
    label: str | None = None
    depends_on: list[str] = field(default_factory=list)
    cites: list[str] = field(default_factory=list)
    finite_claims: list[str] = field(default_factory=list)
    finite_checked: bool = False
    start: int = 0

    @property
    def display(self) -> str:
        return f"{self.kind.capitalize()} {self.index}"

    def as_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "index": self.index,
            "title": self.title, "proved": self.proved, "label": self.label,
            "depends_on": list(self.depends_on), "cites": list(self.cites),
            "finite_claims": list(self.finite_claims),
            "finite_checked": self.finite_checked,
        }


class ClaimGraph:
    """The parsed dependency structure of one proof document."""

    def __init__(self, claims: list[Claim]) -> None:
        self.claims = claims
        self._by_id = {c.id: c for c in claims}

    # ── basics ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.claims)

    def node(self, claim_id: str) -> Claim | None:
        return self._by_id.get(claim_id)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [(c.id, dep) for c in self.claims for dep in c.depends_on]

    def dependents_of(self, claim_id: str) -> set[str]:
        """Every claim that transitively depends on `claim_id`."""
        reverse: dict[str, list[str]] = {c.id: [] for c in self.claims}
        for c in self.claims:
            for dep in c.depends_on:
                if dep in reverse:
                    reverse[dep].append(c.id)
        seen: set[str] = set()
        stack = list(reverse.get(claim_id, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(reverse.get(cur, []))
        seen.discard(claim_id)
        return seen

    def unresolved_downstream(self, claim_id: str) -> int:
        """Dependency impact: unproved claims that rest on this one.

        The paper's tiebreak within a severity level — a gap under three
        unfinished lemmas is worth more than one under none.
        """
        return sum(1 for cid in self.dependents_of(claim_id)
                   if not self._by_id[cid].proved)

    def roots(self) -> list[Claim]:
        """Claims nothing else depends on — the tops of the dependency chains."""
        depended_on = {dep for c in self.claims for dep in c.depends_on}
        return [c for c in self.claims if c.id not in depended_on]

    def main_claims(self) -> list[Claim]:
        """The document's headline results.

        Distinct from :meth:`roots`: every chain has a root, including an
        orphan lemma nobody uses, so "is a root" cannot separate "invalidates
        the main conclusion" (P0) from "blocks a required lemma" (P1). A
        theorem-kind root is the main result; failing that, the root with the
        largest dependency subtree — the one the most work feeds into.
        """
        roots = self.roots()
        if not roots:
            return []
        theorems = [c for c in roots if c.kind == "theorem"]
        if theorems:
            return theorems
        best = max(roots, key=lambda c: len(self._descendants(c.id)))
        return [best]

    def _descendants(self, claim_id: str) -> set[str]:
        """Everything this claim transitively depends on."""
        seen: set[str] = set()
        stack = list(self._in_document_deps(self._by_id[claim_id])) if claim_id in self._by_id else []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            node = self._by_id.get(cur)
            if node:
                stack.extend(self._in_document_deps(node))
        return seen

    def supports_main_result(self, claim_id: str) -> bool:
        """True when a headline result rests on this claim (or is this claim)."""
        mains = {c.id for c in self.main_claims()}
        if claim_id in mains:
            return True
        return bool(self.dependents_of(claim_id) & mains)

    def terminal_claims(self) -> list[Claim]:
        """Leaves: claims reachable from a root that depend on nothing else.

        These are where the argument bottoms out. A document whose leaves are
        all proved is self-contained; one with an unproved leaf is not, however
        polished the layers above it look.
        """
        reachable = self._reachable_from_roots()
        return [c for c in self.claims
                if c.id in reachable and not self._in_document_deps(c)]

    def _in_document_deps(self, claim: Claim) -> list[str]:
        return [d for d in claim.depends_on if d in self._by_id]

    def _reachable_from_roots(self) -> set[str]:
        seen: set[str] = set()
        stack = [c.id for c in self.roots()]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            node = self._by_id.get(cur)
            if node:
                stack.extend(self._in_document_deps(node))
        return seen or {c.id for c in self.claims}

    def proved_terminal_fraction(self) -> float:
        terminals = self.terminal_claims()
        if not terminals:
            return 0.0
        return round(sum(1 for c in terminals if c.proved) / len(terminals), 3)

    def proved_fraction(self) -> float:
        """Proved fraction over ALL claims — the weaker metric lemma_dag used."""
        if not self.claims:
            return 0.0
        return round(sum(1 for c in self.claims if c.proved) / len(self.claims), 3)

    def cycles(self) -> list[list[str]]:
        """Dependency cycles — circular reasoning, always correctness-critical."""
        found: list[list[str]] = []
        colour: dict[str, int] = {}
        path: list[str] = []

        def visit(node_id: str) -> None:
            colour[node_id] = 1
            path.append(node_id)
            for dep in self._in_document_deps(self._by_id[node_id]):
                if colour.get(dep, 0) == 0:
                    visit(dep)
                elif colour.get(dep) == 1:
                    cycle = path[path.index(dep):]
                    if cycle and sorted(cycle) not in [sorted(c) for c in found]:
                        found.append(list(cycle))
            path.pop()
            colour[node_id] = 2

        for claim in self.claims:
            if colour.get(claim.id, 0) == 0:
                visit(claim.id)
        return found

    def unchecked_finite_claims(self) -> list[tuple[Claim, str]]:
        """Per-claim finite/numeric assertions with no runnable check attached.

        Anchored to the claim, unlike the document-global single issue this
        replaces: three unchecked computations are three obligations.
        """
        out = []
        for claim in self.claims:
            if claim.finite_checked:
                continue
            for phrase in claim.finite_claims:
                out.append((claim, phrase))
        return out

    def summary(self) -> dict:
        return {
            "nodes": len(self.claims),
            "edges": len(self.edges),
            "proved": sum(1 for c in self.claims if c.proved),
            "proved_fraction": self.proved_fraction(),
            "terminal_nodes": len(self.terminal_claims()),
            "proved_terminal_fraction": self.proved_terminal_fraction(),
            "unproved_titles": [c.display for c in self.claims if not c.proved],
            "cycles": self.cycles(),
            "external_citations": sorted({k for c in self.claims for k in c.cites}),
            "unchecked_finite": [
                {"claim": c.id, "phrase": p} for c, p in self.unchecked_finite_claims()
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# parsing
# ─────────────────────────────────────────────────────────────────────────────
def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


# \newtheorem{qsixlemma}{Lemma}      \newtheorem*{note}{Note}
# \newtheorem{lem}[thm]{Lemma}
_NEWTHEOREM_RE = re.compile(
    r"\\newtheorem\*?\{([^}]+)\}(?:\[[^\]]*\])?\{([^}]+)\}")


def declared_claim_envs(tex: str) -> dict[str, str]:
    """Map custom environment names to the claim kind they render as.

    Real proofs in this repo declare `\\newtheorem{qsixlemma}{Lemma}` so that
    per-problem proofs can be concatenated into a master document without
    counter clashes. A checker that only knows the literal names
    theorem/lemma/claim/... therefore sees ZERO claims in every real proof —
    which is exactly what rma.completeness.lemma_dag did.
    """
    mapping: dict[str, str] = {}
    for env, display in _NEWTHEOREM_RE.findall(tex or ""):
        env = env.strip()
        display_l = display.strip().lower()
        for kind in CLAIM_ENVS:
            if kind in display_l:
                mapping[env] = kind
                break
    return mapping


def _claim_env_pattern(tex: str) -> tuple[str, dict[str, str]]:
    """Regex alternation of every claim-bearing env name, plus env -> kind."""
    env_to_kind = {k: k for k in CLAIM_ENVS}
    env_to_kind.update(declared_claim_envs(tex))
    # Longest first so `qsixlemma` is not partially matched by `lemma`.
    names = sorted(env_to_kind, key=len, reverse=True)
    return "|".join(re.escape(n) for n in names), env_to_kind


def parse_claims(tex: str) -> ClaimGraph:
    """Parse a LaTeX proof document into a claim-dependency graph."""
    text = _strip_comments(tex or "")
    pattern, env_to_kind = _claim_env_pattern(text)
    env_re = re.compile(
        rf"\\begin\{{({pattern})\}}(\[[^\]]*\])?(.*?)\\end\{{\1\}}",
        re.DOTALL | re.IGNORECASE)
    matches = list(env_re.finditer(text))

    claims: list[Claim] = []
    per_kind: dict[str, int] = {}
    label_map: dict[str, str] = {}
    number_map: dict[tuple[str, int], str] = {}

    for i, m in enumerate(matches):
        env_name = m.group(1)
        kind = env_to_kind.get(env_name, env_to_kind.get(env_name.lower(), env_name.lower()))
        per_kind[kind] = per_kind.get(kind, 0) + 1
        index = per_kind[kind]
        claim_id = f"{kind}-{index}"
        body = m.group(3) or ""
        optional = (m.group(2) or "").strip("[]")

        label = None
        lm = re.search(r"\\label\{([^}]*)\}", body)
        if lm:
            label = lm.group(1).strip()
            label_map[label] = claim_id
        number_map[(kind, index)] = claim_id

        statement = re.sub(r"\\label\{[^}]*\}", " ", body).strip()
        title = optional or re.sub(r"\s+", " ", re.sub(
            r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", statement))[:80].strip()

        claims.append(Claim(
            id=claim_id, kind=kind, index=index, title=title,
            statement=statement, proved=False, label=label, start=m.start(),
        ))

    # A claim counts as argued when a proof environment follows it before the
    # next claim environment begins.
    proof_positions = [pm.start() for pm in _PROOF_OPEN.finditer(text)]
    for i, m in enumerate(matches):
        nxt = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        claims[i].proved = any(m.end() <= p < nxt for p in proof_positions)
        # Dependencies come from the claim's PROOF, not from every line between
        # this claim and the next. Intervening prose routinely points forward
        # ("we will need this in Proposition 5"), and counting that as a
        # dependency manufactures cycles that are not in the mathematics.
        _attach_dependencies(claims[i], _proof_segment(text, m.end(), nxt),
                             label_map, number_map)

    return ClaimGraph(claims)


_PROOF_ENV_RE = re.compile(r"\\begin\{proof\}(.*?)\\end\{proof\}", re.DOTALL)


def _proof_segment(text: str, start: int, end: int) -> str:
    """The body of the proof attached to a claim, or "" when it has none.

    Concatenates every proof environment in the window: a claim occasionally
    carries a proof split across two blocks (a construction, then a
    verification).
    """
    window = text[start:end]
    bodies = _PROOF_ENV_RE.findall(window)
    return "\n".join(bodies)


def _attach_dependencies(claim: Claim, segment: str,
                         label_map: dict[str, str],
                         number_map: dict[tuple[str, int], str]) -> None:
    """Extract the edges, citations and finite-claim flags from a claim's proof."""
    deps: list[str] = []

    for raw in _REF_RE.findall(segment):
        for target in (t.strip() for t in raw.split(",")):
            resolved = label_map.get(target)
            if resolved and resolved != claim.id:
                deps.append(resolved)

    for m in _NL_REF_RE.finditer(segment):
        kind = (m.group(1) or m.group(2) or "").lower()
        number = m.group(3)
        if not kind or not number:
            continue  # the "\ref{...}" alternative is handled above
        resolved = number_map.get((kind, int(number)))
        if resolved and resolved != claim.id:
            deps.append(resolved)

    seen: set[str] = set()
    claim.depends_on = [d for d in deps if not (d in seen or seen.add(d))]

    cites: list[str] = []
    for raw in _CITE_RE.findall(segment):
        cites.extend(k.strip() for k in raw.split(",") if k.strip())
    claim.cites = sorted(set(cites))

    claim.finite_claims = sorted({m.group(0) for m in _FINITE_RE.finditer(segment)})
    claim.finite_checked = bool(_CODE_EVIDENCE_RE.search(segment))


# ─────────────────────────────────────────────────────────────────────────────
# recall evaluation against hand-labelled fixtures
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_fixtures(directory) -> dict:
    """Edge-extraction recall/precision against `*.expected.json` ground truth.

    Extraction from informal LaTeX is inherently partial, so the honest thing
    is to measure it rather than assume it. A low number means the
    dependency-impact ordering degrades toward severity-only, which the paper
    itself scores at 5.0 vs 6.0.
    """
    import json
    from pathlib import Path

    directory = Path(directory)
    fixtures = sorted(directory.glob("*.tex"))
    total_gold = total_found = total_predicted = 0
    per_file = []

    for tex_path in fixtures:
        expected_path = tex_path.with_suffix(".expected.json")
        if not expected_path.is_file():
            continue
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        graph = parse_claims(tex_path.read_text(encoding="utf-8"))
        gold = {tuple(e) for e in expected.get("edges", [])}
        predicted = {(a, b) for a, b in graph.edges}
        hit = gold & predicted
        total_gold += len(gold)
        total_found += len(hit)
        total_predicted += len(predicted)
        per_file.append({
            "fixture": tex_path.name,
            "gold": len(gold), "predicted": len(predicted), "matched": len(hit),
            "missed": sorted(gold - predicted), "spurious": sorted(predicted - gold),
        })

    recall = round(total_found / total_gold, 3) if total_gold else 0.0
    precision = round(total_found / total_predicted, 3) if total_predicted else 0.0
    return {"fixtures": len(per_file), "recall": recall, "precision": precision,
            "gold_edges": total_gold, "per_file": per_file}


def run_claims(args) -> int:
    """`rma claims` — dump a proof's graph, or evaluate extraction recall."""
    import json
    from pathlib import Path

    if getattr(args, "eval_dir", None):
        report = evaluate_fixtures(args.eval_dir)
        threshold = float(getattr(args, "min_recall", 0.80))
        print(f"claim-edge extraction over {report['fixtures']} fixtures")
        for row in report["per_file"]:
            print(f"  {row['fixture']:<28} gold={row['gold']:<3} "
                  f"matched={row['matched']:<3} spurious={len(row['spurious'])}")
            for miss in row["missed"]:
                print(f"      MISSED {miss[0]} -> {miss[1]}")
        print(f"edge recall: {report['recall']}")
        print(f"edge precision: {report['precision']}")
        if report["recall"] < threshold:
            print(f"FAIL: recall {report['recall']} below gate {threshold}")
            return 1
        print(f"PASS: recall gate {threshold} met")
        return 0

    path = Path(getattr(args, "tex", ""))
    if not path.is_file():
        print(f"RMA claims\nFAIL: no such file: {path}")
        return 1
    graph = parse_claims(path.read_text(encoding="utf-8", errors="replace"))
    if getattr(args, "json", False):
        print(json.dumps({"summary": graph.summary(),
                          "claims": [c.as_dict() for c in graph.claims]}, indent=2))
        return 0
    s = graph.summary()
    print("RMA claims")
    print(f"nodes: {s['nodes']}  edges: {s['edges']}  proved: {s['proved']}")
    print(f"terminal nodes: {s['terminal_nodes']}  "
          f"proved-terminal fraction: {s['proved_terminal_fraction']}")
    for claim in graph.claims:
        mark = "proved" if claim.proved else "UNPROVED"
        deps = ", ".join(claim.depends_on) or "-"
        print(f"  {claim.id:<16} {mark:<9} depends_on: {deps}")
    if s["cycles"]:
        print(f"CIRCULAR: {s['cycles']}")
    return 0
