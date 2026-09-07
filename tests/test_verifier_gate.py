"""The structural verifier gate must be reachable by a real proof.

Two of the four required_patterns used to demand the literal strings
"Answer and construction" and "Boundary and consistency checks" — headings
emitted by _render_solution_document, the offline skeleton template. No
model-written proof contains them, so `passed` could never be True. Across a
full First Proof B2 run (10 problems x 5 rounds) the early-exit
`if verification["passed"]: break` never fired once.
"""
from __future__ import annotations

import unittest

from rma.solve import _collect_verification_issues

_PARSED = {"title": "Problem 6: Epsilon-light subsets", "problem_id": "q6"}

# A realistic model-written proof: no skeleton boilerplate anywhere.
REAL_PROOF = r"""\documentclass{article}
\usepackage{amsthm,amsmath}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\begin{document}
\begin{lemma}\label{lem:pave}
Every graph Laplacian admits an $r$-paving with $L_{S_i} \preceq (C/r) L$.
\end{lemma}
\begin{proof}
We give a self-contained derivation from the interlacing families method.
%s
\end{proof}
\begin{theorem}\label{thm:main}
There is a universal $c>0$ such that every graph has an $\varepsilon$-light
subset of size at least $c\varepsilon|V|$.
\end{theorem}
\begin{proof}
Apply \ref{lem:pave} with $r=\lceil 2C/\varepsilon\rceil$ and average over the
parts. We check the degenerate cases $\varepsilon=0$ and $\varepsilon=1$
separately, and the empty graph, the single-vertex graph and the disconnected
graph are handled by the same averaging bound.
%s
\end{proof}
\end{document}
""" % ("Filler establishing the estimate. " * 260, "Further detail. " * 260)


def _errors(issues, code=None):
    return [i for i in issues
            if i.get("severity") == "error" and (code is None or i["code"] == code)]


class GateReachableTest(unittest.TestCase):
    def test_real_proof_has_no_missing_required_section(self) -> None:
        issues = _collect_verification_issues(_PARSED, REAL_PROOF)
        self.assertEqual(_errors(issues, "missing_required_section"), [],
                         "a real proof still trips the structural gate")

    def test_boilerplate_headings_are_no_longer_required(self) -> None:
        """The exact strings that used to be mandatory must be absent here."""
        self.assertNotIn("Answer and construction", REAL_PROOF)
        self.assertNotIn("Boundary and consistency checks", REAL_PROOF)
        issues = _collect_verification_issues(_PARSED, REAL_PROOF)
        details = {i.get("detail") for i in issues}
        self.assertNotIn("Answer and construction", details)
        self.assertNotIn("Boundary and consistency checks", details)

    def test_title_mismatch_is_a_warning_not_an_error(self) -> None:
        """Requiring the parsed title verbatim blocked 9 of 10 real problems."""
        issues = _collect_verification_issues(_PARSED, REAL_PROOF)
        mismatch = [i for i in issues if i["code"] == "statement_mismatch"]
        self.assertEqual(len(mismatch), 1)
        self.assertEqual(mismatch[0]["severity"], "warning")

    def test_custom_newtheorem_envs_satisfy_the_claim_requirement(self) -> None:
        tex = (r"\newtheorem{qsixlemma}{Lemma}"
               r"\begin{qsixlemma}A.\end{qsixlemma}\begin{proof}p\end{proof}")
        issues = _collect_verification_issues({"title": ""}, tex)
        self.assertEqual(_errors(issues, "missing_required_section"), [])


class GateStillGuardsTest(unittest.TestCase):
    """Relaxing the gate must not make it toothless."""

    def test_document_with_no_claims_is_rejected(self) -> None:
        tex = r"\documentclass{article}\begin{document}Some prose.\end{document}"
        issues = _collect_verification_issues({"title": ""}, tex)
        self.assertTrue(_errors(issues, "missing_required_section"))

    def test_document_with_no_proof_is_rejected(self) -> None:
        tex = r"\begin{theorem}A claim with no argument.\end{theorem}"
        codes = {i["code"] for i in _errors(_collect_verification_issues({"title": ""}, tex))}
        self.assertIn("missing_required_section", codes)

    def test_offline_skeleton_still_fails(self) -> None:
        """The deterministic template is not a valid proof and must not pass."""
        from rma.solve import _render_solution_document

        parsed = {"problem_id": "q6", "title": "T", "author": "A",
                  "problem_type": "proof", "definitions": ["d"],
                  "boundary_cases": ["b"], "quantifier_summary": ["q"]}
        profile = {"area": "x", "candidate": "c", "construction": "k",
                   "strategy": "s", "verification": "v"}
        skeleton = _render_solution_document(parsed, profile, {"name": "s"}, 1)
        self.assertTrue(_errors(_collect_verification_issues(parsed, skeleton)),
                        "the offline skeleton must still be rejected")


if __name__ == "__main__":
    unittest.main()
