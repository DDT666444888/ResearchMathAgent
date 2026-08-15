"""T3.1-T3.4 — the claim-dependency graph.

Fixtures in tests/fixtures/claims/ are hand-written with hand-labelled edges;
none is taken from outputs/, final_solutions/ or baselines/.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from rma.claims import (
    declared_claim_envs,
    evaluate_fixtures,
    parse_claims,
)

FIXTURES = Path(__file__).parent / "fixtures" / "claims"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class NodeTest(unittest.TestCase):
    def test_nodes_parsed_with_kind_and_index(self) -> None:
        g = parse_claims(_fixture("01_ref_based.tex"))
        self.assertEqual(len(g), 3)
        self.assertEqual([c.id for c in g.claims], ["lemma-1", "lemma-2", "theorem-1"])
        self.assertEqual(g.node("lemma-2").index, 2)

    def test_proved_flag_follows_proof_environment(self) -> None:
        g = parse_claims(_fixture("04_unproved_leaf.tex"))
        self.assertFalse(g.node("lemma-1").proved)
        self.assertTrue(g.node("lemma-2").proved)

    def test_labels_are_captured(self) -> None:
        g = parse_claims(_fixture("01_ref_based.tex"))
        self.assertEqual(g.node("theorem-1").label, "thm:main")

    def test_custom_newtheorem_envs_are_recognised(self) -> None:
        """Real proofs declare \\newtheorem{qsixlemma}{Lemma} so per-problem
        proofs concatenate without counter clashes. Matching only the literal
        env names found ZERO claims in every such document."""
        tex = r"""
        \newtheorem{qsixlemma}{Lemma}
        \newtheorem{qsixproposition}{Proposition}
        \begin{qsixlemma}\label{l1}A statement.\end{qsixlemma}
        \begin{proof}Argument.\end{proof}
        \begin{qsixproposition}\label{p1}Another.\end{qsixproposition}
        \begin{proof}By \ref{l1} it follows.\end{proof}
        """
        self.assertEqual(declared_claim_envs(tex),
                         {"qsixlemma": "lemma", "qsixproposition": "proposition"})
        g = parse_claims(tex)
        self.assertEqual(len(g), 2)
        self.assertEqual({c.id for c in g.claims}, {"lemma-1", "proposition-1"})
        self.assertEqual(g.node("proposition-1").depends_on, ["lemma-1"])

    def test_starred_newtheorem_and_shared_counter(self) -> None:
        tex = r"""
        \newtheorem*{mylem}{Lemma}
        \newtheorem{mythm}[section]{Theorem}
        \begin{mylem}L.\end{mylem}\begin{proof}p\end{proof}
        \begin{mythm}T.\end{mythm}\begin{proof}p\end{proof}
        """
        g = parse_claims(tex)
        self.assertEqual({c.kind for c in g.claims}, {"lemma", "theorem"})


class EdgeTest(unittest.TestCase):
    def test_ref_edges(self) -> None:
        g = parse_claims(_fixture("01_ref_based.tex"))
        self.assertEqual(sorted(g.node("theorem-1").depends_on), ["lemma-1", "lemma-2"])

    def test_natural_language_edges(self) -> None:
        g = parse_claims(_fixture("02_natural_language.tex"))
        self.assertEqual(g.node("lemma-2").depends_on, ["lemma-1"])
        self.assertEqual(g.node("proposition-1").depends_on, ["lemma-2"])

    def test_cref_edges(self) -> None:
        g = parse_claims(_fixture("03_cref_mixed.tex"))
        self.assertEqual(sorted(g.node("corollary-1").depends_on), ["lemma-1", "lemma-2"])

    def test_citations_tracked_separately_from_claim_edges(self) -> None:
        g = parse_claims(_fixture("01_ref_based.tex"))
        self.assertEqual(g.node("lemma-1").cites, ["mss2015"])
        self.assertEqual(g.node("lemma-1").depends_on, [])

    def test_no_self_edges(self) -> None:
        tex = r"""\begin{lemma}\label{a}A.\end{lemma}
        \begin{proof}As in Lemma 1 and \ref{a}, done.\end{proof}"""
        self.assertEqual(parse_claims(tex).node("lemma-1").depends_on, [])

    def test_dependencies_come_from_the_proof_not_surrounding_prose(self) -> None:
        """Forward-looking prose between claims must not create an edge.

        On a real 113k-char proof, counting all intervening text produced 49
        edges and 16 spurious cycles; scoping to the proof body gave 22 edges
        and none.
        """
        tex = r"""
        \begin{lemma}\label{a}A.\end{lemma}
        \begin{proof}Direct.\end{proof}

        We will use this again in \ref{b} below, after a digression.

        \begin{lemma}\label{b}B.\end{lemma}
        \begin{proof}By \ref{a}.\end{proof}
        """
        g = parse_claims(tex)
        self.assertEqual(g.node("lemma-1").depends_on, [],
                         "forward prose reference leaked into the graph")
        self.assertEqual(g.node("lemma-2").depends_on, ["lemma-1"])
        self.assertEqual(g.cycles(), [])

    def test_unproved_claim_has_no_dependencies(self) -> None:
        g = parse_claims(_fixture("04_unproved_leaf.tex"))
        self.assertEqual(g.node("lemma-1").depends_on, [])


class TerminalTest(unittest.TestCase):
    def test_terminal_set_excludes_intermediate_claims(self) -> None:
        g = parse_claims(_fixture("01_ref_based.tex"))
        self.assertEqual(sorted(c.id for c in g.terminal_claims()),
                         ["lemma-1", "lemma-2"])

    def test_proved_terminal_fraction_differs_from_all_claim_fraction(self) -> None:
        g = parse_claims(_fixture("04_unproved_leaf.tex"))
        self.assertEqual(g.proved_terminal_fraction(), 0.5)   # lemma-1 unproved
        self.assertNotEqual(g.proved_terminal_fraction(), g.proved_fraction())

    def test_fraction_is_one_for_complete_fixture(self) -> None:
        self.assertEqual(parse_claims(_fixture("01_ref_based.tex")).proved_terminal_fraction(), 1.0)

    def test_roots_are_the_headline_results(self) -> None:
        g = parse_claims(_fixture("01_ref_based.tex"))
        self.assertEqual([c.id for c in g.roots()], ["theorem-1"])

    def test_empty_document(self) -> None:
        g = parse_claims("")
        self.assertEqual(len(g), 0)
        self.assertEqual(g.proved_terminal_fraction(), 0.0)


class ImpactAndCycleTest(unittest.TestCase):
    def test_impact_counts_match_fixture(self) -> None:
        tex = r"""
        \begin{lemma}\label{base}Base.\end{lemma}

        \begin{lemma}\label{mid}Mid.\end{lemma}
        \begin{proof}By \ref{base}.\end{proof}

        \begin{theorem}\label{top}Top.\end{theorem}
        \begin{proof}By \ref{mid}.\end{proof}
        """
        g = parse_claims(tex)
        # base is unproved; mid and top are proved, so nothing UNRESOLVED
        # depends on base yet.
        self.assertEqual(g.unresolved_downstream("base" if g.node("base") else "lemma-1"), 0)
        self.assertEqual(sorted(g.dependents_of("lemma-1")), ["lemma-2", "theorem-1"])

    def test_impact_counts_unproved_dependents(self) -> None:
        tex = r"""
        \begin{lemma}\label{base}Base.\end{lemma}
        \begin{proof}Done.\end{proof}

        \begin{lemma}\label{mid}Mid.\end{lemma}

        \begin{theorem}\label{top}Top.\end{theorem}
        """
        g = parse_claims(tex)
        g.node("lemma-2").depends_on = ["lemma-1"]
        g.node("theorem-1").depends_on = ["lemma-2"]
        self.assertEqual(g.unresolved_downstream("lemma-1"), 2)

    def test_cycle_detected_and_flagged(self) -> None:
        g = parse_claims(_fixture("05_cycle.tex"))
        cycles = g.cycles()
        self.assertEqual(len(cycles), 1)
        self.assertEqual(sorted(cycles[0]), ["lemma-1", "lemma-2"])

    def test_acyclic_fixtures_report_no_cycles(self) -> None:
        for name in ("01_ref_based.tex", "02_natural_language.tex",
                     "03_cref_mixed.tex", "04_unproved_leaf.tex"):
            self.assertEqual(parse_claims(_fixture(name)).cycles(), [], name)


class FiniteClaimTest(unittest.TestCase):
    def test_three_finite_claims_yield_three_anchored_issues(self) -> None:
        tex = r"""
        \begin{lemma}\label{a}A.\end{lemma}
        \begin{proof}One can verify this directly.\end{proof}
        \begin{lemma}\label{b}B.\end{lemma}
        \begin{proof}By a short computation.\end{proof}
        \begin{lemma}\label{c}C.\end{lemma}
        \begin{proof}An exhaustive check settles it.\end{proof}
        """
        flagged = parse_claims(tex).unchecked_finite_claims()
        self.assertEqual(len(flagged), 3, "finite claims must be anchored per claim")
        self.assertEqual({c.id for c, _ in flagged}, {"lemma-1", "lemma-2", "lemma-3"})

    def test_claim_with_verbatim_evidence_not_flagged(self) -> None:
        g = parse_claims(_fixture("06_finite_claims.tex"))
        flagged = {c.id for c, _ in g.unchecked_finite_claims()}
        self.assertIn("lemma-1", flagged)
        self.assertNotIn("lemma-2", flagged, "a code-backed check was still flagged")


class RecallGateTest(unittest.TestCase):
    def test_edge_recall_meets_the_gate(self) -> None:
        report = evaluate_fixtures(FIXTURES)
        self.assertGreaterEqual(report["fixtures"], 6)
        self.assertGreaterEqual(report["recall"], 0.80,
                                f"edge recall below gate: {report}")

    def test_every_fixture_has_ground_truth(self) -> None:
        for tex in FIXTURES.glob("*.tex"):
            expected = tex.with_suffix(".expected.json")
            self.assertTrue(expected.is_file(), f"{tex.name} has no ground truth")
            json.loads(expected.read_text(encoding="utf-8"))


class LemmaDagCompatTest(unittest.TestCase):
    """completeness.lemma_dag must keep its historical keys and stop
    reporting zero on real, \\newtheorem-declaring proofs."""

    def test_keys_preserved(self) -> None:
        from rma.completeness import lemma_dag

        d = lemma_dag(_fixture("01_ref_based.tex"))
        for key in ("nodes", "proved", "unproved_titles", "completeness_fraction"):
            self.assertIn(key, d)
        self.assertEqual(d["nodes"], 3)

    def test_custom_envs_no_longer_report_zero(self) -> None:
        from rma.completeness import lemma_dag

        tex = r"""
        \newtheorem{qsixlemma}{Lemma}
        \begin{qsixlemma}A.\end{qsixlemma}\begin{proof}p\end{proof}
        """
        self.assertEqual(lemma_dag(tex)["nodes"], 1)

    def test_new_edge_derived_keys_present(self) -> None:
        from rma.completeness import lemma_dag

        d = lemma_dag(_fixture("01_ref_based.tex"))
        self.assertEqual(d["edges"], 2)
        self.assertEqual(d["terminal_nodes"], 2)
        self.assertEqual(d["proved_terminal_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
