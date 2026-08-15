"""T3.5 — P0-P3 severity and the three ordering modes.

The paper reports these as a ladder (FIFO 3.8 -> severity 5.0 ->
severity+impact 6.0), so all three must be genuinely distinct orderings.
"""
from __future__ import annotations

import unittest

from rma.claims import parse_claims
from rma.ranking import (
    ORDER_MODES,
    Issue,
    assign_severity,
    compute_impact,
    issues_from_verifier,
    rank,
    select,
)

# base <- mid <- top ; base and mid unproved, top is the headline result.
_TEX = r"""
\begin{lemma}\label{base}Base estimate.\end{lemma}

\begin{lemma}\label{mid}Middle step.\end{lemma}

\begin{theorem}\label{top}Main result.\end{theorem}
\begin{proof}By \ref{mid}.\end{proof}
"""


def _graph():
    g = parse_claims(_TEX)
    # Wire the chain explicitly: mid's proof is absent, so no edge is parsed.
    g.node("lemma-2").depends_on = ["lemma-1"]
    return g


class SeverityTest(unittest.TestCase):
    def test_p0_assigned_when_main_theorem_depends_on_gap(self) -> None:
        g = _graph()
        issue = Issue(id="i1", code="unproved_lemma", message="base unproved",
                      claim_id="lemma-1")
        self.assertEqual(assign_severity(issue, g), "P0")

    def test_p0_for_the_headline_claim_itself(self) -> None:
        g = _graph()
        issue = Issue(id="i", code="logical_gap", message="main step wrong",
                      claim_id="theorem-1")
        self.assertEqual(assign_severity(issue, g), "P0")

    def test_p1_when_the_headline_result_does_not_depend_on_it(self) -> None:
        """A side branch: something rests on the claim, but the main theorem
        does not. Root-ness alone cannot express this — every chain has a root."""
        tex = r"""
        \begin{theorem}\label{main}Main result.\end{theorem}
        \begin{proof}Self-contained argument.\end{proof}

        \begin{lemma}\label{side}Side lemma.\end{lemma}

        \begin{lemma}\label{useside}Uses the side lemma.\end{lemma}
        \begin{proof}By \ref{side}.\end{proof}
        """
        g = parse_claims(tex)
        self.assertEqual([c.id for c in g.main_claims()], ["theorem-1"])
        issue = Issue(id="i", code="unproved_lemma", message="m", claim_id="lemma-1")
        self.assertEqual(assign_severity(issue, g), "P1")

    def test_p2_for_a_leaf_nothing_rests_on(self) -> None:
        g = _graph()
        # The defect TYPE caps severity: a citation gap is P2 by the paper's
        # definition wherever it sits in the graph.
        issue = Issue(id="i", code="missing_citation", message="cite needed",
                      claim_id="lemma-2")
        self.assertEqual(assign_severity(issue, g), "P2")

    def test_p3_for_notation_only(self) -> None:
        # A presentation-typed code, or a non-substantive code whose message is
        # about notation, is P3.
        self.assertEqual(assign_severity(Issue(id="a", code="notation",
                                               message="inconsistent symbol"), None), "P3")
        self.assertEqual(assign_severity(Issue(id="b", code="clarity",
                                               message="notation is unclear here"), None), "P3")

    def test_substantive_code_not_downgraded_by_message_wording(self) -> None:
        """The reviewer's bug: a main-theorem-breaking gap whose message merely
        contains 'wording'/'hides' must NOT become P3. A substantive code stays
        substantive regardless of how the message is phrased."""
        issue = Issue(id="x", code="logical_gap",
                      message="the wording of the induction step hides a gap that breaks the main theorem")
        self.assertNotEqual(assign_severity(issue, None), "P3")
        self.assertEqual(assign_severity(issue, None), "P1")   # unanchored substantive gap

    def test_circular_reasoning_is_p0(self) -> None:
        g = parse_claims(r"""
        \begin{lemma}\label{a}A.\end{lemma}\begin{proof}By \ref{b}.\end{proof}
        \begin{lemma}\label{b}B.\end{lemma}\begin{proof}By \ref{a}.\end{proof}
        """)
        issue = Issue(id="i", code="logical_gap", message="m", claim_id="lemma-1")
        self.assertEqual(assign_severity(issue, g), "P0")

    def test_unanchored_gap_defaults_to_p1_not_p3(self) -> None:
        """Under-rating a real gap is worse than over-rating a cosmetic one."""
        issue = Issue(id="i", code="unproved_lemma", message="something missing")
        self.assertEqual(assign_severity(issue, None), "P1")


class ImpactTest(unittest.TestCase):
    def test_impact_is_unresolved_downstream_count(self) -> None:
        g = _graph()
        self.assertEqual(compute_impact(Issue(id="i", code="c", message="m",
                                              claim_id="lemma-1"), g), 1)

    def test_impact_zero_without_a_graph_or_anchor(self) -> None:
        self.assertEqual(compute_impact(Issue(id="i", code="c", message="m"), None), 0)
        self.assertEqual(compute_impact(Issue(id="i", code="c", message="m",
                                              claim_id="nope"), _graph()), 0)


class OrderingTest(unittest.TestCase):
    def _issues(self) -> list[Issue]:
        return [
            Issue(id="first", code="notation", message="notation nit"),          # P3
            Issue(id="second", code="unproved_lemma", message="m", claim_id="lemma-1"),
            Issue(id="third", code="missing_citation", message="cite", claim_id="lemma-2"),
            Issue(id="fourth", code="unproved_lemma", message="m", claim_id="theorem-1"),
        ]

    def test_fifo_preserves_creation_order(self) -> None:
        order = [i.id for i in rank(self._issues(), _graph(), mode="fifo")]
        self.assertEqual(order, ["first", "second", "third", "fourth"])

    def test_severity_mode_ignores_impact(self) -> None:
        ranked = rank(self._issues(), _graph(), mode="severity")
        self.assertEqual(ranked[0].severity, "P0")
        self.assertEqual(ranked[-1].severity, "P3")
        p0 = [i.id for i in ranked if i.severity == "P0"]
        # Stable within a severity level: discovery order, not impact order.
        self.assertEqual(p0, ["second", "fourth"])
        self.assertEqual([i.id for i in ranked if i.severity == "P2"], ["third"])

    def test_severity_plus_impact_breaks_ties_by_downstream_count(self) -> None:
        g = _graph()
        a = Issue(id="low_impact", code="unproved_lemma", message="m", claim_id="theorem-1")
        b = Issue(id="high_impact", code="unproved_lemma", message="m", claim_id="lemma-1")
        ranked = rank([a, b], g, mode="severity+impact")
        self.assertEqual(ranked[0].severity, ranked[1].severity)
        self.assertEqual(ranked[0].id, "high_impact",
                         "impact did not break the severity tie")

    def test_three_modes_can_differ(self) -> None:
        g = _graph()
        orders = {m: [i.id for i in rank(self._issues(), g, mode=m)] for m in ORDER_MODES}
        self.assertNotEqual(orders["fifo"], orders["severity"])

    def test_resolved_issues_are_excluded(self) -> None:
        issues = self._issues()
        issues[1].status = "resolved"
        self.assertNotIn("second", [i.id for i in rank(issues, _graph())])

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rank([], None, mode="vibes")


class SelectTest(unittest.TestCase):
    def test_queue_truncated_to_issue_budget(self) -> None:
        issues = [Issue(id=f"i{i}", code="unproved_lemma", message="m") for i in range(12)]
        self.assertEqual(len(select(issues, 5)), 5)

    def test_budget_larger_than_queue(self) -> None:
        self.assertEqual(len(select([Issue(id="a", code="c", message="m")], 5)), 1)

    def test_zero_budget(self) -> None:
        self.assertEqual(select([Issue(id="a", code="c", message="m")], 0), [])


class AdapterTest(unittest.TestCase):
    def test_issues_from_verifier_dicts(self) -> None:
        raw = [{"code": "logical_gap", "message": "m", "detail": "d"}]
        issues = issues_from_verifier(raw)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "logical_gap")


if __name__ == "__main__":
    unittest.main()
