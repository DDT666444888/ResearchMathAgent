"""T0.2 — structured model calls must not go through the LaTeX quality gate.

Regression context: `_model_verify_proof` asks for a JSON array of issues, but
on the default backend it called `call_claude_code`, which refuses any reply
that is not a LaTeX document. The raise was swallowed by a bare `except`, so
the paper's LM gap critic contributed exactly zero issues on every run.
"""
from __future__ import annotations

import unittest
from argparse import Namespace
from unittest.mock import patch

from rma.models import ModelRequestError, ModelResponse, call_json, parse_json_response
from rma.solve import _model_verify_proof


ISSUE_JSON = (
    '[{"code": "logical_gap", "severity": "error", '
    '"message": "Step 3 does not follow from Lemma 2.", "detail": "Lemma 2"}, '
    '{"code": "unjustified_claim", "severity": "error", '
    '"message": "The bound C < 1/42 is asserted.", "detail": "C < 1/42"}]'
)


def _args(model_name: str = "claude-code", provider: str = "auto") -> Namespace:
    return Namespace(model_name=model_name, model_provider=provider)


class CallJsonTest(unittest.TestCase):
    def test_call_json_accepts_non_latex(self) -> None:
        """The exact payload shape that the LaTeX gate rejects must parse."""
        with patch("rma.models.call_claude_code") as cc:
            cc.return_value = ModelResponse(text=ISSUE_JSON, provider="claude-code", model="m")
            out = call_json(model="claude-code", system="s", prompt="p")
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["code"], "logical_gap")

    def test_call_json_requests_json_expectation(self) -> None:
        """call_json must tell the backend to skip the proof-document gate."""
        with patch("rma.models.call_claude_code") as cc:
            cc.return_value = ModelResponse(text="[]", provider="claude-code", model="m")
            call_json(model="claude-code", system="s", prompt="p")
        self.assertEqual(cc.call_args.kwargs["expect"], "json")

    def test_call_json_returns_none_when_unparseable(self) -> None:
        with patch("rma.models.call_claude_code") as cc:
            cc.return_value = ModelResponse(text="I could not do that.", provider="claude-code", model="m")
            self.assertIsNone(call_json(model="claude-code", system="s", prompt="p"))

    def test_parse_json_response_tolerates_fences_and_trailing_prose(self) -> None:
        self.assertEqual(parse_json_response('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_response('{"a": 1}\n\nNote: hope this helps!'), {"a": 1})
        # A wrapper object must not be mistaken for one of its inner arrays.
        self.assertEqual(
            parse_json_response('{"completeness": 4, "missing": ["x"]}'),
            {"completeness": 4, "missing": ["x"]},
        )


class ModelVerifyProofTest(unittest.TestCase):
    def test_model_verify_proof_returns_issues(self) -> None:
        """FakeBackend emits 2 issues -> 2 issues returned (was [] before T0.2)."""
        with patch("rma.solve.call_json") as cj:
            cj.return_value = [
                {"code": "logical_gap", "severity": "error", "message": "m1", "detail": "d1"},
                {"code": "unjustified_claim", "severity": "error", "message": "m2", "detail": "d2"},
            ]
            issues = _model_verify_proof("\\begin{proof}x\\end{proof}", _args())
        self.assertEqual(len(issues), 2)
        self.assertEqual({i["code"] for i in issues}, {"logical_gap", "unjustified_claim"})

    def test_model_verify_proof_survives_latex_gate_path(self) -> None:
        """End to end through call_json with a raw JSON string from the CLI.

        This is the exact scenario that used to raise ModelRequestError inside
        call_claude_code and get swallowed into [].
        """
        with patch("rma.models.call_claude_code") as cc:
            cc.return_value = ModelResponse(text=ISSUE_JSON, provider="claude-code", model="m")
            issues = _model_verify_proof("\\begin{proof}x\\end{proof}", _args())
        self.assertEqual(len(issues), 2)

    def test_model_verify_proof_empty_array_is_empty(self) -> None:
        with patch("rma.solve.call_json") as cj:
            cj.return_value = []
            self.assertEqual(_model_verify_proof("\\begin{proof}x\\end{proof}", _args()), [])

    def test_offline_backend_skips_the_call(self) -> None:
        self.assertEqual(
            _model_verify_proof("x", _args(model_name="rma-skeleton", provider="offline")), []
        )


class IssueNormalizationTest(unittest.TestCase):
    """Second silent-zero in the same function: the strict {"code","message"}
    filter dropped every element when the model answered with different key
    names. Observed live — the critic found 4 real gaps and reported none."""

    def test_alternate_key_names_are_kept(self) -> None:
        from rma.solve import normalize_issue_records

        # The exact shape claude-opus-4-8 returned in a live run.
        raw = [{
            "issue": "UNJUSTIFIED_CLAIM",
            "location": "Lemma proof",
            "quote": "It follows immediately from standard spectral arguments.",
            "explanation": "The claim is asserted with no actual argument.",
        }]
        out = normalize_issue_records(raw)
        self.assertEqual(len(out), 1, "a real finding was silently discarded")
        self.assertEqual(out[0]["code"], "unjustified_claim")
        self.assertEqual(out[0]["severity"], "error")
        self.assertIn("no actual argument", out[0]["message"])

    def test_canonical_schema_still_works(self) -> None:
        from rma.solve import normalize_issue_records

        out = normalize_issue_records([
            {"code": "logical_gap", "severity": "error", "message": "m", "detail": "d"}
        ])
        self.assertEqual(out, [{"code": "logical_gap", "severity": "error",
                                "message": "m", "detail": "d"}])

    def test_bare_string_finding_is_kept(self) -> None:
        from rma.solve import normalize_issue_records

        out = normalize_issue_records(["Lemma 2 is never proved"])
        self.assertEqual(len(out), 1)
        self.assertIn("Lemma 2", out[0]["message"])

    def test_empty_records_are_dropped(self) -> None:
        from rma.solve import normalize_issue_records

        self.assertEqual(normalize_issue_records([{}, {"code": "logical_gap"}, None]), [])

    def test_empty_list_stays_empty(self) -> None:
        from rma.solve import normalize_issue_records

        self.assertEqual(normalize_issue_records([]), [])

    def test_end_to_end_alternate_schema_reaches_verifier(self) -> None:
        with patch("rma.solve.call_json") as cj:
            cj.return_value = [
                {"issue": "LOGICAL_GAP", "explanation": "step 3 unsupported", "location": "p2"},
                {"issue": "HYPOTHESIS_NOT_VERIFIED", "explanation": "paving hypotheses unchecked",
                 "location": "Thm 1"},
            ]
            issues = _model_verify_proof("\\begin{proof}x\\end{proof}", _args())
        self.assertEqual(len(issues), 2)
        self.assertEqual({i["code"] for i in issues},
                         {"logical_gap", "hypothesis_not_verified"})


class LatexGateRegressionTest(unittest.TestCase):
    """The gate must still protect proof generation — this is a quality control,
    not dead weight. Only structured calls opt out of it."""

    def test_the_bug_was_real_json_fails_the_latex_gate(self) -> None:
        """Proves T0.2 fixed something: the issue-list payload the critic asks
        for is exactly what the LaTeX gate rejects. Without `expect="json"`
        this reply raises ModelRequestError and the caller swallows it into []."""
        from rma.models import _looks_like_latex, _extract_latex_document

        self.assertFalse(_looks_like_latex(ISSUE_JSON))
        self.assertIsNone(_extract_latex_document(ISSUE_JSON))

    def test_regression_latex_gate_still_guards_proof_calls(self) -> None:
        from rma.models import _looks_like_latex, _extract_latex_document

        narration = "I have finished researching and will now write the proof."
        self.assertFalse(_looks_like_latex(narration))
        self.assertIsNone(_extract_latex_document(narration))

        real_proof = "\\documentclass{article}\\begin{document}\\end{document}"
        self.assertIsNotNone(_extract_latex_document(real_proof))

    def test_default_expect_is_latex(self) -> None:
        """A caller that does not opt in keeps the proof-document guarantee."""
        import inspect

        from rma.models import call_claude_code

        self.assertEqual(
            inspect.signature(call_claude_code).parameters["expect"].default, "latex"
        )


if __name__ == "__main__":
    unittest.main()
