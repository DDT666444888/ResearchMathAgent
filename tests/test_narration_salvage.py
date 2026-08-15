"""Narration-only replies must not discard a document the agent wrote to disk.

Observed live on First Proof B2 prob-05: the model wrote and compiled the
proof with file tools, then replied "Writing the complete document now...".
The LaTeX gate rejected the reply, run_solve caught the error and broke out of
the round loop, so the problem got 1 round instead of 5 — while the finished
proof sat in the output file untouched.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.models import _looks_like_latex, _salvage_latex_document

NARRATION = (
    "I have confirmed the reference and verified the computation. "
    "Writing the complete document now. Now let me verify the document compiles. "
    "`\\Kappa` is not a defined command. Let me replace it with `K_0`."
)
DOCUMENT = (
    "\\documentclass{article}\n\\begin{document}\n"
    "\\begin{theorem}A real result.\\end{theorem}\n"
    "\\begin{proof}A real argument.\\end{proof}\n\\end{document}"
)


class SalvageTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_narration_really_does_fail_the_gate(self) -> None:
        """Confirms the bug is real, so the salvage path is not dead code."""
        self.assertFalse(_looks_like_latex(NARRATION))

    def test_document_recovered_from_fallback_file(self) -> None:
        fallback = self.root / "prob-05_solution.tex"
        fallback.write_text(DOCUMENT, encoding="utf-8")
        self.assertEqual(_salvage_latex_document(fallback), DOCUMENT)

    def test_document_recovered_from_partial_file(self) -> None:
        partial = self.root / "partial_output.tex"
        partial.write_text(DOCUMENT, encoding="utf-8")
        self.assertEqual(_salvage_latex_document(None, partial), DOCUMENT)

    def test_first_candidate_wins(self) -> None:
        a = self.root / "a.tex"
        b = self.root / "b.tex"
        a.write_text(DOCUMENT, encoding="utf-8")
        b.write_text(DOCUMENT.replace("A real result", "Older result"), encoding="utf-8")
        self.assertIn("A real result", _salvage_latex_document(a, b))

    def test_narration_on_disk_is_not_salvaged(self) -> None:
        """The gate exists to stop prose masquerading as a proof; salvaging
        must not become a hole in it."""
        junk = self.root / "partial_output.tex"
        junk.write_text(NARRATION, encoding="utf-8")
        self.assertIsNone(_salvage_latex_document(junk))

    def test_missing_and_empty_files_are_skipped(self) -> None:
        empty = self.root / "empty.tex"
        empty.write_text("   \n", encoding="utf-8")
        self.assertIsNone(_salvage_latex_document(self.root / "nope.tex", empty, None))

    def test_bare_latex_without_documentclass_is_accepted(self) -> None:
        """A fragment that is clearly LaTeX still beats throwing the turn away."""
        frag = self.root / "f.tex"
        frag.write_text("\\begin{theorem}T\\end{theorem}\n\\begin{proof}P\\end{proof}",
                        encoding="utf-8")
        self.assertIsNotNone(_salvage_latex_document(frag))


if __name__ == "__main__":
    unittest.main()
