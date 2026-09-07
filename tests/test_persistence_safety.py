"""T0.5 — five silent-data-loss holes.

Each test fails against the pre-T0.5 code. These are not hypothetical: every
one of them loses or fabricates research state without raising anything.
"""
from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from webapp.issues import create_issue, list_issues


class LongProofTest(unittest.TestCase):
    """`content[:60_000]` on the artifact silently dropped the tail of a proof."""

    def test_long_proof_not_silently_clipped(self) -> None:
        from webapp.agent import _artifact_from_workspace
        from webapp.tools import ToolContext

        long_proof = "\\documentclass{article}\\begin{document}\n" + ("x" * 70_000) + "\n\\end{document}"
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "solution.tex").write_text(long_proof, encoding="utf-8")
            event = _artifact_from_workspace(ToolContext(repo_root=ws, workspace=ws))

        self.assertIsNotNone(event)
        content = event.data["content"]
        self.assertEqual(len(content), len(long_proof.strip()),
                         "proof was truncated instead of emitted whole")
        self.assertTrue(content.rstrip().endswith("\\end{document}"),
                        "the tail of the proof was lost")
        self.assertEqual(event.data.get("oversized_chars"), len(long_proof.strip()),
                         "oversize must be reported, not hidden")

    def test_no_silent_60k_clip_remains_on_artifact_paths(self) -> None:
        from webapp import agent, claude_code

        for mod in (agent, claude_code):
            src = inspect.getsource(mod)
            self.assertNotIn('"content": text[:60_000]', src)


class WorkingProofScopeTest(unittest.TestCase):
    """Two datasets sharing a problem id shared one working proof."""

    def test_working_proof_path_is_dataset_scoped(self) -> None:
        from webapp.issue_agents import working_proof_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = working_proof_path(root, "prob-01", "first_proof_2")
            b = working_proof_path(root, "prob-01", "erdos_problems")
            self.assertNotEqual(a, b, "distinct datasets must not share a proof file")
            self.assertIn("first_proof_2", str(a))
            self.assertIn("erdos_problems", str(b))

    def test_signature_accepts_dataset(self) -> None:
        from webapp.issue_agents import get_working_proof, save_working_proof, working_proof_path

        for fn in (working_proof_path, get_working_proof, save_working_proof):
            self.assertIn("dataset", inspect.signature(fn).parameters, fn.__name__)

    def test_legacy_unscoped_file_still_readable(self) -> None:
        """Existing un-scoped proofs must not become invisible."""
        from webapp.issue_agents import get_working_proof

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "webapp" / "issues" / "q6" / "working_solution.tex"
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text("legacy proof", encoding="utf-8")
            self.assertEqual(get_working_proof(root, "q6", "first_proof_1"), "legacy proof")


class ConcurrentIssueIdTest(unittest.TestCase):
    """_short_id is a max+1 scan; concurrent creates collided and lost writes."""

    def test_concurrent_issue_ids_do_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with ThreadPoolExecutor(max_workers=8) as pool:
                issues = list(pool.map(
                    lambda i: create_issue(root, "q6", title=f"issue {i}"),
                    range(20),
                ))
            ids = [i["id"] for i in issues]
            self.assertEqual(len(set(ids)), 20, f"id collision: {sorted(ids)}")
            on_disk = list_issues(root, "q6", "first_proof_1")
            self.assertEqual(len(on_disk), 20, "a concurrent write was lost")

    def test_writes_are_atomic_no_tmp_residue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_issue(root, "q6", title="t")
            leftovers = list((root / "webapp" / "issues").rglob("*.tmp"))
            self.assertEqual(leftovers, [], f"temp files left behind: {leftovers}")


class ListDoesNotMutateTest(unittest.TestCase):
    """Reading a collection must not write to it."""

    def test_list_issues_does_not_create(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            issues = list_issues(root, "q6", "first_proof_1")
            self.assertEqual(issues, [])
            files = list((root / "webapp" / "issues").rglob("*.json"))
            self.assertEqual(files, [], f"listing fabricated records: {files}")

    def test_seeding_is_available_when_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            issues = list_issues(root, "q6", "first_proof_1", seed_if_empty=True)
            self.assertEqual(len(issues), 1)
            self.assertTrue(issues[0]["title"].startswith("Proof:"))

    def test_ui_endpoint_still_seeds(self) -> None:
        """The UI wants a starting issue; that behaviour is preserved explicitly."""
        from webapp import server

        self.assertIn("seed_if_empty=True", inspect.getsource(server.list_issues_ep))


class ApiBaseTest(unittest.TestCase):
    """Hard-coded :8000 made every agent curl fail silently on the dev server."""

    def test_api_base_is_configurable(self) -> None:
        import importlib

        with patch.dict(os.environ, {"RMA_API_BASE": "http://localhost:8001"}):
            from webapp import issue_agents

            importlib.reload(issue_agents)
            self.assertEqual(issue_agents._API_BASE, "http://localhost:8001")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RMA_API_BASE", None)
            from webapp import issue_agents

            importlib.reload(issue_agents)
            self.assertEqual(issue_agents._API_BASE, "http://localhost:8000")


if __name__ == "__main__":
    unittest.main()
