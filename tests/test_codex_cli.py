from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("anthropic"), "webapp extras are not installed")
class CodexCliProviderTest(unittest.TestCase):
    def test_binary_override_is_used(self) -> None:
        from webapp.codex_cli import codex_available
        with tempfile.NamedTemporaryFile() as binary:
            path = Path(binary.name)
            path.chmod(0o755)
            with patch.dict("os.environ", {"RMA_CODEX_BIN": str(path)}, clear=False):
                self.assertEqual(codex_available(), str(path))

    def test_missing_binary_is_a_well_formed_error(self) -> None:
        from webapp.agent import AgentConfig
        from webapp.codex_cli import run_codex_agent
        cfg = AgentConfig(problem_id="q1", problem_text="x")
        with patch("webapp.codex_cli.codex_available", return_value=None):
            events = list(run_codex_agent(cfg))
        self.assertEqual([event.type for event in events], ["error", "done"])
        self.assertIn("codex login", events[0].data["message"])

    def test_jsonl_command_events_are_visible_to_the_ui(self) -> None:
        from webapp.codex_cli import _translate_event
        started = list(_translate_event({
            "type": "item.started",
            "item": {"id": "cmd-1", "type": "command_execution", "command": "ls"},
        }))
        completed = list(_translate_event({
            "type": "item.completed",
            "item": {"id": "cmd-1", "type": "command_execution", "aggregated_output": "problem.tex"},
        }))
        self.assertEqual(started[0].type, "tool_use")
        self.assertEqual(started[0].data["name"], "Shell")
        self.assertEqual(completed[0].type, "tool_result")
        self.assertEqual(completed[0].data["output"], "problem.tex")

    def test_only_complete_latex_is_used_as_a_fallback_artifact(self) -> None:
        from webapp.codex_cli import _latex_document
        tex = "note\\n\\documentclass{article}\\begin{document}x\\end{document}"
        self.assertTrue(_latex_document(tex).startswith("\\documentclass"))
        self.assertIsNone(_latex_document("A short textual summary."))

    def test_runner_streams_cli_operations_and_emits_solution_artifact(self) -> None:
        from webapp.agent import AgentConfig
        from webapp.codex_cli import run_codex_agent
        script = """#!/bin/sh
printf '%s\\n' '{"type":"item.started","item":{"id":"cmd-1","type":"command_execution","command":"pwd"}}'
printf '%s\\n' '{"type":"item.completed","item":{"id":"cmd-1","type":"command_execution","aggregated_output":"/tmp/work"}}'
printf '%s\\n' '{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":3}}'
printf '%s\\n' '\\documentclass{article}\\begin{document}ok\\end{document}' > solution.tex
exit 0
"""
        with tempfile.TemporaryDirectory() as tmp, tempfile.NamedTemporaryFile("w", delete=False) as binary:
            binary.write(script)
            binary.flush()
            path = Path(binary.name)
            path.chmod(0o755)
            cfg = AgentConfig(problem_id="q1", problem_text="x", workspace=Path(tmp))
            with patch.dict("os.environ", {"RMA_CODEX_BIN": str(path)}, clear=False):
                events = list(run_codex_agent(cfg))
            path.unlink()
        self.assertIn("tool_use", [event.type for event in events])
        self.assertIn("tool_result", [event.type for event in events])
        self.assertIn("usage", [event.type for event in events])
        artifact = next(event for event in events if event.type == "artifact")
        self.assertIn("\\documentclass", artifact.data["content"])


if __name__ == "__main__":
    unittest.main()
