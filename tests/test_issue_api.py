"""T0.3 — the P0-P3 severity model must survive a round trip.

`webapp/issues.py` has always defined PRIORITY_LEVELS and accepted priority/
issue_type, but the POST /api/issues/{pid} endpoint dropped both from the
payload. Since the critic agent creates issues through that endpoint, every
issue on disk was P2-by-default or had no priority field at all.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from webapp.issues import (
    ISSUE_TYPES,
    PRIORITY_LEVELS,
    _seed_issue_direct,
    create_issue,
    get_issue,
    list_issues,
)


class IssuePriorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_priority_round_trips(self) -> None:
        """P0 in must be P0 out — not silently downgraded to the P2 default."""
        created = create_issue(
            self.root, "q6", title="Crux lemma unproved",
            author="critic-agent", dataset="first_proof_1",
            issue_type="proof-gap", priority="P0",
        )
        self.assertEqual(created["priority"], "P0")
        reread = get_issue(self.root, "q6", created["id"], "first_proof_1")
        self.assertEqual(reread["priority"], "P0")
        self.assertEqual(reread["issue_type"], "proof-gap")

    def test_endpoint_forwards_priority_and_issue_type(self) -> None:
        """The endpoint must pass both through to create_issue.

        Asserted at the call boundary so the test does not need a live server.
        """
        import inspect

        from webapp import server

        src = inspect.getsource(server.create_issue_ep)
        self.assertIn("priority", src)
        self.assertIn("issue_type", src)

    def test_all_priority_levels_accepted(self) -> None:
        for level in PRIORITY_LEVELS:
            issue = create_issue(self.root, "q7", title=f"t{level}", priority=level)
            self.assertEqual(issue["priority"], level)

    def test_invalid_priority_falls_back_to_p2(self) -> None:
        issue = create_issue(self.root, "q8", title="t", priority="URGENT")
        self.assertEqual(issue["priority"], "P2")

    def test_issue_type_validated_against_taxonomy(self) -> None:
        good = create_issue(self.root, "q9", title="t", issue_type="logical-error")
        self.assertEqual(good["issue_type"], "logical-error")
        self.assertIn(good["issue_type"], ISSUE_TYPES)
        bad = create_issue(self.root, "q9", title="t2", issue_type="not-a-type")
        self.assertIsNone(bad["issue_type"])

    def test_seeded_issue_has_priority_and_dataset(self) -> None:
        """Seeded records were the one shape missing dataset/priority/issue_type,
        so consumers could never read the schema uniformly."""
        issue = _seed_issue_direct(self.root, "q6", "first_proof_1")
        self.assertEqual(issue["dataset"], "first_proof_1")
        self.assertIn(issue["priority"], PRIORITY_LEVELS)
        self.assertEqual(issue["priority"], "P0")
        self.assertIn(issue["issue_type"], ISSUE_TYPES)

    def test_every_issue_on_disk_has_the_full_schema(self) -> None:
        create_issue(self.root, "q6", title="a", priority="P1", issue_type="missing-case")
        _seed_issue_direct(self.root, "q7", "first_proof_1")
        for pid in ("q6", "q7"):
            for issue in list_issues(self.root, pid, "first_proof_1"):
                for field in ("id", "problem_id", "dataset", "title", "status",
                              "priority", "issue_type", "created_at", "created_by"):
                    self.assertIn(field, issue, f"{pid}/{issue.get('id')} missing {field}")


if __name__ == "__main__":
    unittest.main()
