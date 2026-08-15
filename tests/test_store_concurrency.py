"""T1.4 — concurrent writers must not lose records or leave partial files.

The export driver runs several per-problem pipelines at once, and a round runs
up to b=5 solvers; if two of them write the store at the same time and one
silently wins, research state disappears with no error anywhere.
"""
from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rma.store import ResearchStore


class ConcurrencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parallel_writers_lose_nothing(self) -> None:
        """8 threads x 25 records -> exactly 200 on disk, all ids distinct."""
        store = ResearchStore.open(self.root, "q6", "first_proof_1")

        def write(worker: int) -> list[str]:
            local = ResearchStore.open(self.root, "q6", "first_proof_1")
            return [local.add("M", "meeting_record", f"w{worker}-r{i}").id
                    for i in range(25)]

        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = [rid for batch in pool.map(write, range(8)) for rid in batch]

        self.assertEqual(len(ids), 200)
        self.assertEqual(len(set(ids)), 200, "duplicate record ids were issued")
        on_disk = store.reopen().records("M")
        self.assertEqual(len(on_disk), 200, "a concurrent write was lost")
        self.assertEqual(len({r.id for r in on_disk}), 200)

    def test_parallel_writers_across_components(self) -> None:
        store = ResearchStore.open(self.root, "q6", "first_proof_1")

        def write(spec: tuple[str, int]) -> None:
            component, i = spec
            ResearchStore.open(self.root, "q6", "first_proof_1").add(
                component, "x", f"{component}-{i}")

        work = [(c, i) for c in ("M", "H") for i in range(20)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, work))

        fresh = store.reopen()
        self.assertEqual(len(fresh.records("M")), 20)
        self.assertEqual(len(fresh.records("H")), 20)

    def test_no_partial_files_after_writes(self) -> None:
        store = ResearchStore.open(self.root, "q6", "first_proof_1")
        for i in range(10):
            store.add("M", "meeting_record", f"note {i}")
        leftovers = list(store.store_dir.rglob("*.tmp"))
        self.assertEqual(leftovers, [], f"temp files left behind: {leftovers}")

    def test_sidecar_survives_a_corrupt_line(self) -> None:
        """One bad line must not make the whole component unreadable."""
        store = ResearchStore.open(self.root, "q6", "first_proof_1")
        store.add("M", "meeting_record", "good one")
        path = store._sidecar_path("M")
        path.write_text(path.read_text() + "{not json\n", encoding="utf-8")
        records = store.reopen().records("M")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].body, "good one")


if __name__ == "__main__":
    unittest.main()
