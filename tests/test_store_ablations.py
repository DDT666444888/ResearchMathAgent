"""T1.5 — the paper's memory ablations, as runnable store variants.

Figure 5(d): stateless (1.8) -> last-round-only (3.4) -> full persistence (6.0).
Each must genuinely hide prior-round records, not merely deprioritise them,
or the ablation measures nothing.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.store import (
    LastRoundOnlyStore,
    ResearchStore,
    StatelessStore,
    StoreError,
    memory_model_for,
    open_store,
)


class MemoryModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed_rounds(self, store: ResearchStore, n: int = 3) -> None:
        for r in range(n):
            store.begin_round(r)
            store.add("H", "insight", f"insight from round {r}")

    def test_stateless_sees_nothing_from_prior_rounds(self) -> None:
        store = open_store(self.root, "q6", "first_proof_1", memory="stateless")
        self._seed_rounds(store)
        store.begin_round(2)
        bodies = [r.body for r in store.records("H")]
        self.assertEqual(bodies, ["insight from round 2"])

    def test_last_round_only_sees_r_minus_1_not_r_minus_2(self) -> None:
        store = open_store(self.root, "q6", "first_proof_1", memory="last-round-only")
        self._seed_rounds(store)
        store.begin_round(2)
        bodies = sorted(r.body for r in store.records("H"))
        self.assertEqual(bodies, ["insight from round 1", "insight from round 2"])

    def test_full_memory_sees_everything(self) -> None:
        store = open_store(self.root, "q6", "first_proof_1", memory="full")
        self._seed_rounds(store)
        store.begin_round(2)
        self.assertEqual(len(store.records("H")), 3)

    def test_ablated_records_are_still_written(self) -> None:
        """Hiding is a read-side policy: the artifacts must survive for
        telemetry and for the final proof."""
        store = open_store(self.root, "q6", "first_proof_1", memory="stateless")
        self._seed_rounds(store)
        full = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.assertEqual(len(full.records("H")), 3)

    def test_stateless_hides_the_previous_proof(self) -> None:
        store = open_store(self.root, "q6", "first_proof_1", memory="stateless")
        store.begin_round(0)
        store.add_proof_revision("round 0 proof", produced_by="proposer")
        store.begin_round(1)
        self.assertIsNone(store.current_proof(),
                          "a stateless store must not carry the proof forward")

    def test_open_store_returns_the_right_class(self) -> None:
        self.assertIsInstance(open_store(self.root, "q6", memory="full"), ResearchStore)
        self.assertIsInstance(open_store(self.root, "q6", memory="stateless"), StatelessStore)
        self.assertIsInstance(
            open_store(self.root, "q6", memory="last-round-only"), LastRoundOnlyStore)

    def test_unknown_memory_model_rejected(self) -> None:
        with self.assertRaises(StoreError):
            open_store(self.root, "q6", memory="telepathic")


class ConfigMappingTest(unittest.TestCase):
    """RunConfig ablation names must select the store variant."""

    def test_default_is_full(self) -> None:
        self.assertEqual(memory_model_for(RunConfig()), "full")

    def test_stateless_ablation(self) -> None:
        cfg = RunConfig(ablations=frozenset({"store.stateless"}))
        self.assertEqual(memory_model_for(cfg), "stateless")

    def test_last_round_only_ablation(self) -> None:
        cfg = RunConfig(ablations=frozenset({"store.last-round-only"}))
        self.assertEqual(memory_model_for(cfg), "last-round-only")


if __name__ == "__main__":
    unittest.main()
