"""T2.1 / T2.2 / T2.5 — token counting and PrefixToBudget."""
from __future__ import annotations

import os
import random
import unittest
from unittest.mock import patch

from rma.budget import (
    Chunk,
    Section,
    count_tokens,
    mandatory_floor_tokens,
    prefix_to_budget,
    tokenizer_name,
)


def _ctx(n_records: int = 6, size: int = 40) -> list[Section]:
    """A candidate context C in the paper's five-part order."""
    return [
        Section("instructions", [Chunk("INSTRUCTIONS " + "i" * size, droppable=False)]),
        Section("query", [Chunk("QUERY " + "q" * size, droppable=False)]),
        Section("current_proof", [Chunk("PROOF " + "p" * size, record_id="Pi-1",
                                        priority=100)]),
        Section("linked_records", [
            Chunk(f"LINKED{i} " + "l" * size, record_id=f"L-{i}", priority=90 - i)
            for i in range(n_records // 2)
        ]),
        Section("recent_outputs", [
            Chunk(f"RECENT{i} " + "r" * size, record_id=f"R-{i}", priority=40 - i)
            for i in range(n_records // 2)
        ]),
    ]


class CountTest(unittest.TestCase):
    def test_count_is_deterministic(self) -> None:
        text = "\\begin{lemma}For every graph $G$...\\end{lemma}"
        self.assertEqual(count_tokens(text), count_tokens(text))
        self.assertGreater(count_tokens(text), 0)

    def test_count_grows_with_text(self) -> None:
        self.assertGreater(count_tokens("x" * 4000), count_tokens("x" * 40))

    def test_count_empty_is_zero(self) -> None:
        self.assertEqual(count_tokens(""), 0)

    def test_count_works_offline(self) -> None:
        """No network, no API key — the counter must still work."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            self.assertGreater(count_tokens("offline counting"), 0)

    def test_fallback_when_tiktoken_missing(self) -> None:
        import rma.budget as budget

        with patch.dict(os.environ, {"RMA_DISABLE_TIKTOKEN": "1"}):
            budget._ENCODER = None
            budget._ENCODER_TRIED = False
            try:
                n = budget.count_tokens("x" * 400)
                self.assertEqual(n, 100)
                self.assertTrue(budget.tokenizer_name().startswith("chars/"))
            finally:
                budget._ENCODER = None
                budget._ENCODER_TRIED = False

    def test_tokenizer_reported(self) -> None:
        self.assertIn(tokenizer_name(), ("cl100k_base", "chars/4"))


class PrefixToBudgetTest(unittest.TestCase):
    def test_result_never_exceeds_budget(self) -> None:
        """Property test: 200 random contexts, none may overflow."""
        rng = random.Random(42)
        for _ in range(200):
            sections = _ctx(n_records=rng.randint(2, 12), size=rng.randint(5, 200))
            budget = rng.randint(60, 400)
            obs = prefix_to_budget(sections, budget)
            # The floor includes rendering overhead (headers, separators), not
            # just the non-droppable chunk text — a naive chunk-sum floor
            # under-counts and makes this assertion wrong, not the code.
            if mandatory_floor_tokens(sections) <= budget:
                self.assertLessEqual(obs.tokens, budget,
                                     f"overflow: {obs.tokens} > {budget}")

    def test_over_floor_budget_is_reported_not_hidden(self) -> None:
        """When even the mandatory head overflows, say so rather than slicing
        the instructions into nonsense."""
        sections = _ctx(n_records=4, size=400)
        floor = mandatory_floor_tokens(sections)
        obs = prefix_to_budget(sections, floor // 2)
        self.assertGreater(obs.tokens, obs.budget)
        self.assertIn("INSTRUCTIONS", obs.text)
        self.assertIn("QUERY", obs.text)

    def test_floor_is_below_full_context(self) -> None:
        sections = _ctx(n_records=6, size=50)
        full = prefix_to_budget(sections, 10**9)
        self.assertLess(mandatory_floor_tokens(sections), full.tokens)

    def test_drops_whole_records_not_characters(self) -> None:
        """Every surviving record must appear byte-identical."""
        sections = _ctx(n_records=8, size=60)
        obs = prefix_to_budget(sections, 120)
        for section in sections:
            for chunk in section.chunks:
                if chunk.record_id and chunk.record_id in obs.included:
                    self.assertIn(chunk.text, obs.text,
                                  f"{chunk.record_id} was cut mid-record")

    def test_drop_order_is_lowest_priority_last_first(self) -> None:
        sections = [
            Section("instructions", [Chunk("HEAD", droppable=False)]),
            Section("records", [
                Chunk("A " + "a" * 400, record_id="high", priority=90),
                Chunk("B " + "b" * 400, record_id="mid", priority=50),
                Chunk("C " + "c" * 400, record_id="low", priority=10),
            ]),
        ]
        obs = prefix_to_budget(sections, 120)
        self.assertEqual(obs.dropped[0], "low", "did not drop the lowest priority first")
        if len(obs.dropped) > 1:
            self.assertEqual(obs.dropped[1], "mid")
        self.assertNotIn("high", obs.dropped[:1])

    def test_instructions_and_query_are_never_dropped(self) -> None:
        sections = _ctx(n_records=10, size=300)
        obs = prefix_to_budget(sections, 30)  # far too small
        self.assertIn("INSTRUCTIONS", obs.text)
        self.assertIn("QUERY", obs.text)

    def test_nothing_dropped_when_it_already_fits(self) -> None:
        sections = _ctx(n_records=4, size=10)
        obs = prefix_to_budget(sections, 100_000)
        self.assertEqual(obs.dropped, [])
        self.assertEqual(len(obs.included), 5)

    def test_dropped_ids_are_reported(self) -> None:
        obs = prefix_to_budget(_ctx(n_records=8, size=80), 100)
        self.assertGreater(obs.n_dropped, 0)
        self.assertEqual(len(set(obs.dropped)), len(obs.dropped))
        self.assertFalse(set(obs.dropped) & set(obs.included),
                         "a record was reported both included and dropped")

    def test_section_order_follows_the_paper(self) -> None:
        obs = prefix_to_budget(_ctx(n_records=4, size=10), 100_000)
        self.assertEqual(obs.sections,
                         ["instructions", "query", "current_proof",
                          "linked_records", "recent_outputs"])

    def test_empty_sections_are_omitted(self) -> None:
        sections = [Section("instructions", [Chunk("HEAD", droppable=False)]),
                    Section("linked_records", [])]
        obs = prefix_to_budget(sections, 1000)
        self.assertEqual(obs.sections, ["instructions"])

    def test_as_dict_is_telemetry_shaped(self) -> None:
        obs = prefix_to_budget(_ctx(), 100)
        d = obs.as_dict()
        for key in ("context_tokens", "records_included", "records_dropped", "mode", "budget"):
            self.assertIn(key, d)


class ContextModesTest(unittest.TestCase):
    """T2.5 — the paper's dump / truncate / budget ladder."""

    def setUp(self) -> None:
        self.sections = _ctx(n_records=10, size=120)
        self.budget = 150

    def test_dump_exceeds_budget(self) -> None:
        obs = prefix_to_budget(self.sections, self.budget, mode="dump")
        self.assertGreater(obs.tokens, self.budget)
        self.assertEqual(obs.dropped, [])

    def test_truncate_cuts_mid_record(self) -> None:
        obs = prefix_to_budget(self.sections, self.budget, mode="truncate")
        whole = [c.text for s in self.sections for c in s.chunks]
        self.assertTrue(any(t not in obs.text for t in whole),
                        "truncation should have severed at least one record")

    def test_budget_drops_whole_records(self) -> None:
        obs = prefix_to_budget(self.sections, self.budget, mode="budget")
        self.assertLessEqual(obs.tokens, self.budget)
        self.assertGreater(obs.n_dropped, 0)
        for s in self.sections:
            for c in s.chunks:
                if c.record_id in obs.included:
                    self.assertIn(c.text, obs.text)

    def test_modes_produce_different_observations(self) -> None:
        texts = {m: prefix_to_budget(self.sections, self.budget, mode=m).text
                 for m in ("dump", "truncate", "budget")}
        self.assertEqual(len(set(texts.values())), 3,
                         "the three context modes are not actually different")

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            prefix_to_budget(self.sections, 100, mode="vibes")


if __name__ == "__main__":
    unittest.main()
