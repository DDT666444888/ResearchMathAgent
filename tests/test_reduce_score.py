"""The local scorer must reproduce the Arena's published per-problem scores."""
import unittest
from rma.reduction.score import Axes, arena_tokens, objective, parse_official, problem_score, run_score

# Official rows from the round-4 server result (submission 20260910T230421Z).
ROWS = [
    {'score': '65.02%', 'length': '277 of 407', 'heartbeats': '510 of 1,383', 'zero_shot': '4/4'},
    {'score': '72.91%', 'length': '318 of 408', 'heartbeats': '1,996 of 59,830', 'zero_shot': '3/3'},
    {'score': '70.28%', 'length': '170 of 313', 'heartbeats': '1,804 of 5,175', 'zero_shot': '1/1'},
    {'score': '91.13%', 'length': '434 of 1,754', 'heartbeats': '2,497 of 134,499', 'zero_shot': '2/2'},
]


class ScoreTests(unittest.TestCase):
    def test_formula_reproduces_official_problem_scores(self):
        for row in ROWS:
            o = parse_official(row)
            axes = problem_score(tokens=o['length'], heartbeats=o['heartbeats'], ref_tokens=o['ref_length'],
                                 ref_heartbeats=o['ref_heartbeats'], passed=o['zero_passed'],
                                 listed=o['zero_listed'])
            with self.subTest(row=row['score']):
                self.assertAlmostEqual(axes.score, o['score'], delta=0.006)

    def test_slower_proof_has_negative_heartbeat_axis_and_can_lose(self):
        faster = problem_score(tokens=300, heartbeats=500, ref_tokens=400, ref_heartbeats=1000, passed=1, listed=1)
        shorter_but_slow = problem_score(tokens=250, heartbeats=1600, ref_tokens=400, ref_heartbeats=1000,
                                         passed=1, listed=1)
        self.assertLess(shorter_but_slow.heartbeat_pct, 0)
        self.assertLess(shorter_but_slow.score, faster.score)

    def test_zero_shot_loss_and_missing_problems_count(self):
        full = problem_score(tokens=100, heartbeats=100, ref_tokens=200, ref_heartbeats=200, passed=3, listed=3)
        partial = problem_score(tokens=100, heartbeats=100, ref_tokens=200, ref_heartbeats=200, passed=2, listed=3)
        self.assertAlmostEqual(full.score - partial.score, 100 / 9)
        self.assertAlmostEqual(run_score({'a': 90.}, ['a', 'b', 'c']), 30.)

    def test_length_ignores_comments_and_whitespace(self):
        self.assertEqual(arena_tokens('by\n  simp [foo, bar]'), arena_tokens('by  -- note\n simp  [foo,bar]'))
        self.assertEqual(arena_tokens('by exact ⟨h.1, .refl _⟩'), 8)   # h.1 and .refl are one identifier each

    def test_objective_states_the_rule_and_exchange_rate(self):
        axes = Axes(30., 60., 100.)
        text = objective(axes=axes, tokens=280, heartbeats=550, ref_tokens=400, ref_heartbeats=1000,
                         versions=['v4.31.0', 'v4.32.0'])
        self.assertIn('mean(length reduction %, heartbeat reduction %, zero-shot compatibility %)', text)
        self.assertIn('1 token = 0.083 points', text)
        self.assertIn('v4.31.0, v4.32.0', text)


class UndefinedScoreTests(unittest.TestCase):
    def test_unknown_heartbeats_or_reference_is_an_error_not_a_perfect_axis(self):
        for hb, ref in [(None, 1000), (500, None), (500, 0)]:
            with self.subTest(hb=hb, ref=ref), self.assertRaises(ValueError):
                problem_score(tokens=100, heartbeats=hb, ref_tokens=200, ref_heartbeats=ref, passed=1, listed=1)
        self.assertAlmostEqual(Axes(30., -30., 90.).score, 30.)
