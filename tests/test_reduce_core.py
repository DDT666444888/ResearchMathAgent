"""Algorithm-1 reduction mode: every RMA unit runs, reads the store, and changes what happens next."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from rma.reduction.backend import BudgetExceeded
from rma.reduction.core import CorePipeline, apply_patch, compiler_findings, drop_simp_arg, structural_findings
from rma.reduction.lean import Measurement, declaration
from rma.reduction.pipeline import ReductionStore
from rma.reduction.score import arena_tokens

ENTRY = {'name': 'demo', 'source': 'demo', 'statement': 'theorem demo : True',
         'src': 'theorem demo : True := by trivial\n'}
START = 'by\n  simp [foo, bar]\n  have h : True := True.intro\n  exact h\n'
INITIAL = declaration(ENTRY, START)
UNUSED = 'This simp argument is unused:\n  bar\n\nHint: Omit it from the simp argument list.'


class LintingVerifier:
    """Anything without 'fail' compiles; cost grows with length; `bar` in a simp list is linted unused."""
    repos = {'demo': [Path('.')]}

    def verify(self, entry, body, measure=True, version=0):
        code = declaration(entry, body)
        warnings = [{'rel_line': i+1, 'col': 2, 'severity': 'warning', 'message': UNUSED}
                    for i, line in enumerate(code.splitlines()) if 'simp [' in line and 'bar' in line]
        ok = 'fail' not in body
        tokens = arena_tokens(body)
        return Measurement(ok, 10*tokens, tokens, 0 if ok else 1, [], 'log' if ok else
                           'RMAReduce.lean:3:2: error: unknown identifier \'Foo.missing_lemma\'', 'test',
                           hashlib.sha256(code.encode()).hexdigest(), measure,
                           [] if ok else [{'rel_line': 3, 'col': 2, 'severity': 'error',
                                           'message': "unknown identifier 'Foo.missing_lemma'"}],
                           0, tokens, warnings)


class Backend:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, unit, observation):
        self.calls.append((unit, observation))
        reply = self.replies[unit]
        return reply(observation) if callable(reply) else reply


CRITIC = json.dumps([{'kind': 'redundant_step', 'excerpt': 'have h : True := True.intro\n  exact h',
                      'proposal': 'Close the goal directly with `exact True.intro`.',
                      'tokens_saved': 6, 'heartbeats_saved': 60}])
PATCH = json.dumps({'replaced_text': 'have h : True := True.intro\n  exact h',
                    'replacement': 'exact True.intro', 'rationale': 'the have is used once'})
MEETING = json.dumps({'record': 'agreed', 'action_plan': {'summary': 'Replace everything by trivial',
                      'steps': ['use trivial']}, 'insights': ['True goals close by trivial']})
REVISE = '```lean\nby\n  trivial\n```'


class CoreLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_core(self, backend, **kw):
        return CorePipeline(self.root, LintingVerifier(), backend, model='gpt-6-astra', **kw)

    def test_units_run_in_algorithm_order_and_every_component_is_used(self):
        backend = Backend({'reduce.critic': CRITIC, 'reduce.solver': PATCH,
                           'reduce.meeting': MEETING, 'reduce.revise': REVISE})
        result = self.run_core(backend, rounds=1, issues_per_round=2).problem('01', ENTRY, INITIAL)
        self.assertTrue(result['improved'])
        self.assertEqual(result['proof'], declaration(ENTRY, 'by\n  trivial\n'))
        self.assertEqual([u for u, _ in backend.calls],
                         ['reduce.critic', 'reduce.solver', 'reduce.meeting', 'reduce.revise'])
        store = ReductionStore.open(self.root/'problems/01/research', '01', 'lean_reduce')
        counts = store.counts()
        for component in ('I', 'Pi', 'E', 'M', 'H'):
            self.assertGreater(counts.get(component, 0), 0, component)
        kinds = {r.meta.get('kind') for r in store.issues}
        self.assertTrue({'unused_simp_arg', 'redundant_step'} <= kinds)
        self.assertFalse([r for r in store.open_issues() if r.meta.get('status') != 'wontfix'])
        # The LM critic saw the compiler's finding and the public objective; the meeting saw E.
        self.assertIn('Remove the unused simp argument `bar`', backend.calls[0][1])
        self.assertIn('problem score = mean(length reduction %', backend.calls[0][1])
        self.assertIn(' E/evaluation', backend.calls[2][1])

    def test_compiler_suggested_fix_needs_no_model_call(self):
        backend = Backend({})
        result = self.run_core(backend, rounds=1, issues_per_round=1,
                               ablations={'critic.lm', 'meeting'}).problem('01', ENTRY, INITIAL)
        self.assertEqual(backend.calls, [])
        self.assertTrue(result['improved'])
        self.assertNotIn('bar', result['proof'])
        state = json.loads((self.root/'problems/01/state.json').read_text())
        self.assertTrue(state['attempts'][0]['deterministic'])

    def test_failed_move_is_recorded_grounded_and_the_issue_abandoned(self):
        bad = json.dumps({'replaced_text': 'exact h', 'replacement': 'exact Foo.missing_lemma fail'})
        backend = Backend({'reduce.critic': json.dumps([{'kind': 'library_lemma', 'excerpt': 'exact h',
                                                         'proposal': 'Use Foo.missing_lemma',
                                                         'tokens_saved': 1, 'heartbeats_saved': 0}]),
                           'reduce.solver': bad})
        # Repairs are exercised separately; here each round makes one move so memory across rounds is visible.
        pipeline = self.run_core(backend, rounds=2, issues_per_round=2, ablations={'meeting', 'critic.structural'},
                                 repair_threshold=float('inf'))
        pipeline.problem('01', ENTRY, INITIAL)
        store = ReductionStore.open(self.root/'problems/01/research', '01', 'lean_reduce')
        failed = [r for r in store.concepts if r.kind == 'failed_move']
        self.assertTrue(failed and 'unknown identifier' in failed[0].body)
        issue = next(r for r in store.issues if r.meta.get('kind') == 'library_lemma')
        self.assertEqual(issue.meta.get('status'), 'wontfix')
        self.assertTrue(any('Foo.missing_lemma' in r.body for r in store.literature))
        # The second round's solver sees the recorded failure instead of repeating blindly.
        solver_calls = [obs for unit, obs in backend.calls if unit == 'reduce.solver']
        self.assertEqual(len(solver_calls), 2)
        self.assertIn('failed_move', solver_calls[1])

    def test_rma_ablations_switch_units_off_and_unknown_names_fail(self):
        backend = Backend({'reduce.solver': PATCH})
        self.run_core(backend, rounds=1, ablations={'critic.lm', 'meeting', 'literature', 'concepts'}) \
            .problem('01', ENTRY, INITIAL)
        self.assertEqual({u for u, _ in backend.calls}, set())
        with self.assertRaises(ValueError):
            self.run_core(backend, ablations={'telepathy'})

    def test_budget_stop_keeps_the_verified_best(self):
        def broke(unit, obs): raise BudgetExceeded('full')
        result = self.run_core(broke, rounds=1, ablations={'critic.structural'}).problem('01', ENTRY, INITIAL)
        self.assertEqual(result['stop_reason'], 'budget_exhausted'); self.assertEqual(result['proof'], INITIAL)


class CoreHelperTests(unittest.TestCase):
    def test_lint_and_structure_findings(self):
        found = compiler_findings([{'rel_line': 2, 'message': UNUSED}])
        self.assertEqual(found[0]['excerpt'], 'bar'); self.assertEqual(found[0]['line'], 2)
        repeated = structural_findings('by\n  simp only [foo, bar]\n  simp only [foo, bar]\n')
        self.assertEqual(repeated[0]['kind'], 'duplicate_branch')

    def test_simp_argument_removal_and_anchored_patch(self):
        code = 'theorem t : True := by\n  simp [foo, bar]\n'
        self.assertEqual(drop_simp_arg(code, 2, 'bar'), 'theorem t : True := by\n  simp [foo]\n')
        self.assertEqual(drop_simp_arg('x := by\n  simp [bar]', 2, 'bar'), 'x := by\n  simp')
        self.assertIsNone(drop_simp_arg(code, 9, 'bar'))
        self.assertEqual(apply_patch('by\n  a\n  b', {'replaced_text': 'a', 'replacement': 'c'}), 'by\n  c\n  b')
        with self.assertRaises(ValueError):
            apply_patch('by a a', {'replaced_text': 'a', 'replacement': 'c'})


class NameGroundingTests(unittest.TestCase):
    def test_field_notation_local_hypotheses_and_issue_kinds_are_not_looked_up(self):
        from rma.reduction.core import candidate_names
        proof = 'by\n  intro hpos\n  have key : 0 = 0 := rfl\n  exact hpos.ne\' key\n'
        texts = ["Use `real_inner_self_pos.mpr hx0` and `mul_ne_zero_iff.mp hx`; then `hpos.ne' key`.",
                 'Move for issue 10-I-2 (library_lemma: ...) failed: unknown identifier \'Foo.bar_baz\'']
        names = candidate_names(texts, proof)
        self.assertIn('real_inner_self_pos', names); self.assertIn('mul_ne_zero_iff', names)
        self.assertIn('Foo.bar_baz', names)
        self.assertNotIn('library_lemma', names)
        self.assertFalse([n for n in names if n.startswith('hpos')])
        self.assertFalse([n for n in names if n.endswith(('.mpr', '.mp'))])


class RepairTurnTests(unittest.TestCase):
    def test_high_value_failed_patch_gets_one_repair_with_compiler_feedback(self):
        replies = iter([json.dumps({'replaced_text': 'exact h', 'replacement': 'exact Foo.missing_lemma fail'}), PATCH])
        backend = Backend({'reduce.critic': CRITIC, 'reduce.solver': lambda obs: next(replies)})
        with tempfile.TemporaryDirectory() as d:
            result = CorePipeline(Path(d), LintingVerifier(), backend, model='gpt-6-astra', rounds=1,
                                  issues_per_round=1, ablations={'meeting', 'critic.structural'}).problem('01', ENTRY, INITIAL)
            state = json.loads((Path(d)/'problems/01/state.json').read_text())
        solver = [obs for unit, obs in backend.calls if unit == 'reduce.solver']
        self.assertEqual(len(solver), 2)
        self.assertIn('did not compile', solver[1]); self.assertIn('unknown identifier', solver[1])
        self.assertTrue(result['improved'])
        self.assertEqual([(a['tag'], a['status']) for a in state['attempts'] if a['unit'] == 'reduce.solver'],
                         [('r1-solver-1', 'measured'), ('r1-solver-1-repair', 'promoted')])

    def test_low_value_issue_gets_no_repair(self):
        bad = json.dumps({'replaced_text': 'exact h', 'replacement': 'exact fail'})
        backend = Backend({'reduce.critic': json.dumps([{'kind': 'other', 'excerpt': 'exact h', 'proposal': 'tweak',
                                                         'tokens_saved': 0, 'heartbeats_saved': 0}]),
                           'reduce.solver': bad})
        with tempfile.TemporaryDirectory() as d:
            CorePipeline(Path(d), LintingVerifier(), backend, model='gpt-6-astra', rounds=1, issues_per_round=1,
                         ablations={'meeting', 'critic.structural'}).problem('01', ENTRY, INITIAL)
        self.assertEqual(sum(1 for unit, _ in backend.calls if unit == 'reduce.solver'), 1)


class BudgetSchedulingTests(unittest.TestCase):
    def test_critic_and_meeting_are_not_bought_without_room_for_their_follow_up(self):
        from rma.reduction.backend import Ledger
        with tempfile.TemporaryDirectory() as d:
            calls = []
            class Metered:
                ledger = Ledger(Path(d)/'usage.jsonl', 4.0)     # < one 10k-token hold at $500/M
                scope, max_output = None, 10000
                def __call__(self, unit, obs): calls.append(unit); return MEETING
            result = CorePipeline(Path(d), LintingVerifier(), Metered(), model='gpt-6-astra', rounds=1,
                                  issues_per_round=2).problem('01', ENTRY, INITIAL)
        self.assertEqual(calls, [])                  # no paid unit started
        self.assertTrue(result['improved'])          # the compiler-derived fix still ran for free


class CausalContextTests(unittest.TestCase):
    """Store records must change what the model is actually sent, not only be written."""

    def test_patch_prompt_has_one_output_protocol(self):
        backend = Backend({'reduce.critic': CRITIC, 'reduce.solver': PATCH})
        with tempfile.TemporaryDirectory() as d:
            CorePipeline(Path(d), LintingVerifier(), backend, model='gpt-6-astra', rounds=1, issues_per_round=1,
                         ablations={'meeting', 'critic.structural'}).problem('01', ENTRY, INITIAL)
        sent = next(obs for unit, obs in backend.calls if unit == 'reduce.solver')
        self.assertIn('Return ONLY JSON', sent)
        self.assertNotIn('code fence', sent.lower())

    def failed_then_retry(self, ablations):
        """Round 1 fails a move; returns what round 2's critic and solver were sent."""
        bad = json.dumps({'replaced_text': 'exact h', 'replacement': 'exact Foo.missing_lemma fail'})
        critic = json.dumps([{'kind': 'library_lemma', 'excerpt': 'exact h', 'proposal': 'Use Foo.missing_lemma',
                              'tokens_saved': 1, 'heartbeats_saved': 0}])
        backend = Backend({'reduce.critic': critic, 'reduce.solver': bad})
        with tempfile.TemporaryDirectory() as d:
            CorePipeline(Path(d), LintingVerifier(), backend, model='gpt-6-astra', rounds=2, issues_per_round=1,
                         ablations={'meeting', 'critic.structural'} | ablations,
                         repair_threshold=float('inf')).problem('01', ENTRY, INITIAL)
        return [obs for unit, obs in backend.calls][2:]

    def test_recorded_failure_changes_the_next_rounds_context_and_ablation_removes_it(self):
        with_memory = self.failed_then_retry(set())
        self.assertTrue(with_memory and all("unknown identifier 'Foo.missing_lemma'" in obs for obs in with_memory))
        without = self.failed_then_retry({'concepts'})
        self.assertFalse(any('K/failed_move' in obs for obs in without))

    def test_meeting_plan_reaches_the_revise_request(self):
        plan = json.dumps({'record': 'r', 'action_plan': {'summary': 'Collapse to a single closing tactic',
                           'steps': ['Use the term True.intro directly']}, 'insights': []})
        backend = Backend({'reduce.critic': '[]', 'reduce.meeting': plan, 'reduce.revise': REVISE})
        with tempfile.TemporaryDirectory() as d:
            CorePipeline(Path(d), LintingVerifier(), backend, model='gpt-6-astra', rounds=1,
                         ablations={'critic.structural'}).problem('01', ENTRY, INITIAL)
        sent = next(obs for unit, obs in backend.calls if unit == 'reduce.revise')
        self.assertIn('Collapse to a single closing tactic', sent)
        self.assertIn('Use the term True.intro directly', sent)
        self.assertIn('M/action_plan', sent)


class LeanLintTests(unittest.TestCase):
    def test_noop_tactic_reported_by_lean_is_deleted_without_a_model_call(self):
        class Mathlibish(LintingVerifier):
            def verify(self, entry, body, measure=True, version=0, options=()):
                m = super().verify(entry, body, measure, version)
                if 'linter.unusedTactic' in options:
                    code = declaration(entry, body)
                    m.warnings = m.warnings + [{'rel_line': i+1, 'col': 2, 'severity': 'warning',
                                                'message': "'skip' tactic does nothing"}
                                               for i, line in enumerate(code.splitlines()) if line.strip() == 'skip']
                return m
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)/'repo'; (repo/'Mathlib').mkdir(parents=True)
            verifier = Mathlibish(); verifier.repos = {'demo': [repo]}
            start = declaration(ENTRY, 'by\n  skip\n  have h : True := True.intro\n  exact h\n')
            backend = Backend({})
            result = CorePipeline(Path(d)/'out', verifier, backend, model='gpt-6-astra', rounds=1, issues_per_round=1,
                                  ablations={'critic.lm', 'meeting'}).problem('01', ENTRY, start)
        self.assertEqual(backend.calls, [])
        self.assertTrue(result['improved']); self.assertNotIn('skip', result['proof'])

    def test_drop_tactic_handles_lines_and_chains(self):
        from rma.reduction.core import drop_tactic
        self.assertEqual(drop_tactic('x := by\n  skip\n  rfl', 2, 2, 'skip'), 'x := by\n  rfl')
        self.assertEqual(drop_tactic('x := by\n  simp <;> done', 2, 10, 'done'), 'x := by\n  simp')
        self.assertIsNone(drop_tactic('x := by\n  rfl', 2, 2, 'skip'))

    def test_linter_options_are_refused_on_measured_compiles(self):
        from rma.reduction.lean import LeanVerifier
        verifier = LeanVerifier({'demo': [Path('.')]}, Path('/usr/bin/false'))
        with self.assertRaises(ValueError):
            verifier.verify(dict(ENTRY, file_path='x'), 'by trivial', measure=True, options=('linter.unusedTactic',))


class HybridTests(unittest.TestCase):
    def test_strategy_retrieval_prefers_the_axis_with_more_headroom(self):
        from rma.reduction import strategies
        body = 'by\n  constructor\n  · grind\n  · aesop'
        heavy = strategies.retrieve(body, length_headroom=5, heartbeat_headroom=20)
        self.assertEqual(heavy[0]['id'], 'explicit-over-shotgun')
        longish = strategies.retrieve(body, length_headroom=20, heartbeat_headroom=5)
        self.assertNotEqual(longish[0]['id'], 'explicit-over-shotgun')

    def test_opening_rewrite_runs_first_and_sees_its_strategy(self):
        backend = Backend({'reduce.revise': REVISE, 'reduce.critic': '[]'})
        with tempfile.TemporaryDirectory() as d:
            result = CorePipeline(Path(d), LintingVerifier(), backend, model='gpt-6-astra', rounds=1,
                                  ablations={'meeting', 'critic.structural'}, rewrite_first=True).problem('01', ENTRY, INITIAL)
        self.assertEqual([u for u, _ in backend.calls][:2], ['reduce.revise', 'reduce.critic'])
        self.assertIn('Strategy [', backend.calls[0][1])
        self.assertTrue(result['improved'])

    def test_local_search_closes_the_round_without_model_calls(self):
        backend = Backend({})
        with tempfile.TemporaryDirectory() as d:
            result = CorePipeline(Path(d), LintingVerifier(), backend, model='gpt-6-astra', rounds=1,
                                  ablations={'critic.lm', 'meeting', 'critic.structural'},
                                  local_search_attempts=6).problem('01', ENTRY, INITIAL)
            state = json.loads((Path(d)/'problems/01/state.json').read_text())
        self.assertEqual(backend.calls, [])
        self.assertTrue(result['improved'])
        search = next(a for a in state['attempts'] if a['unit'] == 'local_search')
        self.assertEqual(search['status'], 'promoted'); self.assertGreater(search['local_attempts'], 0)


class ChainOfStatesTests(unittest.TestCase):
    def test_probe_inserts_trace_state_before_plain_steps_only(self):
        from rma.reduction.core import instrument_states
        probe, anchors = instrument_states('by\n  intro h\n  cases h with\n  | inl a => exact a\n  exact h\n')
        self.assertEqual(anchors, [2, 3, 5])                   # before each plain step; never inside a `| inl` arm
        self.assertEqual(probe.count('trace_state'), 3)

    def test_states_reach_the_critic_and_the_solver(self):
        class Stateful(LintingVerifier):
            def verify(self, entry, body, measure=True, version=0, options=()):
                m = super().verify(entry, body, measure, version)
                if 'trace_state' in body:
                    m.log = ('RMAReduce.lean:2:2: information: ⊢ True\n'
                             'RMAReduce.lean:4:2: information: h : True\n⊢ True\n')
                return m
        backend = Backend({'reduce.critic': CRITIC, 'reduce.solver': PATCH})
        with tempfile.TemporaryDirectory() as d:
            CorePipeline(Path(d), Stateful(), backend, model='gpt-6-astra', rounds=1, issues_per_round=1,
                         ablations={'meeting'}).problem('01', ENTRY, INITIAL)
        critic = next(obs for unit, obs in backend.calls if unit == 'reduce.critic')
        solver = next(obs for unit, obs in backend.calls if unit == 'reduce.solver')
        self.assertIn('Proof states (Lean, before each step', critic)
        self.assertIn('state before line', critic); self.assertIn('⊢ True', critic)
        self.assertIn('Proof states (Lean, before each step', solver)

    def test_uncompilable_probe_is_silently_skipped(self):
        class Broken(LintingVerifier):
            def verify(self, entry, body, measure=True, version=0, options=()):
                m = super().verify(entry, body, measure, version)
                if 'trace_state' in body: m.valid = False
                return m
        from rma.reduction.core import chain_of_states
        self.assertEqual(chain_of_states(Broken(), ENTRY, START), '')
