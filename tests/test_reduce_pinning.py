"""Heartbeats are counted on the toolchain the Arena measures THIS problem on.

Configured repositories are ordered by `--repo` on the command line, which has nothing to do with the
version the Arena grades a problem on. The same proof can differ by tens of heartbeats between listed
versions (problem 12 reads 179 heartbeats on v4.26.0 against an official 229 on v4.29.1), so measuring
on the wrong one makes a local margin meaningless in either direction.
"""
import hashlib
import tempfile
import unittest
from pathlib import Path

from rma.reduction.lean import Measurement, declaration
from rma.reduction.pipeline import ReductionPipeline
from rma.reduction.score import arena_tokens

VERSIONS = ['v4.25.0', 'v4.26.0', 'v4.27.0']
ENTRY = {'name': 'demo', 'source': 'demo', 'statement': 'theorem demo : True',
         'src': 'theorem demo : True := by trivial\n',
         'version_info': [['v4.25.0', 'v4.26.0'], ['v4.27.0']]}
START = 'by\n  trivial\n'
INITIAL = declaration(ENTRY, START)


class RecordingVerifier:
    """Records which repository index every call used; heartbeats differ per version, so the
    measurement itself reveals which toolchain produced the score."""

    def __init__(self, repos):
        self.repos = {'demo': repos}
        self.measured, self.checked = [], []

    def verify(self, entry, body, measure=True, version=0, options=()):
        (self.measured if measure else self.checked).append(version)
        code = declaration(entry, body)
        tokens = arena_tokens(body)
        return Measurement(True, 1000 * (version + 1), tokens, 0, [], 'log', VERSIONS[version],
                           hashlib.sha256(code.encode()).hexdigest(), measure, [], 0, tokens, [])


class CountingBackend:
    """Records every model call, so a refusal can be shown to happen before anything is bought."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, unit, observation):
        self.calls.append(unit)
        return self.reply


class PinnedMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.repos = []
        for v in VERSIONS:
            repo = Path(self.dir.name) / v
            repo.mkdir()
            (repo / 'lean-toolchain').write_text(f'leanprover/lean4:{v}\n')
            self.repos.append(repo)
        self.addCleanup(self.dir.cleanup)

    def build(self, pins, tag):
        verifier = RecordingVerifier(self.repos)
        backend = CountingBackend(START)
        pipeline = ReductionPipeline(Path(self.dir.name) / ('run-' + tag), verifier, backend,
                                     model='gpt-6-astra', rounds=1, strategies=1, repairs=0)
        pipeline.measure_versions = pins
        return pipeline, verifier, backend

    def run_problem(self, pins):
        pipeline, verifier, backend = self.build(pins, '-'.join(pins.values() or ['none']))
        self.backend = backend
        return verifier, pipeline.problem('12', ENTRY, INITIAL)

    def test_pin_naming_an_unconfigured_version_is_refused_not_silently_defaulted(self):
        # The whole point of the pin is to stop measurement landing on index 0 by accident; an
        # explicit but unusable pin must therefore fail, not fall back to the very default it replaces.
        pipeline, verifier, backend = self.build({'12': 'v9.9.9'}, 'invalid')
        with self.assertRaises(ValueError) as caught:
            pipeline.problem('12', ENTRY, INITIAL)
        self.assertIn('v9.9.9', str(caught.exception))
        # Nothing was compiled and nothing was bought before the refusal.
        self.assertEqual([], verifier.measured)
        self.assertEqual([], verifier.checked)
        self.assertEqual([], backend.calls)

    def test_pinned_problem_is_measured_on_its_pinned_toolchain(self):
        verifier, _ = self.run_problem({'12': 'v4.27.0'})
        self.assertTrue(verifier.measured, 'no measured compile happened')
        # Index 2 is v4.27.0; without the pin this would be index 0 purely because of --repo order.
        self.assertEqual(set(verifier.measured), {2})

    def test_cross_version_checks_skip_the_pinned_version_not_index_zero(self):
        verifier, _ = self.run_problem({'12': 'v4.27.0'})
        # The pinned version is already covered by the measured compile, so the compatibility
        # checks must cover the OTHER listed versions instead of blindly skipping index 0.
        self.assertNotIn(2, verifier.checked)
        self.assertEqual({0, 1}, set(verifier.checked) & {0, 1, 2})

    def test_baseline_versions_lead_with_the_measured_toolchain(self):
        self.run_problem({'12': 'v4.27.0'})
        self.assertEqual('v4.27.0', self.read_state()['baseline_versions'][0])

    def read_state(self):
        import json
        run = next(p for p in Path(self.dir.name).glob('run-*/problems/12/state.json'))
        return json.loads(run.read_text())

    def test_unpinned_problem_still_defaults_to_the_first_configured_repo(self):
        verifier, _ = self.run_problem({})
        self.assertEqual({0}, set(verifier.measured))
        self.assertNotIn(0, verifier.checked)


class PinValidationCliTests(unittest.TestCase):
    """A pin the run cannot honour is refused by the CLI, before preflight and before a plan is written.

    The pipeline-level guard above is the backstop; this is the gate that has to fail while the user is
    still looking at the command, including under --dry-run.
    """

    def setUp(self):
        import json
        import os
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        repo = root / 'repo'
        repo.mkdir()
        src = 'theorem t1 (a b : Nat) : a + b = b + a := by\n  rw [Nat.add_comm]\n'
        (repo / 'Demo.lean').write_text('import Mathlib\n\n' + src)
        (repo / 'lean-toolchain').write_text('leanprover/lean4:v4.26.0\n')
        lake = root / 'lake'
        lake.write_text('#!/bin/sh\necho "Used 100 heartbeats"\necho "does not depend on any axioms"\n')
        os.chmod(lake, 0o755)
        # v4.27.0 is LISTED by the Arena but no --repo provides it: the two halves of the check differ.
        entry = {'name': 't1', 'source': 'demo', 'file_path': 'Demo.lean', 'src': src,
                 'statement': src.split(' :=')[0], 'version_info': [['v4.26.0', 'v4.27.0']]}
        (root / 'bench.jsonl').write_text(json.dumps(entry) + '\n')
        (root / 'base.jsonl').write_text(json.dumps({'name': 't1', 'proof': src}) + '\n')
        self.root, self.repo, self.lake = root, repo, lake

    def run_cli(self, *extra):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from rma.cli import build_parser

        class Fake:
            def __init__(self):
                self.probes = 0

            def __call__(self, unit, observation):
                return 'by\n  omega\n'

            def scoped(self, pid):
                return self

            def probe(self):
                self.probes += 1
                return {'model': 'gpt-6-astra', 'url': 'test'}

        fake = Fake()
        argv = ['reduce', '--mode', 'flat', '--benchmark', str(self.root / 'bench.jsonl'),
                '--baseline', str(self.root / 'base.jsonl'), '--repo', f'demo={self.repo}',
                '--lake', str(self.lake), '--credentials-file', str(self.root / 'unused'),
                '--rounds', '1', '--strategies', '1', '--repairs', '0',
                '--out', str(self.root / 'run'), *extra]
        args = build_parser().parse_args(argv)
        # A rejected run is reported by printing and returning 1, not by propagating the exception.
        printed = io.StringIO()
        with patch('rma.reduction.cli.make_backend', return_value=fake), redirect_stdout(printed):
            return args.func(args), printed.getvalue(), fake

    def assert_refused(self, message, *extra):
        code, printed, fake = self.run_cli('--dry-run', '--measure-version', *extra)
        self.assertEqual(1, code)
        self.assertIn(message, printed)
        # No plan was written and the backend was never probed: the refusal precedes preflight,
        # so a bad pin cannot reach a paid call.
        self.assertFalse((self.root / 'run/plan.json').exists())
        self.assertEqual(0, fake.probes)

    def test_dry_run_refuses_a_toolchain_the_arena_does_not_list(self):
        self.assert_refused('v9.9.9', '01=v9.9.9')

    def test_dry_run_refuses_a_listed_toolchain_that_no_repo_provides(self):
        self.assert_refused('v4.27.0', '01=v4.27.0')

    def test_unknown_problem_id_is_still_refused(self):
        self.assert_refused('unknown problem id', '99=v4.26.0')

    def test_a_configured_and_listed_pin_is_accepted_and_recorded(self):
        import json
        code, _, fake = self.run_cli('--dry-run', '--measure-version', '01=v4.26.0')
        self.assertEqual(0, code)
        self.assertEqual({'01': 'v4.26.0'},
                         json.loads((self.root / 'run/plan.json').read_text())['measure_versions'])


if __name__ == '__main__':
    unittest.main()
