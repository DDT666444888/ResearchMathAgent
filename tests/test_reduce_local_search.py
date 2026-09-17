import tempfile
import unittest
from pathlib import Path
from rma.reduction.local_search import candidates, LocalSearch
from rma.reduction.lean import Measurement, declaration
import hashlib

ENTRY={'name':'demo','source':'demo','statement':'theorem demo : True','src':'theorem demo : True := by trivial\n'}
BODY='by\n  classical\n  exact True.intro\n'
class Verifier:
    repos={'demo':[Path('v1'),Path('v2')]}
    def __init__(self,fail_version=False):self.fail_version=fail_version;self.calls=[]
    def verify(self,entry,body,measure=True,version=0):
        self.calls.append((body,measure,version))
        valid='exact True.intro' in body and not (self.fail_version and version==1)
        return Measurement(valid,10, len(body),0 if valid else 1,[], 'kernel',str(version),hashlib.sha256(declaration(entry,body).encode()).hexdigest(),measure)
class LocalTests(unittest.TestCase):
    def test_simple_simp_arguments_only(self):
        edits=list(candidates('by\n  simp only [foo, bar]\n'))
        self.assertTrue(any('[bar]' in x.body for x in edits))
        self.assertTrue(any('[foo]' in x.body for x in edits))
        edits=list(candidates('by\n  simp [f (a, b), c]\n'))
        self.assertFalse(any('simp argument' in x.label for x in edits))
    def test_empty_simp_chain_stage(self):
        body='by\n  induction h <;>\n    simp only [] <;>\n    assumption\n'
        edits=list(candidates(body))
        self.assertTrue(any(x.label=='drop empty simp stage' and 'simp' not in x.body
                            and 'induction h <;>' in x.body for x in edits))
    def test_kernel_gate_and_all_versions(self):
        with tempfile.TemporaryDirectory() as d:
            v=Verifier();r=LocalSearch(v,lambda m:-m.tokens_proxy,max_attempts=5).run(ENTRY,BODY,Path(d))
            self.assertTrue(r['improved']);self.assertNotIn('classical',r['proof']);self.assertIn('exact True.intro',r['proof'])
            self.assertTrue(any(not measure and version==1 for _,measure,version in v.calls))
    def test_failed_version_never_promoted(self):
        with tempfile.TemporaryDirectory() as d:
            r=LocalSearch(Verifier(True),lambda m:-m.tokens_proxy,max_attempts=5).run(ENTRY,BODY,Path(d))
            self.assertFalse(r['improved'])
    def test_budget_and_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            search=LocalSearch(Verifier(),lambda m:-m.tokens_proxy,max_attempts=0)
            r=search.run(ENTRY,BODY,Path(d));self.assertEqual(r['attempts'],[])
            with self.assertRaises(ValueError):search.run(ENTRY,BODY,Path(d))
