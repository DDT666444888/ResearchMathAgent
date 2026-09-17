"""Safety and orchestration regressions for the Lean reduction workflow."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from rma.reduction.backend import Ledger, BudgetExceeded, ResponsesBackend
from rma.reduction.lean import extract_body, proof_body, Measurement, declaration
from rma.reduction.pipeline import ReductionPipeline, ReductionStore
from rma.models import ModelRequestError


ENTRY={'name':'demo','source':'demo','statement':'theorem demo : True',
       'src':'theorem demo : True := by trivial\n'}
INITIAL=declaration(ENTRY,'by\n  have h : True := True.intro\n  exact h\n')


class FakeVerifier:
    repos={'demo':[Path('.')]}
    def __init__(self):self.calls=[]
    def verify(self,entry,body,measure=True,version=0):
        import hashlib
        self.calls.append((body,measure))
        good='fail' not in body
        short='exact True.intro' in body
        return Measurement(good,50 if short else 1000,3 if short else 20,
            0 if good else 1,[],'compiler error' if not good else 'audited','test',
            hashlib.sha256(declaration(entry,body).encode()).hexdigest(),measure)


class ReduceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_nested_comments_do_not_hide_admissions(self):
        with self.assertRaises(ValueError):extract_body('by /- x /- nested -/ -/ sorry')
        self.assertEqual(extract_body('```lean\nby exact True.intro\n```'),'by exact True.intro\n')

    def test_statement_and_metaprogramming_gate(self):
        for code in ['by run_tac unsafeCast ()','by native_decide','by\n  exact True.intro\nend X',
                     'by\n  exact True.intro\naxiom bad : False']:
            with self.subTest(code=code),self.assertRaises(ValueError):extract_body(code)
        with self.assertRaises(ValueError):proof_body(ENTRY['statement'],'theorem demo : False := by trivial')

    def test_budget_import_and_concurrency(self):
        path=self.root/'ledger.jsonl';path.write_text('{"reserved_usd":0.4}\n')
        ledger=Ledger(path,1,input_rate=1e6,output_rate=1e6)
        errors=[]
        def reserve():
            try:ledger.reserve('',1)
            except BudgetExceeded:errors.append(True)
        workers=[threading.Thread(target=reserve) for _ in range(5)]
        for w in workers:w.start()
        for w in workers:w.join()
        self.assertEqual(len(errors),5);self.assertEqual(ledger.reserved(),.4)
        # Exactly one reservation fits across competing threads.
        ledger=Ledger(self.root/'second.jsonl',1,input_rate=1e6,output_rate=1e6)
        successes=[]
        def one():
            try:successes.append(ledger.reserve('',1))
            except BudgetExceeded:pass
        workers=[threading.Thread(target=one) for _ in range(5)]
        for w in workers:w.start()
        for w in workers:w.join()
        self.assertEqual(len(successes),1)

    def test_corrupt_ledger_fails_closed(self):
        p=self.root/'broken';p.write_text('{bad')
        with self.assertRaises(json.JSONDecodeError):Ledger(p,100).reserve('x',1)

    def test_responses_secret_not_in_argv_or_artifacts(self):
        secret='secret-for-test';ledger=Ledger(self.root/'usage',10)
        backend=ResponsesBackend(endpoint='https://example.services.ai.azure.com/api/projects/a',
            key=secret,model='gpt-6-astra',ledger=ledger,artifacts=self.root/'api',max_output=100)
        response={'status':'completed','model':'gpt-6-astra','usage':{'input_tokens':1,'output_tokens':2},
                  'output':[{'content':[{'type':'output_text','text':'by trivial'}]}]}
        def request(argv,**kw):
            self.assertNotIn(secret,' '.join(argv));self.assertIn(secret,kw['input'])
            self.assertIn('gpt-6-astra',kw['input'])
            from subprocess import CompletedProcess
            return CompletedProcess(argv,0,json.dumps(response)+'\n200','')
        with patch('rma.reduction.backend.subprocess.run',side_effect=request):
            self.assertEqual(backend('solver','proof'),'by trivial')
        for p in self.root.rglob('*'):
            if p.is_file():self.assertNotIn(secret,p.read_text())
        self.assertEqual(ledger.entries()[0]['status'],'completed')

    def test_uncertain_network_keeps_reservation(self):
        ledger=Ledger(self.root/'usage',10)
        backend=ResponsesBackend(endpoint='https://api.openai.com/v1',key='k',model='gpt-6-astra',
            ledger=ledger,artifacts=self.root/'api',max_output=100)
        with patch('rma.reduction.backend.subprocess.run',side_effect=OSError('network')):
            with self.assertRaises(ModelRequestError):backend('solver','x')
        self.assertGreater(ledger.reserved(),0);self.assertEqual(ledger.entries()[0]['status'],'unknown')

    def test_verified_short_proof_is_not_rejected_by_latex_heuristic(self):
        store=ReductionStore.open(self.root,'01','lean_reduce')
        store.add_proof_revision('by\n'+('  -- padding\n'*100)+'  trivial',produced_by='baseline',meta={'lean_verified':True})
        store.add_proof_revision('by trivial',produced_by='reduce',meta={'lean_verified':True})
        store.add('H','candidate','by sorry')
        self.assertEqual(store.current_proof().body,'by trivial')

    def test_repair_promotion_memory_and_resume_no_replay(self):
        replies=iter(['by fail','by exact True.intro'])
        calls=[]
        def backend(unit,text):calls.append((unit,text));return next(replies)
        verifier=FakeVerifier()
        pipeline=ReductionPipeline(self.root,verifier,backend,model='gpt-6-astra',rounds=1,strategies=1,repairs=1)
        result=pipeline.problem('01',ENTRY,INITIAL)
        self.assertTrue(result['improved']);self.assertIn('compiler error',calls[1][1])
        store=ReductionStore.open(self.root/'problems/01/research','01','lean_reduce')
        self.assertEqual(len(store.proofs),2)
        self.assertGreater(len(store.evaluations),0)
        before=len(calls);result=pipeline.problem('01',ENTRY,INITIAL,resume=True)
        self.assertEqual(len(calls),before);self.assertTrue(result['improved'])
        self.assertTrue(any(not measure for _,measure in verifier.calls))

    def test_invalid_candidate_never_promoted(self):
        pipeline=ReductionPipeline(self.root,FakeVerifier(),lambda u,o:'by fail',
            model='gpt-6-astra',rounds=1,strategies=1,repairs=0)
        result=pipeline.problem('01',ENTRY,INITIAL)
        self.assertFalse(result['improved']);self.assertEqual(result['proof'],INITIAL)

    def test_budget_stop_still_exports_best(self):
        def exhausted(u,o):raise BudgetExceeded('full')
        result=ReductionPipeline(self.root,FakeVerifier(),exhausted,model='gpt-6-astra').problem('01',ENTRY,INITIAL)
        self.assertEqual(result['stop_reason'],'budget_exhausted');self.assertEqual(result['proof'],INITIAL)


class MoreReduceTests(unittest.TestCase):
    def test_nonfinite_budget_is_rejected(self):
        for value in [float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):Ledger(Path('/tmp/unused'),value)
            with self.assertRaises(ValueError):Ledger(Path('/tmp/unused'),100,output_rate=value)

    def test_library_cannot_retrieve_later_target(self):
        from rma.reduction.library import retrieve
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            entry=dict(ENTRY,file_path='Main.lean')
            (root/'Main.lean').write_text('lemma before : True := by trivial\n'+ENTRY['src']+
                                        'lemma after : True := by trivial\n')
            facts=retrieve(entry,'by exact before; exact after',root)
            self.assertIn('lemma before',facts);self.assertNotIn('lemma after',facts)

    def test_mandatory_context_overflow_makes_no_model_call(self):
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def backend(u,o):calls.append(o);return 'by trivial'
            pipeline=ReductionPipeline(Path(d),FakeVerifier(),backend,model='gpt-6-astra',
                                       context_budget=1,rounds=1,strategies=1,repairs=0)
            result=pipeline.problem('01',ENTRY,INITIAL)
            self.assertFalse(result['improved']);self.assertEqual(calls,[])


class SizeVerifier:
    """Stands in for lake+lean: anything without 'fail' compiles, shorter is better."""
    repos={'demo':[Path('.')]}
    def __init__(self):self.calls=[]
    def verify(self,entry,body,measure=True,version=0):
        import hashlib
        from rma.reduction.lean import proxy_tokens
        self.calls.append((body,measure))
        tokens=proxy_tokens(body);ok='fail' not in body
        return Measurement(ok,10*tokens,tokens,0 if ok else 1,[],
            'compiler error' if not ok else 'audited','test',
            hashlib.sha256(declaration(entry,body).encode()).hexdigest(),measure)


class LedgerSettlementTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def ledger(self,name,**kw):
        return Ledger(self.root/name,10,input_rate=1e6,output_rate=1e6,**kw)

    def test_completed_call_settles_hold_down_to_metered_usage(self):
        ledger=self.ledger('a',settle_input_rate=1,settle_output_rate=1)
        rec=ledger.reserve('',1)
        self.assertAlmostEqual(ledger.reserved(),1.0)
        ledger.finish(rec['id'],status='completed',settle_usage={'input_tokens':10,'output_tokens':20})
        entry=ledger.entries()[0]
        self.assertTrue(entry['settled']);self.assertAlmostEqual(entry['estimate_usd'],1.0)
        self.assertAlmostEqual(ledger.reserved(),30/1e6)
        self.assertEqual(ledger.summary()['unsettled_calls'],0)

    def test_unknown_or_implausible_usage_keeps_the_upper_bound(self):
        ledger=self.ledger('b',settle_input_rate=1,settle_output_rate=1)
        lost=ledger.reserve('',1);ledger.finish(lost['id'],status='unknown')
        absurd=ledger.reserve('',1)
        ledger.finish(absurd['id'],status='completed',settle_usage={'output_tokens':10**9,'input_tokens':0})
        partial=ledger.reserve('',1)
        ledger.finish(partial['id'],status='completed',settle_usage={'output_tokens':'many'})
        self.assertAlmostEqual(ledger.reserved(),3.0)
        self.assertEqual(ledger.summary()['unsettled_calls'],3)

    def test_per_problem_scope_cannot_drain_the_shared_budget(self):
        ledger=self.ledger('c',scope_limit=1.5)
        ledger.reserve('',1,scope='01')
        with self.assertRaises(BudgetExceeded):ledger.reserve('',1,scope='01')
        ledger.reserve('',1,scope='02')          # a different problem still has its share
        self.assertAlmostEqual(ledger.reserved(),2.0)
        with self.assertRaises(ValueError):self.ledger('d',scope_limit=0)

    def test_probe_is_a_small_call_and_leaves_the_run_budget_intact(self):
        ledger=Ledger(self.root/'probe',10)
        backend=ResponsesBackend(endpoint='https://example.services.ai.azure.com/api/projects/p',
            key='k',model='gpt-6-astra',ledger=ledger,artifacts=self.root/'api',max_output=12000)
        sent={}
        def request(argv,**kw):
            from subprocess import CompletedProcess
            sent['body']=json.loads(kw['input'].split('data = ',1)[1].strip().strip('"')
                                    .encode().decode('unicode_escape'))
            payload={'status':'completed','usage':{'input_tokens':11,'output_tokens':5},
                     'output':[{'content':[{'type':'output_text','text':'OK'}]}]}
            return CompletedProcess(argv,0,json.dumps(payload)+'\n200','')
        with patch('rma.reduction.backend.subprocess.run',side_effect=request):
            report=backend.probe()
        self.assertEqual(report['reply'],'OK')
        self.assertEqual(sent['body']['max_output_tokens'],2000)
        self.assertEqual(sent['body']['reasoning']['effort'],'low')
        self.assertEqual(backend.max_output,12000)     # the probe never mutates the run backend
        self.assertLess(ledger.reserved(),0.01)


class DiagnosticsTests(unittest.TestCase):
    LOG=('/repo/RMAReduce1.lean:12:4: error: unknown identifier "foo"\n  context line\n'
         '/repo/RMAReduce1.lean:20:0: warning: unused variable\n'
         '/repo/RMAReduce1.lean:13:2: error: unsolved goals\n')

    def test_errors_are_anchored_to_the_submitted_declaration(self):
        from rma.reduction.lean import diagnostics
        found=diagnostics(self.LOG,base_line=10)
        self.assertEqual([d['rel_line'] for d in found],[2,3])
        self.assertIn('unknown identifier',found[0]['message'])
        self.assertTrue(all(d['severity']=='error' for d in found))

    def test_feedback_points_at_the_failing_line_and_falls_back_to_the_log(self):
        from rma.reduction.lean import diagnostics
        code='theorem t : True :=\nby\n  exact foo\n'
        located=Measurement(False,None,3,1,[],self.LOG,'t','sha',True,diagnostics(self.LOG,10))
        text=located.feedback(code)
        self.assertIn('>>>',text);self.assertIn('exact foo',text);self.assertIn('unknown identifier',text)
        silent=Measurement(False,None,3,1,[],'deterministic timeout','t','sha',True,[])
        self.assertIn('deterministic timeout',silent.feedback(code))


class SearchPolicyTests(unittest.TestCase):
    def test_bandit_explores_untried_then_exploits_measured_gain(self):
        from rma.reduction.pipeline import select_strategies
        self.assertEqual(select_strategies({'0':{'n':1,'gain':-5.}},2),[1,2])
        stats={'0':{'n':3,'gain':30.},'1':{'n':3,'gain':-9.},'2':{'n':3,'gain':0.}}
        self.assertEqual(select_strategies(stats,1),[0])
        self.assertNotIn(1,select_strategies(stats,2))
        # The ablation restores the fixed portfolio order.
        self.assertEqual(select_strategies(stats,2,bandit=False),[0,1])


class BeamAndPlaybookTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def script(self,replies):
        calls=[]
        answers=iter(replies)
        def backend(unit,observation):calls.append(observation);return next(answers)
        return backend,calls

    def test_second_slot_explores_an_alternate_verified_parent(self):
        backend,calls=self.script(['by\n  exact True.intro\n','by\n  apply True.intro\n',
                                   'by\n  exact trivial\n','by\n  exact id True.intro\n'])
        pipeline=ReductionPipeline(self.root,SizeVerifier(),backend,model='gpt-6-astra',
                                   rounds=2,strategies=2,repairs=0,beam=2)
        result=pipeline.problem('01',ENTRY,INITIAL)
        self.assertTrue(result['improved']);self.assertEqual(result['pool_size'],2)
        second_round=calls[3]
        self.assertIn('apply True.intro',second_round)
        self.assertIn('alternative verified proof',second_round)
        self.assertNotIn('alternative verified proof',calls[2])

    def test_greedy_ablation_always_reduces_the_current_best(self):
        backend,calls=self.script(['by\n  exact True.intro\n','by\n  apply True.intro\n',
                                   'by\n  exact trivial\n','by\n  exact id True.intro\n'])
        pipeline=ReductionPipeline(self.root,SizeVerifier(),backend,model='gpt-6-astra',
                                   rounds=2,strategies=2,repairs=0,beam=2,ablate={'beam'})
        pipeline.problem('01',ENTRY,INITIAL)
        self.assertTrue(all('alternative verified proof' not in c for c in calls))

    def test_unknown_ablation_is_rejected(self):
        with self.assertRaises(ValueError):
            ReductionPipeline(self.root,SizeVerifier(),lambda u,o:'by trivial',
                              model='gpt-6-astra',ablate={'telepathy'})

    def test_promoted_moves_reach_the_next_problem_unless_ablated(self):
        from rma.reduction.playbook import Playbook
        book=Playbook(self.root/'playbook.jsonl')
        backend,calls=self.script(['by\n  exact True.intro\n','by\n  exact trivial\n'])
        pipeline=ReductionPipeline(self.root,SizeVerifier(),backend,model='gpt-6-astra',
                                   rounds=1,strategies=1,repairs=0,playbook=book)
        pipeline.problem('01',ENTRY,INITIAL)
        pipeline.problem('02',ENTRY,INITIAL)
        self.assertIn('Verified reduction moves',calls[1])
        self.assertIn('exact',calls[1])
        self.assertNotIn('exact True.intro\n\nProof body',calls[1])  # vocabulary, not the proof
        lessons=book.entries()
        self.assertEqual([r['id'] for r in lessons],['01','02'])
        self.assertTrue(all(r['tokens_gain']<0 for r in lessons))
        self.assertIn('01',book.lessons(exclude='02'));self.assertNotIn('02',book.lessons(exclude='02'))
        silent,quiet=self.script(['by\n  exact trivial\n'])
        ReductionPipeline(self.root,SizeVerifier(),silent,model='gpt-6-astra',rounds=1,
                          strategies=1,repairs=0,playbook=book,ablate={'playbook'}).problem('03',ENTRY,INITIAL)
        self.assertNotIn('Verified reduction moves',quiet[0])

    def test_localized_compiler_feedback_reaches_the_repair_turn(self):
        log='/repo/RMAReduce1.lean:2:2: error: unknown identifier "fail"\n'
        class Failing(SizeVerifier):
            def verify(self,entry,body,measure=True,version=0):
                m=super().verify(entry,body,measure,version)
                if not m.valid:
                    from rma.reduction.lean import diagnostics
                    m.log=log;m.errors=diagnostics(log,0)
                return m
        backend,calls=self.script(['by\n  exact fail\n','by\n  exact True.intro\n'])
        result=ReductionPipeline(self.root,Failing(),backend,model='gpt-6-astra',rounds=1,
                                 strategies=1,repairs=1).problem('01',ENTRY,INITIAL)
        self.assertTrue(result['improved'])
        self.assertIn('Compiler errors located',calls[1]);self.assertIn('>>>',calls[1])


class ReportTests(unittest.TestCase):
    def run_dir(self, root, name, *, ablate, tokens, held=0.):
        folder=root/name;folder.mkdir(parents=True)
        (folder/'results.json').write_text(json.dumps({
            'plan':{'beam':2,'ablate':ablate},
            'ledger':{'reserved_usd':1.25,'held_usd':held,'unsettled_calls':1 if held else 0},
            'errors':[{'name':'broken','error':'baseline does not compile'}],
            'results':[{'id':'01','improved':tokens<20,'attempts':3,'promoted':1,
                        'stop_reason':'stalled','baseline':{'tokens_proxy':20,'heartbeats':100},
                        'measurement':{'tokens_proxy':tokens,'heartbeats':80}}]}))
        return folder

    def test_report_compares_runs_without_inventing_scores(self):
        from rma.reduction.report import load, render
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            full=self.run_dir(root,'full',ablate=[],tokens=10,held=0.5)
            ablated=self.run_dir(root,'no-beam',ablate=['beam'],tokens=20)
            text=render(load(full),[load(ablated)])
            self.assertIn('| full | none | beam 2 | 1/1 | -50.0% | -20.0% | 1.2500 | 0.5000 |',text)
            self.assertIn('| no-beam | beam | beam 2 | 0/1 |',text)
            self.assertIn('baseline does not compile',text)
            self.assertIn('public Arena rule on local measurements',text)
            self.assertNotIn('official_score',text)

    def test_missing_measurements_do_not_fabricate_a_change(self):
        from rma.reduction.report import gains
        rows=gains({'results':[{'id':'01','baseline':{'tokens_proxy':0,'heartbeats':None},
                                'measurement':{'tokens_proxy':5,'heartbeats':None}}]})
        self.assertIsNone(rows[0]['tokens']);self.assertIsNone(rows[0]['heartbeats'])


class ResumeAfterBudgetStopTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_refused_repair_is_retried_with_its_compiler_context(self):
        def first(unit,obs):
            if 'Failed candidate' in obs:raise BudgetExceeded('scope full')
            return 'by\n  exact fail\n'
        pipe=ReductionPipeline(self.root,SizeVerifier(),first,model='gpt-6-astra',
                               rounds=1,strategies=1,repairs=1)
        self.assertEqual(pipe.problem('01',ENTRY,INITIAL)['stop_reason'],'budget_exhausted')
        calls=[]
        def second(unit,obs):calls.append(obs);return 'by\n  exact True.intro\n'
        pipe.backend=second
        result=pipe.problem('01',ENTRY,INITIAL,resume=True)
        self.assertEqual(len(calls),1)
        self.assertIn('Failed candidate',calls[0]);self.assertIn('exact fail',calls[0])
        self.assertIn('compiler error',calls[0])
        self.assertTrue(result['improved']);self.assertEqual(result['stop_reason'],'round_limit')

    def test_finished_slots_are_not_reopened_after_a_budget_stop(self):
        replies=iter(['by\n  exact True.intro\n'])
        def first(unit,obs):
            try:return next(replies)
            except StopIteration:raise BudgetExceeded('full')
        pipe=ReductionPipeline(self.root,SizeVerifier(),first,model='gpt-6-astra',
                               rounds=1,strategies=2,repairs=1)
        pipe.problem('01',ENTRY,INITIAL)
        calls=[]
        def second(unit,obs):calls.append(obs);return 'by\n  trivial\n'
        pipe.backend=second
        pipe.problem('01',ENTRY,INITIAL,resume=True)
        self.assertEqual(len(calls),1)          # only the refused second slot runs


class SubsetResumeCliTests(unittest.TestCase):
    """A run resumed on a different subset keeps every earlier promotion in its export."""
    def test_subset_resume_exports_all_promotions(self):
        import os
        from rma.cli import build_parser
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);repo=root/'repo';repo.mkdir()
            srcs=[f'theorem t{i} (a b : Nat) : a + b = b + a := by\n  rw [Nat.add_comm]\n' for i in (1,2)]
            (repo/'Demo.lean').write_text('import Mathlib\n\n'+'\n'.join(srcs))
            (repo/'lean-toolchain').write_text('leanprover/lean4:v4.26.0\n')
            lake=root/'lake';lake.write_text('#!/bin/sh\necho "Used 100 heartbeats"\n'
                                             'echo "does not depend on any axioms"\n')
            os.chmod(lake,0o755)
            entries=[{'name':f't{i}','source':'demo','file_path':'Demo.lean','statement':s.split(' :=')[0],
                      'src':s} for i,s in zip((1,2),srcs)]
            (root/'bench.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in entries))
            (root/'base.jsonl').write_text(''.join(json.dumps({'name':e['name'],'proof':e['src']})+'\n' for e in entries))
            class Fake:
                def __call__(self,unit,obs):return 'by\n  omega\n'
                def scoped(self,pid):return self
                def probe(self):return {'model':'gpt-6-astra','url':'test'}
            def run(ids,resume):
                # This exercises the CLI's subset-resume export, which is mode-independent; pin the
                # flat pipeline it was written for (the rma path is covered by tests/test_reduce_core.py).
                argv=['reduce','--mode','flat','--benchmark',str(root/'bench.jsonl'),'--baseline',str(root/'base.jsonl'),
                      '--repo',f'demo={repo}','--lake',str(lake),'--credentials-file',str(root/'unused'),
                      '--rounds','1','--strategies','1','--repairs','0','--out',str(root/'run'),'--ids',*ids]
                args=build_parser().parse_args(argv+(['--resume'] if resume else []))
                with patch('rma.reduction.cli.make_backend',return_value=Fake()):
                    return args.func(args)
            self.assertEqual(run(['01'],False),0)
            self.assertEqual(run(['02'],True),0)
            rows=[json.loads(l) for l in (root/'run/submission.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['proof'].rstrip().endswith('omega') for r in rows))
            results=json.loads((root/'run/results.json').read_text())
            self.assertEqual(sorted(r['id'] for r in results['results']),['01','02'])
            self.assertEqual(results['improved'],2)


class LocateDeclarationTests(unittest.TestCase):
    """Another listed version may prove the same statement differently."""
    STMT='theorem t (n : Nat) : n + 0 = n'
    ENTRY={'name':'t','statement':STMT,'src':STMT+' := by\n  simp\n'}

    def test_pinned_text_is_used_when_present(self):
        from rma.reduction.lean import locate_declaration
        self.assertEqual(locate_declaration('import X\n'+self.ENTRY['src']+'theorem u : True := trivial\n',
                                            self.ENTRY),self.ENTRY['src'])

    def test_older_proof_is_found_by_statement_and_ends_at_next_command(self):
        from rma.reduction.lean import locate_declaration
        older=(self.STMT+' := by\n  induction n with\n  | zero => rfl\n  | succ k ih => rfl\n'
               'termination_by n\n\n/-- next -/\ntheorem u : True := trivial\n')
        found=locate_declaration('import X\n'+older,self.ENTRY)
        self.assertTrue(found.startswith(self.STMT+' := by'))
        self.assertIn('termination_by n',found);self.assertNotIn('theorem u',found);self.assertNotIn('next',found)
        at_end=locate_declaration('import X\n'+self.STMT+' := by\n  omega',self.ENTRY)
        self.assertTrue(at_end.endswith('omega'))

    def test_changed_or_duplicated_statement_is_refused(self):
        from rma.reduction.lean import locate_declaration
        with self.assertRaises(ValueError):
            locate_declaration('theorem t (m : Nat) : m + 0 = m := by simp\n',self.ENTRY)
        with self.assertRaises(ValueError):
            locate_declaration(self.STMT+' := by simp\n'+self.STMT+' := by omega\n',self.ENTRY)


class AttemptReportTests(unittest.TestCase):
    def test_report_lists_every_attempt_failure_cost_and_net_gain(self):
        from rma.reduction.report import load, render
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'arm';(run/'problems/02').mkdir(parents=True)
            (run/'results.json').write_text(json.dumps({'plan':{'beam':2,'ablate':[]},'ledger':{'reserved_usd':3.5},
                'results':[],'estimated_score':{'run_lower':79.2,'run_upper':79.2,'baseline_run_lower':79.06,
                'baseline_run_upper':79.06,'net_gain_lower':0.14,'net_gain_upper':0.14}}))
            (run/'problems/02/state.json').write_text(json.dumps({'attempts':[
                {'tag':'r1-s1-a0','strategy':0,'status':'promoted','score':{'score':70.1},
                 'measurement':{'tokens_arena':190,'heartbeats':600}},
                {'tag':'r1-s2-a0','strategy':1,'status':'measured','measurement':{'tokens_arena':180,'heartbeats':None}},
                {'tag':'r2-s1-a0','strategy':2,'status':'rejected','error':'Disallowed Lean construct in proof body'}]}))
            (run/'usage.jsonl').write_text(json.dumps({'scope':'02','reserved_usd':2.25})+'\n'+
                                          json.dumps({'scope':None,'reserved_usd':0.01})+'\n')
            text=render(load(run),[])
            self.assertIn('| 02 | r1-s1-a0 | 0 | promoted | 190 | 600 | 70.10 |',text)
            self.assertIn('| 02 | r2-s1-a0 | 2 | rejected | - | - | - | Disallowed Lean construct',text)
            self.assertIn('02 2.2500',text)
            self.assertIn('net change against its starting proofs +0.14 to +0.14 points',text)


class ModuleHeaderAuditTests(unittest.TestCase):
    """`#print axioms` is illegal inside a `module`; that must not read as a proof failure."""
    SRC = 'theorem t : True := by trivial\n'
    ENTRY = {'name': 't', 'source': 'demo', 'file_path': 'M.lean',
             'statement': 'theorem t : True', 'src': SRC}

    def build(self, root, stub_output):
        import os
        repo = root/'repo'; repo.mkdir()
        (repo/'M.lean').write_text('module\n\n' + self.SRC)
        (repo/'lean-toolchain').write_text('leanprover/lean4:v4.29.1\n')
        lake = root/'lake'; lake.write_text('#!/bin/sh\n' + stub_output + '\nexit 0\n'); os.chmod(lake, 0o755)
        from rma.reduction.lean import LeanVerifier
        return LeanVerifier({'demo': [repo, repo]}, lake)

    def test_cross_version_check_compiles_without_the_axiom_audit(self):
        with tempfile.TemporaryDirectory() as d:
            verifier = self.build(Path(d), 'true')            # compiler prints nothing at all
            checked = verifier.verify(self.ENTRY, 'by trivial', measure=False, version=1)
            self.assertTrue(checked.valid)                    # compiled unchanged = what zero-shot measures
            self.assertFalse(checked.audited)

    def test_pinned_version_still_requires_the_audit(self):
        with tempfile.TemporaryDirectory() as d:
            verifier = self.build(Path(d), 'true')
            pinned = verifier.verify(self.ENTRY, 'by trivial', measure=False, version=0)
            self.assertFalse(pinned.valid)                    # no audit line emitted ⇒ not promotable
            self.assertTrue(pinned.audited)

    def test_audit_still_gates_forbidden_axioms(self):
        with tempfile.TemporaryDirectory() as d:
            verifier = self.build(Path(d), "echo \"'t' depends on axioms: [sorryAx]\"")
            pinned = verifier.verify(self.ENTRY, 'by trivial', measure=False, version=0)
            self.assertFalse(pinned.valid); self.assertEqual(pinned.axioms, ['sorryAx'])
