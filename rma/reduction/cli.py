"""CLI entry point; outputs are isolated from historical research solutions."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
from pathlib import Path
import zipfile
from .backend import Ledger, ResponsesBackend, atomic_json
from .lean import LeanVerifier, declaration, proof_body
from .pipeline import ABLATIONS, ReductionPipeline
from .playbook import Playbook
from .score import listed_versions, parse_official, run_score
from webapp.locks import file_lock


def register(subparsers):
    p=subparsers.add_parser('reduce',help='Reduce Lean proofs with RMA state, GPT Responses, and kernel feedback.')
    p.add_argument('--benchmark',type=Path,help='Arena benchmark JSONL, including exact source declarations.')
    p.add_argument('--baseline',type=Path,help='Explicit user-provided best submission JSONL.')
    p.add_argument('--baseline-results',type=Path,help='Official results JSON for calibrated local selection.')
    p.add_argument('--manifest',type=Path,help='Map stable problem ids to names; defaults to baseline sibling manifest.json.')
    p.add_argument('--repo',action='append',default=[],metavar='CORPUS=PATH',help='Pinned source repo; repeat a corpus for cross-version checks.')
    p.add_argument('--measure-version',action='append',default=[],metavar='ID=VERSION',
                   help='Count heartbeats for this problem on the given toolchain, i.e. the version the Arena '
                        'measures it on. Default: the corpus\'s first --repo, which is an arbitrary choice and can '
                        'differ from the graded version by tens of heartbeats.')
    p.add_argument('--lake',type=Path,help='lake executable of the pinned toolchain.')
    p.add_argument('--elan-home',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ids',nargs='+',help='Manifest ids or exact theorem names; default all.')
    p.add_argument('--credentials-file',type=Path,help='External endpoint/key file; never copied into artifacts.')
    p.add_argument('--model',default='gpt-6-astra',help='Exact deployment name; never silently substituted.')
    p.add_argument('--effort',choices=['low','medium','high','xhigh','max'],default='high')
    p.add_argument('--ledger',type=Path,help='Shared reservation JSONL; reuse to continue an existing budget.')
    p.add_argument('--budget-usd',type=float,default=100)
    p.add_argument('--per-problem-usd',type=float,help='Cap one problem so it cannot drain the shared budget.')
    p.add_argument('--input-rate',type=float,default=100,help='Conservative USD per million input-token upper bound.')
    p.add_argument('--output-rate',type=float,default=500,help='Conservative USD per million maximum output tokens.')
    p.add_argument('--settle-input-rate',type=float,help='USD per million ACTUAL input tokens used to settle a completed call.')
    p.add_argument('--settle-output-rate',type=float,help='USD per million ACTUAL output tokens used to settle a completed call.')
    p.add_argument('--max-output-tokens',type=int,default=12000)
    p.add_argument('--context-budget',type=int,default=24000)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--strategies',type=int,choices=[1,2,3],default=2)
    p.add_argument('--repairs',type=int,default=1)
    p.add_argument('--beam',type=int,default=2,help='Kernel-verified parents kept per problem; 1 is greedy hill-climbing.')
    p.add_argument('--ablate',action='append',default=[],choices=list(ABLATIONS),
                   help='Disable one reduction mechanism for a controlled comparison; repeatable.')
    p.add_argument('--mode',choices=['flat','rma'],default='rma',
                   help='rma (default): Algorithm 1 — a strategy-driven rewrite each round, critic issues from Lean '
                        'lints/proof states/an LM critic, localized patches with one compiler-feedback repair, grounded '
                        'library lookup, meeting, revise, concepts, and API-free local search. flat: the older '
                        'portfolio of full rewrites with beam/bandit/playbook.')
    p.add_argument('--issues-per-round',type=int,default=2,help='rma mode: issues the solver patches per round (b).')
    p.add_argument('--rma-ablate',action='append',default=[],
                   choices=['critic.lm','critic.structural','literature','meeting','concepts','insights','evaluator-feedback'],
                   help='rma mode: switch one RMA unit off for a controlled comparison; repeatable.')
    p.add_argument('--no-rewrite-first',action='store_true',help='rma mode: skip the strategy-driven opening rewrite.')
    p.add_argument('--local-search-attempts',type=int,default=8,
                   help='rma mode: API-free local-search compiles on each round\'s best (0 disables).')
    p.add_argument('--playbook',type=Path,help='Shared cross-problem lesson file; defaults to OUT/playbook.jsonl.')
    p.add_argument('--jobs',type=int,default=1,help='Concurrent problem workers; each has an isolated store and Lean file.')
    p.add_argument('--timeout',type=int,default=90,help='Lean process-group timeout in seconds.')
    p.add_argument('--api-timeout',type=int,default=900)
    p.add_argument('--hints',type=Path,help='Explicit JSON map id -> scoped library facts allowed in model context.')
    p.add_argument('--local-context',action='store_true',help='Retrieve only referenced declarations preceding the target in its own source file.')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Validate inputs and emit a plan without compiling or calling API.')
    p.add_argument('--probe',action='store_true',help='Make one minimal live call to check endpoint, key and deployment, then exit.')
    p.add_argument('--report',action='store_true',help='Render OUT/report.md from finished runs; no API call, no Lean.')
    p.add_argument('--compare',action='append',default=[],type=Path,help='Further finished run directories for the summary table; repeatable.')
    p.add_argument('--no-preflight',action='store_true',help='Skip the automatic pre-run endpoint probe.')
    p.set_defaults(func=run_reduce)


def load_rows(path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({x['name'] for x in rows})!=len(rows): raise ValueError(f'Duplicate theorem in {path}')
    return rows


def run_reduce(args):
    try:
        if args.report: return _report(args)
        return _probe(args) if args.probe else _run(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f'rma reduce: {exc}')
        return 1


def make_ledger(args, out: Path) -> Ledger:
    return Ledger(args.ledger or out/'usage.jsonl',args.budget_usd,args.input_rate,args.output_rate,
                  settle_input_rate=args.settle_input_rate,settle_output_rate=args.settle_output_rate,
                  scope_limit=args.per_problem_usd)


def make_backend(args, out: Path, ledger: Ledger) -> ResponsesBackend:
    if not args.credentials_file: raise ValueError('--credentials-file required for paid runs')
    return ResponsesBackend.from_file(args.credentials_file,model=args.model,ledger=ledger,
        artifacts=out/'api',effort=args.effort,max_output=args.max_output_tokens,timeout=args.api_timeout)


def _probe(args):
    """One metered live call; the cheapest way to find out the deployment is wrong."""
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    ledger=make_ledger(args,out)
    result=make_backend(args,out,ledger).probe()
    atomic_json(out/'preflight.json',dict(result,ledger=ledger.summary()))
    print(json.dumps(dict(result,ledger=ledger.summary()),indent=2))
    return 0


def _report(args):
    from .report import load, render
    out=args.out.resolve()
    text=render(load(out),[load(path) for path in args.compare])
    (out/'report.md').write_text(text);print(text)
    return 0


def _run(args):
    if min(args.rounds,args.jobs,args.max_output_tokens,args.context_budget,args.timeout,args.api_timeout,args.beam)<1 or args.repairs<0:
        raise ValueError('Round, job, beam, token and timeout limits must be positive; repairs nonnegative')
    for flag in ('benchmark','baseline','lake'):
        if getattr(args,flag) is None: raise ValueError(f'--{flag} is required for a reduction run')
    entries=load_rows(args.benchmark);baseline=load_rows(args.baseline)
    byname={x['name']:x for x in entries};base={x['name']:x['proof'] for x in baseline}
    if set(base)!=set(byname): raise ValueError('Baseline must contain exactly the benchmark theorem set')
    for name,proof in base.items():proof_body(byname[name]['statement'],proof)
    manifest_path=args.manifest or args.baseline.parent/'manifest.json'
    if manifest_path.exists(): ids={x['name']:str(x['id']) for x in json.loads(manifest_path.read_text())}
    else: ids={x['name']:f'{i:02}' for i,x in enumerate(entries,1)}
    if set(ids)!=set(byname) or len(set(ids.values()))!=len(ids):raise ValueError('Manifest does not match benchmark')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',i) for i in ids.values()):raise ValueError('Unsafe problem id')
    selected=[e for e in entries if not args.ids or e['name'] in args.ids or ids[e['name']] in args.ids]
    if args.ids and set(args.ids)-{v for e in selected for v in [e['name'],ids[e['name']]]}:
        raise ValueError('Unknown requested problem id/name')
    repos={}
    for spec in args.repo:
        name,path=spec.split('=',1);repos.setdefault(name,[]).append(Path(path).resolve())
    if any(e['source'] not in repos for e in selected):raise ValueError('Missing --repo for selected corpus')
    # Per-problem measurement toolchain (see ReductionPipeline.problem). Parsed and validated before the
    # run starts -- and before --dry-run prints the plan -- so a typo fails fast instead of silently
    # measuring on whichever --repo happened to come first.
    # An explicit pin that cannot be honoured must NOT fall back to index 0: that is precisely the
    # accidental measurement version the flag exists to remove. Both checks run here, before the plan
    # is written, so --dry-run fails on a bad pin too.
    pins={};byid={ids[e['name']]:e for e in entries}
    for spec in args.measure_version:
        i,ver=spec.split('=',1)
        if i not in byid:raise ValueError(f'--measure-version for unknown problem id {i}')
        entry=byid[i];listed=listed_versions(entry)
        if listed and ver not in listed:
            raise ValueError(f'--measure-version {i}={ver}: the Arena does not list that toolchain for '
                             f'this problem (listed: {listed})')
        configured=[]
        for repo in repos.get(entry['source'],[]):
            try:configured.append((repo/'lean-toolchain').read_text().strip().split(':')[-1])
            except OSError:pass
        # A pin for a problem outside this run keeps its listed-version check but needs no repository.
        if repos.get(entry['source']) and ver not in configured:
            raise ValueError(f'--measure-version {i}={ver}: no --repo for corpus {entry["source"]} '
                             f'provides that toolchain (configured: {configured})')
        pins[i]=ver
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    fingerprint={'benchmark':hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
                 'baseline':hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                 'model':args.model,'repos':{k:[str(p) for p in v] for k,v in repos.items()},
                 # Heartbeats from different toolchains are not comparable, so a resume that changes a
                 # problem's measurement version would silently mix them inside one run's state.
                 'measure_versions':pins,'sources':{}}
    for e in selected:
        for repo in repos[e['source']]:
            source_path=(repo/e['file_path']).resolve()
            if not source_path.is_relative_to(repo):raise ValueError('Benchmark path escapes repository')
            source=(e['header']+e['src']) if e['source']=='putnambench' else source_path.read_text()
            fingerprint['sources'][str(repo)+'/'+e['name']]=hashlib.sha256(
                (source+(repo/'lean-toolchain').read_text()).encode()).hexdigest()
    with file_lock(out,'reduce-run'):
        configfile=out/'inputs.json'
        if configfile.exists():
            # A resume may select a different subset of the SAME run: shared inputs must be
            # identical and every source already fingerprinted must still match.
            stored=json.loads(configfile.read_text())
            clash=[k for k,v in fingerprint['sources'].items() if stored['sources'].get(k,v)!=v]
            # A run recorded before --measure-version existed has no such key, so an absent entry and
            # "no pins" must compare equal; any real change of measurement version still fails the resume.
            if clash or any(stored.get(k)!=fingerprint[k] for k in ('benchmark','baseline','model','repos')) \
                     or (stored.get('measure_versions') or {})!=(fingerprint['measure_versions'] or {}):
                raise ValueError('Run inputs/model/source changed; use a new output directory')
            fingerprint['sources']={**stored['sources'],**fingerprint['sources']}
        if (out/'results.json').exists() and not args.resume and not args.dry_run:
            raise ValueError('Run exists; use --resume')
        atomic_json(configfile,fingerprint)
        plan={'model':args.model,'ids':[ids[e['name']] for e in selected],'measure_versions':pins,
              'rounds':args.rounds,'strategies':args.strategies,'repairs':args.repairs,'jobs':args.jobs,
              'beam':args.beam,'ablate':sorted(set(args.ablate)),'mode':args.mode,
              'issues_per_round':args.issues_per_round,'rma_ablate':sorted(set(args.rma_ablate)),
              'rewrite_first':not args.no_rewrite_first,'local_search_attempts':args.local_search_attempts,
              'context_budget':args.context_budget,'max_output_tokens':args.max_output_tokens,
              'budget_usd':args.budget_usd,'per_problem_usd':args.per_problem_usd,
              'provider':'Responses (external credentials)',
              'score_kind':'public Arena formula computed locally: fitted Arena length lexer, local '
                           '#count_heartbeats, zero-shot verified on prepared versions (others as bounds)'}
        atomic_json(out/'plan.json',plan)
        if args.dry_run:print(json.dumps(plan,indent=2));return 0
        ledger=make_ledger(args,out)
        backend=make_backend(args,out,ledger)
        if not args.no_preflight:
            # Fail before compiling baselines if the deployment or key is wrong.
            check=backend.probe();atomic_json(out/'preflight.json',dict(check,ledger=ledger.summary()))
            print(f'preflight ok: {check["model"]} at {check["url"]}',flush=True)
        verifier=LeanVerifier(repos,args.lake,args.elan_home,args.timeout)
        hints=json.loads(args.hints.read_text()) if args.hints else {}
        if args.local_context:
            from .library import retrieve
            for e in selected:
                pid=ids[e['name']]
                hints[pid]=hints.get(pid,'')+'\n'+retrieve(e,base[e['name']],repos[e['source']][0])
            atomic_json(out/'retrieved-context.json',hints)
        playbook=Playbook(args.playbook or out/'playbook.jsonl')
        if args.mode=='rma':
            from .core import CorePipeline
            pipeline=CorePipeline(out,verifier,backend,model=args.model,context_budget=args.context_budget,
                rounds=args.rounds,issues_per_round=args.issues_per_round,hints=hints,
                ablations=frozenset(args.rma_ablate),rewrite_first=not args.no_rewrite_first,
                local_search_attempts=args.local_search_attempts)
        else:
            pipeline=ReductionPipeline(out,verifier,backend,model=args.model,context_budget=args.context_budget,
                strategies=args.strategies,repairs=args.repairs,rounds=args.rounds,hints=hints,
                playbook=playbook,beam=args.beam,ablate=frozenset(args.ablate))
        pipeline.measure_versions=pins
        official={}
        if args.baseline_results:
            official={x['name']:x for x in json.loads(args.baseline_results.read_text())['rows']}
        previous={}
        if args.resume and (out/'results.json').exists():
            previous={r['id']:r for r in json.loads((out/'results.json').read_text()).get('results',[])}
        results=[];errors=[]
        def export():
            # Every problem's kernel-verified best lives in its state file, so problems
            # improved by an earlier invocation of this run are never exported as baseline.
            winners={}
            for e in entries:
                statefile=out/'problems'/ids[e['name']]/'state.json'
                if statefile.exists():
                    winners[e['name']]=declaration(e,json.loads(statefile.read_text())['best_body'])
            merged=list({**previous,**{r['id']:r for r in results}}.values())
            # Every problem counts in the run score. A problem without local state keeps its
            # official row, which describes the baseline proof only if that proof was the one scored.
            per={}
            for e in entries:
                statefile=out/'problems'/ids[e['name']]/'state.json'
                score=json.loads(statefile.read_text()).get('score') if statefile.exists() else None
                if score:
                    per[ids[e['name']]]={'lower':score['best']['lower']['score'],'upper':score['best']['upper']['score'],
                                         'baseline_lower':score['baseline']['lower']['score'],
                                         'baseline_upper':score['baseline']['upper']['score'],
                                         'basis':'local measurement','unverified_versions':score['unverified_versions']}
                elif e['name'] in official:
                    value=parse_official(official[e['name']])['score']
                    per[ids[e['name']]]={'lower':value,'upper':value,'baseline_lower':value,'baseline_upper':value,
                                         'basis':'official row of the baseline proof'}
            estimate={'run_lower':run_score({k:v['lower'] for k,v in per.items()},ids.values()),
                      'run_upper':run_score({k:v['upper'] for k,v in per.items()},ids.values()),
                      'problems_without_score':sorted(set(ids.values())-set(per)),'per_problem':per}
            # Net change against the run's own starting proofs, on the same public-rule footing.
            for bound in ('lower','upper'):
                start=run_score({k:v['baseline_'+bound] for k,v in per.items()},ids.values())
                estimate['baseline_run_'+bound]=start;estimate['net_gain_'+bound]=estimate['run_'+bound]-start
            rows=[{'name':e['name'],'proof':winners.get(e['name'],base[e['name']])} for e in entries]
            target=out/'submission.jsonl';tmp=target.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));tmp.replace(target)
            atomic_json(out/'results.json',{'results':merged,'errors':errors,'plan':plan,
                        'reserved_usd':ledger.reserved(),'ledger':ledger.summary(),
                        'improved':sum(1 for r in merged if r['improved']),
                        'estimated_score':estimate,'submission_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
            return estimate
        estimate=export()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures={pool.submit(pipeline.problem,ids[e['name']],e,base[e['name']],
                        official=official.get(e['name']),resume=args.resume):e for e in selected}
            for future in as_completed(futures):
                e=futures[future]
                try:
                    result=future.result();results.append(result)
                    print(json.dumps({'id':result['id'],'improved':result['improved'],
                        'tokens_proxy':result['measurement']['tokens_proxy'],
                        'heartbeats':result['measurement']['heartbeats'],
                        'baseline':result['baseline'],'attempts':result['attempts'],
                        'stop':result['stop_reason']}),flush=True)
                except Exception as exc:
                    errors.append({'name':e['name'],'error':str(exc)});print(f'{ids[e["name"]]} failed: {exc}',flush=True)
                estimate=export()
        with zipfile.ZipFile(out/'submission.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['submission.jsonl','results.json','plan.json','inputs.json']:z.write(out/name,name)
        usage=ledger.summary()
        print(f'Exported {len(entries)} proofs to {out / "submission.jsonl"}; '
              f'{sum(1 for r in results if r["improved"])}/{len(selected)} improved; '
              f'{usage["reserved_usd"]:.4f} USD committed '
              f'({usage["held_usd"]:.4f} held on {usage["unsettled_calls"]} unsettled calls); '
              f'estimated Arena score {estimate["run_lower"]:.2f}% (unverified versions failing) to '
              f'{estimate["run_upper"]:.2f}% (unverified versions passing)')
        return 1 if errors else 0
