"""Lean admission gate, pinned compilation, explicitly approximate scoring, and
compiler diagnostics turned into a localized repair observation."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile


def strip_comments(text: str) -> str:
    out=[]; i=0; depth=0
    while i<len(text):
        if depth:
            if text.startswith('/-',i): depth+=1; i+=2
            elif text.startswith('-/',i): depth-=1; i+=2
            else: i+=1
        elif text.startswith('/-',i): depth=1; i+=2; out.append(' ')
        elif text.startswith('--',i):
            j=text.find('\n',i); i=len(text) if j<0 else j
        elif text[i]=='"':
            # Strings are unnecessary for submitted proof bodies and could hide diagnostics tricks.
            raise ValueError('String literals are not supported in reduction candidates')
        else: out.append(text[i]); i+=1
    if depth: raise ValueError('Unclosed block comment')
    return ''.join(out)


FORBIDDEN = re.compile(r'\b(sorry|admit|axiom|unsafe|native_decide|ofReduceBool|ofReduceNat|'
    r'run_tac|run_elab|run_cmd|elab|elab_rules|macro|macro_rules|syntax|notation|'
    r'set_option|attribute|implemented_by|extern|initialize|builtin_initialize|'
    r'theorem|lemma|opaque|namespace|section|end|import|export|instance)\b|#|\bIO\.')


def extract_body(text: str) -> str:
    blocks=re.findall(r'```(?:lean4?|Lean)?\s*\n(.*?)```',text,re.S)
    if blocks:
        if len(blocks)!=1: raise ValueError('Expected exactly one Lean proof block')
        text=blocks[0]
    body=text.strip()+'\n'
    clean=strip_comments(body)
    if not re.match(r'^\s*by\b',clean): raise ValueError('Expected a proof body beginning with by')
    if FORBIDDEN.search(clean): raise ValueError('Disallowed Lean construct in proof body')
    return body


def proxy_tokens(body: str) -> int:
    return len(re.findall(r'\w+|[^\w\s]',strip_comments(body)))


def model_tokens(body: str) -> int:
    """A second length estimate from RMA's tokenizer. The Arena counts proof length
    with its own tokenizer; two disagreeing proxies are reported, neither is claimed."""
    try:
        from rma.budget import count_tokens
        return int(count_tokens(strip_comments(body)))
    except Exception:
        return 0


def proof_body(statement: str, proof: str) -> str:
    if not proof.startswith(statement): raise ValueError('Theorem statement changed')
    tail=proof[len(statement):]
    match=re.match(r'\s*:=\s*',tail)
    if not match: raise ValueError('Missing proof separator')
    return extract_body(tail[match.end():])


def declaration(entry: dict, body: str) -> str:
    sep=re.match(r'\s*:=\s*',entry['src'][len(entry['statement']):])
    if not sep: raise ValueError('Invalid benchmark declaration')
    return entry['statement']+sep[0]+body.strip()+'\n'


# Lines that begin a new top-level command; a proof body ends before the first of them.
# `termination_by`, `decreasing_by`, `where` and match arms stay with the declaration.
_TOPLEVEL=re.compile(r'(?m)^(?:@\[|/--|/-!|#|(?:(?:private|protected|public|noncomputable|partial|unsafe|'
    r'scoped|local)\s+)*(?:theorem|lemma|def|abbrev|instance|example|structure|inductive|class|'
    r'namespace|section|end|open|variable|universe|set_option|attribute|macro|syntax|notation|elab|'
    r'alias|opaque|axiom|mutual|omit|include|export|irreducible_def)\b)')


def locate_declaration(source: str, entry: dict) -> str:
    """The target's original declaration text in THIS version of its source file.

    The benchmark's `src` is the declaration at the pinned commit. Another listed
    version may carry a different proof of the same statement, so when `src` is
    absent the declaration is found by its unique statement and ends before the next
    top-level command. The statement itself is never guessed."""
    if source.count(entry['src'])==1: return entry['src']
    statement=entry['statement']
    if source.count(statement)!=1: raise ValueError('Original declaration not uniquely located')
    start=source.index(statement)
    sep=re.match(r'\s*:=',source[start+len(statement):])
    if not sep: raise ValueError('Declaration separator not found after the statement')
    following=_TOPLEVEL.search(source,start+len(statement)+sep.end())
    original=source[start:following.start() if following else len(source)].rstrip()
    if source.count(original)!=1: raise ValueError('Original declaration not uniquely located')
    return original


DIAGNOSTIC=re.compile(r'(?m)^\S*?\.lean:(\d+):(\d+): (error|warning|information|info): ')


def diagnostics(log: str, base_line: int=0, *, limit=6, severities=('error',)) -> list[dict]:
    """Lean's `file:line:col: severity: message` stream, re-anchored onto the
    submitted declaration so a repair request can point at the failing tactic."""
    found=list(DIAGNOSTIC.finditer(log)); out=[]
    for i,m in enumerate(found):
        if m[3] not in severities: continue
        end=found[i+1].start() if i+1<len(found) else len(log)
        out.append({'line':int(m[1]),'rel_line':int(m[1])-base_line,'col':int(m[2]),
                    'severity':m[3],'message':log[m.end():end].strip()[:600]})
    return out[:limit]


def annotate(code: str, errors: list[dict], *, window=2, limit=3) -> str:
    """The failing declaration with its error lines marked, instead of a raw log tail."""
    lines=code.splitlines(); shown=[]
    for err in errors[:limit]:
        idx=err.get('rel_line',0)-1
        if not 0<=idx<len(lines): continue
        lo,hi=max(0,idx-window),min(len(lines),idx+window+1)
        block=[('>>> ' if j==idx else '    ')+f'{j+1:>3}| '+lines[j] for j in range(lo,hi)]
        shown.append('\n'.join(block)+f'\n    error at column {err["col"]}: '+err['message'])
    return '\n\n'.join(shown)


@dataclass
class Measurement:
    valid: bool
    heartbeats: int | None
    tokens_proxy: int
    exit_code: int
    axioms: list[str]
    log: str
    toolchain: str
    proof_sha256: str
    instrumented: bool
    errors: list = field(default_factory=list)
    tokens_model: int = 0
    tokens_arena: int = 0
    warnings: list = field(default_factory=list)
    audited: bool = True

    def summary(self):
        data=asdict(self); data.pop('log'); return data

    def feedback(self, code: str, *, fallback=9000) -> str:
        """Localized compiler feedback for the repair turn, with the raw tail only
        when nothing parseable was emitted (timeouts, kernel panics, fatal errors)."""
        marked=annotate(code,self.errors) if self.errors else ''
        if marked:
            return 'Compiler errors located in the submitted declaration:\n'+marked
        return 'Compiler log (tail):\n'+self.log[-fallback:]


def read_source(repo: Path, relpath: str) -> str:
    """The benchmark source file, snapshotted from `git show HEAD:<relpath>` when the
    repo is a git working tree, falling back to a plain disk read otherwise (e.g. in
    unit-test fixtures that are not git checkouts).

    Several corpora used by `rma reduce` are shared checkouts that OTHER concurrent
    processes patch in place (a competing solver track, a different session's probe
    run). A live `path.read_text()` on such a tree can silently grab a half-patched or
    corrupted file mid-edit, which then reads as "Baseline does not compile" even
    though the real, git-committed source is fine. Reading through git instead makes
    the baseline immune to any uncommitted, in-progress mutation of the shared tree."""
    try:
        proc=subprocess.run(['git','-C',str(repo),'show',f'HEAD:{relpath}'],
                             capture_output=True,text=True,timeout=30)
        if proc.returncode==0: return proc.stdout
    except Exception: pass
    return (repo/relpath).read_text()


class LeanVerifier:
    def __init__(self, repos: dict[str, list[Path]], lake: Path, elan_home: Path | None=None,
                 timeout=90):
        self.repos={k:[Path(p).resolve() for p in v] for k,v in repos.items()}
        self.lake=Path(lake).resolve(); self.elan_home=elan_home; self.timeout=timeout

    def verify(self, entry: dict, body: str, *, measure=True, version=0, options=()) -> Measurement:
        """`options` are boolean `set_option … true in` switches (e.g. linters) for an
        unmeasured probe compile; they are never combined with heartbeat measurement."""
        if options and measure:
            raise ValueError('Linter options are only allowed on unmeasured probe compiles')
        body=extract_body(body)
        repo=self.repos[entry['source']][version]
        toolchain=(repo/'lean-toolchain').read_text().strip()
        declared={v for row in entry.get('version_info',[]) for v in row}
        if declared and toolchain.split(':')[-1] not in declared:
            raise ValueError(f'Unlisted toolchain for {entry["name"]}: {toolchain}')
        if entry['source']=='putnambench': source=entry['header']+entry['src']
        else:
            path=(repo/entry['file_path']).resolve()
            if not path.is_relative_to(repo): raise ValueError('Benchmark path escapes repository')
            source=read_source(repo,entry['file_path'])
        original=locate_declaration(source,entry)
        decl=code=declaration(entry,body)
        if measure:
            pos=source.index(original); doc=source.rfind('/--',0,pos)
            if doc>=0 and not source[doc:pos].split('-/',1)[1].strip():
                source=source[:doc]+source[doc:].replace('/--','/-',1)
            if entry['source']=='strata':
                harness='''open Lean Elab Command in
elab "#reduce_count " "in" cmd:command : command => do
  let start ← IO.getNumHeartbeats
  elabCommand cmd
  let finish ← IO.getNumHeartbeats
  logInfo m!"Used {(finish - start) / 1000} heartbeats"
'''
                at=source.index('namespace ');source=source[:at]+harness+'\n'+source[at:]
                marker='#reduce_count in\n'
            else:
                source,n=re.subn(r'(?m)^(public )?import ',lambda m:(m[1] or '')+
                    'import Mathlib.Util.CountHeartbeats\n'+m[0],source,count=1)
                if not n: raise ValueError('No import position for measurement')
                marker='#count_heartbeats in\n'
            code='set_option Elab.async false in\nset_option maxHeartbeats 2000000 in\n'+marker+code
        code=''.join(f'set_option {o} true in\n' for o in options)+code
        # `#print axioms` is rejected inside a `module` header. That is a property of the file,
        # not of the proof, so a CROSS-VERSION compatibility check (version>0) drops the audit and
        # measures what the Arena's zero-shot axis measures: does this proof still compile unchanged.
        # The pinned version keeps the audit, so the promotion gate never loses its axiom guarantee.
        audited=not (version and re.search(r'(?m)^\s*module\b',source))
        source=source.replace(original,code,1)+(f'\n#print axioms {entry["name"]}\n' if audited else '')
        # Diagnostics are re-anchored onto the declaration itself, so the
        # instrumentation prefix never shifts the reported tactic line.
        base=source[:source.index(decl)].count('\n')
        env=os.environ.copy()
        if self.elan_home: env['ELAN_HOME']=str(self.elan_home)
        with tempfile.NamedTemporaryFile(mode='w',suffix='.lean',prefix='RMAReduce',dir=repo,delete=False) as f:
            f.write(source); path=Path(f.name)
        try:
            with tempfile.TemporaryFile(mode='w+') as log:
                proc=subprocess.Popen([str(self.lake),'env','lean',str(path)],cwd=repo,env=env,
                    text=True,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try: proc.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                log.seek(0); text=log.read()
        finally: path.unlink(missing_ok=True)
        matches=re.findall(r'depends on axioms:\s*\[([^\]]*)\]',text)
        axioms=[x.strip() for x in matches[-1].split(',') if x.strip()] if matches else []
        has_audit=bool(matches) or 'does not depend on any axioms' in text
        counts=re.findall(r'Used (\d+) heartbeats',text)
        hb=int(counts[-1]) if counts else None
        valid=(proc.returncode==0 and (has_audit if audited else True)
               and set(axioms)<= {'propext','Classical.choice','Quot.sound'}
               and (not measure or hb is not None))
        from .score import arena_tokens
        return Measurement(valid,hb,proxy_tokens(body),proc.returncode,axioms,text,toolchain,
                           hashlib.sha256(declaration(entry,body).encode()).hexdigest(),measure,
                           diagnostics(text,base),model_tokens(body),arena_tokens(body),
                           diagnostics(text,base,limit=20,severities=('warning',)),audited)


def local_score(measurement: Measurement, baseline: Measurement) -> float:
    """A local length/HB utility; NOT the official Arena token count or score."""
    if not measurement.valid: return float('-inf')
    return 100*((1-measurement.tokens_proxy/max(1,baseline.tokens_proxy))+
                (1-(measurement.heartbeats or 0)/max(1,baseline.heartbeats or 0)))/2
