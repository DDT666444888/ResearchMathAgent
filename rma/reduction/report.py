"""Turn finished runs into a comparison table.

An ablation is only evidence if two runs can be put side by side, so the same
reader that prints one run's per-problem outcome also prints the summary line
for every `--compare` run. Everything comes from results.json; nothing is
recomputed from the model, and no official score is inferred.
"""
from __future__ import annotations
import json
from pathlib import Path


def load(path: Path) -> dict:
    data=json.loads((Path(path)/'results.json').read_text())
    data['name']=Path(path).name
    # Every attempt, including failures, and the metered API cost per problem scope.
    data['attempts']=[]
    for statefile in sorted((Path(path)/'problems').glob('*/state.json')):
        for a in json.loads(statefile.read_text()).get('attempts',[]):
            m=a.get('measurement') or {}
            data['attempts'].append({'id':statefile.parent.name,'tag':a.get('tag'),'strategy':a.get('strategy'),
                'status':a.get('status'),'length':m.get('tokens_arena') or m.get('tokens_proxy'),
                'heartbeats':m.get('heartbeats'),'score':(a.get('score') or {}).get('score'),
                'error':(a.get('error') or '')[:80]})
    data['cost_by_problem']={}
    ledger=Path(path)/'usage.jsonl'
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                row=json.loads(line)
                if row.get('scope'):
                    data['cost_by_problem'][row['scope']]=data['cost_by_problem'].get(row['scope'],0.)+float(row['reserved_usd'])
    return data


def gains(run: dict) -> list[dict]:
    rows=[]
    for result in run.get('results',[]):
        base,best=result.get('baseline') or {},result.get('measurement') or {}
        def change(key):
            before,after=base.get(key),best.get(key)
            if not before or after is None: return None
            return after/before-1
        rows.append({'id':result.get('id'),'improved':result.get('improved'),
                     'tokens':change('tokens_proxy'),'heartbeats':change('heartbeats'),
                     'attempts':result.get('attempts'),'promoted':result.get('promoted'),
                     'stop':result.get('stop_reason')})
    return sorted(rows,key=lambda r:str(r['id']))


def mean(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def percent(value):
    return '-' if value is None else f'{value:+.1%}'


def summary_row(run: dict) -> str:
    rows=gains(run)
    ledger=run.get('ledger') or {}
    plan=run.get('plan') or {}
    ablate=','.join(plan.get('ablate') or []) or 'none'
    return (f"| {run['name']} | {ablate} | beam {plan.get('beam','?')} | "
            f"{sum(1 for r in rows if r['improved'])}/{len(rows)} | "
            f"{percent(mean(r['tokens'] for r in rows))} | "
            f"{percent(mean(r['heartbeats'] for r in rows))} | "
            f"{ledger.get('reserved_usd',run.get('reserved_usd',0)):.4f} | "
            f"{ledger.get('held_usd',0):.4f} |")


def render(primary: dict, others: list[dict]) -> str:
    lines=['# rma reduce report','',
           '| run | ablations | search | improved | mean length | mean heartbeats | USD committed | USD held |',
           '|---|---|---|---|---|---|---|---|']
    lines+= [summary_row(run) for run in [primary]+others]
    for run in [primary]+others:
        estimate=run.get('estimated_score')
        if estimate:
            lines.append(f"\n{run['name']}: estimated Arena score {estimate['run_lower']:.2f}% "
                         f"to {estimate['run_upper']:.2f}% (public formula, computed locally; the range "
                         f"is the zero-shot share of versions not verified locally)")
            if 'net_gain_lower' in estimate:
                lines.append(f"{run['name']}: net change against its starting proofs "
                             f"{estimate['net_gain_lower']:+.2f} to {estimate['net_gain_upper']:+.2f} points "
                             f"(from {estimate['baseline_run_lower']:.2f}–{estimate['baseline_run_upper']:.2f}%)")
    lines+=['','## Per problem: '+primary['name'],'',
            '| id | improved | length | heartbeats | attempts | promoted | stop |','|---|---|---|---|---|---|---|']
    for row in gains(primary):
        lines.append(f"| {row['id']} | {'yes' if row['improved'] else 'no'} | {percent(row['tokens'])} | "
                     f"{percent(row['heartbeats'])} | {row['attempts']} | {row['promoted']} | {row['stop']} |")
    for run in [primary]+others:
        if run.get('attempts'):
            lines+=['',f"## Attempts: {run['name']}",'',
                    '| id | attempt | strategy | status | length | heartbeats | problem score | error |',
                    '|---|---|---|---|---|---|---|---|']
            for a in run['attempts']:
                score='-' if a['score'] is None else f"{a['score']:.2f}"
                lines.append(f"| {a['id']} | {a['tag']} | {a['strategy']} | {a['status']} | {a['length'] or '-'} | "
                             f"{a['heartbeats'] or '-'} | {score} | {a['error'] or ''} |")
        if run.get('cost_by_problem'):
            lines.append('\nAPI cost by problem (USD, conservative planning rates on metered usage): '+
                         ', '.join(f'{k} {v:.4f}' for k,v in sorted(run['cost_by_problem'].items())))
    errors=primary.get('errors') or []
    if errors:
        lines+=['','## Failed problems','']+[f"- {e.get('name')}: {e.get('error')}" for e in errors]
    lines+=['','Scores use the public Arena rule on local measurements; the official run remains the record.','']
    return '\n'.join(lines)
