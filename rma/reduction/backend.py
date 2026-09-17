"""Responses transport and a durable, cross-process reservation ledger.

Unknown requests keep their reservation. No automatic retry can double-charge a
request whose response was lost. Planning rates are upper bounds, not billing
claims; a *completed* response settles its hold down to metered usage so a
conservative upper bound cannot silently consume the whole budget.
"""
from __future__ import annotations
import copy
import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from webapp.locks import file_lock
from rma.models import ModelRequestError, ModelConfigurationError


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path, limit: float, input_rate=100., output_rate=500.,
                 settle_input_rate=None, settle_output_rate=None, scope_limit=None):
        rates = (limit, input_rate, output_rate, settle_input_rate if settle_input_rate is not None else input_rate,
                 settle_output_rate if settle_output_rate is not None else output_rate)
        if not all(math.isfinite(x) and x > 0 for x in rates):
            raise ValueError('Budget and planning rates must be positive')
        if scope_limit is not None and not (math.isfinite(scope_limit) and scope_limit > 0):
            raise ValueError('Per-scope limit must be positive')
        self.path, self.limit = Path(path).resolve(), limit
        self.input_rate, self.output_rate = input_rate, output_rate
        # Settlement rates price ACTUAL metered tokens; planning rates price the
        # worst case that must be held before the request is dispatched.
        self.settle_input_rate = input_rate if settle_input_rate is None else settle_input_rate
        self.settle_output_rate = output_rate if settle_output_rate is None else settle_output_rate
        self.scope_limit = scope_limit

    def entries(self):
        if not self.path.exists():
            return []
        # Fail closed on a corrupt ledger; never treat unreadable history as zero.
        rows=[json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if any(not math.isfinite(float(r['reserved_usd'])) or float(r['reserved_usd'])<0 for r in rows):
            raise ValueError('Invalid historical reservation')
        return rows

    def reserved(self):
        return sum(float(e['reserved_usd']) for e in self.entries())

    def summary(self):
        entries=self.entries()
        settled=sum(float(e['reserved_usd']) for e in entries if e.get('settled'))
        total=sum(float(e['reserved_usd']) for e in entries)
        return {'calls':len(entries),'reserved_usd':total,'settled_usd':settled,
                'held_usd':total-settled,'limit_usd':self.limit,'remaining_usd':self.limit-total,
                'unsettled_calls':sum(1 for e in entries if not e.get('settled'))}

    def reserve(self, prompt: str, max_output: int, *, scope=None, **meta):
        if max_output < 1:
            raise ValueError('max_output must be positive')
        cost = (len(prompt.encode('utf-8')) * self.input_rate + max_output * self.output_rate) / 1e6
        with file_lock(self.path.parent, self.path.name + '-budget'):
            entries = self.entries()
            used = sum(float(e['reserved_usd']) for e in entries)
            if used + cost > self.limit + 1e-9:
                raise BudgetExceeded(f'Reservation {cost:.4f} exceeds remaining {self.limit-used:.4f} USD')
            if scope is not None and self.scope_limit is not None:
                # One problem cannot drain a shared run budget before the others start.
                spent = sum(float(e['reserved_usd']) for e in entries if e.get('scope') == scope)
                if spent + cost > self.scope_limit + 1e-9:
                    raise BudgetExceeded(f'Reservation {cost:.4f} exceeds scope {scope} remaining '
                                         f'{self.scope_limit-spent:.4f} USD')
            rec = dict(meta, id=uuid.uuid4().hex, reserved_usd=cost, status='reserved', scope=scope,
                       max_output_tokens=max_output, created_at=time.time())
            # Preserve the legacy JSONL format so an existing arena budget is reusable.
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(e) + '\n' for e in entries + [rec]))
            os.replace(tmp, self.path)
            return rec

    @staticmethod
    def _metered(usage):
        """Actual (input, output) tokens, or None when the report is not trustworthy."""
        if not isinstance(usage, dict):
            return None
        values = usage.get('input_tokens'), usage.get('output_tokens')
        # Do not coerce fractional, boolean or non-finite reports into cheaper
        # integer counts: malformed metering must retain the full reservation.
        if not all(type(v) is int and v >= 0 for v in values):
            return None
        return values

    def finish(self, rid, *, settle_usage=None, **data):
        with file_lock(self.path.parent, self.path.name + '-budget'):
            entries = self.entries()
            entry = next(e for e in entries if e.get('id') == rid)
            entry.update(data)
            metered = self._metered(settle_usage)
            if metered is not None and not entry.get('settled'):
                actual = (metered[0]*self.settle_input_rate + metered[1]*self.settle_output_rate) / 1e6
                # Settlement can only release budget. An implausible usage report that
                # exceeds the pre-dispatch upper bound keeps the conservative hold.
                if actual <= float(entry['reserved_usd']) + 1e-12:
                    entry.update(estimate_usd=entry['reserved_usd'], reserved_usd=actual, settled=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(e) + '\n' for e in entries))
            os.replace(tmp, self.path)


class ResponsesBackend:
    def __init__(self, *, endpoint: str, key: str, model: str, ledger: Ledger,
                 artifacts: Path, effort='high', max_output=12000, timeout=900, scope=None):
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ModelConfigurationError('Endpoint must be HTTPS without embedded credentials/query')
        if not key.strip() or not model.strip():
            raise ModelConfigurationError('Key and deployment/model name are required')
        if '/api/projects/' in parsed.path:
            if not parsed.hostname.endswith('.services.ai.azure.com'):
                raise ModelConfigurationError('Unrecognized Azure project endpoint')
            endpoint = f'https://{parsed.netloc}/openai/v1'
        self.url = endpoint.rstrip('/') + '/responses'
        self.azure = parsed.hostname.endswith(('.azure.com', '.openai.azure.com'))
        self._key, self.model, self.ledger = key.strip(), model, ledger
        self.artifacts = Path(artifacts)
        self.effort, self.max_output, self.timeout = effort, max_output, timeout
        self.scope = scope

    @classmethod
    def from_file(cls, path: Path, **kwargs):
        # Supported legacy two-line file, or JSON {endpoint, api_key}.
        raw = path.read_text().strip()
        if raw.startswith('{'):
            cfg = json.loads(raw); endpoint, key = cfg['endpoint'], cfg['api_key']
        else:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if len(lines) != 2:
                raise ModelConfigurationError('Credential file must contain endpoint and key only')
            endpoint, key = lines
        return cls(endpoint=endpoint, key=key, **kwargs)

    def scoped(self, scope: str):
        """A view charging the same ledger under a per-problem scope cap."""
        clone = copy.copy(self); clone.scope = scope; return clone

    def probe(self):
        """One minimal live call: credentials, deployment name and transport, before any
        Lean compilation or paid reduction work. Costs a settled handful of tokens."""
        cheap = copy.copy(self); cheap.effort = 'low'; cheap.max_output = min(self.max_output, 2000)
        text = cheap('preflight', 'Reply with exactly: OK')
        # Identify this call's own entry: a shared ledger may be appended to concurrently.
        mine = [e for e in self.ledger.entries() if e.get('unit') == 'preflight'][-1]
        return {'model': self.model, 'url': self.url, 'reply': text.strip()[:40],
                'usage': mine.get('usage', {}), 'reserved_usd': float(mine['reserved_usd'])}

    def __call__(self, unit: str, observation: str) -> str:
        rec = self.ledger.reserve(observation, self.max_output, scope=self.scope,
                                  model=self.model, unit=unit)
        folder = self.artifacts / rec['id']; folder.mkdir(parents=True)
        body = dict(model=self.model, input=observation, max_output_tokens=self.max_output,
                    reasoning={'effort': self.effort}, store=False)
        atomic_json(folder/'request.json', body)
        header = ('api-key: ' if self.azure else 'Authorization: Bearer ') + self._key
        # Secrets go over stdin, never argv, shell expansion, logs, or artifacts.
        config = '\n'.join(['url = ' + json.dumps(self.url), 'header = ' + json.dumps(header),
                            'header = "Content-Type: application/json"',
                            'data = ' + json.dumps(json.dumps(body))])
        try:
            proc = subprocess.run(['curl', '--config', '-', '--silent', '--show-error',
                                   '--max-time', str(self.timeout), '--write-out', '\n%{http_code}'],
                                  input=config, text=True, capture_output=True, timeout=self.timeout+10)
            raw, _, status = proc.stdout.rpartition('\n')
            if proc.returncode or not status.startswith('2'):
                self.ledger.finish(rec['id'], status='transport_error', http_status=status)
                raise ModelRequestError('Responses request failed; reservation retained; no automatic retry')
            result = json.loads(raw)
            sanitized = json.loads(json.dumps(result).replace(self._key, '[REDACTED]'))
            atomic_json(folder/'response.json', sanitized)
            completed = result.get('status') == 'completed'
            # A settled hold only shrinks, and only when the server reported a completed
            # response with metered token counts; anything else keeps the upper bound.
            self.ledger.finish(rec['id'], status=result.get('status', 'unknown'),
                               usage=result.get('usage', {}), response_id=result.get('id'),
                               settle_usage=result.get('usage') if completed else None)
            if not completed:
                raise ModelRequestError('Model did not complete a candidate; see response artifact')
            return '\n'.join(c['text'] for o in result.get('output', [])
                             for c in o.get('content', []) if c.get('type') == 'output_text')
        except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
            self.ledger.finish(rec['id'], status='unknown')
            raise ModelRequestError('Transport/response failure; reservation retained') from exc
