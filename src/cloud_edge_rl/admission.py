"""Local fail-closed approval and durable reservation boundary; never releases charges."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

from .config import source_identity, spec_to_dict
from .comparisons.acceptance import _first_artifact, _first_json_equal, _first_json_object


def frozen_payload(spec):
    """Approval contains the complete spec except its own witness descriptor."""
    payload = spec_to_dict(spec)
    payload['config']['native_acceptance'].pop('execution_admission', None)
    return payload


def check_admission(specs, root):
    """Validate all specs BEFORE app creation/open, and again before reserving."""
    if not specs:
        raise ValueError('Explicit approved specs required')
    current = source_identity(Path(root))['source_contents']
    approved = []
    for spec in specs:
        if (type(spec.seed) is not int or type(spec.inputs) is not dict
                or any(type(value) is not str or not value
                       for value in (spec.run_id, spec.stage, spec.method, spec.risk_version))):
            raise ValueError('RunSpec fields require explicit exact types')
        descriptor = spec.config['native_acceptance']['execution_admission']
        witness = _first_artifact(descriptor)
        if (witness.get('approved') is not True or type(witness.get('owner')) is not str
                or not witness['owner'] or witness.get('purpose') != 'fixedcandidate_independent_validation'):
            raise ValueError('Explicit owner stage approval required')
        payload = frozen_payload(spec)
        if not any(_first_json_equal(payload, item['spec']) for item in witness['runs']):
            raise ValueError('Spec does not match stage approval')
        if (spec.provenance.get('identity_kind') != 'source_contents_no_hash'
                or not _first_json_equal(current, spec.provenance['source_contents'])):
            raise ValueError('Source contents changed since stage freeze')
        for field in ('execution', 'data'):
            if type(spec.config.get(field)) is not dict or not spec.config[field]:
                raise ValueError(f'Explicit {field} required')
        isolation = _first_artifact(witness['truth_isolation'])
        if (isolation.get('freeze_id') != witness['freeze_id']
                or type(isolation.get('observed', {}).get('truth_isolation')) is not int
                or isolation['observed']['truth_isolation'] != 1):
            raise ValueError('Truth isolation witness required')
        freeze = _first_artifact(witness['freeze_manifest'])
        if (type(witness.get('freeze_id')) is not str or not witness['freeze_id']
                or freeze.get('freeze_id') != witness['freeze_id']
                or not _first_json_equal(freeze.get('runs'), witness['runs'])):
            raise ValueError('Stage freeze must contain the exact approved runs')
        ledger = _first_artifact(witness['ledger_manifest'])
        if type(ledger) is not list:
            raise ValueError('Explicit historical budget ledger required')
        log = Path(witness['reservation_log'])
        if not log.is_absolute():
            raise ValueError('Absolute durable reservation log required')
        run = next(item for item in witness['runs'] if _first_json_equal(payload, item['spec']))
        approved.append((witness, run, ledger, log))
    return approved


@contextmanager
def _locked(log):
    """OS lock is released on death; journal is never reset after uncertain spawn."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.with_suffix(log.suffix + '.lock').open('a+b') as lock:
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            if lock.read(1) == b'':
                lock.write(b'\0')
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _rows(log):
    if not log.exists():
        return []
    return [json.loads(line, object_pairs_hook=_first_json_object) for line in
            log.read_text(encoding='utf-8').splitlines()]


def _append(log, row):
    with log.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def reserve(spec, root):
    """Return an old bound call or reserve before spawn; unknown attempts cannot retry."""
    witness, run, ledger, log = check_admission([spec], root)[0]
    with _locked(log):
        rows = _rows(log)
        for row in rows:
            if row['event'] not in ('reserved', 'bound'):
                raise ValueError('Unknown reservation event')
            if row['event'] == 'reserved' and not _first_json_equal(
                    row['ledger_manifest'], witness['ledger_manifest']):
                raise ValueError('Reservation history baseline cannot reset or change')
        old = [row for row in rows if row['run_id'] == spec.run_id]
        if old:
            if (not _first_json_equal(old[0]['spec'], frozen_payload(spec))
                    or not _first_json_equal(old[0]['charges'], run['charges'])):
                raise ValueError('Reserved spec changed')
            bound = [row for row in old if row['event'] == 'bound']
            if len(bound) == 1:
                return bound[0]['call_id']
            raise ValueError('Unknown paid spawn remains reserved; manual reconciliation required')
        charges = run['charges']
        if type(charges) is not list or not charges:
            raise ValueError('Explicit group charges required, including zero-interaction validation')
        totals, configurations = {}, {}
        if len({row['call_id'] for row in ledger}) != len(ledger):
            raise ValueError('Duplicate historical budget call')
        for row in ledger:
            key = (row['method'], row['seed'], row['partition'])
            amount = row.get('actual_interactions')
            counters = ('learned_steps', 'warmup_steps', 'collection_interactions', 'resume_offset')
            if any(name in row for name in counters):
                counts = [row[name] for name in counters]
                if (any(type(v) is not int or v < 0 for v in counts)
                        or counts[3] > sum(counts[:2])):
                    raise ValueError('Invalid historical interaction counters')
                counted = sum(counts[:3]) - counts[3]
                if amount is not None and (type(amount) is not int or amount != counted):
                    raise ValueError('Conflicting historical interaction counters')
                amount = counted
            if (type(row['seed']) is not int or type(amount) is not int or amount < 0
                    or row['state'] not in ('completed', 'failed', 'cancelled', 'canceled', 'interrupted')
                    or (row['state'] != 'completed' and row.get('consumption_verified') is not True)):
                raise ValueError('Unreconciled historical consumption')
            totals[key] = totals.get(key, 0) + amount
            configurations.setdefault((key[0], key[2]), set()).add(row['configuration_id'])
        for row in rows:
            if row['event'] == 'reserved':
                for charge in row['charges']:
                    key = (charge['method'], charge['seed'], charge['partition'])
                    totals[key] = totals.get(key, 0) + charge['actual_interactions']
        for charge in charges:
            key = (charge['method'], charge['seed'], charge['partition'])
            amount = charge['actual_interactions']
            if (type(charge['seed']) is not int or type(amount) is not int or amount < 0
                    or charge['configuration_id'] not in configurations.get((key[0], key[2]), set())):
                raise ValueError('No new configuration tuning is authorized')
            totals[key] = totals.get(key, 0) + amount
            if amount and totals[key] > 3000:
                raise ValueError('Cumulative group interaction limit exceeded')
        _append(log, {'event': 'reserved', 'run_id': spec.run_id, 'spec': frozen_payload(spec),
                      'charges': charges, 'freeze_id': witness['freeze_id'],
                      'ledger_manifest': witness['ledger_manifest']})
    return None


def bind_call(spec, root, call_id):
    """Append call binding only after spawn; a failure here leaves the reservation intact."""
    _, _, _, log = check_admission([spec], root)[0]
    if type(call_id) is not str or not call_id:
        raise ValueError('Explicit call ID required')
    with _locked(log):
        rows = [row for row in _rows(log) if row['run_id'] == spec.run_id]
        if len(rows) != 1 or rows[0]['event'] != 'reserved':
            raise ValueError('Call binding requires exactly one reservation')
        _append(log, {'event': 'bound', 'run_id': spec.run_id, 'call_id': call_id})
