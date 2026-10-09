"""Local fail-closed approval and append-only durable reservation boundary."""

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


def check_admission(specs, root, *, _saved_source=False):
    """Validate all specs BEFORE app creation/open, and again before reserving."""
    if not specs:
        raise ValueError('Explicit approved specs required')
    current = None if _saved_source else source_identity(Path(root))['source_contents']
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
                or (not _saved_source and not _first_json_equal(current, spec.provenance['source_contents']))):
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


def _validate_no_spawn(spec, root, proof_descriptor, reservation):
    witness, run, _, log = check_admission([spec], root, _saved_source=True)[0]
    proof = _first_artifact(proof_descriptor)
    if (proof.get('approved') is not True or proof.get('owner') != witness['owner']
            or proof.get('run_id') != spec.run_id or proof.get('freeze_id') != witness['freeze_id']
            or proof.get('verified_no_app_opened') is not True
            or proof.get('verified_no_spawn') is not True
            or type(proof.get('actual_interactions')) is not int or proof['actual_interactions'] != 0
            or reservation['event'] != 'reserved'
            or reservation['freeze_id'] != witness['freeze_id']
            or not _first_json_equal(reservation['spec'], frozen_payload(spec))
            or not _first_json_equal(reservation['charges'], run['charges'])
            or not _first_json_equal(reservation['ledger_manifest'], witness['ledger_manifest'])):
        raise ValueError('Invalid manual verified-not-spawned proof')
    error = _first_artifact(proof['stage_error'])
    message = "_Function.with_options() got an unexpected keyword argument 'startup_timeout'"
    exact_error = ((error.get('error_type') == 'TypeError' and error.get('message') == message)
                   or error.get('error') == 'TypeError: ' + message)
    if (not exact_error
            or 'app_id' not in error or error['app_id'] is not None
            or 'call_id' not in error or error['call_id'] is not None):
        raise ValueError('Proof requires the exact pre-app startup_timeout TypeError')
    controller = _first_artifact(proof['controller_source'])
    source = controller.get('source')
    if type(source) is not str:
        raise ValueError('Preserved controller source required')
    options, startup, app, spawn = (source.find(text) for text in
                                  ('.with_options(', 'startup_timeout', 'app.run(', '.spawn('))
    if not (0 <= options < startup < app < spawn):
        raise ValueError('Controller must show startup_timeout failure before app.run and spawn')
    watchdog = _first_artifact(proof['watchdog'])
    if (type(watchdog.get('exit_code')) is not int or watchdog['exit_code'] != 1
            or type(watchdog.get('return_code')) is not int or watchdog['return_code'] != 1
            or watchdog.get('timed_out') is not False
            or watchdog.get('process_tree_clean') is not True
            or watchdog.get('reaped_pids') != [] or watchdog.get('surviving_pids') != []):
        raise ValueError('Clean local failure watchdog required')
    return log


def _journal(rows, log):
    reservations, bound, closed = {}, set(), set()
    for row in rows:
        run_id, event = row['run_id'], row['event']
        if event == 'reserved':
            if run_id in reservations:
                raise ValueError('Duplicate reservation')
            reservations[run_id] = row
        elif event == 'bound':
            if run_id not in reservations or run_id in bound or run_id in closed:
                raise ValueError('Contradictory call binding')
            bound.add(run_id)
        elif event == 'verified_not_spawned':
            if run_id not in reservations or run_id in bound or run_id in closed:
                raise ValueError('Contradictory verified-not-spawned closure')
            if type(row.get('actual_interactions')) is not int or row['actual_interactions'] != 0:
                raise ValueError('Invalid closure consumption')
            # Revalidate the preserved approval and proof, never the current source.
            saved = row['approval']
            witness = _first_artifact(saved)
            from .config import RunSpec
            payload = json.loads(json.dumps(reservations[run_id]['spec']))
            spec = RunSpec(**payload)
            spec.config['native_acceptance']['execution_admission'] = saved
            _validate_no_spawn(spec, None, row['proof'], reservations[run_id])
            if witness['reservation_log'] != row['reservation_log'] or Path(row['reservation_log']) != log:
                raise ValueError('Closure journal changed')
            closed.add(run_id)
        else:
            raise ValueError('Unknown reservation event')
    return reservations, bound, closed


def verify_not_spawned(spec, root, proof_descriptor):
    """Manually close only the proven pre-app SDK failure; never retry this run ID."""
    witness, _, _, log = check_admission([spec], root, _saved_source=True)[0]
    with _locked(log):
        rows = _rows(log)
        reservations, bound, closed = _journal(rows, log)
        if spec.run_id not in reservations or spec.run_id in bound:
            raise ValueError('Closure requires an unbound reservation')
        _validate_no_spawn(spec, root, proof_descriptor, reservations[spec.run_id])
        if spec.run_id in closed:
            old = next(row for row in rows if row['run_id'] == spec.run_id
                       and row['event'] == 'verified_not_spawned')
            if not _first_json_equal(old['proof'], proof_descriptor):
                raise ValueError('Closure proof changed')
            return
        _append(log, {'event': 'verified_not_spawned', 'run_id': spec.run_id,
                      'actual_interactions': 0, 'proof': proof_descriptor,
                      'approval': spec.config['native_acceptance']['execution_admission'],
                      'reservation_log': str(log)})


def reserve(spec, root):
    """Return an old bound call or reserve before spawn; unknown attempts cannot retry."""
    witness, run, ledger, log = check_admission([spec], root)[0]
    with _locked(log):
        rows = _rows(log)
        _, _, closed = _journal(rows, log)
        for row in rows:
            if row['event'] == 'reserved' and not _first_json_equal(
                    row['ledger_manifest'], witness['ledger_manifest']):
                raise ValueError('Reservation history baseline cannot reset or change')
        old = [row for row in rows if row['run_id'] == spec.run_id]
        if old:
            if spec.run_id in closed:
                raise ValueError('Closed run ID cannot retry')
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
            if row['event'] == 'reserved' and row['run_id'] not in closed:
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
