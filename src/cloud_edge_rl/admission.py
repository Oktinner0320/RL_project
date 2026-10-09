"""Local fail-closed approval and append-only durable reservation boundary."""

import json
import os
from copy import deepcopy
from math import isfinite
from contextlib import contextmanager
from pathlib import Path

from .config import source_identity, spec_to_dict
from .comparisons.acceptance import _first_artifact, _first_json_equal, _first_json_object


def frozen_payload(spec):
    """Approval contains the complete spec except its own witness descriptor."""
    payload = spec_to_dict(spec)
    payload['config']['native_acceptance'].pop('execution_admission', None)
    return payload


_RESOURCE_PURPOSE = 'bounded_paired_resource_feedback_training'
_RESOURCE_SCHEMA = 'resource_pair_shared_source_v1'


def resource_frozen_payload(spec):
    """Keep the complete approved spec except the separately witnessed source contents."""
    payload = frozen_payload(spec)
    payload['provenance'].pop('source_contents')
    return payload


def _resource_shared_source(witness):
    if 'artifact_schema' not in witness:
        return False
    if (witness.get('purpose') != _RESOURCE_PURPOSE
            or witness['artifact_schema'] != _RESOURCE_SCHEMA):
        raise ValueError('Unknown admission artifact schema')
    return True


def _resource_training(witness):
    """Authorize only the two explicitly approved settings and their net continuations."""
    compact = _resource_shared_source(witness)
    authorization = witness.get('configuration_authorization')
    if type(authorization) is not dict:
        raise ValueError('Direct configuration authorization required')
    exact = {'approved': True, 'owner': witness['owner'], 'freeze_id': witness['freeze_id'],
             'approved_additional_slots': 2, 'canonical_method': 'ppo', 'seed': 1201,
             'partition': 'train', 'total_steps_per_arm': 1200, 'max_compute_calls': 4,
             'resource_stage_episodes': 8, 'dual_step_size': 1.0,
             'historical_configuration_count_exact': None,
             'historical_configuration_count_unknown_retained': True}
    if any(not _first_json_equal(authorization.get(key), value)
           or key not in authorization for key, value in exact.items()):
        raise ValueError('Invalid explicit paired configuration authorization')
    lower = authorization.get('historical_configuration_lower_bound')
    ids = authorization.get('configuration_ids_by_mode')
    risk_version = authorization.get('risk_version')
    if (type(lower) is not int or lower < 10 or type(ids) is not dict
            or set(ids) != {'fixed', 'dual'}
            or any(type(value) is not str or not value for value in ids.values())
            or len(set(ids.values())) != 2
            or type(risk_version) is not str or not risk_version):
        raise ValueError('Exactly two additional configuration IDs required')
    runs = witness.get('runs')
    if type(runs) is not list or len(runs) != 4:
        raise ValueError('Exactly four frozen compute runs required')
    stages, normalized, run_ids = {}, [], set()
    for run in runs:
        payload = run['spec']
        if compact and (payload.get('provenance', {}).get('identity_kind') != 'source_contents_no_hash'
                        or 'source_contents' in payload['provenance']):
            raise ValueError('Shared-source runs require intact compact source identity')
        config, inputs = payload['config'], payload['inputs']
        rl, modal = config['rl'], config['modal']
        feedback = rl.get('resource_feedback')
        keys = {'mode', 'stage_episodes', 'time_limit', 'energy_limit', 'dual_step_size'}
        if (type(feedback) is not dict or set(feedback) != keys
                or feedback['mode'] not in ids
                or type(feedback['stage_episodes']) is not int or feedback['stage_episodes'] != 8
                or type(feedback['dual_step_size']) is not float or feedback['dual_step_size'] != 1.0
                or any(type(feedback[key]) not in (int, float)
                       or not isfinite(feedback[key]) or feedback[key] <= 0
                       for key in ('time_limit', 'energy_limit'))):
            raise ValueError('Invalid frozen resource feedback settings')
        mode = feedback['mode']
        limit = rl.get('stage_interaction_limit')
        if (type(limit) is not int or limit not in (30, 1170)
                or type(rl.get('total_steps')) is not int or rl['total_steps'] != 1200
                or payload['stage'] != 'train_policy' or payload['method'] != 'ppo'
                or type(payload['seed']) is not int or payload['seed'] != 1201
                or payload['risk_version'] != risk_version
                or type(payload['run_id']) is not str or not payload['run_id']
                or payload['run_id'] in run_ids or (mode, limit) in stages):
            raise ValueError('Invalid paired training run or net interaction budget')
        expected_modal = {'gpu': 'H100', 'cpu': 16.0, 'memory_mb': 65536,
                          'timeout_seconds': 180 if limit == 30 else 900,
                          'startup_timeout_seconds': 180, 'max_containers': 1}
        if any(not _first_json_equal(modal.get(key), value) for key, value in expected_modal.items()):
            raise ValueError('Frozen bounded H100 compute settings required')
        if (config['execution'].get('backend') != 'modal'
                or config['execution'].get('split') != 'train'
                or config['risk'].get('mode') != 'fitted'
                or config['components'].get('risk') != 'researcher'
                or config['risk'].get('mean_calibration') is not True
                or config['risk'].get('calibration_nonnegative') is not True
                or not _first_json_equal(config['risk'].get('calibration_strength'), 0.5)
                or type(inputs) is not dict
                or set(inputs) != {'risk_checkpoint', 'previous_training_checkpoint'}
                or type(inputs['risk_checkpoint']) is not str or not inputs['risk_checkpoint']):
            raise ValueError('Frozen fitted positive calibrated R3 inputs required')
        charge = {'method': 'ppo', 'seed': 1201, 'partition': 'train',
                  'configuration_id': ids[mode], 'actual_interactions': limit}
        if not _first_json_equal(run.get('charges'), [charge]):
            raise ValueError('Exactly one canonical net charge per run required')
        stages[mode, limit] = payload
        run_ids.add(payload['run_id'])
        scientific = deepcopy(config)
        scientific['rl']['resource_feedback'].pop('mode')
        scientific['rl'].pop('stage_interaction_limit')
        for key in ('timeout_seconds',):
            scientific['modal'].pop(key)
        scientific.get('experiment', {}).pop('variant', None)
        # Admission metadata is not a scientific setting; all other native fields stay frozen.
        scientific.get('native_acceptance', {}).pop('execution_admission', None)
        normalized.append([scientific, inputs['risk_checkpoint'], payload.get('provenance')])
    if any(not _first_json_equal(normalized[0], item) for item in normalized[1:]):
        raise ValueError('Paired arms differ beyond the approved feedback mode')
    for mode in ids:
        pre = stages[mode, 30]
        continuation = stages[mode, 1170]
        if (pre['inputs']['previous_training_checkpoint'] not in (None, '')
                or continuation['inputs']['previous_training_checkpoint']
                != f"runs/{pre['run_id']}/training.pt"):
            raise ValueError('Continuation must restore its own arm precheck checkpoint')
    return set(ids.values())


def _resource_history(ledger):
    """The whole frozen pair must fit before the first app is opened."""
    allocated, calls = 0, set()
    for row in ledger:
        if row['call_id'] in calls:
            raise ValueError('Duplicate historical budget call')
        calls.add(row['call_id'])
        amount = row.get('actual_interactions')
        counters = ('learned_steps', 'warmup_steps', 'collection_interactions', 'resume_offset')
        if any(name in row for name in counters):
            counts = [row[name] for name in counters]
            if (any(type(value) is not int or value < 0 for value in counts)
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
        if (row['method'], row['seed'], row['partition']) == ('ppo', 1201, 'train'):
            allocated += amount
    if allocated > 600:
        raise ValueError('Cumulative group interaction limit exceeded by full frozen pair')


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
                or not witness['owner'] or witness.get('purpose') not in
                ('fixedcandidate_independent_validation', _RESOURCE_PURPOSE)):
            raise ValueError('Explicit owner stage approval required')
        if witness['purpose'] == _RESOURCE_PURPOSE:
            _resource_training(witness)
        compact = _resource_shared_source(witness)
        if compact and not _first_json_equal(
                _first_artifact(witness['source_contents_manifest']), spec.provenance['source_contents']):
            raise ValueError('Shared source contents do not match actual spec')
        payload = resource_frozen_payload(spec) if compact else frozen_payload(spec)
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
        if witness['purpose'] == _RESOURCE_PURPOSE:
            _resource_history(ledger)
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
            or not _first_json_equal(reservation['spec'], run['spec'])
            or not _first_json_equal(reservation['charges'], run['charges'])
            or not _first_json_equal(reservation['ledger_manifest'], witness['ledger_manifest'])):
        raise ValueError('Invalid manual verified-not-spawned proof')
    error = _first_artifact(proof['stage_error'])
    message = "_Function.with_options() got an unexpected keyword argument 'startup_timeout'"
    exact_error = ((error.get('error_type') == 'TypeError' and error.get('message') == message)
                   or (error.get('exception_type') == 'TypeError' and error.get('exception') == message)
                   or error.get('error') == 'TypeError: ' + message)
    if (not exact_error
            or 'app_id' not in error or error['app_id'] is not None
            or 'call_id' not in error or error['call_id'] is not None):
        raise ValueError('Proof requires the exact pre-app startup_timeout TypeError')
    controller = _first_artifact(proof['controller_source'])
    source = controller.get('source', controller.get('source_text'))
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
            if _resource_shared_source(witness):
                payload['provenance']['source_contents'] = _first_artifact(witness['source_contents_manifest'])
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
        if any(not _first_json_equal(row['ledger_manifest'], witness['ledger_manifest'])
               for row in reservations.values()):
            raise ValueError('Reservation history baseline cannot reset or change')
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


def bound_call(spec, root):
    """Read a verified durable binding without authorizing or submitting new compute."""
    witness, run, _, log = check_admission([spec], root, _saved_source=True)[0]
    with _locked(log):
        rows = _rows(log)
        reservations, bound, closed = _journal(rows, log)
        saved = reservations.get(spec.run_id)
        if (saved is None or spec.run_id not in bound or spec.run_id in closed
                or saved['freeze_id'] != witness['freeze_id']
                or not _first_json_equal(saved['spec'], run['spec'])
                or not _first_json_equal(saved['charges'], run['charges'])
                or not _first_json_equal(saved['ledger_manifest'], witness['ledger_manifest'])):
            raise ValueError('Retained call requires an exact durable reservation binding')
        return next(row['call_id'] for row in rows
                    if row['event'] == 'bound' and row['run_id'] == spec.run_id)


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
            if (not _first_json_equal(old[0]['spec'], run['spec'])
                    or not _first_json_equal(old[0]['charges'], run['charges'])):
                raise ValueError('Reserved spec changed')
            bound = [row for row in old if row['event'] == 'bound']
            if len(bound) == 1:
                return bound[0]['call_id']
            raise ValueError('Unknown paid spawn remains reserved; manual reconciliation required')
        resource_ids = None
        if witness['purpose'] == _RESOURCE_PURPOSE:
            resource_ids = _resource_training(witness)
            reservations = [row for row in rows if row['event'] == 'reserved']
            if len(reservations) >= 4:
                raise ValueError('Maximum four compute reservations exceeded')
            if any(row['freeze_id'] != witness['freeze_id'] for row in reservations):
                raise ValueError('Paired training freeze cannot reset or change')
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
                    or (charge['configuration_id'] not in
                        (resource_ids if resource_ids is not None else
                         configurations.get((key[0], key[2]), set())))):
                raise ValueError('No new configuration tuning is authorized')
            totals[key] = totals.get(key, 0) + amount
            if amount and totals[key] > 3000:
                raise ValueError('Cumulative group interaction limit exceeded')
        _append(log, {'event': 'reserved', 'run_id': spec.run_id, 'spec': run['spec'],
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
