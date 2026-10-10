"""Local, fresh-only aggregate DAG admission; never opens an app or binds a call.

Graph approval schema ``fresh_graph_admission_v1`` explicitly binds graph_snapshot,
legacy_admission and source_contents_manifest descriptors plus owner/freeze_id.
Future producers need not exist: execution must independently enforce verified
dependency barriers. Spent IDs cannot be imported by this first-stage adapter.
"""
from copy import deepcopy
from pathlib import Path

from . import admission
from .comparisons.acceptance import _first_artifact, _first_json_equal
from .config import source_identity
from .experiment_graph import expand, validate_graph
from .transport import read_json, write_json


SCHEMA = 'fresh_graph_admission_v1'
_FIELDS = {'schema', 'approved', 'owner', 'freeze_id', 'purpose', 'experiment_id',
           'graph_snapshot', 'legacy_admission', 'source_contents_manifest',
           'official_test_allowed', 'adaptive_tuning_allowed', 'automatic_retry',
           'historical_consumption_reset'}


def _normalized(payload):
    payload = deepcopy(payload)
    payload['config']['native_acceptance'].pop('execution_admission', None)
    payload['provenance'].pop('source_contents', None)
    return payload


def _limit(key):
    return 3090 if key[2] == 'train' else 180 if key[0] in ('dqn', 'ppo', 'pqn') else 30


def _charges(runs):
    totals, collected = {}, 0
    for run in runs:
        for charge in run['charges']:
            amount = charge['actual_interactions']
            if type(amount) is not int or amount < 0:
                raise ValueError('Exact nonnegative planned physical interactions required')
            key = (charge['method'], charge['seed'], charge['partition'])
            totals[key] = totals.get(key, 0) + amount
            if charge['method'] == 'round_robin':
                collected += amount
    # Charge every future shared collection to each learned-policy beneficiary.
    for method in ('dqn', 'ppo', 'pqn'):
        for seed in (8101, 8102, 8103):
            key = (method, seed, 'train')
            totals[key] = totals.get(key, 0) + collected
    return totals, collected


def _rows(totals):
    return [{'method': m, 'seed': s, 'partition': p, 'interactions': value}
            for (m, s, p), value in sorted(totals.items())]


def validate_and_reserve_graph(graph: dict, root: Path, controller_dir: Path) -> dict:
    """Validate explicit fresh graph approval and durably reserve aggregate caps.

    Returns a reservation record, NOT a per-node paid call binding. Repeating the
    exact graph/config/history is idempotent. No execution or retry is authorized
    without a true owner approval, exact legacy contracts and current source.
    """
    root, directory = Path(root).resolve(), Path(controller_dir).resolve()
    validate_graph(graph)
    if not _first_json_equal(read_json(directory / 'snapshot.json'), graph):
        raise ValueError('Graph differs from immutable controller snapshot')
    configuration = read_json(directory / 'configuration.json')
    descriptor = configuration['workflow']['experiment'].get('graph_admission')
    if type(descriptor) is not dict or set(descriptor) != {'path', 'snapshot'}:
        raise ValueError('Explicit owner graph admission descriptor required')
    approval = _first_artifact(descriptor)
    if (type(approval) is not dict or set(approval) != _FIELDS
            or approval['schema'] != SCHEMA or approval['approved'] is not True
            or approval['purpose'] != admission._FRESH_PURPOSE
            or approval['experiment_id'] != graph['experiment_id']
            or any(approval[key] is not False for key in ('official_test_allowed',
                   'adaptive_tuning_allowed', 'automatic_retry', 'historical_consumption_reset'))
            or type(approval['owner']) is not str or not approval['owner']
            or type(approval['freeze_id']) is not str or not approval['freeze_id']):
        raise ValueError('Exact explicit fresh graph approval required')
    if not _first_json_equal(_first_artifact(approval['graph_snapshot']), graph):
        raise ValueError('Approval graph snapshot mismatch')
    witness = _first_artifact(approval['legacy_admission'])
    if (witness.get('approved') is not True or witness.get('purpose') != admission._FRESH_PURPOSE
            or any(witness.get(key) != approval[key] for key in ('owner', 'freeze_id'))):
        raise ValueError('Trusted legacy owner/freeze approval required')
    contract = admission._fresh_contract(witness)
    freeze = _first_artifact(witness['freeze_manifest'])
    isolation = _first_artifact(witness['truth_isolation'])
    if (freeze.get('freeze_id') != witness['freeze_id']
            or not _first_json_equal(freeze.get('runs'), witness['runs'])
            or isolation.get('freeze_id') != witness['freeze_id']
            or type(isolation.get('observed', {}).get('truth_isolation')) is not int
            or isolation['observed']['truth_isolation'] != 1):
        raise ValueError('Exact legacy freeze and truth isolation required')
    approved_source = _first_artifact(approval['source_contents_manifest'])
    if (not _first_json_equal(approved_source, _first_artifact(witness['source_contents_manifest']))
            or not _first_json_equal(source_identity(root)['source_contents'], approved_source)):
        raise ValueError('Trusted source contents mismatch')
    approved_runs = {r['spec']['run_id']: r for r in witness['runs']}
    dag = {n['run_id']: n for n in contract['dag']}
    selected = []
    for node in graph['nodes']:
        payload = expand(graph['pool'], node['spec_ref'])
        run = approved_runs.get(node['run_id'])
        if (run is None or not _first_json_equal(_normalized(payload), run['spec'])
                or set(node['dependencies']) != set(dag[node['run_id']]['dependencies'])
                or node['verifier'] != 'cloud_edge_rl.fresh_benchmark:verify_fresh_artifacts'
                or payload['provenance'].get('identity_kind') != 'source_contents_no_hash'
                or not _first_json_equal(payload['provenance'].get('source_contents'), approved_source)):
            raise ValueError('Graph spec/source/dependencies differ from trusted frozen fresh runs')
        selected.append(run)
    output = Path(contract['output_root']).resolve()
    log = Path(witness['reservation_log'])
    if not log.is_absolute() or not log.resolve().is_relative_to(output) or not directory.is_relative_to(output):
        raise ValueError('Fresh graph reservation must stay in approved output scope')
    reservation_path = directory / 'graph-reservation.json'
    registry_path = log.with_suffix(log.suffix + '.graph-reservations.json')
    # Share the legacy OS lock, but never append new events to its historical log.
    with admission._locked(log):
        ledger = _first_artifact(witness['ledger_manifest'])
        if type(ledger) is not list:
            raise ValueError('Explicit historical ledger required')
        totals, collected = admission._fresh_history(witness, ledger)
        journal = admission._rows(log)
        reservations, bound, closed = admission._journal(journal, log)
        retained = admission._fresh_recovery(witness, ledger, log, journal)
        active_ids = {r['spec']['run_id'] for r in selected}
        spent = {r.get('run_id') for r in ledger}
        if active_ids & (spent | set(retained) | set(reservations) | closed):
            raise ValueError('Spent, failed or unknown reserved graph run IDs cannot be reused')
        known = dict(approved_runs, **retained)
        ledger_calls = {r['call_id'] for r in ledger}
        bindings = {r['run_id']: r['call_id'] for r in journal if r['event'] == 'bound'}
        for run_id, saved in reservations.items():
            expected = known.get(run_id)
            if (expected is None or saved['freeze_id'] != witness['freeze_id']
                    or not _first_json_equal(saved['spec'], expected['spec'])
                    or not _first_json_equal(saved['charges'], expected['charges'])):
                raise ValueError('Retained fresh journal contract drift')
            baseline = _first_artifact(saved['ledger_manifest'])
            if not _first_json_equal(baseline, ledger[:len(baseline)]):
                raise ValueError('Historical ledger prefix drift')
            if run_id in closed or bindings.get(run_id) in ledger_calls:
                continue
            old_totals, old_collected = _charges([expected])
            for key, amount in old_totals.items():
                totals[key] = totals.get(key, 0) + amount
            collected += old_collected
        planned, physical = _charges(selected)
        for key, amount in planned.items():
            if totals.get(key, 0) + amount > _limit(key):
                raise ValueError('Fresh aggregate scoped interaction limit exceeded')
        if collected + physical > 90:
            raise ValueError('Fresh aggregate physical collection exceeds 90')
        record = {'schema': 'fresh_graph_reservation_v1', 'experiment_id': graph['experiment_id'],
                  'owner': witness['owner'], 'freeze_id': witness['freeze_id'],
                  'graph_ref': str(directory / 'snapshot.json'), 'configuration_ref': str(directory / 'configuration.json'),
                  'approval_ref': descriptor['path'], 'ledger_ref': witness['ledger_manifest']['path'],
                  'run_ids': [n['run_id'] for n in graph['nodes']], 'planned_charges': _rows(planned),
                  'historical_charges': _rows(totals), 'planned_physical_interactions': physical,
                  'historical_physical_interactions': collected, 'automatic_retry': False,
                  'historical_consumption_reset': False, 'call_id': None}
        registry = read_json(registry_path) if registry_path.exists() else []
        matching = [r for r in registry if r['reservation_ref'] == str(reservation_path)]
        if reservation_path.exists() or matching:
            if (len(matching) != 1 or not reservation_path.exists()
                    or not _first_json_equal(read_json(reservation_path), record)
                    or matching[0]['experiment_id'] != graph['experiment_id']):
                raise ValueError('Aggregate reservation identity/history drift')
            return record
        if registry:
            # A previous one-call graph has unknown consumption until explicitly
            # reconciled; never reset its allocation or assume its call is free.
            raise ValueError('Unreconciled prior aggregate graph reservation remains charged')
        write_json(reservation_path, record)
        write_json(registry_path, [{'reservation_ref': str(reservation_path),
                                   'experiment_id': graph['experiment_id'],
                                   'planned_charges': record['planned_charges'],
                                   'planned_physical_interactions': physical}])
        return record
