"""Explicit manual full-rerun approval; local reservation, never submission.

No legacy approval trees, hashes, result imports or execution path. Descriptors
retain literal JSON once; source bytes are those read by source_identity.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from . import admission
from .comparisons.acceptance import _first_json_equal
from .config import scientific_configuration, source_identity
from .experiment_graph import MAX_METADATA, expand, validate_graph
from .fresh_benchmark import build_plan, build_run_specs
from .transport import write_json

SCHEMA = 'fresh_full_rerun_admission_v1'
_FLAGS = ('official_test_allowed', 'formal_confirmation_allowed',
          'adaptive_tuning_allowed', 'automatic_retry', 'historical_consumption_reset')
_FIELDS = {'schema', 'approved', 'owner', 'authorization_id', 'experiment_id',
           'graph_snapshot', 'configuration', 'source_contents_manifest', 'history',
           'historical_consumption_retained', *_FLAGS}


def _read(path):
    from .experiment_admission import _read_approval
    return _read_approval(Path(path))


def _equal(actual, expected, message):
    if not _first_json_equal(actual, expected):
        raise ValueError(message)


def _descriptor(value):
    if (type(value) is not dict or set(value) != {'path', 'snapshot'}
            or type(value['path']) is not str or not Path(value['path']).is_absolute()):
        raise ValueError('Absolute no-hash path/snapshot descriptor required')
    actual = _read(value['path'])
    _equal(actual, value['snapshot'], 'Bound artifact snapshot drift')
    return actual


def _bounded_write(path, value):
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')) > MAX_METADATA:
        raise ValueError('Rerun metadata bound exceeded')
    write_json(path, value)


def _artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'snapshot': _read(path)}


def _history(descriptor, run_ids):
    history = _descriptor(descriptor)
    if (history.get('schema') != 'fresh_rerun_history_v1'
            or history.get('total_historical_cost_usd', 'missing') is not None
            or type(history.get('additional_completed_calls')) is not list):
        raise ValueError('Explicit retained history with unknown total cost required')
    ledger_descriptor = history.get('retained_ledger')
    # The original raw rows are immutable; later rows cannot replace that prefix.
    if (type(ledger_descriptor) is not dict or set(ledger_descriptor) != {'path', 'snapshot'}
            or type(ledger_descriptor['path']) is not str
            or not Path(ledger_descriptor['path']).is_absolute()
            or type(ledger_descriptor['snapshot']) is not list):
        raise ValueError('Retained raw ledger descriptor required')
    ledger = _read(ledger_descriptor['path'])
    if type(ledger) is not list:
        raise ValueError('Raw historical ledger rows required')
    _equal(ledger[:len(ledger_descriptor['snapshot'])], ledger_descriptor['snapshot'],
           'Historical ledger prefix drift')
    calls = history.get('completed_scope_calls')
    if type(calls) is not list or len(calls) != 5:
        raise ValueError('Five explicit completed scope call evidence rows required')
    call_ids, evidence_runs, scopes = set(), set(), []
    groups, physical, collected = {}, 0, 0
    for row in calls:
        if (type(row) is not dict or any(type(row.get(k)) is not str or not row[k]
                for k in ('run_id', 'call_id', 'method', 'partition'))
                or type(row.get('seed')) is not int
                or type(row.get('actual_interactions')) is not int
                or row['actual_interactions'] <= 0 or type(row.get('shared_collection')) is not bool
                or row['call_id'] in call_ids or row['run_id'] in evidence_runs):
            raise ValueError('Exact distinct completed call/run evidence required')
        call_ids.add(row['call_id'])
        evidence_runs.add(row['run_id'])
        key = row['method'], row['seed'], row['partition']
        amount = row['actual_interactions']
        scopes.append((*key, amount, row['shared_collection']))
        groups[key] = groups.get(key, 0) + amount
        physical += amount
        if row['shared_collection']:
            collected += amount
    expected = [('round_robin', seed, 'train', 30, True) for seed in (8301, 8302, 8303)]
    expected += [('dqn', 8101, 'train', amount, False) for amount in (500, 1000)]
    _equal([list(row) for row in sorted(scopes)], [list(row) for row in sorted(expected)],
           'Known prior completed scope evidence mismatch')
    if type(history.get('known_previous_physical_interactions')) is not int or history['known_previous_physical_interactions'] != physical:
        raise ValueError('Historical physical total must derive from distinct completed calls')
    for method in ('dqn', 'ppo', 'pqn'):
        for seed in (8101, 8102, 8103):
            key = method, seed, 'train'
            groups[key] = groups.get(key, 0) + collected
    from .experiment_admission import _rows
    _equal(history.get('known_previous_group_charges'), _rows(groups),
           'Historical group totals must derive from completed call evidence')
    extra_ids, extra_calls = set(), set()
    evidence = {row['run_id']: row for row in calls}
    for row in history['additional_completed_calls']:
        if (type(row) is not dict or type(row.get('run_id')) is not str
                or not row['run_id'] or type(row.get('call_id')) is not str or not row['call_id']
                or row['run_id'] in extra_ids or row['call_id'] in extra_calls):
            raise ValueError('Distinct additional completed run/call identities required')
        extra_ids.add(row['run_id'])
        extra_calls.add(row['call_id'])
        if row['run_id'] in evidence:
            _equal(row, evidence[row['run_id']], 'Additional completed call evidence drift')
        elif row['call_id'] in call_ids:
            raise ValueError('Completed call identity assigned to another historical run')
    for receipt in history.get('completed_receipts', []):
        _descriptor(receipt)
    old_ids = evidence_runs | extra_ids
    for row in ledger:
        if type(row) is not dict or type(row.get('run_id')) is not str or not row['run_id']:
            raise ValueError('Historical ledger run identities required')
        old_ids.add(row['run_id'])
    if set(run_ids) & old_ids:
        raise ValueError('Rerun identities intersect retained historical runs')
    return groups, physical, ledger


def _check_graph(graph, root, directory, configuration, source):
    from .experiment_admission import _charges
    validate_graph(graph)
    _equal(_read(directory / 'snapshot.json'), graph, 'Controller graph snapshot drift')
    _equal(scientific_configuration(_read(root / 'config/experiment.json')),
           scientific_configuration(configuration), 'Live scientific configuration drift')
    _equal(source_identity(root)['source_contents'], source, 'Current source contents drift')
    plan = build_plan(configuration)
    if len(graph['nodes']) != 101 or plan['planned_physical_interactions_max'] != 29070:
        raise ValueError('Complete 101-node 29070-interaction fresh plan required')
    if configuration['workflow']['experiment'].get('mode') != 'fresh':
        raise ValueError('Fresh rerun configuration required')
    identities = {p['id']: n['run_id'] for p, n in zip(plan['nodes'], graph['nodes'])}
    first = expand(graph['pool'], graph['nodes'][0]['spec_ref'])
    provenance = first['provenance']
    if provenance.get('identity_kind') != 'source_contents_no_hash':
        raise ValueError('Literal source provenance required')
    actual_source = provenance.get('source_contents')
    if actual_source is None and provenance.get('source_contents_path'):
        actual_source = _read(provenance['source_contents_path'])
    _equal(actual_source, source, 'Graph source provenance drift')
    specs = build_run_specs(configuration, root, identities, provenance)
    runs, physical = [], 0
    for p, node, spec in zip(plan['nodes'], graph['nodes'], specs):
        payload = expand(graph['pool'], node['spec_ref'])
        _equal(payload, asdict(spec), 'Canonical fresh specification/input drift')
        _equal(node['dependencies'], sorted(identities[d] for d in p['dependencies']),
               'Canonical fresh DAG dependencies drift')
        if node['verifier'] != 'cloud_edge_rl.fresh_benchmark:verify_fresh_artifacts':
            raise ValueError('Fresh semantic verifier required')
        if payload['config']['native_acceptance'].get('execution_admission') is not None:
            raise ValueError('Legacy per-node authorization forbidden for full rerun')
        amount = p.get('interaction_limit', 0)
        physical += amount
        runs.append({'charges': [{'method': spec.method, 'seed': spec.seed,
                     'partition': spec.config['execution']['split'], 'actual_interactions': amount}]})
    if physical != 29070:
        raise ValueError('Full rerun physical plan mismatch')
    return _charges(runs)[0]


def _approval_path(configuration, root):
    value = configuration['workflow']['experiment'].get('graph_admission')
    if type(value) is not str or not value:
        raise ValueError('Configured explicit graph_admission path required')
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def prepare_rerun_approval(graph, root, controller_dir, *, owner, history_descriptor, authorization_id) -> dict:
    """Manual owner action ONLY: publish approval, not reserve/submit a call."""
    root, directory = Path(root).resolve(), Path(controller_dir).resolve()
    if any(type(x) is not str or not x.strip() for x in (owner, authorization_id)):
        raise ValueError('Explicit owner and one-full-rerun authorization identity required')
    configuration = _read(directory / 'configuration.json')
    first = expand(graph['pool'], graph['nodes'][0]['spec_ref'])['provenance']
    source_path = first.get('source_contents_path')
    if type(source_path) is not str or not Path(source_path).is_absolute():
        raise ValueError('Original graph absolute source descriptor path required')
    approval = {'schema': SCHEMA, 'approved': True, 'owner': owner,
                'authorization_id': authorization_id, 'experiment_id': graph['experiment_id'],
                'graph_snapshot': _artifact(directory / 'snapshot.json'),
                'configuration': _artifact(directory / 'configuration.json'),
                'source_contents_manifest': _artifact(source_path),
                'history': deepcopy(history_descriptor), 'historical_consumption_retained': True,
                **{key: False for key in _FLAGS}}
    _validate(graph, root, directory, approval)
    path = _approval_path(configuration, root)
    if path.exists():
        _equal(_read(path), approval, 'Existing rerun approval identity drift')
    else:
        _bounded_write(path, approval)
    return approval


def _validate(graph, root, directory, approval):
    if (type(approval) is not dict or set(approval) != _FIELDS or approval['schema'] != SCHEMA
            or approval['approved'] is not True or approval['historical_consumption_retained'] is not True
            or any(approval[k] is not False for k in _FLAGS)
            or any(type(approval[k]) is not str or not approval[k].strip() for k in ('owner', 'authorization_id'))
            or approval['experiment_id'] != graph['experiment_id']):
        raise ValueError('Exact explicit owner full-rerun approval required')
    _equal(_descriptor(approval['graph_snapshot']), graph, 'Approved graph drift')
    config = _descriptor(approval['configuration'])
    _equal(config, _read(directory / 'configuration.json'), 'Approved configuration drift')
    source = _descriptor(approval['source_contents_manifest'])
    planned = _check_graph(graph, root, directory, config, source)
    historical, physical, ledger = _history(approval['history'], [n['run_id'] for n in graph['nodes']])
    output = Path(config['native_acceptance']['fresh_benchmark']['output_root'])
    output = (output if output.is_absolute() else root / output).resolve()
    if not directory.is_relative_to(output):
        raise ValueError('Controller must remain under new fresh output root')
    return planned, historical, physical, ledger, output


def reserve_rerun(graph, root, directory, approval_path, approval):
    """Existing dispatcher branch: atomically charge ONE local full-rerun graph."""
    from .experiment_admission import _rows
    config = _read(directory / 'configuration.json')
    output = Path(config['native_acceptance']['fresh_benchmark']['output_root'])
    output = (output if output.is_absolute() else root / output).resolve()
    registry_path = output / 'full-rerun-reservations.json'
    reservation_path = directory / 'graph-reservation.json'
    with admission._locked(registry_path):
        planned, historical, physical, ledger, checked_output = _validate(graph, root, directory, approval)
        if checked_output != output or _approval_path(config, root) != approval_path:
            raise ValueError('Rerun approval/output scope drift')
        record = {'schema': 'fresh_full_rerun_reservation_v1', 'ready': True,
                  'experiment_id': graph['experiment_id'], 'owner': approval['owner'],
                  'authorization_id': approval['authorization_id'], 'approval_ref': str(approval_path),
                  'reservation_ref': str(reservation_path), 'run_ids': [n['run_id'] for n in graph['nodes']],
                  'planned_charges': _rows(planned), 'historical_charges': _rows(historical),
                  'cumulative_known_charges': _rows({k: v + historical.get(k, 0) for k, v in planned.items()}),
                  'planned_physical_interactions': 29070, 'historical_physical_interactions': physical,
                  'cumulative_known_physical_interactions': 29070 + physical,
                  'historical_global_total_known': False, 'total_historical_cost_usd': None,
                  'historical_ledger_rows_retained': len(ledger), 'historical_consumption_reset': False,
                  'automatic_retry': False, 'call_id': None}
        # Registry is authoritative, published first: an interrupted local mirror
        # can be repaired, but cannot free an already reserved allocation.
        if registry_path.exists():
            registry = _read(registry_path)
            _equal(registry, record, 'Prior full-rerun reservation blocks another graph/authorization')
        else:
            if reservation_path.exists():
                raise ValueError('Unregistered existing graph reservation blocks rerun')
            _bounded_write(registry_path, record)
        if reservation_path.exists():
            _equal(_read(reservation_path), record, 'Rerun reservation drift')
        else:
            _bounded_write(reservation_path, record)
        return record
