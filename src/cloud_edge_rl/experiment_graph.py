"""Flat immutable experiments and a serial, restartable, cloud-owned execution graph.

The pool interns literal JSON values, not file digests. Historical approvals are
resolved on the submitting host; no host path is used for cloud execution.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from .config import RunSpec
from .transport import read_json, run_bounded, safe_run_id, write_json

SCHEMA = 'experiment_graph_v1'
MAX_METADATA = 32 * 1024 * 1024
VERIFIERS = frozenset({
    'cloud_edge_rl.fresh_benchmark:verify_fresh_artifacts',
    'cloud_edge_rl.training_pipeline:verify_training_artifacts',
    'cloud_edge_rl.parameter_selection_pipeline:verify_validation_artifacts',
    'cloud_edge_rl.experiment_graph:verify_stage_artifacts',
})


def _identity(value):
    safe_run_id(value)
    if any(ord(c) < 32 or c in '<>"|?*' for c in value):
        raise ValueError('Unsafe experiment identity')
    return value


def _relative(value):
    if type(value) is not str or '\\' in value or ':' in value:
        raise ValueError('Portable workspace-relative artifact path required')
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Confined artifact path required')
    return path


class ContentPool:
    """Factor repeated subtrees once, retaining exact JSON types and contents."""
    def __init__(self):
        self.values, self.index = [], {}

    def add(self, value):
        if type(value) is dict:
            entry = ['object', [[k, self.add(v)] for k, v in sorted(value.items())]]
        elif type(value) is list:
            entry = ['array', [self.add(v) for v in value]]
        elif type(value) in (str, int, float, bool, type(None)):
            entry = ['value', value]
        else:
            raise ValueError('Strict JSON execution snapshot required')
        key = json.dumps(entry, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        if key not in self.index:
            self.index[key] = len(self.values)
            self.values.append(entry)
        return self.index[key]


def expand(pool, reference):
    cache = {}

    def get(ref, depth=0):
        if type(ref) is not int or not 0 <= ref < len(pool) or depth > 100:
            raise ValueError('Invalid content reference')
        if ref in cache:
            return cache[ref]
        kind, body = pool[ref]
        if kind == 'value':
            value = body
        else:
            refs = [r for _, r in body] if kind == 'object' else body
            if kind not in ('object', 'array') or any(type(r) is not int or r >= ref for r in refs):
                raise ValueError('Cyclic or forward content reference')
            if kind == 'object':
                if len({k for k, _ in body}) != len(body):
                    raise ValueError('Duplicate pooled object key')
                value = {k: get(r, depth + 1) for k, r in body}
            else:
                value = [get(r, depth + 1) for r in body]
        cache[ref] = value
        return value

    return copy.deepcopy(get(reference))


def input_runs(spec):
    """Discover exact producer IDs from declared paths, never from opaque IDs."""
    found = set()
    for key, value in spec.inputs.items():
        if value is None:
            continue
        if key.endswith('_checkpoint'):
            path = _relative(value)
            if len(path.parts) != 3 or path.parts[0] != 'runs':
                raise ValueError('Checkpoint must name an explicit run artifact')
            found.add(_identity(path.parts[1]))
        elif key.endswith('_run_ids'):
            if type(value) is not list:
                raise ValueError('Explicit input run list required')
            found.update(_identity(x) for x in value)
        elif key.endswith('_run_id'):
            found.add(_identity(value))
    return found


def freeze_graph(specs, *, experiment_id, verifier, node_dependencies=None,
                 node_seconds=1800, verify_seconds=90, graph_seconds=86400):
    """Freeze approved specifications; preparing a graph grants no paid permission."""
    _identity(experiment_id)
    if not specs or len(specs) > 256:
        raise ValueError('One to 256 explicitly bounded nodes required')
    pool, nodes, seen = ContentPool(), [], set()
    for spec in specs:
        run = _identity(spec.run_id)
        if run in seen:
            raise ValueError('Duplicate run identity')
        deps = input_runs(spec) | set((node_dependencies or {}).get(run, []))
        if not deps <= seen:
            raise ValueError(f'Missing or out-of-order producer for {run}: {sorted(deps - seen)}')
        gate = verifier[run] if isinstance(verifier, dict) else verifier
        nodes.append({'run_id': run, 'spec_ref': pool.add(asdict(spec)),
                      'dependencies': sorted(deps), 'verifier': gate,
                      'timeout_seconds': node_seconds, 'verify_seconds': verify_seconds})
        seen.add(run)
    graph = {'schema': SCHEMA, 'experiment_id': experiment_id,
             'pool': pool.values, 'nodes': nodes, 'graph_seconds': graph_seconds,
             'automatic_compute_retry': False, 'historical_consumption_reset': False}
    validate_graph(graph)
    return graph


def validate_graph(graph):
    if (graph.get('schema') != SCHEMA or graph.get('automatic_compute_retry') is not False
            or graph.get('historical_consumption_reset') is not False):
        raise ValueError('Immutable no-retry graph contract required')
    _identity(graph['experiment_id'])
    if not 1 <= len(graph['nodes']) <= 256:
        raise ValueError('Graph node limit exceeded')
    if type(graph['graph_seconds']) is not int or not 60 <= graph['graph_seconds'] <= 86400:
        raise ValueError('Bounded graph duration required')
    if len(json.dumps(graph, allow_nan=False).encode('utf-8')) > MAX_METADATA:
        raise ValueError('Factored graph metadata limit exceeded')
    seen = set()
    for node in graph['nodes']:
        run = _identity(node['run_id'])
        if run in seen or not set(node['dependencies']) <= seen:
            raise ValueError('Invalid graph order or duplicate node')
        spec = RunSpec(**expand(graph['pool'], node['spec_ref']))
        if spec.run_id != run or not input_runs(spec) <= set(node['dependencies']):
            raise ValueError('Producer/specification binding mismatch')
        if node['verifier'] not in VERIFIERS:
            raise ValueError('Unregistered semantic verifier')
        for key in ('timeout_seconds', 'verify_seconds'):
            if type(node[key]) is not int or not 1 <= node[key] <= 7200:
                raise ValueError('Explicit bounded node duration required')
        if node['timeout_seconds'] + node['verify_seconds'] + 60 > graph['graph_seconds']:
            raise ValueError('Graph deadline cannot contain one bounded node')
        split = spec.config['execution']['split']
        if spec.stage == 'validate' and split != 'validation':
            raise ValueError('Validation isolation mismatch')
        if spec.stage == 'evaluate' and split != 'test':
            raise ValueError('Final-test isolation mismatch')
        seen.add(run)


def _node_directory(workspace, graph, run):
    return Path(workspace) / 'experiments' / graph['experiment_id'] / 'nodes' / _identity(run)


def execute_node(graph, node, workspace, phase):
    directory = _node_directory(workspace, graph, node['run_id'])
    directory.mkdir(parents=True, exist_ok=True)
    spec = RunSpec(**expand(graph['pool'], node['spec_ref']))
    if phase == 'compute':
        from .pipeline import run_experiment
        result = run_experiment(spec, Path(workspace))
        write_json(directory / 'result.json', result)
    elif phase == 'verify':
        result = read_json(directory / 'result.json')
        module, name = node['verifier'].split(':')
        proof = getattr(importlib.import_module(module), name)(
            spec, Path(workspace) / 'runs' / spec.run_id, result)
        if not isinstance(proof, dict) or proof.get('verified') is not True:
            raise ValueError('Semantic verifier must return evidence')
        write_json(directory / 'verification.json', proof, replace=True)
    else:
        raise ValueError('Unknown node phase')


def verify_stage_artifacts(spec, artifacts, result):
    """General evaluation closure; specialized native stages use their own gate."""
    from .artifacts import artifact_payload
    stored = read_json(Path(artifacts) / 'config.json')
    # Producer metadata can be either the native RunSpec or its public projection.
    if artifact_payload(stored) != artifact_payload(spec):
        raise ValueError('Actual producer configuration mismatch')
    if read_json(Path(artifacts) / 'status.json').get('state') != 'completed':
        raise ValueError('Stage did not complete')
    if result != read_json(Path(artifacts) / 'summary.json') or result.get('run_id') != spec.run_id:
        raise ValueError('Returned and persisted results differ')
    if spec.stage == 'evaluate':
        if spec.config['execution']['split'] != 'test' or not isinstance(result.get('test'), dict):
            raise ValueError('Held-out test result missing')
    elif spec.stage == 'evaluate_risk':
        if spec.config['execution']['split'] == 'train' or not (Path(artifacts) / 'risk_predictions.csv').is_file():
            raise ValueError('Independent risk evaluation missing')
    else:
        raise ValueError('Use the registered specialized verifier for this stage')
    return {'run_id': spec.run_id, 'stage': spec.stage, 'verified': True,
            'formal_acceptance': 'not_evaluated'}


def run_graph(graph, workspace, *, commit, worker=run_bounded, clock=time.monotonic):
    """Computed nodes reverify; ambiguous/failed compute never automatically reruns."""
    validate_graph(graph)
    base = Path(workspace) / 'experiments' / graph['experiment_id']
    base.mkdir(parents=True, exist_ok=True)
    frozen, state_path = base / 'snapshot.json', base / 'state.json'
    if frozen.exists():
        if read_json(frozen) != graph:
            raise ValueError('Frozen experiment changed')
    else:
        write_json(frozen, graph)
    if state_path.exists():
        state = read_json(state_path)
        if (state.get('experiment_id') != graph['experiment_id']
                or set(state.get('nodes', {})) != {n['run_id'] for n in graph['nodes']}):
            raise ValueError('Durable state identity mismatch')
    else:
        state = {'schema': SCHEMA, 'experiment_id': graph['experiment_id'],
                 'status': 'pending', 'actual_cloud_cost_usd': None,
                 'nodes': {n['run_id']: {'status': 'pending'} for n in graph['nodes']}}

    def save():
        write_json(state_path, state, replace=True)
        commit()

    start = clock()
    state['status'] = 'running'
    save()
    try:
        for node in graph['nodes']:
            record = state['nodes'][node['run_id']]
            if record['status'] == 'verified':
                receipt = read_json(_node_directory(workspace, graph, node['run_id']) / 'verification.json')
                if receipt.get('run_id') != node['run_id'] or receipt.get('verified') is not True:
                    raise ValueError('Verified receipt missing or invalid')
                # No hashes: recheck actual dependencies before consuming them on
                # a later invocation. This phase cannot perform training.
                record['status'] = 'computed'
            if record['status'] not in ('pending', 'computed', 'verification_failed'):
                raise ValueError('Unknown or failed compute requires explicit reconciliation')
            if any(state['nodes'][dep]['status'] != 'verified' for dep in node['dependencies']):
                raise ValueError('Unverified dependency')
            required = (node['timeout_seconds'] if record['status'] == 'pending' else 0) + node['verify_seconds'] + 60
            # Large real runs must leave time to close their bounded export shards.
            if graph['graph_seconds'] >= 1800 and any(
                    expand(graph['pool'], item['spec_ref']).get('config', {}).get('rl', {}).get('total_steps', 0) >= 1000
                    for item in graph['nodes']):
                required += 1800
            if graph['graph_seconds'] - (clock() - start) < required:
                state['status'] = 'paused_deadline'
                break
            directory = _node_directory(workspace, graph, node['run_id'])
            directory.mkdir(parents=True, exist_ok=True)
            phases = ('compute', 'verify') if record['status'] == 'pending' else ('verify',)
            for phase in phases:
                if phase == 'compute':
                    if (Path(workspace) / 'runs' / node['run_id']).exists():
                        raise ValueError('Existing run requires explicit import; never overwrite computation')
                    record.update(status='reserved', attempt_id=uuid.uuid4().hex,
                                  actual_interactions=None)
                    save()  # Committed before the first possible interaction.
                    record['status'] = 'running'
                    save()
                try:
                    worker([sys.executable, '-m', 'cloud_edge_rl.experiment_graph',
                            '--snapshot', str(frozen), '--workspace', str(workspace),
                            '--run-id', node['run_id'], '--phase', phase],
                           node['timeout_seconds'] if phase == 'compute' else node['verify_seconds'],
                           directory / f'{phase}-{uuid.uuid4().hex}.log')
                    required_file = 'result.json' if phase == 'compute' else 'verification.json'
                    receipt = read_json(directory / required_file)
                    if receipt.get('run_id') != node['run_id']:
                        raise ValueError('Worker receipt run identity mismatch')
                    if phase == 'verify' and receipt.get('verified') is not True:
                        raise ValueError('Semantic verification did not pass')
                    record['status'] = 'computed' if phase == 'compute' else 'verified'
                    if phase == 'verify' and 'actual_interactions' in receipt:
                        amount = receipt['actual_interactions']
                        if type(amount) is not int or amount < 0:
                            raise ValueError('Verified interaction count must be a nonnegative integer')
                        record['actual_interactions'] = amount
                    record[phase + '_receipt'] = (directory / required_file).relative_to(workspace).as_posix()
                    record.pop('error', None)
                    save()
                except BaseException as error:
                    record['status'] = 'failed' if phase == 'compute' else 'verification_failed'
                    record['error'] = {'type': type(error).__name__, 'message': str(error)}
                    raise
        else:
            state['status'] = 'cloud_verified'
    except BaseException as error:
        state['status'] = 'blocked'
        state['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        save()
        publish_inventory(graph, Path(workspace), commit=commit)
        commit()
    return state


def publish_inventory(graph, workspace, *, commit=lambda: None):
    """Exact complete dependency closure, including nested checkpoints and logs."""
    from .experiment_export import publish_shards
    return publish_shards(graph, workspace, commit=commit)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--phase', choices=('compute', 'verify'), required=True)
    args = parser.parse_args()
    if args.snapshot.stat().st_size > MAX_METADATA:
        raise ValueError('Graph metadata exceeds bound')
    graph = read_json(args.snapshot)
    validate_graph(graph)
    node = next(n for n in graph['nodes'] if n['run_id'] == args.run_id)
    execute_node(graph, node, args.workspace, args.phase)


if __name__ == '__main__':
    cli()
