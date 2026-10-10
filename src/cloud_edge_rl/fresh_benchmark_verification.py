"""Fail-closed local verification of fresh development artifacts; no inference."""
from dataclasses import asdict
from pathlib import Path, PurePosixPath
import csv
import json
import math

from .config import RunSpec
from .transport import read_json
from .training_pipeline import _same, _state_equal, verify_training_artifacts


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load(path):
    # Only use on trusted, locally downloaded artifacts from the bound completed call.
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _data(spec, artifacts):
    data = read_json(artifacts / 'data_metadata.json')
    split, rows = data['split'], data['rows']
    train, valid, clients = split['train_source_indices'], split['validation_source_indices'], split['client_indices']
    flattened = [index for client in clients for index in client]
    _require(data.get('official_test_included') is False and rows['official_test'] == 0
             and set(data['selected_members']) == {'adult.data'}, 'Official data entered fresh scope')
    _require(_same(split['config'], spec.config['data'])
             and split['config']['split_seed'] == 8501 and split['config']['partition_seed'] == 8502
             and all(type(i) is int and i >= 0 for i in train + valid + flattened)
             and len(set(train)) == len(train) == rows['train']
             and len(set(valid)) == len(valid) == rows['validation']
             and not set(train) & set(valid)
             and len(flattened) == len(set(flattened)) == len(train)
             and set(flattened) == set(range(len(train))), 'Fresh data/client isolation mismatch')
    return data


def _metrics(spec, artifacts):
    with (artifacts / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    _require(rows and len({(r['episode'], r['round']) for r in rows}) == len(rows), 'Missing/duplicate environment steps')
    for row in rows:
        _require(row['risk_version'] == spec.risk_version and all(math.isfinite(float(row[key]))
                 for key in ('elapsed_seconds', 'energy_joules', 'validation_accuracy', 'validation_loss', 'reward')),
                 'Nonfinite or wrong-risk metrics')
    return rows


def _public_uploads(spec, artifacts):
    uploads = _jsonl(artifacts / 'uploads.jsonl')
    ids = [u['upload_id'] for u in uploads]
    _require(len(ids) == len(set(ids)), 'Duplicate public upload IDs')
    for upload in uploads:
        _require(upload['run_id'] == spec.run_id and upload['split'] == 'train', 'Collection upload identity mismatch')
        remote = PurePosixPath(upload['packet_path'])
        _require(remote.parts[:3] == ('runs', spec.run_id, 'updates') and len(remote.parts) == 4
                 and remote.name == upload['upload_id'] + '.pt', 'Historical/public packet path mismatch')
        packet = _load(artifacts / 'updates' / remote.name)
        _require(set(packet) == {'model_state', 'gradients', 'labels', 'batch_size', 'feature_metadata', 'model_config'},
                 'Private reference metadata in public packet')
        def private(value):
            if isinstance(value, dict):
                return any(k in {'sample_ids', 'records', 'train_source_indices', 'client_indices'} or private(v)
                           for k, v in value.items())
            return isinstance(value, (list, tuple)) and any(private(v) for v in value)
        _require(not private(packet['feature_metadata']), 'Private feature metadata in public packet')
    from .fresh_benchmark import bootstrap_target_ids
    selected = bootstrap_target_ids(spec.config)['all']
    available = {u['upload_id']: u for u in uploads}
    _require(len(selected) == 12 and set(selected) <= set(ids)
             and all(available[i]['is_warmup'] is False for i in selected), 'Frozen audit targets absent from actual uploads')
    return selected


def _risk(spec, artifacts, summary):
    import numpy as np
    import torch
    from .environment import FEATURE_NAMES
    outer = _load(artifacts / 'risk.pt')
    state = outer['state']
    _require(outer['risk_version'] == spec.risk_version and outer['implementation'] == spec.config['components']['risk']
             and list(outer['feature_names']) == list(FEATURE_NAMES)
             and state['format'] == 'native_risk_v1' and state['version'] == spec.risk_version
             and _same(state['config'], {**spec.config['risk'], 'seed': spec.seed}), 'Fresh risk header/config mismatch')
    count = len(FEATURE_NAMES)
    mean, scale = np.asarray(state['mean']), np.asarray(state['scale'])
    _require(state['feature_count'] == count and mean.shape == scale.shape == (count,)
             and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()
             and state['model'] and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item()
                                       for v in state['model'].values()), 'Missing/nonfinite trained risk header')
    metadata = state['fit_metadata']
    groups = metadata['train_groups'] + metadata['validation_groups']
    sources = []
    labels = []
    for run_id in spec.inputs['audit_run_ids']:
        audit = read_json(artifacts.parent / run_id / 'config.json')
        _require(audit['stage'] == 'audit' and audit['config']['experiment']['tag'] == spec.config['experiment']['tag'],
                 'Historical risk audit source')
        sources.extend(audit['inputs']['source_run_ids'])
        actual = _jsonl(artifacts.parent / run_id / 'audits.jsonl')
        _require(len(actual) == len(audit['inputs']['target_ids'])
                 and {r['upload_id'] for r in actual} == set(audit['inputs']['target_ids'])
                 and all(r['split'] == 'train' and r['run_id'] in audit['inputs']['source_run_ids'] for r in actual),
                 'Risk labels not from exact completed fresh targets')
        labels.extend(actual)
    _require(metadata['train_groups'] and metadata['validation_groups'] and len(groups) == len(set(groups))
             and len(set(sources)) == 3 and set(groups) == set(sources)
             and _same(metadata, summary['fit_metadata'])
             and type(metadata['best_epoch']) is int and metadata['best_epoch'] > 0
             and math.isfinite(metadata['validation_mse']) and metadata['history'], 'Risk fit source/training metadata mismatch')
    _require(summary['training_uploads'] == len(labels) and summary['training_runs'] == len(set(sources)),
             'Risk fit actual label count mismatch')
    return {'risk_header_trained': True, 'risk_version': spec.risk_version,
            'audit_run_ids': spec.inputs['audit_run_ids'], 'audit_source_run_ids': groups,
            'risk_config': state['config']}


def verify_fresh_artifacts(spec: RunSpec, artifacts: Path, result: dict) -> dict:
    artifacts = Path(artifacts)
    settings = spec.config['native_acceptance']['fresh_benchmark']
    _require(settings['schema'] == 'fresh_seven_method_development_benchmark_v1', 'Wrong fresh purpose')
    _require(_same(read_json(artifacts / 'config.json'), asdict(spec)), 'Actual RunSpec differs from frozen source/config')
    status, summary = read_json(artifacts / 'status.json'), read_json(artifacts / 'summary.json')
    metadata = read_json(artifacts / 'metadata.json')
    _require(status.get('state') == 'completed' and metadata.get('run_id') == spec.run_id
             and metadata.get('stage') == spec.stage and metadata.get('method') == spec.method
             and summary.get('run_id') == spec.run_id and summary.get('stage') == spec.stage
             and _same(summary, result), 'Incomplete or mismatched completed result')
    runtime = read_json(artifacts / 'runtime.json')
    _require(runtime.get('cuda_available') is True and 'H100' in runtime.get('gpu', ''), 'Actual H100 runtime required')
    _require(spec.config['execution']['split'] == ('validation' if spec.stage == 'validate' else 'train'), 'Sealed/incorrect execution split')
    _data(spec, artifacts)
    if spec.config['risk']['mode'] == 'unavailable':
        _require(spec.config['reward']['risk_weight'] == spec.config['reward']['uncertainty_weight'] == 0,
                 'Unavailable-risk reward must be disabled')
    proof = {'verified': True, 'experiment_id': settings['experiment_id'], 'run_id': spec.run_id,
             'physical_interactions': 0, 'actual_interactions': 0, 'source_data_verified': True,
             'source_verified': True, 'formal_acceptance': 'not_evaluated', 'actual_cloud_cost_usd': None}
    if spec.stage == 'train_policy' and spec.method in {'dqn', 'ppo'}:
        proof.update(verify_training_artifacts(spec, artifacts, result))
        proof['physical_interactions'] = proof['actual_interactions']
    elif spec.stage == 'train_policy' and spec.method == 'pqn':
        from .comparisons.pqn import _resume_settings
        rows = _metrics(spec, artifacts)
        outer, agent = _load(artifacts / 'training.pt'), _load(artifacts / 'agent.pt')
        state = outer['state']
        effective = {**spec.config['comparisons']['pqn'], 'seed': spec.seed, 'algorithm': spec.method,
                     'risk_version': spec.risk_version, 'warmup_actions': spec.config['execution']['warmup_actions']}
        _require(outer['run_id'] == spec.run_id and outer['risk_version'] == state['risk_version'] == spec.risk_version
                 and all(type(state[k]) is int and state[k] >= 0 for k in ('environment_steps', 'learned_steps', 'warmup_steps', 'updates'))
                 and state['environment_steps'] == state['learned_steps'] + state['warmup_steps'] == len(rows) == 3000
                 and not spec.inputs['previous_training_checkpoint']
                 and spec.config['comparisons']['pqn']['budget_includes_warmup'] is True
                 and _state_equal(agent['training'], state) and _state_equal(agent['agent'], state['agent'])
                 and state['agent']['algorithm'] == 'pqn_torch', 'PQN actual counters/checkpoints mismatch')
        _require(_same(state['settings'], _resume_settings(effective))
                 and _same(read_json(artifacts / 'policy_training.json'), state['history'])
                 and result['learned_steps'] == state['learned_steps']
                 and result['warmup_steps'] == state['warmup_steps'], 'PQN settings/history mismatch')
        import torch
        _require(all(torch.isfinite(v).all().item() for v in state['agent']['model_state'].values()), 'Nonfinite PQN model')
        proof.update(physical_interactions=len(rows), actual_interactions=len(rows), cumulative_interactions=3000)
    elif spec.stage in {'collect', 'validate'}:
        rows = _metrics(spec, artifacts)
        saved = _load(artifacts / 'environment.pt')
        env = saved['environment']
        # CloudEdgeEnv.history holds per-client statistics, not environment calls.
        _require(saved['episode'] == 0 and type(env['round_index']) is int
                 and env['round_index'] == len(rows) == summary['round'] and 1 <= len(rows) <= 30
                 and env['risk_version'] == spec.risk_version, 'Episode actual step count mismatch')
        for key in ('elapsed_seconds', 'energy_joules', 'validation_accuracy'):
            _require(math.isclose(float(summary[key]), float(rows[-1][key]), rel_tol=1e-8, abs_tol=1e-8), 'Episode summary/CSV mismatch')
        if spec.stage == 'collect':
            proof.update(target_ids=_public_uploads(spec, artifacts), audit_targets_verified=True,
                         csv_rows_verified=len(rows))
        else:
            _require(env['finished'] is True, 'Validation did not naturally finish')
            if spec.method in {'dqn', 'ppo', 'pqn'}:
                path = PurePosixPath(spec.inputs['agent_checkpoint'])
                _require(len(path.parts) == 3 and path.parts[0] == 'runs'
                         and path.name.startswith('agent-step') and path.name.endswith('.pt'), 'Exact validation curve snapshot required')
                snapshot = _load(artifacts.parent / path.parts[1] / path.name)
                step = int(path.name[len('agent-step'):-3])
                producer = read_json(artifacts.parent / path.parts[1] / 'config.json')
                _require(snapshot['format'] == 'policy_snapshot_v1' and snapshot['run_id'] == path.parts[1]
                         and snapshot['risk_version'] == spec.risk_version and snapshot['algorithm'] == spec.method
                         and snapshot['environment_steps'] == step
                         and snapshot['environment_steps'] == snapshot['learned_steps'] + snapshot['warmup_steps']
                         and producer['method'] == spec.method and producer['stage'] == 'train_policy'
                         and producer['config']['experiment']['tag'] == settings['experiment_id'], 'Validation snapshot provenance mismatch')
        proof.update(physical_interactions=len(rows), actual_interactions=len(rows))
    elif spec.stage == 'fit_risk':
        proof.update(_risk(spec, artifacts, summary))
    elif spec.stage in {'audit', 'benchmark_attack'}:
        rows = _jsonl(artifacts / 'audits.jsonl')
        targets = spec.inputs['target_ids']
        _require(len(rows) == len(targets) == summary['calls']
                 and len({r['upload_id'] for r in rows}) == len(rows)
                 and {r['upload_id'] for r in rows} == set(targets), 'Incomplete/duplicate frozen audit labels')
        for row in rows:
            _require(row['run_id'] in spec.inputs['source_run_ids'] and row['split'] == 'train'
                     and all(math.isfinite(float(row[k])) and 0 <= float(row[k]) <= 1
                             for k in ('feature_recovery', 'random_feature_recovery', 'marginal_feature_recovery', 'emr')),
                     'Invalid audit labels/controls')
        proof['privacy_evidence'] = 'development_only_not_safety'
    else:
        raise ValueError('Unsupported fresh artifact stage')
    return proof
