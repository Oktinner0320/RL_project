"""本地核对新开发实验产物，检查失败即阻断，不运行模型推理。
Verify fresh development artifacts locally; fail closed without model inference.
"""
from pathlib import Path, PurePosixPath
import csv
import json
import math

from .config import RunSpec
from .transport import read_json
from .training_pipeline import _same, _state_equal, verify_training_artifacts


def _require(condition, message):
    """条件不满足时抛出核验错误。 / Raise a verification error when the condition is not met."""
    if not condition:
        raise ValueError(message)


def _load(path):
    """在 CPU 上读取可信本地产物中的完整检查点。 / Load a full checkpoint from trusted local artifacts on CPU."""
    # 仅加载已绑定完成调用下载的可信产物，不能用于未知来源文件。 / Use only trusted downloads from the bound completed call, not unknown files.
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def _jsonl(path):
    """逐行解析非空 JSON 记录并返回列表。 / Parse nonempty JSON lines and return their records."""
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _data(spec, artifacts):
    """核对开发数据划分、客户端分区和测试集隔离，返回数据元信息。
    Check development splits, client partitions, and test isolation; return data metadata.
    """
    data = read_json(artifacts / 'data_metadata.json')
    split, rows = data['split'], data['rows']
    train, valid, clients = split['train_source_indices'], split['validation_source_indices'], split['client_indices']
    flattened = [index for client in clients for index in client]
    # 新开发实验只读取 adult.data，不允许混入正式测试数据。 / Read only adult.data for fresh development, excluding official test data.
    _require(data.get('official_test_included') is False and rows['official_test'] == 0
             and set(data['selected_members']) == {'adult.data'}, 'Official data entered fresh scope')
    # 训练/验证使用原始行索引；客户端索引则相对训练子集，必须无重叠且完整覆盖。 / Split indices refer to source rows; client indices must partition the training subset exactly.
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
    """读取环境指标，检查步记录唯一、风险版本一致及关键数值有限。
    Read environment metrics; check unique step records, matching risk version, and finite key values.
    """
    with (artifacts / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    _require(rows and len({(r['episode'], r['round']) for r in rows}) == len(rows), 'Missing/duplicate environment steps')
    for row in rows:
        _require(row['risk_version'] == spec.risk_version and all(math.isfinite(float(row[key]))
                 for key in ('elapsed_seconds', 'energy_joules', 'validation_accuracy', 'validation_loss', 'reward')),
                 'Nonfinite or wrong-risk metrics')
    return rows


def _public_uploads(spec, artifacts):
    """核对公开上传包及冻结审计目标，返回目标编号。 / Check public upload packets and frozen audit targets; return target IDs."""
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
            """递归检测嵌套元信息中禁止公开的参考数据键。 / Recursively detect forbidden private reference keys in nested metadata."""
            if isinstance(value, dict):
                return any(k in {'sample_ids', 'records', 'train_source_indices', 'client_indices'} or private(v)
                           for k, v in value.items())
            return isinstance(value, (list, tuple)) and any(private(v) for v in value)
        _require(not private(packet['feature_metadata']), 'Private feature metadata in public packet')
    from .fresh_benchmark import bootstrap_target_ids
    selected = bootstrap_target_ids(spec.config)['all']
    available = {u['upload_id']: u for u in uploads}
    # 冻结的 12 个目标必须实际上传，且不能来自预热步骤。 / Require all 12 frozen targets to exist in actual non-warmup uploads.
    _require(len(selected) == 12 and set(selected) <= set(ids)
             and all(available[i]['is_warmup'] is False for i in selected), 'Frozen audit targets absent from actual uploads')
    return selected


def _risk(spec, artifacts, summary):
    """核对风险模型、拟合来源和标签计数，返回训练证据，不运行预测。
    Check the risk model, fitting sources, and label counts; return training evidence without prediction.
    """
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
    # 检查缩放形状及缩放参数、模型权重的有限性，不只检查文件是否存在。 / Check scaling shape and finite scaling parameters/model weights, not merely file existence.
    _require(state['feature_count'] == count and mean.shape == scale.shape == (count,)
             and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()
             and state['model'] and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item()
                                       for v in state['model'].values()), 'Missing/nonfinite trained risk header')
    metadata = state['fit_metadata']
    groups = metadata['train_groups'] + metadata['validation_groups']
    sources = []
    labels = []
    for run_id in spec.inputs['audit_run_ids']:
        # 标签必须来自本实验登记的审计来源和完整目标集合。 / Require labels from this experiment's registered audit sources and exact target sets.
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
    # 三个来源运行按组划分，拟合训练组和验证组必须非空、互斥。 / Partition the three source runs into nonempty, disjoint fitting groups.
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
    """按阶段核对下载产物、完成状态和真实计数，返回开发实验核验记录。
    Verify downloaded artifacts, completion status, and actual counters by stage; return development evidence.

    不运行模型推理，也不代表正式验收；云端实际费用未知时保留为 None。
    Run no model inference and do not claim formal acceptance; keep unknown actual cloud cost as None.
    """
    from .artifacts import artifact_payload
    artifacts = Path(artifacts)
    settings = spec.config['native_acceptance']['fresh_benchmark']
    _require(settings['schema'] == 'fresh_seven_method_development_benchmark_v1', 'Wrong fresh purpose')
    _require(_same(artifact_payload(read_json(artifacts / 'config.json')), artifact_payload(spec)),
             'Actual RunSpec differs from frozen source/config')
    status, summary = read_json(artifacts / 'status.json'), read_json(artifacts / 'summary.json')
    metadata = read_json(artifacts / 'metadata.json')
    _require(status.get('state') == 'completed' and metadata.get('run_id') == spec.run_id
             and metadata.get('stage') == spec.stage and metadata.get('method') == spec.method
             and summary.get('run_id') == spec.run_id and summary.get('stage') == spec.stage
             and _same(summary, result), 'Incomplete or mismatched completed result')
    runtime = read_json(artifacts / 'runtime.json')
    # 核对运行记录中的 CUDA/H100 信息，防止混用不符合要求的运行产物。 / Check recorded CUDA/H100 runtime details before accepting artifacts.
    _require(runtime.get('cuda_available') is True and 'H100' in runtime.get('gpu', ''), 'Actual H100 runtime required')
    _require(spec.config['execution']['split'] == ('validation' if spec.stage == 'validate' else 'train'), 'Sealed/incorrect execution split')
    _data(spec, artifacts)
    if spec.config['risk']['mode'] == 'unavailable':
        # 风险模型不可用时，风险和不确定性奖励项必须同时关闭。 / Disable both risk and uncertainty reward terms when risk is unavailable.
        _require(spec.config['reward']['risk_weight'] == spec.config['reward']['uncertainty_weight'] == 0,
                 'Unavailable-risk reward must be disabled')
    proof = {'verified': True, 'experiment_id': settings['experiment_id'], 'run_id': spec.run_id,
             'physical_interactions': 0, 'actual_interactions': 0, 'source_data_verified': True,
             'source_verified': True, 'formal_acceptance': 'not_evaluated', 'actual_cloud_cost_usd': None}
    if spec.stage == 'train_policy' and spec.method in {'dqn', 'ppo'}:
        # 原生 DQN/PPO 复用已有训练核验，不另设一套恢复或预算规则。 / Reuse native DQN/PPO training verification and its budget/resume rules.
        proof.update(verify_training_artifacts(spec, artifacts, result))
        proof['physical_interactions'] = proof['actual_interactions']
    elif spec.stage == 'train_policy' and spec.method == 'pqn':
        from .comparisons.pqn import _resume_settings
        rows = _metrics(spec, artifacts)
        outer, agent = _load(artifacts / 'training.pt'), _load(artifacts / 'agent.pt')
        state = outer['state']
        effective = {**spec.config['comparisons']['pqn'], 'seed': spec.seed, 'algorithm': spec.method,
                     'risk_version': spec.risk_version, 'warmup_actions': spec.config['execution']['warmup_actions']}
        # PQN 必须从头完成 3000 次交互（含预热），且训练与导出检查点内容一致。 / Require 3000 fresh PQN interactions including warmup and matching exported state.
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
        # 环境 history 是客户端统计，不能当作交互数；使用轮次和指标行数核对。 / History stores per-client statistics; count interactions with rounds and metric rows.
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
                # 学习曲线验证必须绑定明确步数的策略快照，不能替换为最终策略。 / Bind curve evaluation to an exact-step snapshot, not the final policy.
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
        # 每个冻结目标必须恰好有一条标签；这里只核对覆盖和数值范围，不宣称隐私安全。 / Check one label per target and valid values, without claiming privacy safety.
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
