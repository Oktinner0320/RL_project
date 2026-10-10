"""Pure plan and configuration bindings for one entirely fresh development batch.

Experiment values come only from the notebook.  Planning is not paid approval.
"""

from copy import deepcopy
import math


_METHODS = ['dqn', 'ppo', 'pqn', 'metacs', 'fixed', 'time_greedy', 'risk_greedy']
_LEARNED = ['dqn', 'ppo', 'pqn']
_SCHEMA = 'fresh_seven_method_development_benchmark_v1'


def _settings(config: dict) -> dict:
    settings = config.get('native_acceptance', {}).get('fresh_benchmark')
    if type(settings) is not dict or settings.get('schema') != _SCHEMA:
        raise ValueError('Explicit fresh benchmark notebook registration required')
    return settings


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def build_plan(config: dict) -> dict:
    """Check the fresh scope and produce dependency nodes without reading artifacts."""
    s = _settings(config)
    if s.get('methods') != _METHODS:
        raise ValueError('Exactly the seven registered methods required')
    seed_fields = ('training_seeds', 'validation_seeds', 'collection_seeds')
    seeds = []
    for key in seed_fields:
        values = s.get(key)
        if (type(values) is not list or len(values) != 3 or
                any(not _integer(value) for value in values) or len(set(values)) != 3):
            raise ValueError('Three distinct exact-integer seeds required per role')
        seeds.extend(values)
    if (not _integer(s.get('risk_fit_seed')) or
            len(set(seeds + [s['risk_fit_seed']])) != len(seeds) + 1):
        raise ValueError('Fresh seed roles must not overlap')
    for key, expected in (('policy_interactions_per_group', 3000),
                          ('shared_collection_limit', 90), ('group_limit', 3090),
                          ('collection_rounds_per_seed', 30), ('panel_uploads_per_batch', 4)):
        if type(s.get(key)) is not int or s[key] != expected:
            raise ValueError(f'{key} must retain the user-approved fresh scope')
    if (s.get('native_stage_interactions') != [500, 1000, 1000, 500] or
            any(type(value) is not int for value in s['native_stage_interactions']) or
            s.get('checkpoint_steps') != [500, 1000, 1500, 2000, 2500, 3000] or
            any(type(value) is not int for value in s['checkpoint_steps'])):
        raise ValueError('Bounded continuation and common real-step curves required')
    for key, expected in (('historical_consumption_retained', True),
                          ('unknown_consumption_is_zero', False),
                          ('new_tuning_search', False), ('official_test_access', False),
                          ('formal_confirmation', False), ('policy_snapshots_enabled', True)):
        if type(s.get(key)) is not bool or s[key] is not expected:
            raise ValueError('History retention, sealed test and fixed design required')
    if (s.get('scientific_inputs') != 'new_experiment_only_except_public_raw_adult' or
            s.get('interaction_authorization') !=
            'explicit_user_3000_policy_plus_90_new_collection_scope_only'):
        raise ValueError('Explicit fresh-input and scoped-budget authorization required')
    if any(not _integer(s.get(key)) for key in ('data_split_seed', 'partition_seed')):
        raise ValueError('New development split and partition identities required')
    if (config['environment']['max_rounds'] != 30 or
            len(config['environment']['client_sets']) *
            len(config['environment']['batch_sizes']) != 12 or
            config['execution']['warmup_actions'] != list(range(12))):
        raise ValueError('Existing finite task and warmup design must remain unchanged')
    if (s.get('collection_actions') != list(range(12)) or
            any(type(value) is not int for value in s['collection_actions'])):
        raise ValueError('New bootstrap must cover every action without adaptive selection')
    limits = s.get('execution_limits', {})
    if (type(limits.get('function_timeout_seconds')) is not int or
            not 1 <= limits['function_timeout_seconds'] <= 1800 or
            type(limits.get('stage_timeout_seconds')) is not int or
            limits['stage_timeout_seconds'] < limits['function_timeout_seconds'] or
            limits.get('startup_timeout_seconds') != 180 or
            type(limits.get('max_containers')) is not int or limits['max_containers'] != 1 or
            type(limits.get('retries')) is not int or limits['retries'] != 0 or
            limits.get('gpu') != 'H100' or limits.get('cpu') != 16.0 or
            type(limits.get('memory_mb')) is not int or limits['memory_mb'] != 65536):
        raise ValueError('Explicit single-container bounded execution required')
    for method in ('dqn', 'ppo'):
        rl = s['rl_configs'][method]
        if (rl.get('algorithm') != method or type(rl.get('total_steps')) is not int or
                rl['total_steps'] != 3000 or type(rl.get('gamma')) not in (int, float) or
                rl['gamma'] != 1 or not math.isfinite(rl['learning_rate'])):
            raise ValueError('Fixed selected method settings and inclusive policy clock required')
    if (s['rl_configs']['dqn']['learning_rate'] != .001 or
            s['rl_configs']['ppo']['learning_rate'] != .0003 or
            s['rl_configs']['ppo']['rollout_steps'] != 64 or
            s['rl_configs']['ppo']['ppo']['rollout_steps'] != 64):
        raise ValueError('D06 and P01 must not be retuned')
    nodes = []

    def add(node_id, phase, method, seed, dependencies, **extra):
        nodes.append({'id': node_id, 'phase': phase, 'method': method, 'seed': seed,
                      'dependencies': list(dependencies), **extra})

    collections = []
    for seed in s['collection_seeds']:
        node = f'collect-s{seed}'
        collections.append(node)
        add(node, 'collect', 'round_robin', seed, [], interaction_limit=30)
    benchmark = 'bootstrap-attack-precheck'
    add(benchmark, 'benchmark_attack', 'gradient_matching', s['risk_fit_seed'],
        [collections[0]], max_uploads=3)
    audit_nodes = []
    for index, node in enumerate(collections):
        audit = f'bootstrap-audit-{index}'
        audit_nodes.append(audit)
        add(audit, 'audit', 'gradient_matching', s['risk_fit_seed'], [node, benchmark],
            max_uploads=9 if index == 0 else 12)
    risk_fit = 'fresh-risk-fit'
    add(risk_fit, 'fit_risk', 'risk_ensemble', s['risk_fit_seed'], audit_nodes)
    final_training = {}
    snapshot_producers = {}
    for method in _LEARNED:
        for seed in s['training_seeds']:
            increments = s['native_stage_interactions'] if method != 'pqn' else [3000]
            previous = None
            total = 0
            for stage_index, increment in enumerate(increments):
                total += increment
                node = f'train-{method}-s{seed}-stage{stage_index}'
                dependencies = [risk_fit, *([previous] if previous else [])]
                add(node, 'train_policy', method, seed, dependencies,
                    interaction_limit=increment, expected_total_steps=total,
                    previous_training_node=previous)
                for step in s['checkpoint_steps']:
                    if total - increment < step <= total:
                        snapshot_producers[method, seed, step] = node
                previous = node
            final_training[method, seed] = previous
    for method in _METHODS:
        for train_seed, validation_seed in zip(s['training_seeds'], s['validation_seeds']):
            steps = s['checkpoint_steps'] if method in _LEARNED else [None]
            for step in steps:
                training_node = (snapshot_producers[method, train_seed, step]
                                 if method in _LEARNED else None)
                add(f'validate-{method}-s{train_seed}-step{step}', 'validate', method,
                    validation_seed, [risk_fit, *([training_node] if training_node else [])],
                    training_seed=train_seed, checkpoint_step=step,
                    final_comparison=step is None or step == 3000,
                    interaction_limit=30, training_node=training_node)
    return {'schema': _SCHEMA, 'experiment_id': s['experiment_id'], 'nodes': nodes,
            'planned_training_groups': 9, 'planned_policy_interactions': 27000,
            'planned_collection_interactions_max': 90, 'planned_validation_jobs': 66,
            'planned_final_comparison_jobs': 21, 'planned_validation_interactions_max': 1980,
            'planned_physical_interactions_max': 29070,
            'per_training_group_limit': 3090, 'paid_execution_approved': False,
            'historical_consumption_reset': False, 'official_test_access': False}


def run_config(config: dict, method: str, stage: str, seed: int,
               *, stage_interaction_limit: int | None = None) -> dict:
    """Bind numeric settings only; never inherit old fitted paths or permissions."""
    build_plan(config)
    s = _settings(config)
    if (method not in _METHODS + ['round_robin', 'gradient_matching', 'risk_ensemble'] or
            stage not in {'collect', 'benchmark_attack', 'audit', 'fit_risk',
                          'train_policy', 'validate'} or not _integer(seed)):
        raise ValueError('Only registered development routes are allowed')
    cfg = {key: deepcopy(config[key]) for key in
           ('experiment', 'data', 'model', 'environment', 'resources', 'baseline',
            'execution', 'modal')}
    cfg['comparisons'] = {key: deepcopy(config['comparisons'][key]) for key in
                          ('modal_source_root', 'attack', 'metacs', 'pqn', 'risk')}
    cfg['experiment'] = {'tag': s['experiment_id'], 'variant': method}
    cfg['data'].update(split_seed=s['data_split_seed'], partition_seed=s['partition_seed'])
    cfg['reward'] = deepcopy(s['reward_config'])
    cfg['risk'] = deepcopy(s['risk_config'])
    cfg['rl'] = deepcopy(s['rl_configs'].get(method, s['rl_configs']['dqn']))
    cfg['rl']['algorithm'] = method
    if stage_interaction_limit is not None:
        if (stage != 'train_policy' or method not in ('dqn', 'ppo') or
                type(stage_interaction_limit) is not int or
                stage_interaction_limit not in s['native_stage_interactions']):
            raise ValueError('Only bounded native policy continuation stages allowed')
        cfg['rl']['stage_interaction_limit'] = stage_interaction_limit
    cfg['comparisons']['pqn'].update(total_steps=3000, budget_includes_warmup=True,
                                     gamma=1.0, lr_decay_steps=3000)
    cfg['components'] = {'backend': 'researcher', 'risk': 'researcher',
                         'attack': s['attack_profile']['component']}
    cfg['execution'].update(backend='modal', split='validation' if stage == 'validate' else 'train')
    cfg['risk']['mode'] = 'fitted' if stage in {'train_policy', 'validate'} else 'learned'
    if stage in {'collect', 'benchmark_attack', 'audit'}:
        cfg['risk']['mode'] = 'unavailable'
        cfg['reward']['risk_weight'] = 0.0
        cfg['reward']['uncertainty_weight'] = 0.0
    if stage == 'collect':
        cfg['baseline']['round_robin_actions'] = list(s['collection_actions'])
    cfg['attack'] = {key: deepcopy(value) for key, value in s['attack_profile'].items()
                     if key not in {'component', 'audit_max_seconds'}}
    cfg['audit'] = deepcopy(config['audit'])
    cfg['audit'].update(policy='benchmark' if stage == 'benchmark_attack' else 'panel',
                        max_calls=3 if stage == 'benchmark_attack' else 12, exclude_warmup=True,
                        max_seconds=s['attack_profile']['audit_max_seconds'],
                        estimated_seconds_by_batch=None)
    cfg['native_acceptance'] = {'version': config['native_acceptance']['version'],
                                'artifact_verification': 'json_snapshot_no_hash',
                                'enabled': False, 'fresh_benchmark': deepcopy(s)}
    limits = s['execution_limits']
    cfg['modal'].update(app_name='cloud-edge-rl-fresh-seven-methods', volume_name=s['volume_name'],
                         gpu=limits['gpu'], cpu=limits['cpu'], memory_mb=limits['memory_mb'],
                         timeout_seconds=limits['function_timeout_seconds'],
                         startup_timeout=limits['startup_timeout_seconds'],
                         startup_timeout_seconds=limits['startup_timeout_seconds'],
                         max_containers=1, retries=0)
    return cfg


def bootstrap_target_ids(config: dict) -> dict:
    """Freeze non-adaptive target IDs; actual public uploads must match before attack."""
    build_plan(config)
    s = _settings(config)
    env = config['environment']
    batches = env['batch_sizes']
    targets = {batch: [] for batch in batches}
    warmup = len(config['execution']['warmup_actions'])
    for round_index in range(warmup, s['collection_rounds_per_seed']):
        action = s['collection_actions'][(round_index - warmup) % len(s['collection_actions'])]
        batch = batches[action % len(batches)]
        for client in env['client_sets'][action // len(batches)]:
            if len(targets[batch]) < s['panel_uploads_per_batch']:
                targets[batch].append(f'e0-r{round_index}-c{client}')
    if any(len(ids) != s['panel_uploads_per_batch'] for ids in targets.values()):
        raise ValueError('All registered batch strata must reach the frozen target count')
    selected = [upload for batch in batches for upload in targets[batch]]
    benchmark = [targets[batch][0] for batch in batches]
    return {'all': selected, 'benchmark': benchmark,
            'remainder': [upload for upload in selected if upload not in benchmark],
            'by_batch': targets, 'selection_uses_labels': False,
            'actual_public_upload_match_required': True}


def build_run_specs(config: dict, root, node_run_ids: dict, provenance: dict) -> list:
    """Bind a complete DAG to preallocated UUIDs, not to historical artifacts."""
    from .config import RunSpec

    plan = build_plan(config)
    if (set(node_run_ids) != {node['id'] for node in plan['nodes']} or
            len(set(node_run_ids.values())) != len(node_run_ids) or
            any(type(value) is not str or not value or '/' in value or '\\' in value
                for value in node_run_ids.values())):
        raise ValueError('Exactly one explicit unique run identity per new DAG node required')
    targets = bootstrap_target_ids(config)
    risk_version = f"{plan['experiment_id']}-risk-fit8401-v1"
    specs = []
    for node in plan['nodes']:
        stage = node['phase']
        cfg = run_config(config, node['method'], stage, node['seed'],
                         stage_interaction_limit=node['interaction_limit']
                         if stage == 'train_policy' and node['method'] != 'pqn' else None)
        dependencies = [node_run_ids[key] for key in node['dependencies']]
        inputs = {}
        if stage in {'benchmark_attack', 'audit'}:
            source = next(key for key in node['dependencies'] if key.startswith('collect-'))
            inputs['source_run_ids'] = [node_run_ids[source]]
            inputs['target_ids'] = list(targets['benchmark'] if stage == 'benchmark_attack'
                                        else targets['remainder'] if node['id'] == 'bootstrap-audit-0'
                                        else targets['all'])
            if stage == 'audit':
                inputs['benchmark_run_id'] = node_run_ids['bootstrap-attack-precheck']
        elif stage == 'fit_risk':
            inputs['audit_run_ids'] = dependencies
        elif stage in {'train_policy', 'validate'}:
            inputs['risk_checkpoint'] = f"runs/{node_run_ids['fresh-risk-fit']}/risk.pt"
            if stage == 'train_policy':
                previous = node['previous_training_node']
                inputs['previous_training_checkpoint'] = (
                    f'runs/{node_run_ids[previous]}/training.pt' if previous else None)
            elif node['training_node'] is not None:
                step = node['checkpoint_step']
                inputs['agent_checkpoint'] = (
                    f"runs/{node_run_ids[node['training_node']]}/agent-step{step:05d}.pt")
        version = 'unavailable' if cfg['risk']['mode'] == 'unavailable' else risk_version
        specs.append(RunSpec(node_run_ids[node['id']], stage, node['method'], node['seed'],
                             version, cfg, inputs, deepcopy(provenance)))
    return specs


def verify_fresh_artifacts(spec, artifacts, result: dict) -> dict:
    """Use the stage-specific semantic gate before accepting a fresh result."""
    from .fresh_benchmark_verification import verify_fresh_artifacts as verify

    return verify(spec, artifacts, result)
