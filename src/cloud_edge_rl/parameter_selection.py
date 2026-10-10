"""Pure, validation-only short-budget parameter-selection contracts.

No execution, artifact loading, seed allocation or formal acceptance is performed.
"""

from copy import deepcopy
import json
import math


_COMMON = {'hidden_sizes', 'learning_rate', 'batch_size', 'max_grad_norm'}
_DQN = {'buffer_size', 'learning_starts', 'train_frequency',
        'target_update_interval', 'epsilon_start', 'epsilon_end',
        'epsilon_decay_steps', 'double_dqn'}
_PPO = {'rollout_steps', 'ppo_epochs', 'clip_range', 'gae_lambda',
        'entropy_coefficient', 'value_coefficient', 'normalize_advantage'}
_ALIASES = {'rollout_steps': 'rollout_steps', 'ppo_epochs': 'epochs',
            'clip_range': 'clip_ratio', 'gae_lambda': 'gae_lambda',
            'entropy_coefficient': 'entropy_weight'}
_IDS = {f'{prefix}{i:02d}': algorithm for prefix, algorithm in
        (('D', 'dqn'), ('P', 'ppo')) for i in range(10)}
_BASELINE = {'hidden_sizes': [128, 128], 'learning_rate': .0003, 'gamma': 1,
             'batch_size': 64, 'buffer_size': 20000, 'learning_starts': 256,
             'train_frequency': 1, 'target_update_interval': 100,
             'epsilon_start': 1., 'epsilon_end': .05, 'epsilon_decay_steps': 2000,
             'max_grad_norm': 10., 'double_dqn': True, 'rollout_steps': 128,
             'ppo_epochs': 4, 'clip_range': .2, 'gae_lambda': .95,
             'entropy_coefficient': .01, 'value_coefficient': .5,
             'normalize_advantage': True,
             'ppo': {'rollout_steps': 128, 'epochs': 4, 'clip_ratio': .2,
                     'gae_lambda': .95, 'entropy_weight': .01}}
_REWARD = {'accuracy_weight': 1., 'time_weight': .05, 'energy_weight': .01,
           'risk_weight': .1, 'time_scale': 1., 'energy_scale': 10.,
           'uncertainty_weight': 0., 'protocol': 'exposure_v1'}


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _frozen_equal(value, expected):
    if type(expected) is dict:
        return (type(value) is dict and value.keys() == expected.keys() and
                all(_frozen_equal(value[k], v) for k, v in expected.items()))
    if type(expected) is list:
        return (type(value) is list and len(value) == len(expected) and
                all(_frozen_equal(a, b) for a, b in zip(value, expected)))
    if type(expected) in (int, float):
        return _number(value) and value == expected
    return type(value) is type(expected) and value == expected


def _settings(config):
    if type(config) is not dict:
        raise ValueError('CONFIG must be a dictionary')
    settings = config.get('native_acceptance', {}).get('parameter_selection')
    if type(settings) is not dict:
        raise ValueError('CONFIG.native_acceptance.parameter_selection required')
    return settings


def _effective(settings, row):
    rl = deepcopy(settings['baseline_rl'])
    rl.update(deepcopy(row['override']))
    for key, alias in _ALIASES.items():
        if key in row['override']:
            rl['ppo'][alias] = deepcopy(rl[key])
    return rl


def build_plan(config: dict) -> dict:
    """Validate frozen design, without interpreting unknown execution facts as zero."""
    s = _settings(config)
    if (type(s.get('id')) is not str or not s['id'] or
            type(s.get('status')) is not str or not s['status']):
        raise ValueError('Explicit plan id and status required')
    for key, expected in (('groups_per_algorithm', 10),
                          ('environment_interactions_per_group', 500),
                          ('historical_configuration_lower_bound', 12)):
        if type(s.get(key)) is not int or s[key] != expected:
            raise ValueError(f'{key} must be {expected}')
    if (not _frozen_equal(s.get('baseline_rl'), _BASELINE) or
            not _frozen_equal(s.get('baseline_reward'), _REWARD)):
        raise ValueError('Historical RL and reward baselines must remain frozen')
    selection = {'accuracy_slack': .01, 'cost_multiplier': 1.15,
                 'reference_method': 'time_greedy',
                 'ordering': ['accuracy_desc', 'time_asc', 'energy_asc',
                              'configuration_id_asc']}
    if not _frozen_equal(s.get('selection'), selection):
        raise ValueError('Selection thresholds and ordering must remain frozen')
    env, execution = config.get('environment', {}), config.get('execution', {})
    if (type(env.get('max_rounds')) is not int or env['max_rounds'] != 30 or
            type(execution.get('warmup_actions')) is not list or
            execution['warmup_actions'] != list(range(12)) or
            any(type(x) is not int for x in execution['warmup_actions']) or
            len(env.get('client_sets', [])) * len(env.get('batch_sizes', [])) != 12):
        raise ValueError('30 rounds and all 12 actions in frozen warmup order required')
    rows = s.get('candidates')
    if type(rows) is not list or len(rows) != 20:
        raise ValueError('Exactly 20 registered candidates required')
    seen, vectors, result = set(), {'dqn': set(), 'ppo': set()}, []
    for row in rows:
        if type(row) is not dict or set(row) != {'id', 'algorithm', 'override', 'hypothesis'}:
            raise ValueError('Invalid candidate contract')
        cid, algorithm, delta = row['id'], row['algorithm'], row['override']
        if type(cid) is not str or cid not in _IDS or cid in seen or algorithm != _IDS[cid]:
            raise ValueError('Candidate IDs must be D00..D09 and P00..P09 exactly once')
        seen.add(cid)
        legal = _COMMON | (_DQN if algorithm == 'dqn' else _PPO)
        if (type(delta) is not dict or len(delta) > 1 or set(delta) - legal or
                (cid.endswith('00') and delta)):
            raise ValueError('Control is unchanged; candidates allow one own-algorithm flat delta')
        if type(row['hypothesis']) is not str or not row['hypothesis'].strip():
            raise ValueError('Candidate hypothesis required')
        for key, value in delta.items():
            baseline = _BASELINE[key]
            valid = (type(value) is bool if type(baseline) is bool else
                     type(value) is list and bool(value) and all(_integer(v, 1) for v in value)
                     if key == 'hidden_sizes' else
                     _integer(value, 1) if type(baseline) is int else _number(value))
            if not valid or value == baseline:
                raise ValueError('Delta must be finite, correctly typed and effective')
            if key in {'learning_rate', 'max_grad_norm', 'clip_range'} and value <= 0:
                raise ValueError('Positive optimizer parameter required')
            if key in {'epsilon_start', 'epsilon_end', 'gae_lambda', 'clip_range'} and value > 1:
                raise ValueError('Probability parameter exceeds one')
        rl = _effective(s, row)
        if rl['epsilon_end'] > rl['epsilon_start']:
            raise ValueError('Invalid epsilon interval')
        vector = json.dumps(rl, sort_keys=True, allow_nan=False)
        if vector in vectors[algorithm]:
            raise ValueError('Duplicate effective candidate vector')
        vectors[algorithm].add(vector)
        result.append({**deepcopy(row), 'rl': rl, 'planned_training_interactions': 500,
                       'planned_evaluation_interactions': 60})
    return {'id': s['id'], 'candidates': result,
            'training_seed': deepcopy(s.get('training_seed')),
            'evaluation_seeds': deepcopy(s.get('evaluation_seeds', [])),
            'selection': deepcopy(selection),
            'planned_training_interactions_by_algorithm': {'dqn': 5000, 'ppo': 5000},
            'planned_evaluation_interactions_by_algorithm': {'dqn': 600, 'ppo': 600},
            'forecast_if_no_early_task_termination': {'warmup_interactions': 204,
                                                      'learned_interactions': 296},
            'task_termination': 'original_loss_time_sample_or_round_limit',
            'historical_configuration_lower_bound': 12, 'executed_new_configurations': 0,
            'formal_acceptance': 'not_evaluated', 'official_test_access': False}


def candidate_config(config: dict, candidate_id: str) -> dict:
    plan = build_plan(config)
    row = next((r for r in plan['candidates'] if r['id'] == candidate_id), None)
    if row is None:
        raise ValueError('Unregistered candidate ID')
    result, s = deepcopy(config), _settings(config)
    result['rl'] = deepcopy(row['rl'])
    result['rl'].update(algorithm=row['algorithm'], device=config['rl']['device'],
                        seed=s.get('training_seed'), total_steps=500, checkpoint_interval=500)
    for key in ('stage_interaction_limit', 'resource_feedback', 'observation_divisors'):
        result['rl'].pop(key, None)
    result['reward'] = deepcopy(s['baseline_reward'])
    result['risk'] = deepcopy(s.get('frozen_risk_config', config['risk']))
    result['risk']['device'] = config['risk']['device']
    result['risk']['mode'] = 'fitted'
    result['execution'].update(backend='modal', split='train')
    result['native_acceptance'].pop('execution_admission', None)
    return result


def execution_blockers(config: dict) -> list[str]:
    try:
        build_plan(config)
    except (ValueError, KeyError, TypeError) as exc:
        return [str(exc)]
    s, errors = _settings(config), []
    if s.get('paid_execution_approved') is not True:
        errors.append('Paid execution is not explicitly approved')
    if type(s.get('tuning_configuration_slots_approved')) is not int or s['tuning_configuration_slots_approved'] != 20:
        errors.append('Exactly 20 tuning configuration slots require explicit approval')
    scoped = s.get('scoped_quota_amendment')
    amended = (type(scoped) is dict and scoped.get('approved') is True
               and scoped.get('scope_id') == s['id']
               and scoped.get('additional_configuration_slots') == 20
               and type(scoped.get('additional_configuration_slots')) is int
               and _frozen_equal(scoped.get('training_interactions_by_algorithm'), {'dqn': 5000, 'ppo': 5000})
               and scoped.get('cumulative_interaction_limit') == 5120
               and type(scoped.get('cumulative_interaction_limit')) is int
               and scoped.get('historical_unknowns_retained') is True
               and scoped.get('official_test_access') is False)
    if not _integer(s.get('historical_configuration_total'), 12) and not amended:
        errors.append('Historical configuration total is unverified')
    seed = s.get('training_seed')
    if not _integer(seed) or seed in {101, 102, 103, 104, 1101, 1102, 1103, 1201}:
        errors.append('Explicit nonnegative unused training seed required; historical/formal seeds forbidden')
    seeds = s.get('evaluation_seeds')
    if (type(seeds) is not list or len(seeds) != 2 or
            any(not _integer(v) or v in {2101, 2102, 2103} for v in seeds) or
            len(set(seeds)) != 2):
        errors.append('Exactly two distinct development-only evaluation seeds required')
    for key in ('frozen_risk_checkpoint', 'frozen_risk_version'):
        if type(s.get(key)) is not str or not s[key].strip():
            errors.append(f'{key} is unverified')
    if type(s.get('frozen_risk_config')) is not dict or not s['frozen_risk_config']:
        errors.append('Frozen risk checkpoint configuration is unverified')
    limit = s.get('interaction_limit')
    if not _integer(limit, 1):
        errors.append('Explicit positive interaction limit required')
    elif limit != 3000 and not (amended and limit == 5120):
        errors.append('Changing CONFIG alone cannot revise the shared 3000 interaction admission contract')
    for algorithm in ('dqn', 'ppo'):
        counts = [s.get(key, {}).get(algorithm) if type(s.get(key)) is dict else None
                  for key in ('historical_interactions', 'allocated_collection_interactions')]
        if any(not _integer(v) for v in counts):
            errors.append(f'{algorithm}: historical/shared interaction counts are unverified')
        if _integer(limit, 1) and 5000 > limit:
            errors.append(f'{algorithm}: 5000 planned training interactions exceed limit {limit}')
        elif _integer(limit, 1) and all(_integer(v) for v in counts) and 5000 + sum(counts) > limit:
            errors.append(f'{algorithm}: historical + shared + 5000 forecast exceeds limit {limit}')
    return errors


def _evaluation_errors(records, seeds):
    errors = []
    if type(records) is not list or len(records) != 2:
        return ['Exactly two evaluation records required']
    actual_seeds = []
    for record in records:
        if type(record) is not dict:
            errors.append('Invalid evaluation record')
            continue
        seed = record.get('seed')
        if not _integer(seed):
            errors.append('Invalid evaluation seed')
        actual_seeds.append(seed)
        if (record.get('official_test_included') is not False or
                not _integer(record.get('actual_interactions'), 1) or record['actual_interactions'] > 30 or
                type(record.get('warmup_steps')) is not int or
                record['warmup_steps'] != min(12, record['actual_interactions']) or
                record.get('task_terminal') is not True):
            errors.append('Evaluation must be a complete validation-only task with its actual warmup count')
        if (not _number(record.get('accuracy')) or record['accuracy'] > 1 or
                not _number(record.get('time_seconds')) or not _number(record.get('energy_joules'))):
            errors.append('Invalid finite evaluation metrics')
    if sorted(actual_seeds, key=repr) != sorted(seeds, key=repr):
        errors.append('Evaluation seeds do not match registered plan')
    return errors


def _means(records):
    return {key: math.fsum(r[key] / 2 for r in records)
            for key in ('accuracy', 'time_seconds', 'energy_joules')}


def rank_results(plan: dict, rows: list[dict], reference: dict | None) -> dict:
    """Rank only complete, verified validation comparisons; never certify privacy."""
    errors, retained = [], {'dqn': [], 'ppo': []}
    seeds = plan.get('evaluation_seeds', [])
    if (type(seeds) is not list or len(seeds) != 2 or
            any(not _integer(v) or v in {2101, 2102, 2103} for v in seeds) or
            len(set(seeds)) != 2):
        errors.append('Two registered development-only seeds required')
        seeds = []
    expected = {r['id']: r['algorithm'] for r in plan.get('candidates', [])}
    if expected != _IDS:
        errors.append('Plan must register all 20 candidates')
    seen = set()
    if type(rows) is not list:
        rows = []
        errors.append('Outcome rows must be a list')
    for row in rows:
        issues = []
        if type(row) is not dict:
            errors.append('Invalid outcome row')
            continue
        cid, algorithm = row.get('configuration_id'), row.get('algorithm')
        if type(cid) is not str or cid not in expected or cid in seen:
            errors.append('Duplicate or unregistered outcome ID')
            continue
        seen.add(cid)
        if algorithm != expected[cid]:
            issues.append('Algorithm does not match registered candidate')
        if row.get('status') != 'completed' or row.get('artifact_verified') is not True:
            issues.append('Candidate is not completed with verified artifacts')
        if type(row.get('actual_interactions')) is not int or row['actual_interactions'] != 500:
            issues.append('actual_interactions must equal 500')
        if (not _integer(row.get('warmup_steps'), 1) or not _integer(row.get('learned_steps'), 1) or
                row['warmup_steps'] + row['learned_steps'] != 500):
            issues.append('Actual warmup plus learned interactions must equal 500')
        if not _integer(row.get('optimizer_updates'), 1):
            issues.append('Invalid optimizer update counter')
        issues.extend(_evaluation_errors(row.get('evaluations'), seeds))
        record = deepcopy(row)
        record['error_reasons'] = issues
        if not issues:
            record['means'] = _means(row['evaluations'])
        retained[expected[cid]].append(record)
        errors.extend(f'{cid}: {issue}' for issue in issues)
    if seen != set(expected):
        errors.append('Missing candidate outcomes')
    if type(reference) is not dict or reference.get('method') != 'time_greedy' or reference.get('artifact_verified') is not True:
        errors.append('Verified time_greedy reference required')
    else:
        errors.extend(f'reference: {e}' for e in _evaluation_errors(reference.get('evaluations'), seeds))
    output = {'status': 'incomplete_comparison' if errors else 'complete_comparison',
              'error_reasons': errors, 'algorithms': {}, 'formal_acceptance': 'not_evaluated',
              'official_test_access': False, 'privacy_winner': None}
    for algorithm, records in retained.items():
        winner, eligible = None, []
        if not errors:
            ref = _means(reference['evaluations'])
            for record in records:
                m = record['means']
                record['eligible'] = (m['accuracy'] >= ref['accuracy'] - .01 and
                                      m['time_seconds'] <= ref['time_seconds'] * 1.15 and
                                      m['energy_joules'] <= ref['energy_joules'] * 1.15)
                if record['eligible']:
                    eligible.append(record)
            eligible.sort(key=lambda r: (-r['means']['accuracy'], r['means']['time_seconds'],
                                         r['means']['energy_joules'], r['configuration_id']))
            if eligible:
                winner = eligible[0]['configuration_id']
        output['algorithms'][algorithm] = {
            'status': 'incomplete_comparison' if errors else
                      'best_observed_short_budget_candidate' if winner else 'no_eligible_candidate',
            'best_observed_short_budget_candidate': winner,
            'rows': records, 'error_reasons': deepcopy(errors)}
    if not errors and all(v['best_observed_short_budget_candidate'] is None for v in output['algorithms'].values()):
        output['status'] = 'no_eligible_candidate'
    return output
