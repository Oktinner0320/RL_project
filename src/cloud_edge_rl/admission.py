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
_SELECTION_PURPOSE = 'bounded_dqn_ppo_parameter_selection'
_SELECTION_SCHEMA = 'parameter_selection_shared_source_v1'
_FRESH_PURPOSE = 'fresh_seven_method_development_benchmark_v1'
_FRESH_METHODS = ['dqn', 'ppo', 'pqn', 'metacs', 'fixed', 'time_greedy', 'risk_greedy']


def resource_frozen_payload(spec):
    """Keep the complete approved spec except the separately witnessed source contents."""
    payload = frozen_payload(spec)
    payload['provenance'].pop('source_contents')
    return payload


def _resource_shared_source(witness):
    if 'artifact_schema' not in witness:
        return False
    if (witness.get('purpose'), witness['artifact_schema']) not in (
            (_RESOURCE_PURPOSE, _RESOURCE_SCHEMA), (_SELECTION_PURPOSE, _SELECTION_SCHEMA),
            (_FRESH_PURPOSE, _FRESH_PURPOSE)):
        raise ValueError('Unknown admission artifact schema')
    return True


def _fresh_contract(witness):
    """A separate fresh scope: no historical scientific artifacts or tuning slots."""
    freeze = _first_artifact(witness['freeze_manifest'])
    contract = freeze.get('fresh_contract', {})
    exact = {'schema': _FRESH_PURPOSE, 'methods': _FRESH_METHODS,
             'training_seeds': [8101, 8102, 8103], 'validation_seeds': [8201, 8202, 8203],
             'collection_seeds': [8301, 8302, 8303], 'risk_fit_seed': 8401,
             'data_split_seed': 8501, 'partition_seed': 8502,
             'policy_interactions_per_group': 3000, 'shared_collection_limit': 90,
             'group_limit': 3090,
             'scientific_inputs': 'new_experiment_only_except_public_raw_adult',
             'checkpoint_steps': [500, 1000, 1500, 2000, 2500, 3000]}
    if (witness.get('artifact_schema') != _FRESH_PURPOSE
            or any(not _first_json_equal(contract.get(k), v) for k, v in exact.items())
            or type(contract.get('experiment_id')) is not str or not contract['experiment_id']):
        raise ValueError('Exact fresh benchmark contract required')
    output = Path(contract.get('output_root', ''))
    prefix = contract.get('volume_prefix')
    if (not output.is_absolute() or type(prefix) is not str or not prefix
            or prefix.startswith('/') or any(p in ('', '.', '..') for p in prefix.split('/'))):
        raise ValueError('New absolute output root and confined volume prefix required')
    runs = witness.get('runs')
    dag = contract.get('dag')
    if type(runs) is not list or not runs or type(dag) is not list:
        raise ValueError('Explicit fresh frozen DAG required')
    ids = [r['spec']['run_id'] for r in runs]
    if (any(type(i) is not str or not i or '/' in i or '\\' in i or i in ('.', '..') for i in ids)
            or len(set(ids)) != len(ids) or len(dag) != len(ids)
            or [d['run_id'] for d in dag] != ids):
        raise ValueError('Unique ordered fresh run IDs required')
    by_id, seen, trains, validations, collections, fits = {}, set(), {}, set(), set(), []
    for entry, run in zip(dag, runs):
        p = run['spec']
        rid, stage, method, seed = p['run_id'], p['stage'], p['method'], p['seed']
        deps = entry.get('dependencies')
        if (type(deps) is not list or len(set(deps)) != len(deps) or not set(deps) <= seen):
            raise ValueError('Only declared new DAG predecessors allowed')
        config, inputs = p['config'], p['inputs']
        execution, data, modal = config['execution'], config['data'], config['modal']
        if (execution.get('backend') != 'modal' or execution.get('split') not in ('train', 'validation')
                or data.get('split_seed') != 8501 or data.get('partition_seed') != 8502
                or p.get('provenance', {}).get('identity_kind') != 'source_contents_no_hash'
                or 'source_contents' in p['provenance']):
            raise ValueError('Fresh split, partition and compact source identity required')
        bounded = {'gpu': 'H100', 'cpu': 16.0, 'memory_mb': 65536,
                   'startup_timeout_seconds': 180, 'max_containers': 1, 'retries': 0}
        if (any(not _first_json_equal(modal.get(k), v) for k, v in bounded.items())
                or type(modal.get('timeout_seconds')) is not int
                or not 1 <= modal['timeout_seconds'] <= 1800):
            raise ValueError('Fresh bounded H100 settings and zero retries required')
        # Input artifacts use suite-relative paths; IDs are resolved only through the DAG.
        for key, value in inputs.items():
            if key.endswith('run_ids'):
                if type(value) is not list or not value or not set(value) <= set(deps):
                    raise ValueError('Fresh audit/source IDs must be declared predecessors')
            elif key.endswith('run_id'):
                if value not in deps:
                    raise ValueError('Fresh source ID must be a declared predecessor')
            elif value not in (None, ''):
                parts = value.split('/') if type(value) is str else []
                if (len(parts) != 3 or parts[0] != 'runs' or parts[1] not in deps
                        or parts[2] not in ('risk.pt', 'agent.pt', 'training.pt')):
                    raise ValueError('Historical or unconfined scientific input forbidden')
                producer = by_id[parts[1]]
                if (parts[2] == 'risk.pt' and producer['stage'] != 'fit_risk'
                        or parts[2] in ('agent.pt', 'training.pt') and
                        (producer['stage'] != 'train_policy' or producer['method'] != method
                         or producer['seed'] != (seed if stage == 'train_policy' else 8101 + seed - 8201))):
                    raise ValueError('Cross-arm or invalid checkpoint producer forbidden')
        amount, partition = 0, execution['split']
        if stage == 'collect':
            amount = config['rl'].get('stage_interaction_limit')
            if (method != 'round_robin' or seed not in (8301, 8302, 8303)
                    or seed in collections or partition != 'train' or inputs
                    or type(amount) is not int or not 1 <= amount <= 30):
                raise ValueError('Exactly three fresh bounded bootstrap collections required')
            collections.add(seed)
        elif stage == 'train_policy':
            rl = config['comparisons']['pqn'] if method == 'pqn' else config['rl']
            amount = rl.get('stage_interaction_limit', rl.get('total_steps'))
            if (method not in ('dqn', 'ppo', 'pqn') or seed not in (8101, 8102, 8103)
                    or partition != 'train' or rl.get('total_steps') != 3000
                    or rl.get('gamma') != 1 or type(amount) is not int
                    or set(inputs) != {'risk_checkpoint', 'previous_training_checkpoint'}
                    or not inputs.get('risk_checkpoint')):
                raise ValueError('Fresh learned policy budget includes all warmup')
            if (method == 'pqn' and (amount != 3000 or rl.get('budget_includes_warmup') is not True)
                    or method == 'dqn' and rl.get('learning_rate') != .001
                    or method == 'ppo' and (rl.get('learning_rate') != .0003 or rl.get('rollout_steps') != 64)):
                raise ValueError('Only registered D06/P01/PQN settings allowed')
            group = trains.setdefault((method, seed), [])
            expected = [3000] if method == 'pqn' else [500, 1000, 1000, 500]
            if len(group) >= len(expected) or amount != expected[len(group)]:
                raise ValueError('Exact native/PQN stage interaction budgets required')
            previous = inputs['previous_training_checkpoint']
            if previous != (f'runs/{group[-1]}/training.pt' if group else None):
                raise ValueError('Fresh training must start from zero then its own predecessor')
            group.append(rid)
        elif stage == 'validate':
            if (method not in _FRESH_METHODS or seed not in (8201, 8202, 8203)
                    or partition != 'validation' or (method, seed) in validations):
                raise ValueError('Exact fresh paired validation matrix required')
            validations.add((method, seed))
            amount = config.get('evaluation', {}).get('episodes')
            if type(amount) is not int or not 1 <= amount <= 30 or not inputs.get('risk_checkpoint'):
                raise ValueError('Bounded fresh validation required')
            if method in ('dqn', 'ppo', 'pqn'):
                group = trains.get((method, 8101 + seed - 8201), [])
                if not group or inputs.get('agent_checkpoint') != f'runs/{group[-1]}/agent.pt':
                    raise ValueError('Validation must use its own final fresh policy')
            elif set(inputs) != {'risk_checkpoint'}:
                raise ValueError('Static methods cannot load policy checkpoints')
        elif stage == 'fit_risk':
            if seed != 8401 or partition != 'train' or set(inputs) != {'audit_run_ids'}:
                raise ValueError('Fresh risk fit must use new audit labels')
            if any(by_id[i]['stage'] != 'audit' for i in inputs['audit_run_ids']):
                raise ValueError('Fresh risk fit requires audit producers')
            fits.append(rid)
        elif stage in ('audit', 'benchmark_attack'):
            if partition != 'train' or not inputs.get('source_run_ids') or any(
                    by_id[i]['stage'] != 'collect' for i in inputs['source_run_ids']):
                raise ValueError('Fresh audits require only fresh collection producers')
        else:
            raise ValueError('Official/sealed or unrelated stage forbidden')
        charges = [{'method': method, 'seed': seed, 'partition': partition,
                    'configuration_id': method, 'actual_interactions': amount}]
        if not _first_json_equal(run.get('charges'), charges):
            raise ValueError('Exact fresh canonical charges required')
        by_id[rid] = p
        seen.add(rid)
    if (collections != {8301, 8302, 8303} or len(fits) != 1
            or set(trains) != {(m, s) for m in ('dqn', 'ppo', 'pqn') for s in (8101, 8102, 8103)}
            or any(len(v) != (1 if k[0] == 'pqn' else 4) for k, v in trains.items())
            or validations != {(m, s) for m in _FRESH_METHODS for s in (8201, 8202, 8203)}):
        raise ValueError('Incomplete fresh seven-method matrix')
    return contract


def _fresh_history(witness, ledger):
    """Retain unknown old costs explicitly; allocate physical collection only once."""
    contract = _fresh_contract(witness)
    if (witness.get('historical_cloud_cost_usd', 'missing') is not None
            or witness.get('historical_cloud_cost_unknown_retained') is not True):
        raise ValueError('Historical cost unknown must remain unknown')
    calls, totals, collected = set(), {}, 0
    for row in ledger:
        if row['call_id'] in calls:
            raise ValueError('Duplicate historical budget call')
        calls.add(row['call_id'])
        amount = row.get('actual_interactions')
        if (type(amount) is not int or amount < 0 or row.get('consumption_verified') is not True
                or row.get('state') not in ('completed', 'failed', 'cancelled', 'canceled', 'interrupted')):
            raise ValueError('Unreconciled historical consumption')
        if row.get('experiment_id') != contract['experiment_id']:
            # Old consumption is retained, but is never transferred into the fresh scope.
            continue
        if row.get('owner') != witness['owner']:
            raise ValueError('Fresh ledger owner mismatch')
        if row.get('method') == 'round_robin':
            if (row.get('seed') not in (8301, 8302, 8303) or row.get('partition') != 'train'
                    or row.get('state') != 'completed' or not 1 <= amount <= 30):
                raise ValueError('Invalid fresh physical collection')
            allocations = [{'method': m, 'seed': s, 'partition': 'train',
                            'collection_interactions': amount}
                           for m in ('dqn', 'ppo', 'pqn') for s in (8101, 8102, 8103)]
            if not _first_json_equal(row.get('scoped_ancestry_allocations'), allocations):
                raise ValueError('Every beneficiary must retain conservative collection allocation')
            collected += amount
            for allocation in allocations:
                key = (allocation['method'], allocation['seed'], 'train')
                totals[key] = totals.get(key, 0) + amount
        else:
            key = (row['method'], row['seed'], row['partition'])
            totals[key] = totals.get(key, 0) + amount
    if collected > 90 or any(v > 3090 for v in totals.values()):
        raise ValueError('Fresh scoped 3090 interaction limit exceeded')
    return totals, collected


def _selection_history(witness, ledger):
    """Reconcile scoped ancestry without claiming globally unknown history is zero."""
    counter = _first_artifact(witness['counter_admission'])
    exact = {'approved': True, 'owner': witness['owner'], 'freeze_id': witness['freeze_id'],
             'ancestry_retained': True,
             'scoped_counters_verified': True, 'historical_cloud_cost_usd': None,
             'historical_cloud_cost_unknown_retained': True,
             'ledger_manifest': witness['ledger_manifest']}
    if witness.get('retained_failed_zero_closures'):
        _selection_recovery(witness)
        exact.update(scoped_no_previous_training_interactions=True,
                     retained_failed_zero_closures=witness['retained_failed_zero_closures'])
        if 'scoped_training_seed_unused' in counter:
            raise ValueError('Failed training seed cannot be declared unused')
    else:
        exact['scoped_training_seed_unused'] = True
    if any(key not in counter or not _first_json_equal(counter[key], value)
           for key, value in exact.items()):
        raise ValueError('Explicit verified scoped counter and retained ancestry proof required')
    totals, calls, physical_total = {}, set(), 0
    for row in ledger:
        if row['call_id'] in calls:
            raise ValueError('Duplicate historical budget call')
        calls.add(row['call_id'])
        amount = row.get('actual_interactions')
        names = ('learned_steps', 'warmup_steps', 'collection_interactions', 'resume_offset')
        if any(name in row for name in names):
            counts = [row[name] for name in names]
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
        key = (row['method'], row['seed'], row['partition'])
        if 'scoped_ancestry_allocations' in row or 'conservative_full_ancestry' in row:
            allocations = [{'method': method, 'seed': 5101, 'partition': 'train',
                            'owner': witness['owner'], 'collection_interactions': amount}
                           for method in ('dqn', 'ppo')]
            if (row.get('conservative_full_ancestry') is not True
                    or row.get('owner') != witness['owner'] or row['state'] != 'completed'
                    or row['method'] != 'round_robin' or row['partition'] != 'train'
                    or row['seed'] in (5101, 6201, 6202)
                    or type(row['call_id']) is not str or not row['call_id']
                    or not 0 < amount <= 30
                    or not _first_json_equal(row['scoped_ancestry_allocations'], allocations)):
                raise ValueError('Invalid conservative physical ancestry allocations')
            manifest = _first_artifact(row['physical_call_manifest'])
            summary = _first_artifact(row['physical_summary'])
            physical = manifest.get('spec', {})
            if (manifest.get('state') != 'completed' or manifest.get('call_id') != row['call_id']
                    or physical.get('method') != row['method'] or physical.get('seed') != row['seed']
                    or physical.get('stage') != 'collect'
                    or physical.get('config', {}).get('execution', {}).get('split') != row['partition']
                    or summary.get('run_id') != physical.get('run_id')
                    or summary.get('stage') != 'collect'
                    or type(summary.get('round')) is not int or summary['round'] != amount
                    or type(row.get('warmup_included')) is not int
                    or not 0 <= row['warmup_included'] <= amount):
                raise ValueError('Physical ancestry call identity or interaction evidence mismatch')
            physical_total += amount
            for method in ('dqn', 'ppo'):
                scoped = (method, 5101, 'train')
                totals[scoped] = totals.get(scoped, 0) + amount
        if key[1] in (5101, 6201, 6202):
            if row.get('owner') != witness['owner']:
                raise ValueError('Scoped ledger owner mismatch')
            if key[1] == 5101:
                if (key[0] not in ('dqn', 'ppo') or key[2] != 'train'
                        or row.get('collection_interactions') != amount
                        or row.get('learned_steps') != 0 or row.get('warmup_steps') != 0
                        or row.get('resume_offset') != 0):
                    raise ValueError('Unused training seed permits retained collection only')
            totals[key] = totals.get(key, 0) + amount
    collection = {method: totals.get((method, 5101, 'train'), 0) for method in ('dqn', 'ppo')}
    if (physical_total > 120 or any(value > 120 for value in collection.values())
            or not _first_json_equal(counter.get('scoped_collection_interactions_by_algorithm'), collection)):
        raise ValueError('Retained collection counters must be verified and at most 120 per algorithm')
    expected = [{'method': method, 'seed': 5101, 'partition': 'train',
                 'limit': 5000 + collection[method]} for method in ('dqn', 'ppo')]
    expected += [{'method': method, 'seed': seed, 'partition': 'validation', 'limit': 3000}
                 for method in ('dqn', 'ppo', 'time_greedy') for seed in (6201, 6202)]
    if not _first_json_equal(witness.get('group_interaction_limits'), expected):
        raise ValueError('Exact scoped training and unchanged validation limits required')
    limits = {(row['method'], row['seed'], row['partition']): row['limit'] for row in expected}
    for run in witness['runs']:
        for charge in run['charges']:
            key = (charge['method'], charge['seed'], charge['partition'])
            totals[key] = totals.get(key, 0) + charge['actual_interactions']
    if any(key not in limits or amount > limits[key] for key, amount in totals.items()):
        raise ValueError('Full frozen matrix exceeds scoped group interaction limit')
    return limits


def _selection_physical_ancestry(ledger):
    """Budget both descendants without duplicating physical calls or warmup."""
    totals = {}
    for row in ledger:
        if row.get('conservative_full_ancestry') is True:
            for allocation in row['scoped_ancestry_allocations']:
                key = (allocation['method'], allocation['seed'], allocation['partition'])
                totals[key] = totals.get(key, 0) + allocation['collection_interactions']
    return totals


def _selection_training(witness):
    """Validate the entire 62-call matrix before admitting any individual call."""
    from .parameter_selection import candidate_config
    if not _resource_shared_source(witness):
        raise ValueError('Parameter selection requires compact shared source schema')
    authorization = witness.get('configuration_authorization', {})
    ids = [f'{prefix}{index:02d}' for prefix in ('D', 'P') for index in range(10)]
    exact = {'approved': True, 'owner': witness['owner'], 'freeze_id': witness['freeze_id'],
             'approved_additional_slots': 20, 'training_seed': 5101,
             'evaluation_seeds': [6201, 6202], 'training_interactions_per_algorithm': 5000,
             'max_compute_calls': 62, 'historical_configuration_count_exact': None,
             'historical_configuration_count_unknown_retained': True, 'configuration_ids': ids}
    if any(key not in authorization or not _first_json_equal(authorization[key], value)
           for key, value in exact.items()):
        raise ValueError('Invalid explicit parameter-selection configuration authorization')
    lower, risk = authorization.get('historical_configuration_lower_bound'), authorization.get('risk_version')
    if type(lower) is not int or lower < 12 or type(risk) is not str or not risk:
        raise ValueError('Retained historical lower bound and frozen risk version required')
    plan = _first_artifact(witness['plan_config'])
    if 'execution_admission' in plan.get('native_acceptance', {}):
        raise ValueError('Plan CONFIG snapshot must exclude execution admission')
    plan = deepcopy(plan)
    plan.get('native_acceptance', {}).pop('execution_admission', None)
    settings = plan['native_acceptance']['parameter_selection']
    if (settings.get('training_seed') != 5101 or settings.get('evaluation_seeds') != [6201, 6202]
            or settings.get('frozen_risk_version') != risk
            or type(settings.get('frozen_risk_checkpoint')) is not str
            or not settings['frozen_risk_checkpoint']):
        raise ValueError('Plan CONFIG seed and risk freeze mismatch')
    runs = witness.get('runs')
    if type(runs) is not list or len(runs) != 62:
        raise ValueError('Exactly 62 frozen parameter-selection runs required')
    trains, evaluations, references, run_ids = {}, {}, {}, set()
    def scientific(config):
        value = deepcopy(config)
        value.get('native_acceptance', {}).pop('execution_admission', None)
        value.get('experiment', {}).pop('variant', None)
        value['execution'].pop('split', None)
        value['modal'].pop('timeout_seconds', None)
        return value
    source = None
    for run in runs:
        payload = run['spec']
        config, inputs = payload['config'], payload['inputs']
        cid = config.get('experiment', {}).get('variant')
        reference = cid == 'reference-time_greedy'
        method = 'time_greedy' if reference else 'dqn' if cid in ids[:10] else 'ppo'
        if (cid not in ids and not reference or payload['method'] != method
                or type(payload['seed']) is not int
                or type(payload['run_id']) is not str or not payload['run_id']
                or payload['run_id'] in run_ids or payload['risk_version'] != risk
                or inputs.get('risk_checkpoint') != settings['frozen_risk_checkpoint']
                or config['execution'].get('backend') != 'modal'
                or payload.get('provenance', {}).get('identity_kind') != 'source_contents_no_hash'
                or 'source_contents' in payload['provenance']):
            raise ValueError('Invalid frozen selection run, source, configuration or risk')
        if source is None:
            source = payload['provenance']
        if not _first_json_equal(source, payload['provenance']):
            raise ValueError('Selection matrix provenance differs')
        run_ids.add(payload['run_id'])
        modal = config['modal']
        bounded = {'gpu': 'H100', 'cpu': 16.0, 'memory_mb': 65536,
                   'startup_timeout_seconds': 180, 'max_containers': 1}
        timeout = modal.get('timeout_seconds')
        if (any(not _first_json_equal(modal.get(key), value) for key, value in bounded.items())
                or type(timeout) is not int or not 1 <= timeout <= 900
                or type(modal.get('retries', 0)) is not int or modal.get('retries', 0) != 0):
            raise ValueError('Bounded H100 settings and zero retries required')
        expected = candidate_config(plan, 'D00' if reference else cid)
        if not _first_json_equal(scientific(config), scientific(expected)):
            raise ValueError('Selection run differs from sole scientific CONFIG design')
        stage, seed = payload['stage'], payload['seed']
        if stage == 'train_policy':
            if (reference or cid in trains or seed != 5101
                    or config['execution'].get('split') != 'train'
                    or set(inputs) != {'risk_checkpoint', 'previous_training_checkpoint'}
                    or inputs['previous_training_checkpoint'] not in (None, '')):
                raise ValueError('Invalid 500-interaction no-resume training run')
            trains[cid] = payload
            amount, partition = 500, 'train'
        elif stage == 'validate':
            if seed not in (6201, 6202) or config['execution'].get('split') != 'validation':
                raise ValueError('Invalid paired validation seed or split')
            if reference:
                if seed in references or set(inputs) != {'risk_checkpoint'}:
                    raise ValueError('Invalid fixed time-greedy reference')
                references[seed] = payload
            else:
                if (cid, seed) in evaluations or set(inputs) != {'risk_checkpoint', 'agent_checkpoint'}:
                    raise ValueError('Invalid or duplicate candidate evaluation')
                evaluations[cid, seed] = payload
            amount, partition = 30, 'validation'
        else:
            raise ValueError('Unrelated selection stage forbidden')
        charge = {'method': method, 'seed': seed, 'partition': partition,
                  'configuration_id': cid, 'actual_interactions': amount}
        if not _first_json_equal(run.get('charges'), [charge]):
            raise ValueError('Exact canonical interaction charge required')
    if set(trains) != set(ids) or len(evaluations) != 40 or set(references) != {6201, 6202}:
        raise ValueError('Incomplete selection matrix')
    for (cid, _), payload in evaluations.items():
        if payload['inputs']['agent_checkpoint'] != f"runs/{trains[cid]['run_id']}/agent.pt":
            raise ValueError('Evaluation must load its own exact training agent')
    return set(ids) | {'reference-time_greedy'}


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
                 ('fixedcandidate_independent_validation', _RESOURCE_PURPOSE, _SELECTION_PURPOSE)):
            raise ValueError('Explicit owner stage approval required')
        if witness['purpose'] == _RESOURCE_PURPOSE:
            _resource_training(witness)
        if witness['purpose'] == _SELECTION_PURPOSE:
            _selection_training(witness)
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
        if witness['purpose'] == _SELECTION_PURPOSE:
            _selection_history(witness, ledger)
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


def _validate_failed_zero(approval, proof_descriptor, reservation, binding, log):
    """Only the preserved constructor failure can release a bound selection charge."""
    witness, proof = _first_artifact(approval), _first_artifact(proof_descriptor)
    run_id = reservation['run_id']
    payload = reservation['spec']
    if (witness.get('approved') is not True or witness.get('purpose') != _SELECTION_PURPOSE
            or type(witness.get('owner')) is not str or not witness['owner']
            or Path(witness['reservation_log']) != log
            or reservation['freeze_id'] != witness['freeze_id']
            or not _first_json_equal(reservation['ledger_manifest'], witness['ledger_manifest'])
            or not any(_first_json_equal(item, {'spec': payload, 'charges': reservation['charges']})
                       for item in witness['runs'])
            or payload.get('stage') != 'train_policy' or payload.get('method') != 'dqn'
            or type(payload.get('seed')) is not int or payload['seed'] != 5101
            or payload['config'].get('experiment', {}).get('variant') != 'D00'
            or not _first_json_equal(reservation['charges'], [{'method': 'dqn', 'seed': 5101,
                    'partition': 'train', 'configuration_id': 'D00', 'actual_interactions': 500}])):
        raise ValueError('Failed-zero closure requires exact saved D00 selection approval')
    freeze = _first_artifact(witness['freeze_manifest'])
    if (freeze.get('freeze_id') != witness['freeze_id']
            or not _first_json_equal(freeze.get('runs'), witness['runs'])):
        raise ValueError('Saved failed-zero freeze mismatch')
    _first_artifact(witness['ledger_manifest'])
    source = _first_artifact(witness['source_contents_manifest'])
    exact = {'approved': True, 'owner': witness['owner'], 'purpose': _SELECTION_PURPOSE,
             'freeze_id': witness['freeze_id'], 'run_id': run_id,
             'call_id': binding['call_id'], 'classification': 'FAILED_ZERO_INTERACTIONS',
             'actual_environment_interactions': 0, 'optimizer_updates': 0,
             'failed_call_actual_cloud_cost_usd': None, 'no_cost_imputed_zero': True}
    if any(key not in proof or not _first_json_equal(proof[key], value) for key, value in exact.items()):
        raise ValueError('Invalid bound failed-zero proof identity or counters')
    app_id = proof.get('app_id')
    if type(app_id) is not str or not app_id or type(binding.get('call_id')) is not str or not binding['call_id']:
        raise ValueError('Explicit bound call and app required')
    config = _first_artifact(proof['config'])
    expected = deepcopy(payload)
    expected['provenance']['source_contents'] = source
    actual = deepcopy(config)
    actual.get('config', {}).get('native_acceptance', {}).pop('execution_admission', None)
    if not _first_json_equal(actual, expected):
        raise ValueError('Original failed config does not match saved approval')
    metadata = _first_artifact(proof['metadata'])
    if any(not _first_json_equal(metadata.get(key), expected[key])
           for key in ('run_id', 'stage', 'method', 'seed', 'risk_version', 'provenance')):
        raise ValueError('Original failed metadata mismatch')
    if (expected['config'].get('risk', {}).get('mode') != 'unavailable'
            or not _first_json_equal(expected['config'].get('reward', {}).get('risk_weight'), 0.1)):
        raise ValueError('Only unavailable risk with positive frozen reward is reconciled')
    status = _first_artifact(proof['status'])
    error = '无风险模型时风险权重必须为 0 / Unavailable risk requires risk_weight=0'
    if status.get('state') != 'failed' or status.get('error') != error:
        raise ValueError('Exact failed pre-environment constructor error required')
    runtime = _first_artifact(proof['runtime'])
    if (type(runtime.get('python')) is not str or type(runtime.get('packages')) is not dict
            or runtime.get('cuda_available') is not True or runtime.get('gpu') != 'NVIDIA H100 80GB HBM3'):
        raise ValueError('Original remote H100 runtime required')
    inspection = _first_artifact(proof['inspection'])
    inspected = {'run_id': run_id, 'call_id': binding['call_id'], 'app_id': app_id,
                 'status': 'failed_before_environment_construction_no_training_interactions',
                 'remote_status': status, 'risk_mode_in_frozen_spec': 'unavailable',
                 'risk_weight_in_frozen_spec': 0.1, 'environment_constructor_failed_before_reset': True,
                 'actual_environment_interactions': 0, 'optimizer_updates': 0, 'new_audits': 0,
                 'new_compute_calls_in_recovery': 0, 'failed_call_actual_cloud_cost_usd': None,
                 'no_cost_imputed_zero': True, 'same_failed_run_retry_authorized': False}
    files = inspection.get('observed_remote_files')
    if (any(key not in inspection or not _first_json_equal(inspection[key], value)
            for key, value in inspected.items()) or type(files) is not list
            or len(files) != 6 or set(files) != {'config.json', 'data_metadata.json', 'error.txt',
                                               'metadata.json', 'runtime.json', 'status.json'}):
        raise ValueError('Original inspection must exclude all partial training artifacts')
    shutdown = _first_artifact(proof['shutdown'])
    receipt = shutdown.get('observed_app_row', {})
    stopped = {'run_id': run_id, 'call_id': binding['call_id'], 'app_id': app_id,
               'verified': True, 'stopped': True, 'new_compute_calls': 0,
               'automatic_retry_authorized': False}
    if (any(key not in shutdown or not _first_json_equal(shutdown[key], value)
            for key, value in stopped.items()) or receipt.get('app_id') != app_id
            or receipt.get('state') != 'stopped' or receipt.get('tasks') != '0'
            or type(receipt.get('stopped_at')) is not str or not receipt['stopped_at']):
        raise ValueError('Exact terminal stopped app receipt required')
    return witness


def verify_failed_zero_interactions(approval_descriptor, proof_descriptor):
    """Append a durable bound-call closure using old approval, not current source."""
    witness = _first_artifact(approval_descriptor)
    log = Path(witness['reservation_log'])
    if not log.is_absolute():
        raise ValueError('Absolute durable reservation log required')
    with _locked(log):
        rows = _rows(log)
        reservations, bound, closed = _journal(rows, log)
        proof = _first_artifact(proof_descriptor)
        run_id = proof['run_id']
        if run_id not in reservations or run_id not in bound:
            raise ValueError('Failed-zero closure requires an existing bound reservation')
        binding = next(row for row in rows if row['event'] == 'bound' and row['run_id'] == run_id)
        _validate_failed_zero(approval_descriptor, proof_descriptor, reservations[run_id], binding, log)
        closure = {'event': 'verified_failed_zero_interactions', 'run_id': run_id,
                   'actual_interactions': 0, 'failed_call_actual_cloud_cost_usd': None,
                   'approval': approval_descriptor, 'proof': proof_descriptor, 'reservation_log': str(log)}
        if run_id in closed:
            if not any(_first_json_equal(row, closure) for row in rows):
                raise ValueError('Failed-zero closure proof changed')
            return closure
        _append(log, closure)
        return closure


def _failed_zero_summary(closure):
    approval = _first_artifact(closure['approval'])
    proof = _first_artifact(closure['proof'])
    return {'event': closure['event'], 'run_id': closure['run_id'],
            'actual_interactions': closure['actual_interactions'],
            'failed_call_actual_cloud_cost_usd': closure['failed_call_actual_cloud_cost_usd'],
            'owner': approval['owner'], 'purpose': approval['purpose'],
            'freeze_id': approval['freeze_id'], 'app_id': proof['app_id'],
            'call_id': proof['call_id'], 'reservation_log': closure['reservation_log']}


def failed_zero_closure_summary(closure):
    """Return a small exact summary only for a revalidated full durable closure."""
    log = Path(closure['reservation_log'])
    if not log.is_absolute():
        raise ValueError('Absolute durable reservation log required')
    with _locked(log):
        rows = _rows(log)
        _journal(rows, log)
        matches = [row for row in rows if row.get('event') == 'verified_failed_zero_interactions'
                   and row.get('run_id') == closure.get('run_id')]
        if len(matches) != 1 or not _first_json_equal(matches[0], closure):
            raise ValueError('Summary requires an exact unique full durable closure')
        return _failed_zero_summary(matches[0])


def _selection_recovery(witness):
    """A new matrix retains the exact old ledger and one closed D00 call, not new slots."""
    descriptors = witness.get('retained_failed_zero_closures')
    amendment = witness.get('failed_zero_amendment', {})
    exact = {'approved': True, 'owner': witness['owner'], 'purpose': _SELECTION_PURPOSE,
             'freeze_id': witness['freeze_id'], 'max_total_compute_calls': 63,
             'max_active_matrix_calls': 62, 'approved_additional_slots': 20,
             'no_counter_reset': True, 'retained_failed_zero_closures': descriptors}
    if (type(descriptors) is not list or len(descriptors) != 1
            or any(key not in amendment or not _first_json_equal(amendment[key], value)
                   for key, value in exact.items())):
        raise ValueError('Explicit same-owner failed-zero amendment required')
    log = Path(witness['reservation_log'])
    rows = _rows(log)
    reservations, _, closed = _journal(rows, log)
    retained = []
    for descriptor in descriptors:
        saved = _first_artifact(descriptor)
        matches = [row for row in rows if row.get('event') == 'verified_failed_zero_interactions'
                   and row.get('run_id') == saved.get('run_id')]
        if len(matches) != 1:
            raise ValueError('Exact unique durable failed-zero closure required')
        closure = matches[0]
        # Full descriptors remain compatible; compact summaries must match every field.
        if not (_first_json_equal(saved, closure)
                or _first_json_equal(saved, _failed_zero_summary(closure))):
            raise ValueError('Exact durable failed-zero closure descriptor required')
        retained.append(closure)
        old = _first_artifact(closure['approval'])
        if (old['owner'] != witness['owner'] or old['purpose'] != witness['purpose']
                or old['freeze_id'] == witness['freeze_id']
                or not _first_json_equal(old['ledger_manifest'], witness['ledger_manifest'])
                or not _first_json_equal(old['configuration_authorization']['configuration_ids'],
                                         witness['configuration_authorization']['configuration_ids'])
                or old['configuration_authorization']['historical_configuration_lower_bound']
                   != witness['configuration_authorization']['historical_configuration_lower_bound']):
            raise ValueError('Recovery cannot reset owner, ancestry or configuration slots')
    retained_ids = {row['run_id'] for row in retained}
    matrix_ids = {row['spec']['run_id'] for row in witness['runs']}
    if retained_ids & matrix_ids or any(run_id not in matrix_ids and
            (run_id not in closed or run_id not in retained_ids) for run_id in reservations):
        raise ValueError('Foreign or unknown spawn cannot be excluded by new freeze')
    return retained_ids


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
        elif event == 'verified_failed_zero_interactions':
            if run_id not in reservations or run_id not in bound or run_id in closed:
                raise ValueError('Contradictory bound failed-zero closure')
            if (type(row.get('actual_interactions')) is not int or row['actual_interactions'] != 0
                    or 'failed_call_actual_cloud_cost_usd' not in row
                    or row['failed_call_actual_cloud_cost_usd'] is not None
                    or Path(row['reservation_log']) != log):
                raise ValueError('Invalid failed-zero durable closure')
            binding = next(item for item in rows if item['event'] == 'bound' and item['run_id'] == run_id)
            _validate_failed_zero(row['approval'], row['proof'], reservations[run_id], binding, log)
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
        retained = (_selection_recovery(witness) if witness['purpose'] == _SELECTION_PURPOSE
                    and witness.get('retained_failed_zero_closures') else set())
        for row in rows:
            if row['event'] == 'reserved' and not _first_json_equal(
                    row['ledger_manifest'], witness['ledger_manifest']):
                raise ValueError('Reservation history baseline cannot reset or change')
            if (witness['purpose'] == _SELECTION_PURPOSE and row['event'] == 'reserved'
                    and row['freeze_id'] != witness['freeze_id'] and row['run_id'] not in retained):
                raise ValueError('Selection freeze cannot reset or change')
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
        group_limits = None
        if witness['purpose'] == _SELECTION_PURPOSE:
            resource_ids = _selection_training(witness)
            group_limits = _selection_history(witness, ledger)
            reservations = [row for row in rows if row['event'] == 'reserved']
            if len(reservations) >= 62 + len(retained):
                raise ValueError('Maximum 62 compute reservations exceeded')
            approved_runs = {item['spec']['run_id']: item for item in witness['runs']}
            if any(row['run_id'] not in approved_runs
                   or not _first_json_equal(row['spec'], approved_runs[row['run_id']]['spec'])
                   or not _first_json_equal(row['charges'], approved_runs[row['run_id']]['charges'])
                    for row in reservations if row['run_id'] not in retained):
                raise ValueError('Selection journal contains foreign or changed runs')
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
        if witness['purpose'] == _SELECTION_PURPOSE:
            for key, amount in _selection_physical_ancestry(ledger).items():
                totals[key] = totals.get(key, 0) + amount
        for charge in charges:
            key = (charge['method'], charge['seed'], charge['partition'])
            amount = charge['actual_interactions']
            if (type(charge['seed']) is not int or type(amount) is not int or amount < 0
                    or (charge['configuration_id'] not in
                        (resource_ids if resource_ids is not None else
                         configurations.get((key[0], key[2]), set())))):
                raise ValueError('No new configuration tuning is authorized')
            totals[key] = totals.get(key, 0) + amount
            if amount and totals[key] > (group_limits[key] if group_limits is not None else 3000):
                raise ValueError('Cumulative group interaction limit exceeded')
        _append(log, {'event': 'reserved', 'run_id': spec.run_id, 'spec': run['spec'],
                      'charges': charges, 'freeze_id': witness['freeze_id'],
                      'ledger_manifest': witness['ledger_manifest']})
    return None


def bind_call(spec, root, call_id):
    """Append call binding only after spawn; a failure here leaves the reservation intact."""
    witness, _, _, log = check_admission([spec], root)[0]
    if type(call_id) is not str or not call_id:
        raise ValueError('Explicit call ID required')
    with _locked(log):
        journal = _rows(log)
        if (witness['purpose'] == _SELECTION_PURPOSE
                and any(row['event'] == 'bound' and row['call_id'] == call_id for row in journal)):
            raise ValueError('Duplicate compute call binding')
        rows = [row for row in journal if row['run_id'] == spec.run_id]
        if len(rows) != 1 or rows[0]['event'] != 'reserved':
            raise ValueError('Call binding requires exactly one reservation')
        _append(log, {'event': 'bound', 'run_id': spec.run_id, 'call_id': call_id})
