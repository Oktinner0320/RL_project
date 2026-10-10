"""Internal, isolated stage execution; admission belongs to the ablation runner."""

import csv
import json
from pathlib import Path

import torch

from cloud_edge_rl.artifacts import RunStore, artifact_payload
from cloud_edge_rl.config import RunSpec
from cloud_edge_rl.algorithms.policy import load_agent, _settings
from cloud_edge_rl import pipeline


def _inside(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or '\\' in str(value):
        raise ValueError('Artifact paths must be workspace-relative')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Artifact path escapes ablation workspace')
    return resolved


def _guard(spec, workspace):
    root = Path(workspace).resolve(strict=True)
    marker = _inside(root, 'ablation-workspace.json')
    identity = json.loads(marker.read_text(encoding='utf-8'))
    experiment = identity.get('experiment_id')
    if (identity.get('schema') != 'ablation_workspace_v1' or not experiment
            or spec.provenance.get('ablation_experiment_id') != experiment):
        raise ValueError('Isolated ablation workspace identity mismatch')
    if spec.method != 'dqn' or spec.stage not in ('train_policy', 'validate'):
        raise ValueError('Only DQN train_policy/validate stages are allowed')
    if (not isinstance(spec.run_id, str) or not spec.run_id
            or Path(spec.run_id).name != spec.run_id or spec.run_id in ('.', '..')
            or '/' in spec.run_id or '\\' in spec.run_id or ':' in spec.run_id):
        raise ValueError('Invalid new run identity')
    _inside(root, f'runs/{spec.run_id}')
    cfg = spec.config
    rl = cfg['rl']
    if (rl.get('algorithm') != 'dqn' or type(rl.get('n_step')) is not int
            or rl['n_step'] not in (1, 3) or rl.get('gamma') != 1
            or rl.get('resource_feedback') is not None or rl.get('total_steps') != 1200):
        raise ValueError('Invalid DQN ablation settings')
    if spec.stage == 'train_policy' and rl.get('stage_interaction_limit') not in (500, 700):
        raise ValueError('Training must be bounded to 500 or 700 interactions')
    if cfg['execution'].get('split') != ('train' if spec.stage == 'train_policy' else 'validation'):
        raise ValueError('Official/test splits are forbidden')
    if cfg['data'].get('include_official_test', False):
        raise ValueError('Official test loading is forbidden')
    if cfg['risk'].get('mode') != 'fitted':
        raise ValueError('A frozen fitted risk checkpoint is required')
    allowed = {'risk_checkpoint', 'previous_training_checkpoint'} if spec.stage == 'train_policy' else {'risk_checkpoint', 'agent_checkpoint'}
    if set(spec.inputs) != allowed or spec.inputs['risk_checkpoint'] != 'inputs/risk.pt':
        raise ValueError('Only explicit workspace risk and policy inputs are allowed')
    for key, value in spec.inputs.items():
        if key == 'previous_training_checkpoint' and value is None:
            continue
        if key != 'risk_checkpoint':
            parts = Path(value).parts
            expected = 'training.pt'
            if len(parts) != 3 or parts[0] != 'runs' or parts[-1] != expected or parts[1] == spec.run_id:
                raise ValueError('Policy input must be a prior workspace run artifact')
        if not _inside(root, value).is_file():
            raise ValueError('Missing workspace artifact: ' + str(value))
    return root, experiment


def _config(spec, training_seed=None):
    return {**spec.config['rl'], 'risk_version': spec.risk_version,
            'seed': spec.seed if training_seed is None else training_seed,
            'warmup_actions': spec.config['execution']['warmup_actions'], 'algorithm': 'dqn'}


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _training(envelope, spec, experiment):
    config = _config(spec)
    state = envelope['state']
    settings = {**_settings(config), 'total_steps': config['total_steps']}
    if (envelope.get('ablation_experiment_id') != experiment
            or envelope.get('risk_version') != spec.risk_version
            or state.get('risk_version') != spec.risk_version
            or state.get('format') != 'ablation_dqn_training_v1'
            or state.get('settings') != settings):
        raise ValueError('Prior training experiment/risk/seed/n-step/settings mismatch')
    for key in ('environment_steps', 'learned_steps', 'warmup_steps', 'updates'):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError('Invalid actual training counter')
    if state['environment_steps'] != state['learned_steps'] + state['warmup_steps']:
        raise ValueError('Warmup must be included in the interaction budget')
    load_agent(state['agent'], config)
    return state


def execute_stage(spec: RunSpec, workspace: Path) -> dict:
    """LOW-LEVEL internal worker AFTER runner admission, not an unguarded notebook launcher.

    Requires an isolated marked workspace, frozen risk input, and explicit prior
    training/agent input. Validation rl.seed is the training seed; spec.seed is
    the evaluation seed. This function supplies no approval or submission API.
    """
    root, experiment = _guard(spec, workspace)
    previous = None
    if spec.stage == 'train_policy' and spec.inputs['previous_training_checkpoint']:
        previous = _load(_inside(root, spec.inputs['previous_training_checkpoint']))
        _training(previous, spec, experiment)
    with RunStore(root, spec) as store:
        store.write_json('runtime.json', pipeline.runtime_info())
        data = pipeline.prepare_adult(spec.config['data'], store.path / 'data', include_official_test=False)
        store.write_json('data_metadata.json', data.metadata)
        env = pipeline._make_environment(spec, data, root, store)
        try:
            if spec.stage == 'train_policy':
                from ablations.variants.dqn_nstep import train_agent

                def save_training(state):
                    store.save_checkpoint({'state': state, 'risk_version': spec.risk_version,
                                           'run_id': spec.run_id, 'ablation_experiment_id': experiment}, 'training.pt')

                trained = train_agent(env, _config(spec), previous, save_training)
                save_training(trained.checkpoint)
                state = trained.checkpoint
                store.save_checkpoint({'agent': trained.agent.checkpoint(), 'risk_version': spec.risk_version,
                                       'training_seed': spec.seed, 'n_step': spec.config['rl']['n_step'],
                                       'ablation_experiment_id': experiment}, 'agent.pt')
                store.write_json('policy_training.json', state['history'])
                before = previous['state']['environment_steps'] if previous else 0
                result = {'trained': True, 'new_environment_steps': state['environment_steps'] - before,
                          **{key: state[key] for key in ('environment_steps', 'learned_steps', 'warmup_steps', 'updates')}}
            else:
                saved = _load(_inside(root, spec.inputs['agent_checkpoint']))
                seed = spec.config['rl']['seed']
                from dataclasses import replace

                state = _training(saved, replace(spec, seed=seed), experiment)
                if state['environment_steps'] != 1200:
                    raise ValueError('Validation requires the completed training budget')
                policy = load_agent(state['agent'], _config(spec, seed))
                if policy.observation_space != env.observation_space or policy.action_count != env.action_space.n:
                    raise ValueError('Validation environment space mismatch')
                result = pipeline._run_episode(env, policy, spec.seed)
                store.save_checkpoint(env.snapshot(), 'environment.pt')
            result = {'run_id': spec.run_id, 'stage': spec.stage, 'risk_version': spec.risk_version,
                      'ablation_experiment_id': experiment, 'risk_measurement': 'predicted_risk', **result}
            store.write_json('summary.json', result)
        finally:
            env.close()
    return result


def verify_stage(spec: RunSpec, workspace: Path, result: dict) -> dict:
    """Read actual artifacts and counters; not formal acceptance or true-risk auditing."""
    root, experiment = _guard(spec, workspace)
    path = _inside(root, f'runs/{spec.run_id}')
    read = lambda name: json.loads(_inside(root, f'runs/{spec.run_id}/{name}').read_text(encoding='utf-8'))
    if read('status.json').get('state') != 'completed' or read('config.json') != artifact_payload(spec) or read('summary.json') != result:
        raise ValueError('Completed status/config/summary content mismatch')
    with _inside(root, f'runs/{spec.run_id}/metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or result.get('risk_measurement') != 'predicted_risk':
        raise ValueError('Missing actual metrics or predicted-risk labeling')
    if spec.stage == 'train_policy':
        envelope = _load(_inside(root, f'runs/{spec.run_id}/training.pt'))
        state = _training(envelope, spec, experiment)
        from ablations.variants.dqn_nstep import _validate

        _validate(_config(spec), envelope)
        agent = _load(_inside(root, f'runs/{spec.run_id}/agent.pt'))
        if (agent.get('risk_version') != spec.risk_version or agent.get('training_seed') != spec.seed
                or agent.get('n_step') != spec.config['rl']['n_step']
                or agent.get('ablation_experiment_id') != experiment):
            raise ValueError('Exported agent binding mismatch')
        exported = load_agent(agent['agent'], _config(spec))
        actual = load_agent(state['agent'], _config(spec))
        if any(not torch.equal(value, exported.network.state_dict()[key])
               for key, value in actual.network.state_dict().items()):
            raise ValueError('Exported agent differs from trained checkpoint')
        prior_path = spec.inputs['previous_training_checkpoint']
        before = _training(_load(_inside(root, prior_path)), spec, experiment)['environment_steps'] if prior_path else 0
        target = min(1200, before + spec.config['rl']['stage_interaction_limit'])
        delta = state['environment_steps'] - before
        if (state['environment_steps'] != target or delta != len(rows) or delta != result.get('new_environment_steps')
                or state['updates'] <= 0 or read('policy_training.json') != state['history']
                or any(result.get(k) != state[k] for k in ('environment_steps', 'learned_steps', 'warmup_steps', 'updates'))):
            raise ValueError('Actual trained checkpoint/counters/stage target mismatch')
        warmup_before = _load(_inside(root, prior_path))['state']['warmup_steps'] if prior_path else 0
        if sum(row.get('is_warmup') == 'True' for row in rows) != state['warmup_steps'] - warmup_before:
            raise ValueError('Actual warmup metric count mismatch')
    else:
        if rows[-1].get('terminated') != 'True' or rows[-1].get('truncated') != 'False':
            raise ValueError('Validation must reach a true terminal, not a budget truncation')
        _load(_inside(root, f'runs/{spec.run_id}/environment.pt'))
    return {'verified': True, 'run_id': spec.run_id, 'actual_interactions': len(rows),
            'risk_measurement': 'predicted_risk', 'formal_acceptance': False}
