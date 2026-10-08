"""真实同目标样本门；不是最终多种子验收。 / Real common-target sample gate, not final acceptance."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..algorithms.attack import GradientAttack
from ..algorithms.federated import FederatedBackend
from ..metrics import reconstruction_metrics, sample_marginal_records, sample_random_records
from ..types import Action
from .attack import TabularGradientAttack
from .federated import TabularFedSGD


def backend_gate(data, config: dict, source_root: Path, store) -> dict:
    """全动作、配对种子验证 FedSGD；不打开测试集。 / All actions and seeds, without test evaluation."""
    settings = config['native_acceptance']
    tolerance = settings['backend_numerical_tolerance']
    if (not 0 < tolerance < 1 or not settings['seeds'] or
            len(set(settings['seeds'])) != len(settings['seeds'])):
        raise ValueError('Backend gate needs unique seeds and numerical tolerance in (0,1)')
    actions = [Action(i, tuple(clients), batch)
               for i, (clients, batch) in enumerate(
                   (c, b) for c in config['environment']['client_sets']
                   for b in config['environment']['batch_sizes'])]
    rows = []
    for seed in settings['seeds']:
        model_config = {**config['model'], 'seed': seed}
        if len(set(model_config['hidden_sizes'])) != 1:
            raise ValueError('The reference MLP requires equal hidden widths')
        native = FederatedBackend(data, model_config)
        reference = TabularFedSGD(source_root, data, {
            **model_config, 'architecture': 'tabular_gia_mlp',
            'hidden_size': model_config['hidden_sizes'][0],
            'hidden_layers': len(model_config['hidden_sizes']),
        })
        native.reset(seed)
        reference.reset(seed)
        with torch.no_grad():
            parameters = dict(native.model.named_parameters())
            for name, parameter in reference.model.named_parameters():
                parameter.copy_(parameters[name])
        for action in actions:
            durations, results = [], []
            for backend in (native, reference):
                if native.device.type == 'cuda':
                    torch.cuda.synchronize()
                started = perf_counter()
                results.append(backend.step_round(action))
                if native.device.type == 'cuda':
                    torch.cuda.synchronize()
                durations.append(perf_counter() - started)
            n, r = results
            samples_equal = len(n.updates) == len(r.updates) and all(
                np.array_equal(a.sample_ids, b.sample_ids)
                for a, b in zip(n.updates, r.updates))
            gradient_difference = max(
                (a.attack_input.gradients[name] - b.attack_input.gradients[name]).abs().max().item()
                for a, b in zip(n.updates, r.updates) for name in a.attack_input.gradients)
            parameters = dict(reference.model.named_parameters())
            parameter_difference = max((p - parameters[name]).abs().max().item()
                                       for name, p in native.model.named_parameters())
            loss_difference = abs(n.summary.validation_loss - r.summary.validation_loss)
            accuracy_difference = abs(n.summary.validation_accuracy - r.summary.validation_accuracy)
            passed = samples_equal and all(np.isfinite(value) and value <= tolerance for value in
                (gradient_difference, parameter_difference, loss_difference, accuracy_difference))
            rows.append({'seed': seed, 'action': asdict(action),
                         'partition': config['data']['partition'],
                         'samples_equal': samples_equal,
                         'gradient_difference': gradient_difference,
                         'parameter_difference': parameter_difference,
                         'validation_loss_difference': loss_difference,
                         'validation_accuracy_difference': accuracy_difference,
                         'native': asdict(n.summary), 'reference': asdict(r.summary),
                         'native_seconds': durations[0], 'reference_seconds': durations[1],
                         'status': 'pass' if passed else 'fail'})
            store.write_json('backend_progress.json', {'rows': rows})
    return {'scope': 'FedSGD paired all-action numerical gate; not overall acceptance',
            'device': str(native.device), 'partition': config['data']['partition'],
            'seeds': settings['seeds'], 'cases': len(rows), 'rows': rows,
            'status': 'pass' if rows and all(row['status'] == 'pass' for row in rows) else 'fail',
            'final_acceptance': False}


def sample_gate(data, config: dict, source_root: Path, store) -> dict:
    """共享参数、采样和梯度，真值仅供评价。 / Share weights/samples/gradients; truth is evaluator-only."""
    settings = config['native_acceptance']
    seed = settings['sample_seed']
    torch.set_num_threads(config['comparisons']['risk']['threads'])
    model_config = {**config['model'], 'seed': seed}
    native = FederatedBackend(data, model_config)
    reference = TabularFedSGD(source_root, data, {
        **model_config, 'architecture': 'tabular_gia_mlp',
        'hidden_size': model_config['hidden_sizes'][0],
        'hidden_layers': len(model_config['hidden_sizes']),
    })
    if len(set(model_config['hidden_sizes'])) != 1:
        raise ValueError('The reference MLP requires equal hidden widths')
    native.reset(seed)
    reference.reset(seed)
    # 对照的初始化策略不同；显式复制共同初始权重。 / Explicit common initial weights.
    with torch.no_grad():
        source_parameters = dict(native.model.named_parameters())
        for name, parameter in reference.model.named_parameters():
            parameter.copy_(source_parameters[name])
    actions = [Action(i, tuple(clients), batch)
               for i, (clients, batch) in enumerate(
                   (c, b) for c in config['environment']['client_sets']
                   for b in config['environment']['batch_sizes'])]
    chosen = actions[settings['sample_action']]
    rows, backends = [], []
    previous_round = 0
    for rounds in settings['model_rounds']:
        if rounds < previous_round:
            raise ValueError('Sample model rounds must be increasing')
        for index in range(previous_round, rounds):
            action = actions[config['execution']['warmup_actions'][index % len(config['execution']['warmup_actions'])]]
            native.step_round(action)
            reference.step_round(action)
        # 目标轮在采集后不改变后续轨迹。 / Restore after target acquisition.
        native_state, reference_state = native.checkpoint(), reference.checkpoint()
        target, target_reference = native.step_round(chosen), reference.step_round(chosen)
        maximum_gradient_difference = max(
            (a.attack_input.gradients[name] - b.attack_input.gradients[name]).abs().max().item()
            for a, b in zip(target.updates, target_reference.updates)
            for name in a.attack_input.gradients)
        if not all(np.array_equal(a.sample_ids, b.sample_ids)
                   for a, b in zip(target.updates, target_reference.updates)):
            raise RuntimeError('Paired backend sampled different private rows')
        backends.append({'model_round': rounds, 'native': asdict(target.summary),
                         'reference': asdict(target_reference.summary),
                         'maximum_gradient_difference': maximum_gradient_difference})
        native.load_checkpoint(native_state)
        reference.load_checkpoint(reference_state)
        previous_round = rounds
        observed = target.updates[0].attack_input
        store.save_checkpoint(observed, f'common_target_round{rounds}.pt')
        store.write_json(f'private_reference_round{rounds}.json',
                         {'sample_ids': target.updates[0].sample_ids.tolist()})
        attack_config = {**config['attack'], 'seed': seed,
                         'iterations': settings['iterations'], 'restarts': settings['restarts']}
        for name, attack in [
            ('native', GradientAttack(attack_config)),
            ('reference', TabularGradientAttack({**attack_config, **config['comparisons']['attack'],
                'source_root': str(source_root), 'artifact_directory': str(store.path / 'attack_details')})),
        ]:
            torch.cuda.synchronize()
            started = perf_counter()
            reconstructed = attack.reconstruct(observed)
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            reconstructed.to_csv(store.path / f'{name}_round{rounds}.csv', index=False)
            # 攻击执行后才读取真值；原始记录从未传给攻击者。 / Truth is read only after inversion.
            original = data.train_records.iloc[target.updates[0].sample_ids].reset_index(drop=True)
            metrics = reconstruction_metrics(original, reconstructed, observed.feature_metadata,
                                              config['audit']['numeric_tolerance'])
            row = {'model_round': rounds, 'method': name, 'seed': seed,
                   'batch_size': observed.batch_size, 'seconds': elapsed,
                   'diagnostics': attack.last_diagnostics, **metrics}
            for baseline_name, baseline in [
                ('random', sample_random_records(data.train_records, observed.feature_metadata,
                                                  observed.batch_size, seed)),
                ('marginal', sample_marginal_records(data.train_records, observed.batch_size, seed)),
            ]:
                score = reconstruction_metrics(original, baseline, observed.feature_metadata,
                                               config['audit']['numeric_tolerance'])
                row[f'{baseline_name}_feature_recovery'] = score['feature_recovery']
            rows.append(row)
            store.write_json('sample_progress.json', {'backend': backends, 'attacks': rows})
    comparisons = []
    for rounds in settings['model_rounds']:
        n, r = [next(row for row in rows if row['model_round'] == rounds and row['method'] == name)
                for name in ('native', 'reference')]
        effective = r['feature_recovery'] > max(r['random_feature_recovery'], r['marginal_feature_recovery'])
        comparisons.append({'model_round': rounds,
            'feature_recovery_ratio': n['feature_recovery'] / r['feature_recovery'] if r['feature_recovery'] else None,
            'relative_feature_recovery_pass': n['feature_recovery'] >= settings['ratio'] * r['feature_recovery'],
            'reference_beats_guessing': effective,
            'emr_status': 'inconclusive_zero_reference' if r['emr'] == 0 else
                ('pass' if n['emr'] >= settings['ratio'] * r['emr'] else 'fail')})
    return {'scope': 'one-seed two-target feasibility gate; not final performance acceptance',
            'backend': backends, 'attacks': rows, 'comparisons': comparisons,
            'status': 'ready_for_expansion' if all(c['reference_beats_guessing'] and
                      c['relative_feature_recovery_pass'] and c['emr_status'] == 'pass'
                      for c in comparisons) else 'blocked_sample_gate',
            'final_acceptance': False}
