"""整体验收的审计覆盖与配对门槛。 / Coverage and paired gates; never turn missing data into a pass."""

from pathlib import PurePosixPath
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math

import numpy as np


def training_allocation(prior: list[dict], method: str, seed: int, partition: str,
                        limit: int, collection_interactions: int,
                        previous_checkpoint: str | None, risk_version: str) -> dict:
    """按真实新增交互计费，恢复计数不能重复收费。 / Charge new interactions, not cumulative resume counters."""
    if (type(limit) is not int or limit <= 0 or type(collection_interactions) is not int
            or not 0 <= collection_interactions < limit):
        raise ValueError('Interaction limit and collection charge must be valid integer counts')
    matching = [row for row in prior if row['spec']['stage'] == 'train_policy'
                and row['spec']['method'] == method and row['spec']['seed'] == seed
                and row['spec']['config']['data']['partition'] == partition]
    known, seen, used = [], set(), 0
    for row in matching:
        # 恢复日志证明该请求尚未调用训练器；失败与计算费用仍保留在账本。
        # Recovered artifacts can prove a request failed before the trainer; never erase its monetary cost.
        proof = row.get('verified_pretraining_failure', {})
        if (row['state'] == 'failed' and
                row.get('error') == {'type': 'KeyError', 'message': "'previous_training_checkpoint'"} and
                'previous_training_checkpoint' not in row['spec']['inputs'] and
                proof.get('call_id') == row.get('call_id') and proof.get('call_id') and
                proof.get('run_id') == row['spec']['run_id'] and
                type(proof.get('transition_rows')) is int and proof['transition_rows'] == 0 and
                proof.get('checkpoint_created') is False and
                isinstance(proof.get('error_sha256'), str) and len(proof['error_sha256']) == 64):
            continue
        # 下载失败仍然可能已消耗完整训练预算；未知进度必须先恢复检查点。
        # A failed download can follow completed compute; unknown progress must be recovered first.
        if row['state'] not in ('completed', 'remote_completed') or 'result' not in row:
            raise ValueError('Unknown consumed interactions: recover interrupted training before scheduling')
        result = row['result']
        learned, warmup = result['learned_steps'], result['warmup_steps']
        offset = row.get('interaction_resume_offset', 0)
        if (any(type(value) is not int or value < 0 for value in (learned, warmup, offset))
                or offset > learned + warmup):
            raise ValueError('Invalid recorded interaction counters')
        if collection_interactions < row['spec']['config']['experiment'].get('charged_collection_interactions', 0):
            raise ValueError('Collection charge cannot decrease within a method/seed/partition')
        identity = row.get('call_id')
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        used += learned + warmup - offset
        known.append(row)
    remaining = limit - collection_interactions - used
    if remaining <= 0:
        raise ValueError('Method/seed/partition interaction allocation is exhausted; do not restart training')
    resume_offset = 0
    if previous_checkpoint:
        path = PurePosixPath(previous_checkpoint)
        if len(path.parts) != 3 or path.parts[0] != 'runs' or path.parts[2] != 'training.pt':
            raise ValueError('Resume needs a recorded runs/<run_id>/training.pt checkpoint')
        parents = [row for row in known if row['spec']['run_id'] == path.parts[1]]
        if len(parents) != 1:
            raise ValueError('Resume checkpoint needs exactly one known matching parent run')
        parent = parents[0]
        if parent['spec']['risk_version'] == risk_version:
            resume_offset = parent['result']['learned_steps'] + parent['result']['warmup_steps']
    return {'consumed_interactions': used, 'remaining_interactions': remaining,
            'resume_offset': resume_offset, 'total_steps': remaining + resume_offset,
            'collection_interactions': collection_interactions}


def complete_trajectory_risk(uploads: list[dict], audits: list[dict]) -> dict:
    """仅完整自有轨迹可给出实际暴露风险。 / Only complete own-trajectory audits yield actual exposure risk."""
    uploads = [row for row in uploads if not row.get('is_warmup', False)]
    expected = {(row['run_id'], row['upload_id']): row for row in uploads}
    if not expected or len(expected) != len(uploads):
        raise ValueError('Trajectory uploads must be nonempty and unique')
    scores = {}
    for row in audits:
        key = (row['run_id'], row['upload_id'])
        if key not in expected:
            if row.get('is_warmup', False):
                continue
            raise ValueError('Audit is not part of this trajectory')
        if key in scores:
            raise ValueError('Duplicate audit: select exactly one frozen attack protocol')
        score = row['feature_recovery']
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Audit recovery must be finite in [0,1]')
        scores[key] = score
    missing = sorted(set(expected) - set(scores))
    if missing:
        return {'status': 'incomplete', 'missing_uploads': missing,
                'coverage': len(scores) / len(expected), 'exposure_risk': None}
    weights = np.asarray([row['batch_size'] for row in expected.values()], dtype=float)
    if (weights <= 0).any():
        raise ValueError('Exposure weights must be positive')
    values = np.asarray([scores[key] for key in expected])
    clients = sorted({row['client_id'] for row in expected.values()})
    client_risks = {str(client): float(np.average(
        values[[row['client_id'] == client for row in expected.values()]],
        weights=weights[[row['client_id'] == client for row in expected.values()]]))
        for client in clients}
    return {'status': 'complete', 'coverage': 1., 'exposures': int(weights.sum()),
            'exposure_risk': float(np.average(values, weights=weights)),
            'equal_upload_risk': float(values.mean()),
            'worst_client_risk': max(client_risks.values()), 'client_risks': client_risks}


def paired_gate(native: dict, reference: dict, seeds: list[int], ratio: float,
                higher_is_better: bool, absolute_tolerance: float | None = None) -> dict:
    """种子是独立单位；零或负参考值禁止比例验收。 / Seeds are independent; nonpositive refs require absolute tolerance."""
    if not seeds or len(set(seeds)) != len(seeds) or not 0 < ratio <= 1:
        raise ValueError('Gate needs unique seeds and ratio in (0,1]')
    if set(native) != set(seeds) or set(reference) != set(seeds):
        return {'status': 'incomplete', 'expected_seeds': seeds}
    n, r = np.asarray([native[s] for s in seeds]), np.asarray([reference[s] for s in seeds])
    if not np.isfinite(n).all() or not np.isfinite(r).all():
        raise ValueError('Acceptance metrics must be finite')
    nm, rm = float(n.mean()), float(r.mean())
    if rm <= 0:
        if absolute_tolerance is None:
            return {'status': 'inconclusive', 'reason': 'nonpositive_reference',
                    'native_mean': nm, 'reference_mean': rm}
        if absolute_tolerance < 0:
            raise ValueError('Absolute tolerance must be nonnegative')
        passed = nm >= rm - absolute_tolerance if higher_is_better else nm <= rm + absolute_tolerance
    else:
        passed = nm >= ratio * rm if higher_is_better else nm <= rm / ratio
    difference = n - r
    return {'status': 'pass' if passed else 'fail', 'n_pairs': len(seeds),
            'native_mean': nm, 'reference_mean': rm,
            'paired_differences': difference.tolist(), 'difference_mean': float(difference.mean()),
            'difference_std': float(difference.std(ddof=1)) if len(seeds) > 1 else None}


def scheduler_gate(native: dict, reference: dict, seeds: list[int], ratio: float,
                   accuracy_slack: float, cost_multiplier: float,
                   absolute_tolerance: float | None = None) -> dict:
    """约束必须同时满足；风险必须来自完整实际审计。 / All constraints hold together, using fully audited actual risk."""
    if not seeds or len(set(seeds)) != len(seeds) or accuracy_slack < 0 or cost_multiplier < 1:
        raise ValueError('Invalid scheduler acceptance protocol')
    if set(native) != set(seeds) or set(reference) != set(seeds):
        return {'status': 'incomplete', 'expected_seeds': seeds}
    if any(row.get('risk_status') != 'complete' for rows in (native, reference) for row in rows.values()):
        return {'status': 'incomplete', 'reason': 'own_trajectory_audits_incomplete'}
    fields = ('accuracy', 'elapsed_seconds', 'energy_joules', 'exposure_risk')
    n = np.asarray([[native[s][key] for key in fields] for s in seeds], dtype=float)
    r = np.asarray([[reference[s][key] for key in fields] for s in seeds], dtype=float)
    if (not np.isfinite(n).all() or not np.isfinite(r).all() or
            (n < 0).any() or (r < 0).any() or
            (n[:, [0, 3]] > 1).any() or (r[:, [0, 3]] > 1).any()):
        raise ValueError('Invalid scheduler metrics')
    nm, rm = n.mean(axis=0), r.mean(axis=0)
    checks = {'accuracy': bool(nm[0] >= rm[0] - accuracy_slack),
              'time': bool(nm[1] <= cost_multiplier * rm[1]),
              'energy': bool(nm[2] <= cost_multiplier * rm[2])}
    privacy = paired_gate({s: native[s]['exposure_risk'] for s in seeds},
                          {s: reference[s]['exposure_risk'] for s in seeds},
                          seeds, ratio, False, absolute_tolerance)
    status = 'fail' if not all(checks.values()) or privacy['status'] == 'fail' else privacy['status']
    return {'status': status, 'constraints': checks, 'privacy': privacy,
            'native_means': dict(zip(fields, nm.tolist())),
            'reference_means': dict(zip(fields, rm.tolist()))}


def overall_gate(results: dict, required: list[str]) -> dict:
    """缺少任何必测格均不能宣布整体验收。 / Missing mandatory cells can never become overall acceptance."""
    if not required or len(set(required)) != len(required):
        raise ValueError('Overall acceptance requires unique mandatory cells')
    missing = [name for name in required if name not in results]
    statuses = {name: results[name]['status'] for name in required if name in results}
    if any(value not in ('pass', 'fail', 'incomplete', 'inconclusive') for value in statuses.values()):
        raise ValueError('Unknown acceptance status')
    status = ('fail' if 'fail' in statuses.values() else
              'incomplete' if missing or 'incomplete' in statuses.values() else
              'inconclusive' if 'inconclusive' in statuses.values() else 'pass')
    return {'status': status, 'missing': missing, 'cells': statuses}


class FirstAcceptanceContractError(ValueError):
    """Malformed or unverifiable first-acceptance evidence (never a negative experiment)."""


_FIRST_THRESHOLDS = {
    'accuracy_slack': .01, 'cost_multiplier': 1.15, 'risk_multiplier': .95,
    'proxy_mae_multiplier': 1.10, 'calibration_underestimation_multiplier': .95,
    'calibration_mae_multiplier': 1.10, 'required_paired_seeds': 3,
    'minimum_feasible_improving_seeds': 2, 'interaction_limit': 3000,
}
_FIRST_HARD_CHECKS = {
    'fedsgd': ('numerical_equivalence', 'checkpoint_restore'),
    'policy': ('termination', 'restore', 'frozen_stage'),
    'attack': ('leakage_detection', 'prior_comparison', 'emr_report', 'coverage_report'),
    'truth_isolation': ('attack_inputs', 'policy_observations', 'evaluation_order'),
    'exposure_alignment': ('gamma', 'telescope'),
    'reproducibility': ('source_identity', 'lock_identity', 'call_status', 'resume_chain'),
    'execution': ('timeout', 'watchdog', 'concurrency', 'preflight', 'cleanup', 'spending'),
}


def _first_require(condition: bool, message: str) -> None:
    if not condition:
        raise FirstAcceptanceContractError(message)


def _first_number(value, name: str, maximum: float | None = None) -> float:
    _first_require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                   f'{name}: finite nonnegative number required')
    _first_require(maximum is None or value <= maximum, f'{name}: out of range')
    return float(value)


def _first_ids(values: list, name: str, count: int | None = None) -> set:
    _first_require(isinstance(values, list) and bool(values), f'{name}: nonempty list required')
    _first_require(all(type(v) in (str, int) and str(v) for v in values),
                   f'{name}: invalid identities')
    canonical = {str(v) for v in values}
    _first_require(len(canonical) == len(values) and (count is None or len(values) == count),
                   f'{name}: duplicate or incorrect identity count')
    return canonical


def _first_artifact(identity: dict):
    """Read a local JSON witness, verifying its byte identity before interpretation."""
    _first_require(isinstance(identity, dict) and set(identity) == {'path', 'sha256'},
                   'Artifact requires path and sha256')
    _first_require(isinstance(identity['path'], str) and bool(identity['path']) and
                   isinstance(identity['sha256'], str) and len(identity['sha256']) == 64,
                   'Invalid artifact identity')
    raw = Path(identity['path']).read_bytes()
    _first_require(hashlib.sha256(raw).hexdigest() == identity['sha256'],
                   f"Artifact digest mismatch: {identity['path']}")
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
        FirstAcceptanceContractError(f'Nonfinite JSON constant: {value}')))


def _first_time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    _first_require(result.tzinfo is not None, 'Timestamps need timezone offsets')
    return result


def _first_uploads(uploads: list[dict], audits: list[dict], run_id: str,
                   attack_id: str) -> dict:
    _first_require(isinstance(uploads, list) and isinstance(audits, list), 'Upload/audit lists required')
    for row in uploads:
        _first_require(row['run_id'] == run_id and type(row['is_warmup']) is bool,
                       'Upload trajectory identity/warmup flag mismatch')
        _first_require(type(row['batch_size']) is int and row['batch_size'] > 0,
                       'Actual batch size must be a positive integer')
        _first_ids([row['upload_id']], 'upload_id')
        _first_ids([row['client_id']], 'client_id')
        _first_require(isinstance(row['record_ids'], list) and
                       len(row['record_ids']) == row['batch_size'] and
                       all(isinstance(v, str) and v for v in row['record_ids']),
                       'Evaluation-only record identities must account for actual exposures')
    _first_require(len({row['upload_id'] for row in uploads}) == len(uploads), 'Duplicate upload')
    for row in audits:
        _first_require(row['run_id'] == run_id and row['attack_id'] == attack_id and
                       row['state'] == 'completed' and isinstance(row['call_id'], str) and row['call_id'],
                       'Audit identity/protocol/completion mismatch')
    _first_require(len({row['call_id'] for row in audits}) == len(audits), 'Duplicate audit call')
    result = complete_trajectory_risk(uploads, audits)
    _first_require(result['status'] == 'complete', 'Incomplete post-warmup own-trajectory audit')
    records = [v for row in uploads if not row['is_warmup'] for v in row['record_ids']]
    result.update(unique_records=len(set(records)), repeated_exposures=len(records) - len(set(records)))
    return result


def _first_panel(panel: list[dict], prediction_key: str) -> dict:
    errors, underestimates = [], []
    for trajectory in panel:
        rows = trajectory['uploads']
        _first_require(isinstance(rows, list) and bool(rows), 'Empty panel trajectory')
        _first_ids([row['upload_id'] for row in rows], 'panel uploads')
        weights, absolute, positive = [], [], []
        for row in rows:
            _first_require(type(row['batch_size']) is int and row['batch_size'] > 0,
                           'Panel actual batch must be a positive integer')
            truth = _first_number(row['feature_recovery'], 'panel truth', 1)
            prediction = _first_number(row[prediction_key], prediction_key, 1)
            weights.append(row['batch_size'])
            absolute.append(abs(prediction - truth))
            positive.append(max(truth - prediction, 0.))
        errors.append(float(np.average(absolute, weights=weights)))
        underestimates.append(float(np.mean(positive)))
    return {'ewmae': float(np.mean(errors)), 'underestimation': float(np.mean(underestimates)),
            'trajectory_ewmae': errors, 'trajectory_underestimation': underestimates}


def _first_budget(config: dict, evidence: dict, seeds: set, method: str) -> dict:
    """No reset: an explicit pre-run ledger must be a subset of the final ledger."""
    prior = _first_artifact(config['budget_reconciliation'])
    final = _first_artifact(evidence['budget'])
    _first_require(isinstance(prior, list) and isinstance(final, list), 'Budget ledgers must be lists')
    _first_require(all(row in final for row in prior), 'Budget history was removed or rewritten')
    calls = _first_ids([row['call_id'] for row in final], 'budget call identities')
    totals, configurations = {}, {}
    for row in final:
        _first_require(row['state'] == 'completed', 'Unreconciled failed/interrupted budget record')
        key = (row['method'], str(row['seed']), row['partition'])
        counts = [row[name] for name in ('learned_steps', 'warmup_steps', 'collection_interactions', 'resume_offset')]
        _first_require(all(type(v) is int and v >= 0 for v in counts) and counts[3] <= sum(counts[:2]),
                       'Invalid budget counters')
        totals[key] = totals.get(key, 0) + sum(counts[:3]) - counts[3]
        configuration_key = (row['method'], row['partition'])
        configurations.setdefault(configuration_key, set()).add(row['configuration_id'])
    _first_require(all((method, seed, 'iid') in totals for seed in seeds), 'Missing training budget history')
    return {'passed': all(v <= 3000 for v in totals.values()) and
            all(len(v) <= 8 for v in configurations.values()),
            'totals': [{'method': k[0], 'seed': k[1], 'partition': k[2], 'interactions': v}
                       for k, v in totals.items()], 'call_ids': sorted(calls),
            'configuration_counts': [{'method': k[0], 'partition': k[1], 'count': len(v)}
                                     for k, v in configurations.items()], 'ledger': final}


def first_acceptance_report(config: dict, evidence: dict) -> dict:
    """Verify first-acceptance-v1, independently of the legacy component-95% gates.

    Minimal JSON schema (no cloud calls; all paths are local):
    * config is CONFIG['native_acceptance'], with the exact v1 thresholds, enabled
      True, require_complete_audit True and confirmation settings. Every artifact
      descriptor is {'path': nonempty str, 'sha256': hex digest of file bytes}.
      confirmation.freeze_manifest identifies a JSON object containing ALL its
      other fields (including frozen True), plus registered_at (aware ISO time),
      attack_id, method_settings_id, reference_settings_id, risk_version,
      fit_group_ids, calibration_group_ids, development_group_ids, hard_rules.
      IDs identify full frozen settings/protocols, not display names. risk_reference
      must be MLP or TabM; scheduler_reference is fixed/time-greedy/risk-greedy/
      MetaCS/PQN. hard_rules maps each _FIRST_HARD_CHECKS category/check to
      {'op': 'eq'|'le'|'ge', 'target': finite nonnegative numeric value}.
    * confirmation.evidence_manifest identifies a JSON object exactly equal to
      evidence. evidence contains phase='confirmation', freeze_sha256,
      started_at, budget descriptor, hard_requirements (category -> descriptor),
      pairs, panel, calibration. Hard-requirement JSON witnesses contain numeric
      'observed' mappings for all category checks and a nonempty 'artifacts' list
      of content-verified JSON descriptors. No self-reported passing booleans.
    * config.budget_reconciliation identifies a pre-run JSON ledger (explicit []
      allowed). evidence.budget identifies its append-only final ledger. Each row
      has call_id, run_id, method, seed, partition, configuration_id, state,
      learned_steps, warmup_steps, collection_interactions, resume_offset.
      Counts are actual charged interactions (including allocated collection);
      resume_offset removes already charged warmup/learned counters, not collection.
      Include ALL historical calls/settings, including reconciled failed calls.
    * pairs: exactly 3 rows {training_seed, evaluation_seed, native, reference}.
      Each side: run_id, group_id, training_call_id, method, settings_id,
      risk_version, initial_model_id, data_split_id, resource_trace_id,
      resource_clock='absolute_simulated_time', metric_split='official_test',
      accuracy, elapsed_seconds, energy_joules, uploads, audits.
      uploads: {run_id, upload_id, client_id, is_warmup: bool, batch_size: int,
      record_ids: list[str]}; record_ids are evaluation-only provenance, NOT
      attack/observation inputs. audits: {run_id, upload_id, call_id, attack_id,
      state:'completed', feature_recovery:[0,1]}. Expected uploads are the actual
      trajectory inventory, not audited-row IDs or a 'complete' flag.
    * panel: {risk_version, reference, attack_id, coverage_artifact: descriptor,
      trajectories:[{group_id, panel_seed, uploads:[{upload_id,batch_size,
      feature_recovery,prediction,reference_prediction}]}]}; >=3 independent groups
      with registered panel seeds, disjoint from fit/calibration/development/own
      evaluation groups. coverage_artifact JSON has nonempty arrays actions,
      clients, batches, stages describing panel coverage.
    * calibration: {baseline:'raw', variant:'batch', pairs:[{fit_seed,
      model_id, raw_model_id, training_signature, raw_training_signature,
      fit_group_ids, calibration_group_ids, trajectories: same panel identities,
      upload IDs, batch sizes and truths, with raw_prediction,batch_prediction}]}.
      Exactly 3 registered fit seeds; model and training signatures match raw
      and batch (all network/labels/groups/scaling/optimizer/epoch settings).

    Zero reference risk/error/raw underestimation is conservatively insufficient:
    this minimal schema intentionally does not implement pre-registered absolute
    alternatives. Malformed/missing evidence also returns insufficient_evidence;
    complete numeric failures return failed. A witness proves identity/structure,
    not the honesty of its producer; independent review remains necessary.
    """
    report = {'version': 'first-acceptance-v1', 'status': 'insufficient_evidence',
              'status_zh': '证据不足', 'reasons': [], 'checks': {}}
    try:
        _first_require(config['version'] == report['version'] and config['enabled'] is True and
                       config['require_complete_audit'] is True, 'First-acceptance configuration not active')
        _first_require(all(type(config[k]) in (int, float) and config[k] == v
                           for k, v in _FIRST_THRESHOLDS.items()), 'First-acceptance thresholds differ')
        confirmation = config['confirmation']
        freeze = _first_artifact(confirmation['freeze_manifest'])
        for key, value in confirmation.items():
            if key not in ('freeze_manifest', 'evidence_manifest'):
                _first_require(freeze[key] == value, f'Frozen confirmation mismatch: {key}')
        _first_require(confirmation['frozen'] is True and confirmation['main_method'] in ('DQN', 'PPO') and
                       confirmation['scheduler_reference'] in ('fixed', 'time-greedy', 'risk-greedy', 'MetaCS', 'PQN') and
                       confirmation['risk_reference'] in ('MLP', 'TabM'), 'Unique method/references not frozen')
        _first_require((confirmation['dataset'], confirmation['partition'], confirmation['resources']) ==
                       ('adult', 'iid', 'normal'), 'Not the registered first-acceptance scenario')
        seeds = _first_ids(confirmation['training_seeds'], 'training seeds', 3)
        fit_seeds = _first_ids(confirmation['risk_fit_seeds'], 'fit seeds', 3)
        panel_seeds = _first_ids(confirmation['panel_seeds'], 'panel seeds')
        _first_require(len(panel_seeds) >= 3, 'At least three panel seeds required')
        evaluation = confirmation['evaluation_seeds']
        _first_require(set(evaluation) == seeds, 'Evaluation mapping must use string training-seed keys')
        evaluation_seeds = _first_ids(list(evaluation.values()), 'evaluation seeds', 3)
        _first_require(len(seeds | fit_seeds | panel_seeds | evaluation_seeds) ==
                       len(seeds) + len(fit_seeds) + len(panel_seeds) + len(evaluation_seeds),
                       'Registered random-seed namespaces overlap')
        _first_require(_first_artifact(confirmation['evidence_manifest']) == evidence,
                       'Evidence does not match its registered artifact')
        _first_require(evidence['phase'] == 'confirmation' and
                       evidence['freeze_sha256'] == confirmation['freeze_manifest']['sha256'] and
                       _first_time(freeze['registered_at']) < _first_time(evidence['started_at']),
                       'Development diagnostics or unfrozen/post-hoc evidence')
        for key in ('attack_id', 'method_settings_id', 'reference_settings_id', 'risk_version'):
            _first_require(isinstance(freeze[key], str) and freeze[key], f'Missing frozen identity: {key}')
        fit_groups = _first_ids(freeze['fit_group_ids'], 'fit groups')
        cal_groups = _first_ids(freeze['calibration_group_ids'], 'calibration groups')
        dev_groups = _first_ids(freeze['development_group_ids'], 'development groups')
        _first_require(not fit_groups & cal_groups, 'Fit/calibration full-run groups overlap')
        budget = _first_budget(config, evidence, seeds, confirmation['main_method'])
        report['checks']['budget'] = {k: v for k, v in budget.items() if k != 'ledger'}
        ledger = {row['call_id']: row for row in budget['ledger']}
        pairs = evidence['pairs']
        _first_require(_first_ids([p['training_seed'] for p in pairs], 'paired training seeds', 3) == seeds,
                       'Missing independent paired training seeds')
        native, reference, own_groups, own_runs, training_calls = [], [], [], [], []
        for pair in pairs:
            seed = str(pair['training_seed'])
            _first_require(pair['evaluation_seed'] == evaluation[seed], 'Evaluation seed mismatch')
            n, r = pair['native'], pair['reference']
            for key in ('initial_model_id', 'data_split_id', 'resource_trace_id'):
                _first_require(isinstance(n[key], str) and n[key] and n[key] == r[key],
                               f'Paired initial/data/resource identity mismatch: {key}')
            for side, output, method, settings in (
                    (n, native, confirmation['main_method'], freeze['method_settings_id']),
                    (r, reference, confirmation['scheduler_reference'], freeze['reference_settings_id'])):
                _first_require(side['method'] == method and side['settings_id'] == settings and
                               side['risk_version'] == freeze['risk_version'] and
                               side['resource_clock'] == 'absolute_simulated_time' and
                               side['metric_split'] == 'official_test', 'Unfrozen/unpaired evaluation settings')
                call = ledger[side['training_call_id']]
                _first_require(call['method'] == method and str(call['seed']) == seed and
                               call['partition'] == 'iid' and call['run_id'] == side['run_id'],
                               'Training ledger identity mismatch')
                own_groups.append(side['group_id'])
                own_runs.append(side['run_id'])
                training_calls.append(side['training_call_id'])
                risk = _first_uploads(side['uploads'], side['audits'], side['run_id'], freeze['attack_id'])
                _first_require(risk['exposure_risk'] > 0 if side is r else True,
                               'Zero reference risk requires a pre-registered absolute rule')
                output.append([_first_number(side['accuracy'], 'accuracy', 1),
                               _first_number(side['elapsed_seconds'], 'simulated time'),
                               _first_number(side['energy_joules'], 'simulated energy'), risk['exposure_risk']])
                side_report = {'training_seed': pair['training_seed'], 'method': method,
                               'run_id': side['run_id'], 'metrics': output[-1], 'audit': risk}
                report.setdefault('trajectories', []).append(side_report)
        groups = _first_ids(own_groups, 'independent confirmation groups', 6)
        _first_ids(own_runs, 'independent confirmation runs', 6)
        _first_ids(training_calls, 'independent training calls', 6)
        _first_require(not groups & (fit_groups | cal_groups | dev_groups), 'Confirmation reused development/fitting groups')
        n, r = np.asarray(native), np.asarray(reference)
        nm, rm = n.mean(axis=0), r.mean(axis=0)
        mean_checks = {'accuracy': bool(nm[0] >= rm[0] - .01),
                       'time': bool(nm[1] <= 1.15 * rm[1]), 'energy': bool(nm[2] <= 1.15 * rm[2]),
                       'risk': bool(nm[3] <= .95 * rm[3])}
        feasible = (n[:, 0] >= r[:, 0] - .01) & (n[:, 1] <= 1.15 * r[:, 1]) & (n[:, 2] <= 1.15 * r[:, 2])
        improving = feasible & (n[:, 3] < r[:, 3])
        report['checks']['scheduler'] = {'passed': all(mean_checks.values()) and int(improving.sum()) >= 2,
            'mean_constraints': mean_checks, 'native_means': nm.tolist(), 'reference_means': rm.tolist(),
            'paired_differences': (n - r).tolist(), 'difference_std': (n - r).std(axis=0, ddof=1).tolist(),
            'feasible_improving_seeds': [pairs[i]['training_seed'] for i in range(3) if improving[i]]}
        panel = evidence['panel']
        _first_require(panel['risk_version'] == freeze['risk_version'] and
                       panel['reference'] == confirmation['risk_reference'] and
                       panel['attack_id'] == freeze['attack_id'], 'Panel uses unfrozen risk/attack reference')
        trajectories = panel['trajectories']
        panel_groups = _first_ids([t['group_id'] for t in trajectories], 'independent panel groups')
        _first_require(len(panel_groups) >= 3 and
                       _first_ids([t['panel_seed'] for t in trajectories], 'panel seeds') == panel_seeds and
                       not panel_groups & (fit_groups | cal_groups | dev_groups | groups),
                       'Panel independence or registration mismatch')
        coverage = _first_artifact(panel['coverage_artifact'])
        _first_require(all(isinstance(coverage[k], list) and coverage[k]
                           for k in ('actions', 'clients', 'batches', 'stages')), 'Missing panel coverage disclosure')
        proxy, proxy_ref = _first_panel(trajectories, 'prediction'), _first_panel(trajectories, 'reference_prediction')
        _first_require(proxy_ref['ewmae'] > 0, 'Zero reference error needs a pre-registered absolute rule')
        report['checks']['proxy'] = {'passed': proxy['ewmae'] <= 1.10 * proxy_ref['ewmae'],
                                    'native': proxy, 'reference': proxy_ref}
        calibration = evidence['calibration']
        _first_require(calibration['baseline'] == 'raw' and calibration['variant'] == 'batch',
                       'Calibration requires same-model raw -> batch, not global -> batch')
        cpairs = calibration['pairs']
        _first_require(_first_ids([p['fit_seed'] for p in cpairs], 'paired fit seeds', 3) == fit_seeds,
                       'Three registered calibration fit seeds required')
        expected_panel = {t['group_id']: {u['upload_id']: (u['batch_size'], u['feature_recovery'])
                                        for u in t['uploads']} for t in trajectories}
        raw_results, batch_results, models = [], [], []
        for pair in cpairs:
            _first_require(isinstance(pair['model_id'], str) and pair['model_id'] and
                           pair['model_id'] == pair['raw_model_id'] and
                           isinstance(pair['training_signature'], str) and pair['training_signature'] and
                           pair['training_signature'] == pair['raw_training_signature'],
                           'Raw/batch model or training recipe differs')
            models.append(pair['model_id'])
            _first_require(_first_ids(pair['fit_group_ids'], 'calibration fit groups') == fit_groups and
                           _first_ids(pair['calibration_group_ids'], 'calibration holdout groups') == cal_groups,
                           'Calibration fitting/holdout group mismatch')
            actual_panel = pair['trajectories']
            _first_require(_first_ids([t['group_id'] for t in actual_panel], 'calibration panel groups') == panel_groups,
                           'Calibration must use the same complete panel')
            for t in actual_panel:
                _first_require(t['panel_seed'] == next(v['panel_seed'] for v in trajectories if v['group_id'] == t['group_id']),
                               'Calibration panel seed mismatch')
                _first_ids([u['upload_id'] for u in t['uploads']], 'calibration panel uploads')
                _first_require({u['upload_id']: (u['batch_size'], u['feature_recovery']) for u in t['uploads']} ==
                               expected_panel[t['group_id']], 'Calibration panel labels/exposures differ')
            raw_results.append(_first_panel(actual_panel, 'raw_prediction'))
            batch_results.append(_first_panel(actual_panel, 'batch_prediction'))
        _first_ids(models, 'independent fitted model identities', 3)
        raw_u = float(np.mean([v['underestimation'] for v in raw_results]))
        raw_mae = float(np.mean([v['ewmae'] for v in raw_results]))
        batch_u = float(np.mean([v['underestimation'] for v in batch_results]))
        batch_mae = float(np.mean([v['ewmae'] for v in batch_results]))
        _first_require(raw_u > 0 and raw_mae > 0, 'Zero raw underestimation/error needs a pre-registered absolute rule')
        report['checks']['calibration'] = {'passed': batch_u <= .95 * raw_u and batch_mae <= 1.10 * raw_mae,
            'raw_underestimation': raw_u, 'batch_underestimation': batch_u,
            'raw_ewmae': raw_mae, 'batch_ewmae': batch_mae, 'raw_seeds': raw_results, 'batch_seeds': batch_results}
        for category, names in _FIRST_HARD_CHECKS.items():
            witness = _first_artifact(evidence['hard_requirements'][category])
            _first_require(isinstance(witness['artifacts'], list) and witness['artifacts'],
                           f'{category}: actual witness artifacts required')
            for artifact in witness['artifacts']:
                _first_artifact(artifact)
            checks = {}
            for name in names:
                rule = freeze['hard_rules'][category][name]
                actual = _first_number(witness['observed'][name], f'{category}.{name}')
                target = _first_number(rule['target'], f'{category}.{name}.target')
                _first_require(rule['op'] in ('eq', 'le', 'ge'), 'Unknown pre-registered hard rule')
                checks[name] = {'eq': actual == target, 'le': actual <= target, 'ge': actual >= target}[rule['op']]
            if category == 'attack':
                _first_require(checks['leakage_detection'] and checks['prior_comparison'],
                               'Attack ineffectiveness makes privacy benefits unidentifiable')
            report['checks'][category] = {'passed': all(checks.values()), 'constraints': checks}
        report['status'] = 'passed' if all(c['passed'] for c in report['checks'].values()) else 'failed'
        report['status_zh'] = '通过' if report['status'] == 'passed' else '未通过'
        report['reasons'] = [key for key, value in report['checks'].items() if not value['passed']]
    except (FirstAcceptanceContractError, KeyError, TypeError, ValueError, OSError, OverflowError) as error:
        report['reasons'].append(str(error))
    return report
