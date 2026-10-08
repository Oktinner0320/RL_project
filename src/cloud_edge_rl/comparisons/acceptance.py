"""整体验收的审计覆盖与配对门槛。 / Coverage and paired gates; never turn missing data into a pass."""

import numpy as np


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
