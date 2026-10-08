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
