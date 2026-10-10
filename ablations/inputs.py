"""Stage the existing developer risk model by content, without refitting or cloud access."""

from pathlib import Path

from cloud_edge_rl.environment import FEATURE_NAMES
from cloud_edge_rl.transport import publish_file, read_json, write_json

from .runner import _check_preparation, output_directory, project_root

RISK_SOURCE = ('output/fresh-seven-method-rerun-20261010-v1/experiment/partial-artifacts/'
               '121196f0242940a69c558b34a98ba2e3/runs/'
               'fit_risk-fc7e3f5c799f49c2bc74487687245745')


def stage_risk(root: Path | None = None, *, name: str = 'N-preparation-v1') -> dict:
    """Copy a trusted local checkpoint; readiness still requires ancestry/accounting approval."""
    import torch

    from cloud_edge_rl.algorithms.risk import RiskEstimator
    from cloud_edge_rl.experiment_transfer import _check_local

    root = project_root(root)
    destination = output_directory(root, name)
    _check_preparation(root, destination)
    producer = _check_local(root / RISK_SOURCE)
    checkpoint_path = _check_local(producer / 'risk.pt')
    if not checkpoint_path.is_file():
        return {'status': 'missing_local_risk_candidate', 'source': str(checkpoint_path),
                'execution_ready': False}
    if checkpoint_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Risk candidate exceeds the bounded local input limit')
    original = read_json(_check_local(producer / 'config.json'))
    summary = read_json(_check_local(producer / 'summary.json'))
    if read_json(_check_local(producer / 'status.json')).get('state') != 'completed':
        raise ValueError('Risk producer did not complete')
    plan = read_json(destination / 'plan.json')
    common = plan['arm_configurations']['N1']
    if (original['run_id'] != producer.name or original['stage'] != 'fit_risk'
            or original['config']['execution']['split'] != 'train'
            or original['config']['data'] != common['data']
            or original['config']['model'] != common['model']):
        raise ValueError('Developer risk source/data/model mismatch')
    blob = checkpoint_path.read_bytes()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    version = summary['risk_version']
    if (state['risk_version'] != version or tuple(state['feature_names']) != FEATURE_NAMES
            or state['implementation'] != 'researcher'):
        raise ValueError('Risk version or feature contract mismatch')
    # Native loader validates exact scientific risk settings, scaling and calibration.
    estimator = RiskEstimator({**common['risk'], 'device': 'cpu'}, version)
    estimator.load_checkpoint(state['state'])
    fit = estimator.fit_metadata
    if (not fit['train_groups'] or not fit['validation_groups']
            or set(fit['train_groups']) & set(fit['validation_groups'])
            or fit != summary['fit_metadata']):
        raise ValueError('Risk fitting groups or producer metadata mismatch')
    inputs = _check_local(destination / 'inputs')
    inputs.mkdir(exist_ok=True)
    target = _check_local(inputs / 'risk.pt')
    if target.exists():
        if target.read_bytes() != blob:
            raise ValueError('Staged risk input conflicts with the frozen local source')
    else:
        temporary = inputs / '.risk.pt.copy.part'
        with temporary.open('xb') as handle:
            handle.write(blob)
        publish_file(temporary, target)
    if target.read_bytes() != checkpoint_path.read_bytes():
        raise ValueError('Risk content changed during staging')
    record = {'status': 'local_candidate_staged_not_admitted', 'source': str(checkpoint_path),
              'risk_version': version, 'producer_run_id': original['run_id'],
              'producer_inputs': original['inputs'], 'bytes': len(blob),
              'fit_train_groups': fit['train_groups'], 'fit_validation_groups': fit['validation_groups'],
              'source_training_uploads': summary['training_uploads'],
              'source_training_runs': summary['training_runs'],
              'validation': 'native_loader_and_full_byte_content_equality_no_hash',
              'allocation_verified': False, 'attack_strength_verified': False,
              'new_risk_fits': 0, 'new_real_interactions': 0, 'new_cloud_calls': 0,
              'execution_ready': False, 'official_test_access': False}
    receipt = _check_local(inputs / 'risk-source.json')
    if receipt.exists():
        if read_json(receipt) != record:
            raise ValueError('Risk input receipt changed')
    else:
        write_json(receipt, record)
    return record
