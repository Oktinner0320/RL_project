"""开发用同权重 raw→batch 消融。 / Development-only paired risk ablation.

No collection, attack, FedSGD, cloud submission, or official-test feedback is run here.
Call one fit seed per externally bounded stage; aggregate only after all stages finish.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..algorithms.risk import RiskEstimator
from ..types import RiskPrediction


def development_registration(config: dict[str, Any]
                             ) -> tuple[list[int], list[int], dict[str, float]]:
    """读取唯一 CONFIG 并保护确认种子。 / Read sole CONFIG and protect confirmation seeds."""
    acceptance = config['native_acceptance']
    development = acceptance['development']
    if (acceptance['version'] != 'first-acceptance-v1'
            or development['hypothesis_id'] != 'R1-raw-vs-batch'
            or development['diagnostic_only'] is not True
            or development['allow_new_fedsgd_interactions'] is not False):
        raise ValueError('Only registered diagnostic R1-raw-vs-batch without new FedSGD is allowed')
    seeds = list(development['risk_fit_seeds'])
    panels = list(development['panel_seeds'])
    if (len(seeds) != 3 or len(set(seeds)) != 3 or any(type(seed) is not int for seed in seeds)
            or not panels or len(set(panels)) != len(panels)
            or any(type(seed) is not int for seed in panels)):
        raise ValueError('Register three distinct integer fit seeds and distinct development panel seeds')
    confirmation = acceptance['confirmation']
    reserved = set(confirmation['training_seeds']) | set(confirmation['risk_fit_seeds']) | set(confirmation['panel_seeds'])
    reserved.update(int(seed) for seed in confirmation['evaluation_seeds'].values())
    reserved.update(int(seed) for seed in confirmation['evaluation_seeds'])
    if (set(seeds) | set(panels)) & reserved:
        raise ValueError('Development seeds must not overlap reserved confirmation seeds')
    thresholds = {key: acceptance[key] for key in first_acceptance_thresholds()}
    if thresholds != first_acceptance_thresholds():
        raise ValueError('Calibration thresholds must match first-acceptance-v1')
    return seeds, panels, thresholds


def first_acceptance_thresholds() -> dict[str, float]:
    """纯 API 默认首版标准；运行入口显式传 CONFIG 值。 / Pure API defaults, not runtime config."""
    return {'calibration_underestimation_multiplier': .95, 'calibration_mae_multiplier': 1.10}


def development_risk_config(config: dict[str, Any]) -> dict[str, Any]:
    """从唯一 CONFIG 已登记候选派生，不搜索或改写根配置。 / Derive registered overlay only."""
    development_registration(config)
    acceptance = config['native_acceptance']
    development = acceptance['development']
    candidate = development['risk_candidate']
    if (candidate != 'width64-100-b16-batch-calibrated'
            or candidate not in acceptance['risk_candidates']):
        raise ValueError('Register the existing width64-100-b16-batch-calibrated candidate for R1')
    if development['device'] != 'cpu':
        raise ValueError('Development risk ablation is a local CPU diagnostic')
    effective = deepcopy(config['risk'])
    effective.update(deepcopy(acceptance['risk_candidates'][candidate]))
    effective['device'] = development['device']
    if (effective.get('mean_calibration') is not True
            or type(effective.get('calibration_feature_index')) is not int
            or effective['calibration_feature_index'] != 1):
        raise ValueError('Registered candidate must enable batch mean calibration at feature index 1')
    return effective


def state_hash(value: Any) -> str:
    """稳定哈希实际数组/权重/元数据。 / Hash actual weights, arrays and metadata."""
    digest = sha256()

    def visit(item: Any) -> None:
        """递归保留类型、形状和字节。 / Preserve type, shape and bytes recursively."""
        if hasattr(item, 'detach'):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for element in item:
                visit(element)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        digest.update(b'\0')

    visit(value)
    return digest.hexdigest()


def paired_predictions(checkpoint: dict[str, Any], features: np.ndarray
                       ) -> tuple[RiskPrediction, RiskPrediction, dict[str, str]]:
    """只关闭副本校准；不重拟合且不改变输入。 / Disable calibration on a copy only."""
    original_hash = state_hash(checkpoint)
    config = checkpoint['config']
    if not config.get('mean_calibration') or config.get('calibration_feature_index') != 1:
        raise ValueError('A fitted batch-calibrated native checkpoint (feature index 1) is required')
    calibrated = RiskEstimator(deepcopy(config), checkpoint['version'])
    calibrated.load_checkpoint(deepcopy(checkpoint))
    raw = deepcopy(calibrated)
    # predict reads offsets unconditionally; merely changing the boolean is insufficient.
    raw.config['mean_calibration'] = False
    raw.config['calibration_feature_index'] = None
    raw.calibration_offset = 0.0
    raw.calibration_values = np.empty(0)
    raw.calibration_offsets = np.empty(0)
    network_hash = state_hash(calibrated.model.state_dict())
    fit_keys = ('model', 'mean', 'scale', 'feature_count', 'fit_metadata')
    fit_hash = state_hash({key: checkpoint[key] for key in fit_keys})
    raw_state, cal_state = raw.checkpoint(), calibrated.checkpoint()
    if state_hash({key: raw_state[key] for key in fit_keys}) != fit_hash:
        raise RuntimeError('Raw arm changed the shared fit state')
    raw_prediction, cal_prediction = raw.predict(features), calibrated.predict(features)
    if (state_hash(raw.checkpoint()) != state_hash(raw_state)
            or state_hash(calibrated.checkpoint()) != state_hash(cal_state)
            or state_hash(checkpoint) != original_hash):
        raise RuntimeError('Prediction mutated checkpoint state')
    if not np.array_equal(raw_prediction.uncertainty, cal_prediction.uncertainty):
        raise RuntimeError('Calibration changed member dispersion')
    return raw_prediction, cal_prediction, {
        'raw_network_hash': network_hash, 'calibrated_network_hash': network_hash,
        'raw_fitstate_hash': fit_hash, 'calibrated_fitstate_hash': fit_hash,
        'checkpoint_hash': original_hash,
    }


def _content_equal(left: Any, right: Any) -> bool:
    """Compare actual state contents, without serialization or digests."""
    import torch

    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (left.dtype == right.dtype and left.device == right.device
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return (left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right))
    if isinstance(left, dict):
        return (left.keys() == right.keys()
                and all(_content_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _content_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (str, int, float, bool, type(None), np.generic)):
        return bool(left == right)
    raise TypeError(f'Unsupported checkpoint content type: {type(left).__name__}')


def three_arm_predictions(checkpoint: dict[str, Any], features: np.ndarray
                          ) -> tuple[dict[str, RiskPrediction], dict[str, bool]]:
    """Clone one signed batch fit into raw/signed/positive-part arms; never refit."""
    original = deepcopy(checkpoint)
    config = checkpoint['config']
    if (config.get('mean_calibration') is not True
            or type(config.get('calibration_feature_index')) is not int
            or config['calibration_feature_index'] != 1
            or config.get('calibration_nonnegative', False) is not False):
        raise ValueError('A fitted signed batch checkpoint at feature index 1 is required')
    inputs = np.asarray(features)
    input_state = inputs.copy()
    signed = RiskEstimator(deepcopy(config), checkpoint['version'])
    signed.load_checkpoint(deepcopy(checkpoint))
    raw, positive_part = deepcopy(signed), deepcopy(signed)
    raw.config['mean_calibration'] = False
    raw.config['calibration_feature_index'] = None
    raw.config['calibration_nonnegative'] = False
    raw.calibration_offset = 0.0
    raw.calibration_values = np.empty(0)
    raw.calibration_offsets = np.empty(0)
    positive_part.config['calibration_nonnegative'] = True
    arms = {'raw': raw, 'signed': signed, 'positive_part': positive_part}
    states = {name: arm.checkpoint() for name, arm in arms.items()}
    fit_keys = ('model', 'mean', 'scale', 'feature_count', 'fit_metadata')
    fit_state = {key: original[key] for key in fit_keys}
    if any(not _content_equal({key: state[key] for key in fit_keys}, fit_state)
           for state in states.values()):
        raise RuntimeError('An arm changed the shared fit state')
    model_flags = {name: (tuple(module.training for module in arm.model.modules()),
                          tuple(p.requires_grad for p in arm.model.parameters()))
                   for name, arm in arms.items()}

    def unchanged() -> None:
        if (not _content_equal(checkpoint, original)
                or not _content_equal(inputs, input_state)
                or any(not _content_equal(arm.checkpoint(), states[name])
                       or model_flags[name] != (
                           tuple(module.training for module in arm.model.modules()),
                           tuple(p.requires_grad for p in arm.model.parameters()))
                       for name, arm in arms.items())):
            raise RuntimeError('Prediction mutated checkpoint or input state')

    unchanged()
    predictions = {}
    for name, arm in arms.items():
        arm_input = input_state.copy()
        prediction = arm.predict(arm_input)
        unchanged()
        if not _content_equal(arm_input, input_state):
            raise RuntimeError('Prediction mutated checkpoint or input state')
        for value in (prediction.mean, prediction.uncertainty):
            if (not isinstance(value, np.ndarray) or value.shape != (len(inputs),)
                    or value.dtype.kind not in 'fiu' or not np.isfinite(value).all()
                    or ((value < 0) | (value > 1)).any()):
                raise ValueError('Finite [0,1] predictions and uncertainties must match inputs')
        predictions[name] = deepcopy(prediction)
    if any(not np.array_equal(predictions['raw'].uncertainty, prediction.uncertainty)
           for prediction in predictions.values()):
        raise RuntimeError('Calibration changed member dispersion')
    if (not (predictions['positive_part'].mean >= predictions['raw'].mean).all()
            or not (predictions['positive_part'].mean >= predictions['signed'].mean).all()):
        raise RuntimeError('Positive-part calibration violated the underestimation contract')
    return predictions, {'shared_fit_state': True, 'checkpoint_unchanged': True,
                         'uncertainty_equal': True,
                         'nonnegative_underestimation_contract': True}


def validate_frame(frame: pd.DataFrame) -> np.ndarray:
    """严格验证真值与曝光契约。 / Reject invalid labels, exposure and duplicate uploads."""
    required = {'run_id', 'upload_id', 'features', 'feature_recovery', 'batch_size', 'split'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Nonempty complete audit rows are required')
    if frame[['run_id', 'upload_id']].isna().any().any() or frame.duplicated(['run_id', 'upload_id']).any():
        raise ValueError('Missing group/upload identity or duplicate uploads')
    features = np.asarray(frame['features'].tolist(), dtype=float)
    labels = frame['feature_recovery'].to_numpy(dtype=float)
    exposure = frame['batch_size'].to_numpy(dtype=float)
    if (features.ndim != 2 or features.shape[1] < 2 or not np.isfinite(features).all()
            or not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any()
            or not np.isfinite(exposure).all() or (exposure <= 0).any()
            or not np.array_equal(features[:, 1], exposure)):
        raise ValueError('Finite features, labels in [0,1], and positive matching batch exposure required')
    return features


def check_group_isolation(training: pd.DataFrame, panel: pd.DataFrame,
                          metadata: dict[str, Any] | None = None) -> None:
    """按完整运行隔离，包括原生校准留出。 / Enforce complete-run isolation."""
    validate_frame(training)
    validate_frame(panel)
    training_groups, panel_groups = set(training.run_id), set(panel.run_id)
    if not training.split.eq('train').all() or not panel.split.eq('validation').all():
        raise ValueError('Development fitting uses train audits; panels use validation, never test')
    if training_groups & panel_groups:
        raise ValueError('Fit/calibration groups and panel trajectories must be disjoint')
    if metadata is not None:
        fit, holdout = set(metadata['train_groups']), set(metadata['validation_groups'])
        calibration = set(metadata['mean_calibration']['groups'])
        if (not fit or not holdout or fit & holdout or fit | holdout != training_groups
                or calibration != holdout or (fit | holdout) & panel_groups):
            raise ValueError('Invalid complete-run fit/calibration/panel partition')


def trajectory_metrics(frame: pd.DataFrame, raw: np.ndarray,
                       calibrated: np.ndarray, fit_seed: int) -> list[dict[str, Any]]:
    """轨迹内 B 加权误差、上传等权低估。 / Exposure MAE and upload-equal U per trajectory."""
    validate_frame(frame)
    predictions = [np.asarray(raw, dtype=float), np.asarray(calibrated, dtype=float)]
    for prediction in predictions:
        if (prediction.shape != (len(frame),) or not np.isfinite(prediction).all()
                or ((prediction < 0) | (prediction > 1)).any()):
            raise ValueError('Finite [0,1] predictions must match every audit row')
    rows = []
    for group in sorted(frame.run_id.unique()):
        mask = frame.run_id.to_numpy() == group
        truth = frame.feature_recovery.to_numpy(dtype=float)[mask]
        weights = frame.batch_size.to_numpy(dtype=float)[mask]
        result: dict[str, Any] = {'fit_seed': fit_seed, 'trajectory': group,
                                  'uploads': int(mask.sum()), 'exposure': float(weights.sum())}
        for arm, prediction in zip(('raw', 'calibrated'), predictions):
            residual = truth - prediction[mask]
            result[arm] = {'EWMAE': float(np.average(np.abs(residual), weights=weights)),
                           'U': float(np.maximum(residual, 0).mean()),
                           'MAE': float(np.abs(residual).mean()),
                           'worst_underestimation': float(np.maximum(residual, 0).max()),
                           'underestimation_frequency': float((residual > 0).mean())}
        rows.append(result)
    return rows


def summarize_pairs(rows: list[dict[str, Any]], registered_seeds: list[int],
                    thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """先轨迹后拟合种子等权；开发结果不作确认。 / Equal trajectories then fit seeds."""
    if len(registered_seeds) != 3 or len(set(registered_seeds)) != 3:
        raise ValueError('Exactly three distinct preregistered fit seeds required')
    pairs = {(row['fit_seed'], row['trajectory']) for row in rows}
    if len(pairs) != len(rows) or not rows:
        raise ValueError('Duplicate or missing paired trajectories')
    seeds = {row['fit_seed'] for row in rows}
    panels = {row['trajectory'] for row in rows}
    if not seeds <= set(registered_seeds) or pairs != {(s, p) for s in seeds for p in panels}:
        raise ValueError('Each fit seed must use the identical common panel')
    for row in rows:
        for arm in ('raw', 'calibrated'):
            for metric in ('U', 'EWMAE'):
                value = row[arm][metric]
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Paired metrics must be finite and in [0,1]')
    means = {arm: {metric: float(np.mean([
        np.mean([row[arm][metric] for row in rows if row['fit_seed'] == seed])
        for seed in sorted(seeds)])) for metric in ('U', 'EWMAE')}
        for arm in ('raw', 'calibrated')}
    reasons = []
    if seeds != set(registered_seeds):
        reasons.append('Fewer than three registered paired fit seeds')
    if len(panels) < 3:
        reasons.append('Fewer than three independent panel trajectories; do not duplicate trajectories')
    if means['raw']['U'] == 0 or means['raw']['EWMAE'] == 0:
        reasons.append('Zero raw reference; no preregistered absolute improvement rule')
    thresholds = first_acceptance_thresholds() if thresholds is None else dict(thresholds)
    if thresholds != first_acceptance_thresholds():
        raise ValueError('Calibration thresholds must match first-acceptance-v1')
    gates = {'U': means['calibrated']['U'] <= thresholds['calibration_underestimation_multiplier'] * means['raw']['U'],
             'EWMAE': means['calibrated']['EWMAE'] <= thresholds['calibration_mae_multiplier'] * means['raw']['EWMAE']}
    return {'scope': 'development_only_not_confirmation', 'means': means,
            'paired_rows': rows, 'fit_seeds': sorted(seeds), 'panel_trajectories': sorted(panels),
            'numerical_gates': gates, 'thresholds': thresholds, 'limitations': reasons,
            'development_status': 'insufficient_evidence' if reasons else
                ('pass' if all(gates.values()) else 'fail'), 'final_test_opened': False}


def read_development_sources(paths: list[Path], expected_split: str,
                             collection_configs: dict[str, Path],
                             registered_panel_seeds: list[int] | None = None) -> pd.DataFrame:
    """读既有原生审计或同列 CSV；核对采集种子。 / Read existing audits, not new experiments."""
    if expected_split not in ('train', 'validation') or not paths:
        raise ValueError('Only nonempty development sources are allowed')
    frames = []
    for source in paths:
        source = Path(source).resolve()
        if source.suffix == '.jsonl':
            frame = pd.DataFrame([json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()])
        elif source.suffix == '.csv':
            frame = pd.read_csv(source)
            frame['features'] = frame.features.map(json.loads)
        else:
            raise ValueError('Audit source must be JSONL or CSV')
        validate_frame(frame)
        if not frame.split.eq(expected_split).all():
            raise ValueError('Source split mismatch; official test feedback forbidden')
        allowed_seeds = {11, 12, 13, 14} if expected_split == 'train' else set(
            [21, 22] if registered_panel_seeds is None else registered_panel_seeds)
        if expected_split == 'validation' and not allowed_seeds <= {21, 22}:
            raise ValueError('Only known development panels 21/22 may be read by this runner')
        for run_id in frame.run_id.unique():
            # The packet belongs to a collection run; its config identifies the real seed.
            collection = Path(collection_configs[run_id])
            spec = json.loads(collection.read_text(encoding='utf-8'))
            spec = spec.get('spec', spec)
            if (spec['run_id'] != run_id or spec['seed'] not in allowed_seeds
                    or spec['stage'] != 'collect'
                    or spec['config']['execution']['split'] != expected_split):
                raise ValueError('Only known development collections s11-14 and panels 21/22 allowed')
        frame['source_path'] = str(source)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    validate_frame(combined)
    return combined


def batch_diagnostics(frame: pd.DataFrame, raw: np.ndarray, calibrated: np.ndarray,
                       checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    """按 batch 披露残差、偏移、曝光和边界饱和；不用于选参。 / Batch diagnostics, not tuning."""
    validate_frame(frame)
    # Reuse the prediction contract; diagnostics do not alter gate aggregation.
    trajectory_metrics(frame, raw, calibrated, 0)
    truth = frame.feature_recovery.to_numpy(dtype=float)
    batches = frame.batch_size.to_numpy(dtype=float)
    offsets = dict(zip(checkpoint['calibration_values'], checkpoint['calibration_offsets']))
    diagnostics = []
    for batch in sorted(np.unique(batches)):
        mask = batches == batch
        item: dict[str, Any] = {'batch_size': float(batch), 'uploads': int(mask.sum()),
                               'calibration_offset': float(offsets.get(batch, checkpoint['calibration_offset'])),
                               'calibration_source': 'batch_holdout' if batch in offsets else 'global_holdout_fallback',
                               'exposure_share': float(batches[mask].sum() / batches.sum())}
        for arm, prediction in (('raw', np.asarray(raw)), ('calibrated', np.asarray(calibrated))):
            current = prediction[mask]
            item[arm] = {'mean_residual': float((truth[mask] - current).mean()),
                         'lower_boundary_fraction': float((current == 0).mean()),
                         'upper_boundary_fraction': float((current == 1).mean()),
                         'boundary_saturation_fraction': float(((current == 0) | (current == 1)).mean())}
        diagnostics.append(item)
    return diagnostics


def run_development_seed(config: dict[str, Any], *, fit_seed: int, risk_version: str,
                         training_sources: list[Path], panel_sources: list[Path],
                         collection_configs: dict[str, Path],
                         output: Path, repository: Path) -> dict[str, Any]:
    """一个有界阶段只 fit 一次。 / One fit per externally watchdog-bounded stage.

    Pass the complete Notebook CONFIG unchanged; no parallel configuration is created.
    Output must not exist. Failed stages retain status/artifacts; use a new stage path.
    """
    started = perf_counter()
    seeds, panel_seeds, thresholds = development_registration(config)
    if type(fit_seed) is not int or fit_seed not in seeds:
        raise ValueError('Fit seed must match nested Notebook development registration')
    if not {'data', 'model', 'environment', 'risk', 'execution'}.issubset(config):
        raise ValueError('Pass the complete Notebook CONFIG')
    risk_config = development_risk_config(config)
    training = read_development_sources(training_sources, 'train', collection_configs)
    panel = read_development_sources(panel_sources, 'validation', collection_configs, panel_seeds)
    check_group_isolation(training, panel)
    # Complete provenance before fitting; missing lock/source identity is a preflight failure.
    repository = Path(repository).resolve()
    identity_paths = sorted((repository / 'src/cloud_edge_rl').rglob('*.py')) + [repository / 'uv.lock']
    identities = {str(path.relative_to(repository)): sha256(path.read_bytes()).hexdigest()
                  for path in identity_paths if 'test' not in path.relative_to(repository).parts}
    sources = {str(Path(path).resolve()): sha256(Path(path).read_bytes()).hexdigest()
               for path in training_sources + panel_sources + list(collection_configs.values())}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'status.json').write_text(json.dumps({'state': 'running', 'fit_seed': fit_seed}), encoding='utf-8')
    try:
        risk_config['seed'] = fit_seed
        model = RiskEstimator(risk_config, risk_version)
        fit_started = perf_counter()
        model.fit(validate_frame(training), training.feature_recovery.to_numpy(dtype=float),
                  training.run_id.to_numpy())
        fit_seconds = perf_counter() - fit_started
        checkpoint = model.checkpoint()
        check_group_isolation(training, panel, checkpoint['fit_metadata'])
        raw, calibrated, hashes = paired_predictions(checkpoint, validate_frame(panel))
        rows = trajectory_metrics(panel, raw.mean, calibrated.mean, fit_seed)
        detail = panel.drop(columns=['features']).copy()
        detail['raw_mean'], detail['calibrated_mean'] = raw.mean, calibrated.mean
        detail['member_dispersion'] = raw.uncertainty
        detail.to_csv(output / 'predictions.csv', index=False)
        import torch
        torch.save(checkpoint, output / 'risk.pt')
        result = {**summarize_pairs(rows, seeds, thresholds=thresholds), 'fit_seed': fit_seed, 'hashes': hashes,
                  'batch_diagnostics': batch_diagnostics(panel, raw.mean, calibrated.mean, checkpoint),
                  'fit_metadata': checkpoint['fit_metadata'], 'source_sha256': sources,
                  'source_lock_sha256': identities, 'notebook_config': deepcopy(config),
                  'effective_risk_config': deepcopy(risk_config),
                  'risk_candidate': config['native_acceptance']['development']['risk_candidate'],
                  'config_sha256': state_hash(config), 'fit_seconds': fit_seconds,
                  'elapsed_seconds': perf_counter() - started,
                  'new_fedsgd_interactions': 0, 'new_audit_calls': 0,
                  'training_sources': [str(path) for path in training_sources],
                  'panel_sources': [str(path) for path in panel_sources]}
        (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        (output / 'status.json').write_text(json.dumps({'state': 'completed'}), encoding='utf-8')
        return result
    except Exception as error:
        (output / 'status.json').write_text(json.dumps({'state': 'failed', 'error': str(error),
                                                       'elapsed_seconds': perf_counter() - started}), encoding='utf-8')
        raise
