"""Read-only native result exporter. Never imports or executes the scientific pipeline."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# 只导出白名单字段，屏蔽私有数据。 / Export allowlisted fields only; exclude private data.
MAX_BYTES = 16 * 1024 * 1024
MAX_FILES = 4096
MAX_ITEMS = 100000
METRICS = set('episode round validation_accuracy validation_loss elapsed_seconds energy_joules upload_bytes download_bytes processed_samples unique_samples risk_available reward_protocol is_warmup risk_exposures predicted_exposure_risk action_id loss_gain round_time_seconds round_energy_joules predicted_risk worst_predicted_risk round_upload_bytes round_download_bytes total_reward reward terminated truncated'.split())
SUMMARY = METRICS | set('run_id stage risk_version risk_measurement trained new_environment_steps environment_steps learned_steps warmup_steps updates'.split())
TRAINING = set('environment_steps learned_steps warmup_steps updates td_loss q_mean risk_version loss entropy value_loss policy_loss'.split())
UPLOAD = set('upload_id run_id episode round round_index action_id client_id batch_size predicted_risk uncertainty decision_margin risk_version split is_warmup reward_protocol decision_kind'.split())
DECISION = set('episode round kind values margin'.split())


def primitive(value):
    """Reject nonfinite numbers and suppress path-bearing/unbounded string values."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('Nonfinite result value')
        return value
    if isinstance(value, str):
        if len(value) > 256 or '/' in value or '\\' in value or ':' in value:
            return None
        return value
    return None


def selected(row, keys):
    if not isinstance(row, dict):
        raise ValueError('Expected object row')
    return {k: primitive(v) for k, v in row.items()
            if k in keys or (keys is METRICS and re.fullmatch(r'c\d+_(participations|samples|unique)', k))}


class Reader:
    # 限制读取路径、文件大小和记录数。 / Bound input paths, file sizes, and row counts.
    def __init__(self, root):
        self.root = root
        self.files = 0
        self.bytes = 0
        self.items = 0
        self.training_links = {}

    def confined(self, path):
        path = Path(path).absolute()
        if not path.is_relative_to(self.root):
            raise ValueError('Path outside project')
        for part in [path, *path.parents]:
            if part == self.root.parent:
                break
            if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
                raise ValueError('Symlink/junction paths forbidden')
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('Resolved path outside project')
        return path

    def text(self, path):
        path = self.confined(path)
        if not path.exists():
            return None
        size = path.stat().st_size
        self.files += 1
        self.bytes += size
        if size > MAX_BYTES or self.bytes > 128 * MAX_BYTES or self.files > MAX_FILES:
            raise ValueError('Input byte/file limit exceeded')
        with path.open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('Input byte limit exceeded')
        return raw.decode('utf-8-sig')

    def json(self, path, default=None):
        text = self.text(path)
        return default if text is None else json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))

    def rows(self, path, keys, mode='jsonl'):
        text = self.text(path)
        if text is None:
            return []
        if mode == 'csv':
            raw = csv.DictReader(io.StringIO(text))
        elif mode == 'json':
            raw = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
            if not isinstance(raw, list):
                raise ValueError('Expected row array')
        else:
            raw = (json.loads(line) for line in text.splitlines() if line.strip())
        rows = []
        for i, row in enumerate(raw):
            self.items += 1
            if self.items > MAX_ITEMS:
                raise ValueError('Input item limit exceeded')
            if mode == 'csv':
                row = {k: scalar(v) for k, v in row.items()}
            out = selected(row, keys)
            if keys is DECISION and isinstance(row.get('values'), list):
                if len(row['values']) > 1024:
                    raise ValueError('Decision values limit exceeded')
                out['values'] = [primitive(v) for v in row['values']]
            out['source_index'] = i
            rows.append(out)
        return rows

    def source(self, path):
        return {'path': self.confined(path).relative_to(self.root).as_posix()}


def scalar(value):
    if value is None or value == '':
        return None
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    try:
        number = float(value)
    except ValueError:
        return primitive(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite CSV value')
    return int(number) if number.is_integer() else number


def directories(reader, parent):
    parent = reader.confined(parent)
    if not parent.exists():
        return []
    children = []
    for i, child in enumerate(parent.iterdir()):
        if i >= 256:
            raise ValueError('Directory item limit exceeded')
        reader.confined(child)
        if child.is_dir() and not child.name.startswith('.'):
            children.append(child)
    return sorted(children)


def discover(reader, source):
    """Only fixed native locations; no recursive scan, archives, or checkpoints."""
    # 只识别固定产物目录，不扫描权重或归档。 / Read known layouts, not checkpoints or archives.
    if (source / 'phases').exists():
        return [(phase, run) for phase in ('pilot', 'continue')
                for run in directories(reader, source / 'phases' / phase / 'artifacts' / 'runs')]
    if (source / 'summary.json').exists() and (source / 'config.json').exists():
        return [('native', source)]
    parents = [source / 'runs', source / 'artifacts' / 'runs']
    for category in ('artifacts', 'partial-artifacts'):
        for container in directories(reader, source / 'experiment' / category):
            parents.append(container / 'runs')
    return [('native', run) for parent in parents for run in directories(reader, parent)]


def config_summary(config):
    data = config.get('data', {})
    env = config.get('environment', {})
    policy = config.get('rl', config.get('policy', {}))
    dataset = data.get('dataset', data.get('name'))
    if dataset is None and data.get('url') == 'https://archive.ics.uci.edu/static/public/2/adult.zip':
        dataset = 'adult'
    result = {'dataset': primitive(dataset),
              'partition': primitive(data.get('partition')), 'num_clients': primitive(data.get('num_clients')),
              'gamma': primitive(policy.get('gamma')), 'n_step': primitive(policy.get('n_step')),
              'warmup_rounds': primitive(env.get('warmup_rounds', config.get('reward', {}).get('warmup_rounds'))), 'max_rounds': primitive(env.get('max_rounds')),
              'action_space': []}
    # 按生产端的集合×批量顺序还原动作。 / Match the producer's client-set × batch-size order.
    # 生产端 / Producer: src/cloud_edge_rl/environment.py:62-67, enumerate(product(sets, sizes)).
    sets, sizes = env.get('client_sets'), env.get('batch_sizes')
    if isinstance(sets, list) and isinstance(sizes, list) and len(sets) * len(sizes) <= 1024:
        if all(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids) for ids in sets) and all(type(b) is int and b > 0 for b in sizes):
            result['action_space'] = [{'action_id': i, 'client_ids': ids, 'batch_size': b}
                                      for i, (ids, b) in enumerate((ids, b) for ids in sets for b in sizes)]
    return result


def native_run(reader, folder, phase, experiment_id):
    summary = reader.json(folder / 'summary.json', {})
    wrapper = reader.json(folder / 'config.json', {})
    if not summary or not wrapper:
        return None
    config = wrapper.get('config', {})
    stage = summary.get('stage', wrapper.get('stage'))
    if stage not in ('train_policy', 'validate', 'evaluate', 'collect'):
        return None
    run_id = primitive(summary.get('run_id'))
    if not run_id:
        raise ValueError('Missing safe run_id')
    # 检查点路径只用于关联，不读取权重。 / Use checkpoint paths for linkage, never read weights.
    checkpoint = wrapper.get('inputs', {}).get('agent_checkpoint')
    if isinstance(checkpoint, str) and re.fullmatch(r'runs/[A-Za-z0-9_-]+/training\.pt', checkpoint):
        reader.training_links[run_id] = checkpoint.split('/')[1]
    verify = reader.json(folder / 'verification.json', {})
    status = reader.json(folder / 'status.json', {})
    cs = config_summary(config)
    warnings = ['Predicted risk is not audited complete-policy risk.']
    if not cs['action_space']:
        warnings.append('Action mapping unavailable.')
    seed = primitive(wrapper.get('seed'))
    partition = primitive(wrapper.get('partition', config.get('execution', {}).get('split', config.get('data', {}).get('split'))))
    if stage == 'train_policy':
        partition = 'train'
    elif stage == 'validate':
        partition = 'validation'
    elif stage == 'evaluate':
        partition = partition or 'unknown'
    elif stage == 'collect':
        partition = partition or 'unknown'
    return {'run_id': run_id, 'experiment_id': experiment_id, 'method': primitive(wrapper.get('method')),
            'arm': primitive(config.get('experiment', {}).get('variant')), 'phase': phase, 'stage': stage,
            'train_seed': seed if stage == 'train_policy' else None,
            'eval_seed': seed if stage != 'train_policy' else None,
            'partition': partition,
            'risk_measurement': primitive(summary.get('risk_measurement', 'unavailable')),
            'risk_version': primitive(summary.get('risk_version', wrapper.get('risk_version'))),
            'evidence_scope': 'development; single native trajectory',
            'status': {'compute': primitive(status.get('state', 'unknown')), 'transport': 'unknown',
                       'verification': 'verified' if verify.get('verified') is True else 'unknown',
                       'acceptance': 'not_formal' if verify.get('formal_acceptance') is False else 'unknown'},
            'budget': {'new_interactions': primitive(summary.get('new_environment_steps', verify.get('actual_interactions'))),
                       'cumulative_interactions': primitive(summary.get('environment_steps', verify.get('actual_interactions'))),
                       'warmup_interactions': primitive(summary.get('warmup_steps'))},
            'summary': selected(summary, SUMMARY), 'config_summary': cs,
            'metrics': reader.rows(folder / 'metrics.csv', METRICS, 'csv'),
            'training': reader.rows(folder / 'policy_training.json', TRAINING, 'json'),
            'decisions': reader.rows(folder / 'policy_decisions.jsonl', DECISION),
            'uploads': reader.rows(folder / 'uploads.jsonl', UPLOAD),
            'sources': [reader.source(folder / name) for name in ('summary.json', 'config.json', 'metrics.csv', 'policy_training.json', 'policy_decisions.jsonl', 'uploads.jsonl', 'status.json', 'verification.json') if (folder / name).exists()],
            'warnings': warnings}


def atomic_json(path, payload):
    # 完整写入后原子替换，避免暴露半成品。 / Publish atomically after writing the complete file.
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + '\n'
    if len(raw.encode('utf-8')) > 64 * MAX_BYTES:
        raise ValueError('Output byte limit exceeded')
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def resolve_native_training_metadata(reader, runs):
    """Resolve exact native producers before attaching presentation budget fields."""
    # 按生产端 ID 关联累计预算，不从名称猜测。 / Link producer IDs and counters; never guess from names.
    linked_budgets = {}
    for run in runs:
        if run['stage'] != 'train_policy':
            matches = [r for r in runs if r['stage'] == 'train_policy' and r['run_id'] == reader.training_links.get(run['run_id']) and r['arm'] == run['arm'] and r['risk_version'] == run['risk_version'] and r['config_summary'] == run['config_summary']]
            seeds = {r['train_seed'] for r in matches if r['train_seed'] is not None}
            if len(seeds) == 1:
                run['train_seed'] = seeds.pop()
            counters = {r['budget']['cumulative_interactions'] for r in matches
                        if type(r['budget']['cumulative_interactions']) is int and r['budget']['cumulative_interactions'] >= 0}
            if len(counters) > 1:
                raise ValueError('Contradictory native producer cumulative counters: ' + run['run_id'])
            if len(matches) == 1:
                linked_budgets[run['run_id']] = matches[0]['budget']['cumulative_interactions']
    # 匹配完成后再添加派生字段。 / Attach derived fields only after producer matching.
    for run in runs:
        counter = (run['budget']['cumulative_interactions'] if run['stage'] == 'train_policy'
                   else linked_budgets.get(run['run_id']))
        run['config_summary']['training_interaction_budget'] = counter if type(counter) is int and counter >= 0 else None


@contextmanager
def output_lock(reader):
    # 防止多个导出同时更新快照索引。 / Prevent concurrent exports from updating the index.
    directory = reader.confined(reader.root / 'frontend/public/data')
    directory.mkdir(parents=True, exist_ok=True)
    lock = reader.confined(directory / '.export.lock')
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError('Another export is publishing; inspect stale lock before retrying') from None
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def export_snapshot(root: Path, *, sources: list[Path] | None = None, snapshot_id: str | None = None) -> dict:
    # 只读取已有结果；不会触发实验或验收。 / Read existing results without running experiments or gates.
    root = Path(root).absolute()
    reader = Reader(root)
    reader.confined(root)
    if sources is None:
        sources = [Path('ablations/output/N-cloud-v1')]
        seven_report = Path('output/fresh-seven-method-rerun-20261010-v1/report-analysis-20261010-v1')
        if (root / seven_report / 'analysis.json').exists():
            sources.append(seven_report)
        for name in ('risk-R1-raw-vs-batch-cpu-v1', 'risk-R2-positive-part-cpu-v1'):
            candidate = Path('output/first-acceptance-v1-20261009') / name
            if (root / candidate).exists():
                sources.append(candidate)
    generated_at = datetime.now(timezone.utc).isoformat()
    snapshot_id = snapshot_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:12]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', snapshot_id):
        raise ValueError('Invalid snapshot_id')
    payload = {'schema_version': 'results_visualization_v1', 'snapshot_id': snapshot_id, 'generated_at': generated_at,
               'experiments': [], 'runs': [], 'audits': [], 'calibration': [], 'acceptance_reports': [],
               'warnings': [], 'metric_definitions': {
                   'validation_accuracy': {'units': 'fraction', 'scope': 'native validation split'},
                   'elapsed_seconds': {'units': 'simulated s'}, 'energy_joules': {'units': 'simulated J'},
                   'predicted_exposure_risk': {'scope': 'predicted exposure-weighted risk; not audited truth'},
                   'training': {'scope': 'native cumulative diagnostics; resumed logs retain predecessor rows'}}}
    seen = {}
    for source in sources:
        source = reader.confined(root / source)
        if not (source.is_relative_to(root / 'ablations' / 'output') or source.is_relative_to(root / 'output') or source.is_relative_to(root / 'runs')):
            raise ValueError('Unsupported input root')
        try:
            from adapters.seven_methods import read_seven_methods
        except ModuleNotFoundError:
            from figure_assets.scripts.adapters.seven_methods import read_seven_methods
        # 只识别指定公开报告，不接纳任意分析文件。 / Accept this public layout, not arbitrary reports.
        if source.name in ('fresh-seven-method-rerun-20261010-v1', 'report-analysis-20261010-v1'):
            seven = read_seven_methods(reader, source, primitive, METRICS)
            if seven is not None:
                experiment, runs, reports, warnings = seven
                for run in runs:
                    if run['run_id'] in seen:
                        raise ValueError('Repeated seven-method source: ' + run['run_id'])
                    seen[run['run_id']] = run
                payload['experiments'].append(experiment)
                payload['runs'].extend(runs)
                payload['acceptance_reports'].extend(reports)
                payload['warnings'].extend(warnings)
                continue
        plan = reader.json(source / 'plan.json', {})
        experiment_id = primitive(plan.get('experiment_id')) or source.relative_to(root).as_posix().replace('/', '-')
        experiment_runs = []
        try:
            from adapters.calibration import read_calibration
        except ModuleNotFoundError:
            from figure_assets.scripts.adapters.calibration import read_calibration
        calibration, reports = read_calibration(reader, source, primitive)
        payload['calibration'].extend(calibration)
        payload['acceptance_reports'].extend(reports)
        receipts = {}
        for phase in ('pilot', 'continue'):
            phase_dir = source / 'phases' / phase
            receipts[phase] = reader.json(phase_dir / 'transfer.json', {})
        for phase, folder in discover(reader, source):
            run = native_run(reader, folder, phase, experiment_id)
            if run is None:
                payload['warnings'].append('Unsupported native run omitted: ' + folder.relative_to(root).as_posix())
                continue
            receipt = receipts.get(phase, {})
            if receipt.get('transport_complete') is True:
                run['status']['transport'] = 'transport_complete'
                run['sources'].append(reader.source(source / 'phases' / phase / 'transfer.json'))
            if run['run_id'] in seen:
                old = seen[run['run_id']]
                fields = ('summary', 'config_summary', 'metrics', 'training', 'decisions', 'uploads', 'stage', 'train_seed', 'eval_seed')
                if any(old[k] != run[k] for k in fields):
                    raise ValueError('Conflicting safe result copies: ' + run['run_id'])
                old['sources'].extend(s for s in run['sources'] if s not in old['sources'])
                continue
            seen[run['run_id']] = run
            experiment_runs.append(run)
            payload['runs'].append(run)
        resolve_native_training_metadata(reader, experiment_runs)
        gaps = ['Actual cloud cost unavailable.', 'No complete-policy audited risk exported.', 'Historical calibration/audit layouts not recognized are omitted.', 'No new official acceptance gates.']
        if experiment_runs:
            gaps.append('One training seed is not statistical replication.')
        if calibration:
            gaps.append('Model identity unavailable in selected calibration reports; groups are not pooled.')
        if not experiment_runs and not calibration:
            gaps.append('No recognized native run artifacts; source omitted from quantitative comparisons.')
        trains = [r for r in experiment_runs if r['stage'] == 'train_policy']
        vals = [r for r in experiment_runs if r['stage'] == 'validate']
        def total(rows):
            counts = [r['budget']['new_interactions'] for r in rows]
            return sum(counts) if counts and all(type(n) in (int, float) for n in counts) else None
        payload['experiments'].append({'experiment_id': experiment_id, 'name': source.name, 'kind': 'ablation' if plan else 'native_experiment',
            'evidence_scope': 'development, exploratory', 'sources': [reader.source(source)],
            'status': {'compute': 'see_original_run_receipts', 'transport': 'unknown', 'verification': 'see_original_run_receipts', 'acceptance': 'not_inferred'},
            'actual_cloud_cost_usd': None, 'counts': {'physical_train_interactions': total(trains), 'physical_validation_interactions': total(vals),
                'training_seed_count': len({r['train_seed'] for r in trains if r['train_seed'] is not None})}, 'evidence_gaps': gaps})
    with output_lock(reader):
        return _publish_snapshot(reader, root, snapshot_id, payload)


def _publish_snapshot(reader, root, snapshot_id, payload):
    data_dir = reader.confined(root / 'frontend' / 'public' / 'data')
    data_path = reader.confined(data_dir / snapshot_id / 'results.json')
    index_path = reader.confined(data_dir / 'index.json')
    report_path = reader.confined(root / 'figure_assets' / 'reports' / (snapshot_id + '-export.json'))
    if data_path.exists():
        existing = reader.json(data_path)
        payload['generated_at'] = existing.get('generated_at')
        if existing != payload:
            raise ValueError('Inconsistent snapshot reuse')
    else:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(data_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, {'snapshot_id': snapshot_id, 'run_count': len(payload['runs']), 'input_files': reader.files,
                              'input_bytes': reader.bytes, 'warnings': payload['warnings'], 'discrepancies': [],
                              'evidence_gaps': [g for e in payload['experiments'] for g in e['evidence_gaps']]})
    index = reader.json(index_path, {'schema_version': 'results_index_v1', 'snapshots': []})
    if index.get('schema_version') != 'results_index_v1' or not isinstance(index.get('snapshots'), list) or len(index['snapshots']) > 1000:
        raise ValueError('Invalid snapshot index')
    for entry in index['snapshots']:
        identity = entry.get('snapshot_id', '')
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', identity) or entry.get('path') != identity + '/results.json':
            raise ValueError('Unsafe snapshot index entry')
    # 索引只保留公开字段。 / Keep only declared public index fields.
    index['snapshots'] = [{k: e[k] for k in ('snapshot_id', 'generated_at', 'path')} for e in index['snapshots']]
    entries = [e for e in index['snapshots'] if e['snapshot_id'] != snapshot_id]
    entries.append({'snapshot_id': snapshot_id, 'generated_at': payload['generated_at'], 'path': snapshot_id + '/results.json'})
    # 快照完成后才更新入口索引。 / Update the entry index only after the snapshot is complete.
    atomic_json(index_path, {'schema_version': 'results_index_v1', 'current_snapshot': snapshot_id, 'snapshots': entries})
    return {'snapshot_id': snapshot_id, 'data_path': str(data_path), 'index_path': str(index_path), 'report_path': str(report_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, action='append')
    parser.add_argument('--snapshot-id')
    args = parser.parse_args()
    print(json.dumps(export_snapshot(args.root, sources=args.source, snapshot_id=args.snapshot_id)))
