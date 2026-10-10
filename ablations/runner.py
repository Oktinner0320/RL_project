"""Local-only preparation and preflight. No Modal submission or inherited execution guards."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from cloud_edge_rl.config import load_config, scientific_configuration, source_identity
from cloud_edge_rl.experiment_transfer import _check_local
from cloud_edge_rl.transport import read_json, run_bounded, write_json

from .studies import build_plan, study_definition


def project_root(start: Path | None = None) -> Path:
    """Support notebooks launched from either the project or the ablations directory."""
    start = Path.cwd() if start is None else Path(start)
    for path in (start, *start.parents):
        if (path / 'config/experiment.json').is_file() and (path / 'ablations').is_dir():
            return path.resolve()
    raise ValueError('Cannot locate shared config/experiment.json and ablations directory')


def output_directory(root: Path, name: str = 'N-preparation-v1') -> Path:
    if (type(name) is not str or not name or not name.isascii()
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name)):
        raise ValueError('A simple isolated preparation name is required')
    root = Path(root).resolve(strict=True)
    path = root / 'ablations' / 'output' / name
    _check_local(path)
    if not path.resolve().is_relative_to(root / 'ablations' / 'output'):
        raise ValueError('Ablation outputs must stay inside ablations/output')
    return path


def research_sources(root: Path) -> dict:
    base = root / 'ablations'
    paths = sorted(base.glob('*.py')) + sorted((base / 'variants').glob('*.py'))
    result = {p.relative_to(root).as_posix(): p.read_text(encoding='utf-8') for p in paths}
    notebook = base / 'experiment.ipynb'
    if notebook.is_file():
        result['ablations/experiment.ipynb'] = [
            [cell['cell_type'], cell['source']] for cell in read_json(notebook)['cells']
        ]
    return result


def _check_preparation(root: Path, path: Path) -> dict:
    for name in ('preparation.json', 'configuration.json', 'base-source.json',
                 'ablation-source.json', 'plan.json', 'ablation-workspace.json'):
        _check_local(path / name)
    record = read_json(path / 'preparation.json')
    config = load_config(root / 'config/experiment.json')
    if scientific_configuration(config) != read_json(path / 'configuration.json'):
        raise ValueError('Shared scientific configuration changed; old preparation is immutable')
    if source_identity(root)['source_contents'] != read_json(path / 'base-source.json'):
        raise ValueError('Imported base source changed; do not resume this preparation')
    if research_sources(root) != read_json(path / 'ablation-source.json'):
        raise ValueError('Ablation source changed; create a new unsubmitted preparation')
    expected = build_plan(config, record['experiment_id'], record['study_id'])
    if read_json(path / 'plan.json') != expected:
        raise ValueError('Prepared plan differs from the registered study')
    marker = read_json(path / 'ablation-workspace.json')
    if marker != {'schema': 'ablation_workspace_v1', 'experiment_id': record['experiment_id']}:
        raise ValueError('Ablation workspace identity mismatch')
    return record


def prepare(root: Path | None = None, *, study_id: str = 'N',
            name: str = 'N-preparation-v1') -> dict:
    """Freeze literal content in an exclusive directory; never mutate shared configuration."""
    root = project_root(root)
    if study_definition(study_id)['status'] != 'implemented':
        raise ValueError('Requested study is not implemented')
    path = output_directory(root, name)
    if path.exists():
        if _check_preparation(root, path)['study_id'] != study_id:
            raise ValueError('Existing directory belongs to another study')
        return status(root, name=name)
    config = load_config(root / 'config/experiment.json')
    experiment_id = f'ablation-{study_id.lower()}-{uuid.uuid4().hex}'
    plan = build_plan(config, experiment_id, study_id)
    base_sources = source_identity(root)['source_contents']
    addon_sources = research_sources(root)
    # No rollback hides incomplete preparations. A partial directory requires review.
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'configuration.json', scientific_configuration(config))
    write_json(path / 'base-source.json', base_sources)
    write_json(path / 'ablation-source.json', addon_sources)
    write_json(path / 'plan.json', plan)
    write_json(path / 'ablation-workspace.json',
               {'schema': 'ablation_workspace_v1', 'experiment_id': experiment_id})
    write_json(path / 'preparation.json', {
        'schema': 'ablation_preparation_v1', 'experiment_id': experiment_id,
        'study_id': study_id, 'status': 'prepared_not_admitted',
        'shared_config_path': str(root / 'config/experiment.json'),
        'new_real_interactions': 0, 'new_cloud_calls': 0, 'new_cloud_cost_usd': 0,
        'actual_historical_cloud_cost_usd': None,
        'official_test_access': False, 'formal_acceptance': 'not_evaluated',
    })
    return status(root, name=name)


def status(root: Path | None = None, *, name: str = 'N-preparation-v1') -> dict:
    root = project_root(root)
    path = output_directory(root, name)
    if not (path / 'preparation.json').exists():
        return {'status': 'not_prepared', 'directory': str(path), 'execution_ready': False}
    record = read_json(path / 'preparation.json')
    plan = read_json(path / 'plan.json')
    return {**record, 'directory': str(path), 'execution_ready': False,
            'blocking_items': plan['pending_evidence'], 'budget': plan['budget'],
            'local_preflight': (read_json(path / 'preflight.json')
                                if (path / 'preflight.json').exists() else None)}


def preflight(root: Path | None = None, *, name: str = 'N-preparation-v1') -> dict:
    """Run only deterministic synthetic regression fixtures under a process-tree watchdog."""
    root = project_root(root)
    path = output_directory(root, name)
    _check_preparation(root, path)
    if (path / 'preflight.json').exists():
        return read_json(path / 'preflight.json')
    tests = root / 'ablations/test'
    required = ['test_dqn_nstep.py', 'test_execution.py', 'test_preparation.py']
    if any(not (tests / item).is_file() for item in required):
        raise FileNotFoundError('Local ignored regression tests are required for this preflight')
    log = path / 'controller' / f'preflight-{uuid.uuid4().hex}.log'
    # A notebook kernel's sys.path is not inherited by a child launched from ablations/.
    bootstrap = ('import sys; sys.path.insert(0, sys.argv.pop(1)); import pytest; '
                 'raise SystemExit(pytest.main(sys.argv[1:]))')
    outcome = run_bounded([sys.executable, '-c', bootstrap, str(root), '-o', 'addopts=', '-q',
                           *(str(tests / item) for item in required)], 90, log)
    _check_preparation(root, path)
    result = {'status': 'local_synthetic_preflight_passed', 'process': outcome,
              'new_real_interactions': 0, 'new_cloud_calls': 0,
              'live_modal_validated': False, 'execution_ready': False,
              'official_test_access': False, 'formal_acceptance': 'not_evaluated'}
    write_json(path / 'preflight.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'status', 'preflight', 'roadmap'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--name', default='N-preparation-v1')
    args = parser.parse_args()
    if args.action == 'roadmap':
        value = {key: study_definition(key) for key in ('N', 'L', 'Q', 'E', 'W')}
    else:
        value = globals()[args.action](args.root, name=args.name)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
