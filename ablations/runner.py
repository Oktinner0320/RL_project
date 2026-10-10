"""仅提供本地准备与预检，不提交 Modal，也不继承原执行门禁。
Provide local-only preparation and preflight, without Modal submission or inherited execution guards.
"""

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
    """向上查找项目根目录，支持从项目或消融目录启动 Notebook。
    Find the project root upward, supporting notebooks launched from the project or ablations directory.
    """
    start = Path.cwd() if start is None else Path(start)
    for path in (start, *start.parents):
        if (path / 'config/experiment.json').is_file() and (path / 'ablations').is_dir():
            return path.resolve()
    raise ValueError('Cannot locate shared config/experiment.json and ablations directory')


def output_directory(root: Path, name: str = 'N-preparation-v1') -> Path:
    """检查准备目录名称，返回独立输出路径但不创建目录。 / Check the preparation name and return an isolated output path without creating it."""
    if (type(name) is not str or not name or not name.isascii()
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name)):
        raise ValueError('A simple isolated preparation name is required')
    root = Path(root).resolve(strict=True)
    path = root / 'ablations' / 'output' / name
    _check_local(path)
    # 名称检查和路径检查共同阻止输出逃逸或经过本地路径别名。 / Check both name and path to prevent output escapes or local aliases.
    if not path.resolve().is_relative_to(root / 'ablations' / 'output'):
        raise ValueError('Ablation outputs must stay inside ablations/output')
    return path


def research_sources(root: Path) -> dict:
    """记录消融源码及可选 Notebook 单元内容，用于冻结后逐字核对。
    Record ablation source and optional notebook cells for literal checks after freezing.
    """
    base = root / 'ablations'
    paths = sorted(base.glob('*.py')) + sorted((base / 'variants').glob('*.py'))
    result = {p.relative_to(root).as_posix(): p.read_text(encoding='utf-8') for p in paths}
    notebook = base / 'experiment.ipynb'
    if notebook.is_file():
        # 仅冻结单元类型和源码，不把 Notebook 输出或运行元数据计入身份。 / Freeze cell types and source, excluding outputs and execution metadata.
        result['ablations/experiment.ipynb'] = [
            [cell['cell_type'], cell['source']] for cell in read_json(notebook)['cells']
        ]
    return result


def _check_preparation(root: Path, path: Path) -> dict:
    """核对已有准备的配置、源码、计划和身份，返回准备记录。 / Check saved configuration, source, plan, and identity; return the preparation record."""
    for name in ('preparation.json', 'configuration.json', 'base-source.json',
                 'ablation-source.json', 'plan.json', 'ablation-workspace.json'):
        _check_local(path / name)
    record = read_json(path / 'preparation.json')
    config = load_config(root / 'config/experiment.json')
    # 按保存内容核对，不覆盖旧快照；源码注释变化也会使原准备失效。 / Compare saved content without overwriting it; source comment changes also invalidate preparation.
    if scientific_configuration(config) != read_json(path / 'configuration.json'):
        raise ValueError('Shared scientific configuration changed; old preparation is immutable')
    if source_identity(root)['source_contents'] != read_json(path / 'base-source.json'):
        raise ValueError('Imported base source changed; do not resume this preparation')
    if research_sources(root) != read_json(path / 'ablation-source.json'):
        raise ValueError('Ablation source changed; create a new unsubmitted preparation')
    # 用当前登记定义重新生成计划，检测臂设置或预算规则漂移。 / Rebuild the registered plan to detect arm settings or budget-rule changes.
    expected = build_plan(config, record['experiment_id'], record['study_id'])
    if read_json(path / 'plan.json') != expected:
        raise ValueError('Prepared plan differs from the registered study')
    marker = read_json(path / 'ablation-workspace.json')
    if marker != {'schema': 'ablation_workspace_v1', 'experiment_id': record['experiment_id']}:
        raise ValueError('Ablation workspace identity mismatch')
    return record


def prepare(root: Path | None = None, *, study_id: str = 'N',
            name: str = 'N-preparation-v1') -> dict:
    """在独立目录冻结配置与源码，返回准备状态，不修改共享配置。
    Freeze configuration and source in an exclusive directory; return status without changing shared configuration.
    """
    root = project_root(root)
    if study_definition(study_id)['status'] != 'implemented':
        raise ValueError('Requested study is not implemented')
    path = output_directory(root, name)
    if path.exists():
        # 已有准备只核对和读取，不重新生成或覆盖。 / Check and read existing preparation without regenerating or overwriting it.
        if _check_preparation(root, path)['study_id'] != study_id:
            raise ValueError('Existing directory belongs to another study')
        return status(root, name=name)
    config = load_config(root / 'config/experiment.json')
    experiment_id = f'ablation-{study_id.lower()}-{uuid.uuid4().hex}'
    plan = build_plan(config, experiment_id, study_id)
    base_sources = source_identity(root)['source_contents']
    addon_sources = research_sources(root)
    # 不回滚掩盖未完成的准备；部分写入的目录必须检查后处理。 / Do not hide incomplete preparation with rollback; review partial directories.
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'configuration.json', scientific_configuration(config))
    write_json(path / 'base-source.json', base_sources)
    write_json(path / 'ablation-source.json', addon_sources)
    write_json(path / 'plan.json', plan)
    write_json(path / 'ablation-workspace.json',
               {'schema': 'ablation_workspace_v1', 'experiment_id': experiment_id})
    # 准备完成不等于执行获准；历史费用未知时仍记录为 None，而不是零。 / Preparation is not approval; keep unknown historical cost as None, not zero.
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
    """读取准备状态和阻塞项，不重新核验冻结内容，也不授予执行许可。
    Read preparation status and blockers without rechecking frozen content or granting execution approval.
    """
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
    """用带进程树看护的有界子进程运行确定性合成回归测试，返回本地预检记录。
    Run deterministic synthetic regression fixtures in a bounded subprocess with a process-tree watchdog; return its record.

    不训练真实策略，不调用云端，也不构成正式验收。 / No real policy training, cloud calls, or formal acceptance.
    """
    root = project_root(root)
    path = output_directory(root, name)
    _check_preparation(root, path)
    # 先核对冻结内容，再复用已有预检记录，避免重复运行。 / Check frozen content before reusing an existing preflight record.
    if (path / 'preflight.json').exists():
        return read_json(path / 'preflight.json')
    tests = root / 'ablations/test'
    required = ['test_dqn_nstep.py', 'test_execution.py', 'test_preparation.py',
                'test_native_recorder.py']
    if any(not (tests / item).is_file() for item in required):
        raise FileNotFoundError('Local ignored regression tests are required for this preflight')
    log = path / 'controller' / f'preflight-{uuid.uuid4().hex}.log'
    # 从消融目录启动的子进程不继承 Notebook 路径，需显式加入项目根目录。 / A child launched from ablations/ needs an explicit project root on sys.path.
    bootstrap = ('import sys; sys.path.insert(0, sys.argv.pop(1)); import pytest; '
                 'raise SystemExit(pytest.main(sys.argv[1:]))')
    outcome = run_bounded([sys.executable, '-c', bootstrap, str(root), '-o', 'addopts=', '-q',
                           *(str(tests / item) for item in required)], 90, log)
    # 测试结束后再次核对，检测执行期间配置或源码是否改变。 / Recheck after tests to detect configuration or source changes during execution.
    _check_preparation(root, path)
    result = {'status': 'local_synthetic_preflight_passed', 'process': outcome,
              'new_real_interactions': 0, 'new_cloud_calls': 0,
              'live_modal_validated': False, 'execution_ready': False,
              'official_test_access': False, 'formal_acceptance': 'not_evaluated'}
    write_json(path / 'preflight.json', result)
    return result


def main() -> None:
    """解析本地准备命令并输出 JSON，不提供训练或云端提交命令。 / Parse local preparation commands and print JSON; provide no training or cloud submission command."""
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
