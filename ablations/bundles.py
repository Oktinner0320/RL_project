"""在本地构建原始归档，用大块文件持久化和回传消融产物。
Local tar construction and bulk-only durable ablation transport (no cloud SDK).

原生小文件留在临时盘；先提交不可变计算归档，再发布可恢复状态。
Keep native small files on local disk; commit immutable compute bundles before
publishing resumable state. Verification metadata is stored separately from raw tars.
核验元数据独立于原始归档，避免后续核验改写计算证据。
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from cloud_edge_rl.experiment_export import extract_shard, publish_shards
from cloud_edge_rl.experiment_transfer import (
    GraphTransferLimits,
    _check_local,
    _confined,
    _inventory,
    _path,
)
from cloud_edge_rl.transport import publish_file, read_json, safe_run_id, write_json

SECONDS = 900


def _remaining(deadline):
    """检查剩余期限；超时保留已完成产物。 / Check the deadline; retain completed artifacts on timeout."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Ablation bundle deadline; completed artifacts retained')
    return remaining


def _root(path):
    # 只解析明确可信的工作区根别名，后代路径仍禁止符号链接。
    # Resolve only the trusted root alias; descendant paths still reject symlinks.
    return _check_local(Path(path).resolve(strict=True))


def _json(path, value, *, mutable=False):
    """默认写不可变文档；状态覆盖必须显式选择。 / Default to immutable documents; opt in to state updates."""
    path = _check_local(path)
    if path.exists() and not mutable:
        if read_json(path) != value:
            raise ValueError('Immutable ablation document changed')
        return
    write_json(path, value, replace=mutable and path.exists())


def _copy(source, target, deadline):
    """已存在文件逐块核对完整字节，否则有界复制后原子发布。
    Compare existing immutable bytes; otherwise atomically copy bounded chunks.

    不计算哈希；中途失败不将部分文件当作完成产物。
    No hashes are computed; partial copies never count as completed artifacts.
    """
    source, target = _check_local(source), _check_local(target)
    if not source.is_file():
        raise ValueError('Regular ablation source required')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with source.open('rb') as left, target.open('rb') as right:
            while True:
                _remaining(deadline)
                a, b = left.read(1024 * 1024), right.read(1024 * 1024)
                if a != b:
                    raise ValueError('Immutable ablation artifact changed')
                if not a:
                    return
    partial = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.part')
    try:
        with source.open('rb') as reader, partial.open('xb') as writer:
            while True:
                _remaining(deadline)
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        publish_file(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _identity(workflow):
    """检查安全且不重复的实验与运行编号。 / Check safe, unique experiment and run identities."""
    experiment = safe_run_id(workflow['experiment_id'])
    _path(experiment)
    seen = set()
    for row in workflow['nodes']:
        run = safe_run_id(row['spec']['run_id'])
        _path(run)
        if run.casefold() in seen:
            raise ValueError('Duplicate ablation run identity')
        seen.add(run.casefold())
    return experiment


def _status(node):
    # 兼容两种已登记节点状态字段，不推断未知结果。
    # Accept the registered state/status field aliases without inferring unknown outcomes.
    return node.get('state', node.get('status'))


def _state(workflow, root):
    """将状态绑定到准确的工作流和节点集合。 / Bind state to the exact workflow and node set."""
    state = read_json(_confined(root, 'workflow-state.json'))
    if (state.get('experiment_id') != workflow['experiment_id']
            or set(state.get('nodes', {})) != {row['spec']['run_id'] for row in workflow['nodes']}
            or state.get('workflow', workflow) != workflow):
        raise ValueError('Ablation workflow state identity mismatch')
    return state


def _workspace_files(local):
    """登记共享输入与小型来源文档，不扫描历史运行。 / Register shared inputs and small provenance documents."""
    names = ['workflow.json', 'ablation-workspace.json', 'inputs/risk.pt']
    # 来源与许可回执是冻结的小型元数据，不通过遍历运行目录收集。
    # Source/admission receipts are frozen metadata, not run-directory scans.
    for path in sorted(local.glob('*.json')):
        if path.name not in ('workflow-state.json', *names):
            names.append(path.name)
    inputs = _check_local(local / 'inputs')
    for path in sorted(inputs.glob('*.json')):
        names.append(path.relative_to(local).as_posix())
    return names


def _registry(root):
    """验证冻结文件清单和路径范围。 / Validate the frozen file registry and path scope."""
    value = read_json(_confined(root, 'workspace-files.json'))
    names = value.get('files')
    required = {'workflow.json', 'ablation-workspace.json', 'inputs/risk.pt'}
    if (type(names) is not list or any(type(name) is not str for name in names)
            or len(names) != len(set(names)) or not required <= set(names)):
        raise ValueError('Invalid frozen workspace file registry')
    for name in names:
        path = _path(name)
        if name not in required and not (path.suffix == '.json' and (
                len(path.parts) == 1 or (len(path.parts) == 2 and path.parts[0] == 'inputs'))):
            raise ValueError('Unsafe frozen workspace metadata path')
    return names


def _manifest(workflow, run, durable):
    """复用原传输校验器核对归档身份、成员及字节范围。
    Reuse native inventory checks for bundle identity, members, and byte limits.
    """
    value = read_json(_confined(durable, f'bundles/{run}/manifest.json'))
    if (value.get('schema') != 'ablation_bundle_v1'
            or value.get('experiment_id') != workflow['experiment_id']
            or value.get('run_id') != run or not value.get('files') or not value.get('shards')):
        raise ValueError('Ablation bundle identity mismatch')
    # 复用原清单规则检查大小、路径冲突及成员完整覆盖，不另建弱校验。
    # Reuse native size, collision, and coverage rules rather than weaker validation.
    experiment = workflow['experiment_id'] + '-bundle'
    metadata = [{'path': f'experiments/{experiment}/{name}', 'bytes': 0}
                for name in ('snapshot.json', 'state.json')]
    mapped = []
    for index, shard in enumerate(value['shards']):
        if shard['path'] != f'bundles/{run}/run-{index:05d}.tar':
            raise ValueError('Unsafe ablation bundle archive path')
        mapped.append({**shard, 'path': f'experiments/{experiment}/export/run-{index:05d}.tar'})
    if any(not row['path'].startswith(f'runs/{run}/') for row in value['files']):
        raise ValueError('Ablation bundle includes another run')
    validation = {'schema': 'experiment_inventory_v2', 'experiment_id': experiment,
                  'files': metadata + value['files'], 'bytes': sum(row['bytes'] for row in value['files']),
                  'shards': mapped + [{'path': f'experiments/{experiment}/export/base.tar',
                                       'bytes': 10240, 'members': [row['path'] for row in metadata]}]}
    _inventory(validation, experiment, GraphTransferLimits())
    return value


def _bundle(workflow, run, local, durable, deadline, commit):
    """只构建一次计算归档，逐片提交后最后发布清单。
    Build compute bundles once; commit each shard before publishing its manifest.
    """
    manifest = _confined(durable, f'bundles/{run}/manifest.json')
    if manifest.exists():
        # 已冻结的原始计算产物不因后续核验而重新扫描或改写。
        # Never rescan or rewrite frozen compute artifacts after verification.
        return _manifest(workflow, run, durable)
    experiment = workflow['experiment_id'] + '-bundle'
    base = _confined(local, f'experiments/{experiment}')
    _json(base / 'snapshot.json', {'experiment_id': experiment})
    _json(base / 'state.json', {'experiment_id': experiment})
    owner = _confined(local, f'runs/{run}')
    if not owner.is_dir() or not any(owner.iterdir()):
        raise ValueError('Completed ablation run artifacts missing')
    inventory = publish_shards({'experiment_id': experiment, 'nodes': [{'run_id': run}]},
                               local, seconds=_remaining(deadline))
    files = [row for row in inventory['files'] if row['path'].startswith(f'runs/{run}/')]
    shards = []
    for shard in inventory['shards']:
        if not all(member.startswith(f'runs/{run}/') for member in shard['members']):
            continue
        remote = f'bundles/{run}/run-{len(shards):05d}.tar'
        _copy(_confined(local, shard['path']), _confined(durable, remote), deadline)
        shards.append({**shard, 'path': remote})
        commit()
    value = {'schema': 'ablation_bundle_v1', 'experiment_id': workflow['experiment_id'],
             'run_id': run, 'files': files, 'shards': shards}
    _json(manifest, value)
    commit()
    return _manifest(workflow, run, durable)


def persist_workspace(workflow: dict, local: Path, durable: Path, *, commit) -> None:
    """先持久化原始计算归档，再发布可恢复的状态转换。
    Durably publish raw compute bundles BEFORE a resumable state transition.

    durable 状态不能声称 computed/verified 却缺少对应检查点和原始日志。
    Durable state must not claim computed/verified before checkpoints and raw
    logs are safely bundled. A stopped container can then restore exact bytes.
    """
    _identity(workflow)
    local, durable = _root(local), _root(durable)
    deadline = time.monotonic() + SECONDS
    _json(local / 'workflow.json', workflow)
    if read_json(_confined(local, 'ablation-workspace.json')) != {
            'schema': 'ablation_workspace_v1', 'experiment_id': workflow['experiment_id']}:
        raise ValueError('Ablation workspace marker identity mismatch')
    names = _workspace_files(local)
    for name in names:
        _copy(_confined(local, name), _confined(durable, name), deadline)
    _json(durable / 'workspace-files.json', {'files': names})
    state_path = _confined(local, 'workflow-state.json')
    # 初始节点状态创建前，也先保存工作流快照和输入。
    # Commit workflow and inputs even before workflow.py creates initial node state.
    if state_path.exists():
        state = _state(workflow, local)
        for row in workflow['nodes']:
            run = row['spec']['run_id']
            if _status(state['nodes'][run]) in ('computed', 'verified', 'verification_failed'):
                _bundle(workflow, run, local, durable, deadline, commit)
        _json(durable / 'workflow-state.json', state, mutable=True)
    commit()


def hydrate_workspace(workflow: dict, durable: Path, local: Path) -> None:
    """恢复冻结的计算字节，拒绝危险成员和本地内容冲突。
    Restore frozen compute bytes, rejecting unsafe members and local conflicts.

    只恢复确定已完成计算的节点，不能把失败或未知计算变成可自动续跑状态。
    Restore only known computed nodes; failed or uncertain computation must not
    become automatically resumable. Restore later verification proof separately.
    """
    _identity(workflow)
    durable, local = _root(durable), _root(local)
    deadline = time.monotonic() + SECONDS
    if read_json(_confined(durable, 'workflow.json')) != workflow:
        raise ValueError('Immutable workflow snapshot changed')
    for name in _registry(durable):
        _copy(_confined(durable, name), _confined(local, name), deadline)
    state = _state(workflow, durable)
    for row in workflow['nodes']:
        run = row['spec']['run_id']
        node = state['nodes'][run]
        if _status(node) not in ('computed', 'verified', 'verification_failed'):
            # 失败或结果不明的计算仍保持阻断，不能因存在文件而自动续跑。
            # Existing files do not make failed or uncertain compute resumable.
            continue
        manifest = _manifest(workflow, run, durable)
        by_path = {row['path']: row for row in manifest['files']}
        for index, shard in enumerate(manifest['shards']):
            archive = _confined(local, f'.bundles/{run}/run-{index:05d}.tar')
            _copy(_confined(durable, shard['path']), archive, deadline)
            if archive.stat().st_size != shard['bytes']:
                raise ValueError('Ablation bundle archive size mismatch')
            extract_shard(archive, [by_path[name] for name in shard['members']], local,
                          archive.with_suffix('.extract.json'), seconds=_remaining(deadline))
        # 核验在计算之后发生，单独恢复证明，不重写已冻结的计算归档。
        # Verification is post-compute metadata, not a rewrite of frozen compute tars.
        if _status(node) == 'verified':
            _json(_confined(local, f'runs/{run}/verification.json'), node['proof'])
    # 本地状态存在且冲突时直接拒绝，不覆盖隐藏矛盾。
    # Reject conflicting local state instead of overwriting the discrepancy.
    _json(local / 'workflow-state.json', state)


def export_phase(workflow: dict, phase: str, local: Path, durable: Path, *, commit) -> dict:
    """导出截至本阶段的全部已核验节点，最后发布传输清单。
    Publish a transfer-compatible verified prefix, with final inventory LAST.

    包含前序节点，使本地核验能读取续训依赖；清单最后提交表示分片已齐。
    Include predecessors for local continuation checks. Publishing the inventory
    last signals that shards are complete, not that formal algorithm gates passed.
    """
    experiment = _identity(workflow)
    if phase not in workflow['phases']:
        raise ValueError('Unknown ablation phase')
    local, durable = _root(local), _root(durable)
    deadline = time.monotonic() + SECONDS
    state = _state(workflow, local)
    selected = [row for row in workflow['nodes']
                if workflow['phases'].index(row['phase']) <= workflow['phases'].index(phase)]
    if not selected or any(_status(state['nodes'][row['spec']['run_id']]) != 'verified' for row in selected):
        raise ValueError('Ablation phase prefix is not fully verified')
    export_id = safe_run_id(experiment + '-' + phase)
    prefix = f'experiments/{export_id}'
    frozen = _confined(durable, prefix + '/inventory.json')
    proofs = {row['spec']['run_id']: state['nodes'][row['spec']['run_id']] for row in selected}
    export_state = {'experiment_id': export_id, 'status': 'cloud_verified', 'nodes': proofs}
    snapshot = {'schema': 'ablation_phase_v1', 'experiment_id': export_id,
                'ablation_experiment_id': experiment, 'phase': phase,
                'nodes': [{'node_id': row['spec']['run_id'], 'spec': row['spec']} for row in selected]}
    if frozen.exists():
        inventory = read_json(frozen)
        _inventory(inventory, export_id, GraphTransferLimits())
        if read_json(_confined(durable, prefix + '/snapshot.json')) != snapshot:
            raise ValueError('Immutable phase export snapshot changed')
        if read_json(_confined(durable, prefix + '/state.json')) != export_state:
            raise ValueError('Immutable phase export proofs changed')
        return {'experiment_id': experiment, 'phase': phase, 'export_id': export_id,
                'status': 'cloud_verified', 'files': len(inventory['files']),
                'bytes': inventory['bytes'], 'shards': len(inventory['shards'])}
    with tempfile.TemporaryDirectory(prefix='ablation-export-', dir=local) as temporary:
        stage = Path(temporary)
        base = _confined(stage, prefix)
        _json(base / 'snapshot.json', snapshot)
        _json(base / 'state.json', export_state)
        for name in _registry(durable):
            _copy(_confined(local, name), _confined(base, name), deadline)
        _json(base / 'workflow-state.json', state)
        bundles = []
        for row in selected:
            run = row['spec']['run_id']
            node = proofs[run]
            proof = node['proof']
            if proof.get('verified') is not True or proof.get('run_id') != run:
                raise ValueError('Invalid ablation verification proof')
            result = node.get('result', node.get('summary'))
            if type(result) is not dict:
                raise ValueError('Verified ablation node result missing')
            _json(base / 'nodes' / run / 'result.json', result)
            _json(base / 'nodes' / run / 'verification.json', proof)
            bundle = _manifest(workflow, run, durable)
            bundles.append((run, bundle))
            if f'runs/{run}/verification.json' not in {item['path'] for item in bundle['files']}:
                _json(_confined(stage, f'runs/{run}/verification.json'), proof)
        inventory = publish_shards({'experiment_id': export_id,
                                    'nodes': [{'run_id': run} for run, _ in bundles]},
                                   stage, seconds=_remaining(deadline))
        for shard in inventory['shards']:
            _copy(_confined(stage, shard['path']), _confined(durable, shard['path']), deadline)
            commit()
        for run, bundle in bundles:
            inventory['files'].extend(bundle['files'])
            for index, shard in enumerate(bundle['shards']):
                remote = f'{prefix}/export/bundle-{run}-{index:05d}.tar'
                _copy(_confined(durable, shard['path']), _confined(durable, remote), deadline)
                inventory['shards'].append({**shard, 'path': remote})
                commit()
        inventory['bytes'] = sum(row['bytes'] for row in inventory['files'])
        limits = GraphTransferLimits()
        _inventory(inventory, export_id, limits)
        if len(json.dumps(inventory, ensure_ascii=False, indent=2).encode('utf-8')) > limits.max_inventory_bytes:
            raise ValueError('Ablation export inventory document exceeds byte bound')
        # 下载器在解包前先直接读取快照；快照和状态先于最终清单发布。
        # The downloader reads snapshot.json before extraction; publish it before inventory.
        _copy(base / 'snapshot.json', _confined(durable, prefix + '/snapshot.json'), deadline)
        _copy(base / 'state.json', _confined(durable, prefix + '/state.json'), deadline)
        if frozen.exists() and read_json(frozen) != inventory:
            raise ValueError('Immutable phase export inventory changed')
        _json(frozen, inventory)
        commit()
    return {'experiment_id': experiment, 'phase': phase, 'export_id': export_id,
            'status': 'cloud_verified', 'files': len(inventory['files']),
            'bytes': inventory['bytes'], 'shards': len(inventory['shards'])}
