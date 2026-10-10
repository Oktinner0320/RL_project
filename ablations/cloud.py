"""消融 Notebook 的有界云端闭环。 / Bounded cloud lifecycle for the isolated notebook.

准备 -> 显式准入 -> 提交唯一调用 -> 等待 -> 回传 -> 本地语义核验。
Prepare -> explicit admission -> submit one call -> wait -> collect -> verify locally.
云端计算、传输完整与本地核验分别记账，均不代表正式算法验收。
Cloud computation, complete transport, and local verification are distinct states;
none alone implies formal algorithm acceptance or authorizes automatic expansion.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tomllib
import uuid
from pathlib import Path

from cloud_edge_rl.config import RunSpec
from cloud_edge_rl.experiment_transfer import _check_local, collect_experiment
from cloud_edge_rl.transport import (
    publish_file,
    read_json,
    run_bounded,
    stop_and_verify,
    write_json,
)

from .runner import _check_preparation, output_directory, prepare, project_root
from .workflow import PHASES, build_workflow

DEFAULT_NAME = 'N-cloud-v1'
DEADLINES = {'pilot': 4800, 'continue': 9000, 'replicate': 27000}
ENVIRONMENT = 'main'


def _immutable(path, value):
    """重复调用只接受相同内容。 / Repeated calls accept identical content only."""
    path = _check_local(Path(path))
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Immutable ablation document changed: {path}')
    else:
        write_json(path, value)


def phase_limits(phase):
    """返回登记的单容器资源和阶段时限，不授予执行许可。
    Return registered single-container resources and deadlines, not permission.
    """
    if phase not in PHASES:
        raise ValueError('Unknown ablation phase')
    return {'gpu': 'H100', 'cpu': 16, 'memory': 65536, 'max_containers': 1,
            'retries': 0, 'deadline_seconds': DEADLINES[phase]}


def admission_view(workflow, phase):
    """将同一执行图投影到账目；不生成第二套科学配置。 / Project the same graph onto accounting."""
    return {'full_workflow_jobs': len(workflow['nodes']), 'cloud_limits': phase_limits(phase),
            'jobs': [{'phase': row['phase'], 'arm': row['spec']['config']['experiment']['variant'],
                      'seed': row['spec']['seed'],
                      'partition': row['spec']['config']['execution']['split'],
                      'interactions': row['spec']['config']['rl']['stage_interaction_limit']
                      if row['spec']['stage'] == 'train_policy' else 30}
                     for row in workflow['nodes']]}


def prepare_cloud(root=None, *, name=DEFAULT_NAME):
    """冻结并暂存，不提交云任务。 / Freeze and stage without submitting a cloud task."""
    from .inputs import stage_risk

    root = project_root(root)
    prepare(root, name=name)
    directory = output_directory(root, name)
    risk = stage_risk(root, name=name)
    if risk.get('status') != 'local_candidate_staged_not_admitted':
        raise ValueError('A complete local frozen risk input is required')
    workflow = build_workflow(read_json(directory / 'configuration.json'),
                              read_json(directory / 'plan.json'), risk['risk_version'])
    _immutable(directory / 'workflow.json', workflow)
    return Session(root, name=name)


def _source_files(root, workflow):
    """只打包运行代码，不上传本地旧产物或测试。 / Package runtime code, not old outputs/tests."""
    from cloud_edge_rl.modal_runner import _reference_files, _reference_sources

    specs = [RunSpec(**row['spec']) for row in workflow['nodes']]
    _, names, remote_root = _reference_sources(specs)
    files = []
    for path in sorted((root / 'src/cloud_edge_rl').rglob('*.py')):
        if not {'test', 'tests', '__pycache__'}.intersection(path.parts):
            files.append((path, '/root/' + path.relative_to(root / 'src').as_posix()))
    for path in sorted((root / 'ablations').glob('*.py')) + sorted((root / 'ablations/variants').glob('*.py')):
        files.append((path, '/root/' + path.relative_to(root).as_posix()))
    for path in _reference_files(root, names):
        files.append((path, (remote_root / path.relative_to(root).as_posix()).as_posix()))
    return files


def _runtime_sources(root, workflow, source_directory=None):
    """固定实际云端源码，允许本地控制器修复而不改写计算溯源。
    Pin actual cloud sources across local controller repairs.

    已冻结文本来自 sources.json；本地修复不偷偷替换旧调用实际运行的版本。
    Stage frozen sources.json text, not silently updated local controller code.
    路径集合、大小和暂存字节仍须匹配，不把冻结解释为跳过核验。
    Still check path set, size, and staged text; pinning is not a validation bypass.
    """
    files = _source_files(root, workflow)
    current = {remote: source.read_text(encoding='utf-8') for source, remote in files}
    if source_directory is None:
        return files, current
    directory = _check_local(Path(source_directory))
    manifest = directory / 'sources.json'
    frozen = read_json(manifest) if manifest.exists() else current
    if (not isinstance(frozen, dict) or set(frozen) != set(current) or len(frozen) > 4096
            or any(type(value) is not str for value in frozen.values())
            or sum(len(value.encode('utf-8')) for value in frozen.values()) > 32 * 1024**2):
        raise ValueError('Frozen runtime source manifest has invalid paths or size')
    _immutable(manifest, frozen)
    staged = []
    for index, (source, remote) in enumerate(files):
        target = _check_local(directory / 'files' / f'{index:04d}{source.suffix}')
        if target.exists():
            if target.read_text(encoding='utf-8') != frozen[remote]:
                raise ValueError('Staged cloud source changed')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.part')
            part.write_bytes(frozen[remote].encode('utf-8'))
            publish_file(part, target)
        staged.append((target, remote))
    return staged, frozen


def create_app(root, workflow, phase, *, source_directory=None):
    """使用原锁文件构建独立应用；不调用旧执行准入。 / Build an isolated app from the shared lockfile."""
    import modal

    root = Path(root)
    limits = phase_limits(phase)
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    image = modal.Image.debian_slim(python_version=(root / '.python-version').read_text().strip()).uv_sync(
        str(root), frozen=True, extra_options='--no-dev --extra comparisons',
        uv_version=project['tool']['uv']['required-version'].removeprefix('=='))
    # ENV 属于构建步骤，必须先于默认的运行时文件挂载，避免 Modal 镜像次序错误。
    # ENV is a build step and must precede default runtime file mounts in Modal.
    image = image.env({'PYTHONPATH': '/root', 'PYTHONUNBUFFERED': '1'})
    files, sources = _runtime_sources(root, workflow, source_directory)
    for source, remote in files:
        image = image.add_local_file(source, remote)
    volume_name = workflow['experiment_id']
    volume = modal.Volume.from_name(volume_name, create_if_missing=True, environment_name=ENVIRONMENT)
    app = modal.App(volume_name + '-' + phase)
    function = app.function(name='ablation_phase', image=image, volumes={'/workspace': volume},
                            gpu=limits['gpu'], cpu=limits['cpu'], memory=limits['memory'],
                            timeout=limits['deadline_seconds'], startup_timeout=180,
                            max_containers=1, min_containers=0, buffer_containers=0,
                            scaledown_window=2, retries=0)(run_remote)
    return app, function, sources


def run_remote(workflow, phase, attempt_id, risk_bytes, source_contents):
    """本地盘执行，归档后持久化；完整结果与失败回执均保留。
    Compute locally, persist bundles and receipts.

    同一 attempt 的完成回执直接复用；已有工作流状态先从云盘恢复。
    Reuse an existing receipt for the same attempt; hydrate persisted workflow
    state before continuation. Scientific computation never runs in the downloader.
    下载器不执行科学计算；异常只记阻断，不隐式重试训练。
    """
    import modal

    from .bundles import export_phase, hydrate_workspace, persist_workspace
    from .workflow import run_phase

    phase_limits(phase)
    volume = modal.Volume.from_name(workflow['experiment_id'], environment_name=ENVIRONMENT)
    volume.reload()
    # Modal 挂载根可能是可信别名；只在此解析根，归档代码继续限制后代路径。
    # Modal's mount root may be a trusted alias; descendant paths remain confined.
    durable = Path('/workspace').resolve(strict=True)
    receipt_path = durable / 'calls' / f'{phase}-{attempt_id}.json'
    if receipt_path.exists():
        return read_json(receipt_path)
    for name, content in source_contents.items():
        if Path(name).read_text(encoding='utf-8') != content:
            raise ValueError(f'Cloud runtime source content mismatch: {name}')
    _immutable(durable / 'workflow.json', workflow)
    _immutable(durable / 'ablation-workspace.json',
               {'schema': 'ablation_workspace_v1', 'experiment_id': workflow['experiment_id']})
    _immutable(durable / 'runtime-sources.json', source_contents)
    risk = durable / 'inputs/risk.pt'
    risk.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(risk_bytes, bytes) or not 0 < len(risk_bytes) <= 64 * 1024**2:
        raise ValueError('Bounded complete risk input required')
    if risk.exists():
        if risk.read_bytes() != risk_bytes:
            raise ValueError('Frozen cloud risk input changed')
    else:
        from cloud_edge_rl.transport import publish_file
        part = risk.with_name('.risk.pt.part')
        part.write_bytes(risk_bytes)
        publish_file(part, risk)
    volume.commit()
    # FUSE 只接收大块归档和小型账目；大量原生小文件留在容器临时盘。 / Keep native small-file I/O off FUSE.
    with tempfile.TemporaryDirectory(prefix='ablation-') as temporary:
        local = Path(temporary)
        result = {'experiment_id': workflow['experiment_id'], 'phase': phase,
                  'attempt_id': attempt_id, 'status': 'blocked', 'actual_cloud_cost_usd': None,
                  'formal_acceptance': False, 'risk_measurement': 'predicted_risk'}
        try:
            if (durable / 'workflow-state.json').exists():
                hydrate_workspace(workflow, durable, local)
            else:
                if phase != 'pilot':
                    raise ValueError('Later phases require persisted pilot state')
                import shutil
                for name in ('workflow.json', 'ablation-workspace.json', 'runtime-sources.json',
                             'inputs/risk.pt'):
                    target = local / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(durable / name, target)
                persist_workspace(workflow, local, durable, commit=volume.commit)
            outcome = run_phase(workflow, local, phase, commit=lambda: persist_workspace(
                workflow, local, durable, commit=volume.commit))
            result.update(outcome)
            if outcome['status'] == 'cloud_verified':
                result.update(export_phase(workflow, phase, local, durable, commit=volume.commit))
                result['status'] = 'cloud_verified'
        except Exception as error:
            result.update(status='blocked', error=f'{type(error).__name__}: {error}')
        _immutable(receipt_path, result)
        volume.commit()
        return result


class Session:
    """可重复点击的 Notebook 会话；已提交阶段不会再次训练。 / Idempotent notebook facade."""

    def __init__(self, root=None, *, name=DEFAULT_NAME):
        """打开本地冻结会话，不触发网络或训练。 / Open the frozen local session without network or training."""
        self.root = project_root(root)
        self.name = name
        self.directory = output_directory(self.root, name)
        self.workflow = read_json(self.directory / 'workflow.json')

    def _phase(self, phase):
        """定位阶段专属输出目录。 / Locate the isolated output directory for a registered phase."""
        phase_limits(phase)
        path = _check_local(self.directory / 'phases' / phase)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _command(self, action, phase, timeout):
        """用看门狗执行一个动作，将操作结果与进程回执分开保存。
        Run one watched action, keeping operation output separate from process receipts.
        """
        path = self._phase(phase)
        output = path / 'controller' / f'{action}-{uuid.uuid4().hex}.result.json'
        # 支持 Notebook 从 ablations/ 启动，不修改全局环境。 / Support a notebook cwd inside ablations/.
        bootstrap = ('import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); '
                     'runpy.run_module("ablations.cloud",run_name="__main__")')
        # worker.log 的看门狗回执不能覆盖动作写入的 result.json。
        # The worker.log watchdog receipt must not overwrite the action's result.json.
        run_bounded([sys.executable, '-c', bootstrap, str(self.root), action,
                     '--root', str(self.root), '--name', self.name, '--phase', phase,
                     '--output', str(output)], timeout, output.with_suffix('.worker.log'))
        return read_json(output)

    def submit(self, phase='pilot'):
        """提交获准阶段；已有准确调用记录直接复用。 / Submit an admitted phase; reuse an existing bound call."""
        return self._command('submit', phase, 1200)

    def approve_phase(self, phase):
        """Notebook 显式批准下一阶段；此前阶段必须已回传核验。 / Explicit next-phase approval after local closure."""
        from .admission import authorize_phase

        _check_preparation(self.root, self.directory)
        if phase not in ('continue', 'replicate'):
            raise ValueError('Only a future continuation or replication can be approved here')
        prior_phase = 'pilot' if phase == 'continue' else 'continue'
        prior_path = self._phase(prior_phase) / 'closure.json'
        prior = read_json(prior_path)
        if prior.get('state') != 'verified':
            raise ValueError('Prior phase must be downloaded and locally verified first')
        fields = {key: prior[key] for key in ('state', 'jobs', 'call_id', 'app_id')}
        prior = {**prior, 'verification_source': {'path': str(prior_path), 'fields': fields}}
        decision = {'schema': 'ablation_phase_decision_v1', 'experiment_id': self.workflow['experiment_id'],
                    'phase': phase, 'prior_phase': prior_phase, 'prior_call_id': prior['call_id'],
                    'approval': 'explicit_notebook_phase_selection',
                    'automatic_expansion': False, 'formal_acceptance': False}
        decision_path = self._phase(phase) / 'decision.json'
        _immutable(decision_path, decision)
        return authorize_phase(self.directory, phase, workflow=admission_view(self.workflow, phase),
                               evidence={'request_ref': str(decision_path), 'scope': phase,
                                         'prior_result': prior, 'decision_source': {
                                             'path': str(decision_path), 'fields': decision}})

    def status(self, phase='pilot', *, refresh=False):
        """默认只读本地回执；显式刷新才查询云端。 / Read local receipts unless cloud refresh is explicit."""
        if refresh:
            return self._command('status', phase, 90)
        path = self._phase(phase)
        return {'experiment_id': self.workflow['experiment_id'], 'phase': phase,
                **{key: read_json(path / f'{key}.json') for key in
                   ('submission', 'remote', 'shutdown', 'transfer', 'closure')
                   if (path / f'{key}.json').exists()}}

    def wait(self, phase='pilot'):
        """只等待原调用，达到上限后保留记录；不发起新计算。 / Wait for the original bounded call."""
        return self._command('wait', phase, DEADLINES[phase] + 300)

    def collect(self, phase='pilot'):
        """下载已完成阶段，保留完整文件以支持续传。 / Download a completed phase with file-level resume."""
        return self._command('collect', phase, 7350)

    def verify(self, phase='pilot'):
        """本地逐节点核验下载产物，不重新训练。 / Verify downloaded nodes locally without retraining."""
        return self._command('verify', phase, 18 * 90 + 120)

    def cancel(self, phase='pilot'):
        """显式取消原调用并确认其应用停止，不重置消耗。 / Cancel the original call without resetting consumption."""
        return self._command('cancel', phase, 90)

    def summary(self, phase='pilot'):
        """展示完整性及原始阶段结果，不把预测风险改称真实风险。 / Summarize verified stage results without relabeling proxy risk."""
        closure = read_json(self._phase(phase) / 'closure.json')
        root = self._phase(phase) / 'artifacts'
        rows = []
        for row in self.workflow['nodes']:
            if PHASES.index(row['phase']) > PHASES.index(phase):
                continue
            spec = row['spec']
            result = read_json(root / 'runs' / spec['run_id'] / 'summary.json')
            rows.append({'phase': row['phase'], 'arm': spec['config']['experiment']['variant'],
                         'seed': spec['seed'], **result})
        result = {'phase': phase, 'status': closure['state'], 'results': rows,
                  'actual_phase_interactions': closure['actual_interactions'],
                  'algorithm_conclusion': 'insufficient_evidence' if phase == 'pilot' else 'development_only',
                  'formal_acceptance': 'not_evaluated', 'risk_measurement': 'predicted_risk',
                  'actual_cloud_cost_usd': None}
        _immutable(self._phase(phase) / 'summary.json', result)
        return result


def _submit(session, phase):
    """先核对冻结与额度，再保存意图并启动唯一云调用。
    Check frozen sources and quota, persist intent, then launch one cloud call.

    已知 call_id 可复用；意图存在但 call_id 缺失时阻断，不能假定未消费。
    Reuse a known call_id; an intent without one is unresolved, not proof of zero use.
    """
    import modal

    from .admission import bind_call, reserve_phase

    _check_preparation(session.root, session.directory)
    path = session._phase(phase) / 'submission.json'
    if path.exists():
        previous = read_json(path)
        if previous.get('call_id'):
            return previous
        raise ValueError('Submission outcome unknown; no automatic resubmission')
    if not (session.directory / 'preflight.json').is_file():
        raise ValueError('Complete local preflight before cloud submission')
    if read_json(session.directory / 'preflight.json').get('status') != 'local_synthetic_preflight_passed':
        raise ValueError('Successful local preflight required')
    reservation = reserve_phase(session.directory, phase,
                                workflow=admission_view(session.workflow, phase))
    # 已有未绑定的保留记录并不授予重试；独占意图先于网络调用。 / Durable intent precedes every network call.
    record = {'experiment_id': session.workflow['experiment_id'], 'phase': phase,
              'attempt_id': uuid.uuid4().hex, 'status': 'reserved', 'call_id': None,
              'app_id': None, 'volume_name': session.workflow['experiment_id'],
              'environment': ENVIRONMENT, 'reservation': reservation, 'actual_cloud_cost_usd': None}
    write_json(path, record)
    app = None
    try:
        app, function, sources = create_app(session.root, session.workflow, phase,
                                            source_directory=session.directory / 'runtime')
        with modal.enable_output(), app.run(detach=True, environment_name=ENVIRONMENT):
            record.update(status='submitting', app_id=app.app_id)
            write_json(path, record, replace=True)
            call = function.spawn(session.workflow, phase, record['attempt_id'],
                                  (session.directory / 'inputs/risk.pt').read_bytes(), sources)
            record.update(status='submitted', call_id=call.object_id)
            write_json(path, record, replace=True)
            bind_call(session.directory, phase, call_id=call.object_id, app_id=app.app_id)
        return record
    except BaseException:
        app_id = record.get('app_id') or getattr(app, 'app_id', None)
        if app_id:
            stopped = stop_and_verify(app_id, environment=ENVIRONMENT)
            write_json(path.parent / 'shutdown.json', stopped, replace=True)
        raise


def _remote(session, phase, *, wait=False):
    """只查询或等待原调用；SDK 失效时读取同 attempt 的持久回执。
    Query or wait for the original call; fall back to its exact durable receipt.

    网络错误与结果过期不等于任务已完成，无法核清时不停止未知运行。
    Network errors or expired results do not prove completion; do not stop an
    unresolved running call. Stop only the exact app after bound completion.
    """
    import modal

    path = session._phase(phase)
    record = read_json(path / 'submission.json')
    if not record.get('call_id'):
        raise ValueError('No bound cloud call; reconciliation is required')
    deadline = time.monotonic() + (DEADLINES[phase] + 120 if wait else 1)
    result = None
    error = None
    while True:
        try:
            result = modal.FunctionCall.from_id(record['call_id']).get(timeout=30 if wait else 0)
            break
        except modal.exception.FunctionTimeoutError as exc:
            error = f'{type(exc).__name__}: {exc}'
            break
        except TimeoutError:
            if not wait or time.monotonic() >= deadline:
                return {'status': 'pending', 'call_completed': False,
                        'call_id': record['call_id'], 'phase': phase}
            print(f'{phase}: awaiting original call {record["call_id"]}', flush=True)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            # SDK 结果过期时只查同一调用的持久回执。 / Expired SDK results use the exact durable receipt.
            try:
                volume = modal.Volume.from_name(record['volume_name'], environment_name=ENVIRONMENT)
                chunks = bytearray()
                for chunk in volume.read_file(f'calls/{phase}-{record["attempt_id"]}.json'):
                    chunks.extend(chunk)
                    if len(chunks) > 2 * 1024**2:
                        raise ValueError('Completion receipt exceeds byte limit')
                from cloud_edge_rl.transport import _decode_json
                result = _decode_json(chunks.decode('utf-8'))
                error = None
            except Exception as fallback:
                error += f'; receipt: {type(fallback).__name__}: {fallback}'
            break
    if result is not None and (result.get('experiment_id') != session.workflow['experiment_id']
                               or result.get('phase') != phase
                               or result.get('attempt_id') != record['attempt_id']):
        raise ValueError('Foreign cloud completion binding')
    remote = {'status': 'reconciliation_required' if error else result['status'], 'result': result,
              'error': error, 'call_completed': not bool(error), 'call_id': record['call_id'],
              'app_id': record['app_id'], 'attempt_id': record['attempt_id'], 'phase': phase}
    write_json(path / 'remote.json', remote, replace=True)
    if not error:
        stopped = stop_and_verify(record['app_id'], environment=ENVIRONMENT)
        write_json(path / 'shutdown.json', stopped, replace=True)
    return remote


def _collect(session, phase):
    """要求准确完成回执和应用停止证明，再下载固定清单。
    Require exact completion and app shutdown before downloading the pinned inventory.
    """
    path = session._phase(phase)
    remote = read_json(path / 'remote.json') if (path / 'remote.json').exists() else _remote(session, phase)
    submission = read_json(path / 'submission.json')
    stopped = read_json(path / 'shutdown.json') if (path / 'shutdown.json').exists() else {}
    if (remote.get('call_completed') is not True or remote.get('status') != 'cloud_verified'
            or remote.get('attempt_id') != submission['attempt_id']
            or stopped.get('stopped') is not True or stopped.get('app_id') != submission['app_id']):
        raise ValueError('Collection requires exact completed phase and observed app shutdown')
    export_id = session.workflow['experiment_id'] + '-' + phase
    if remote['result'].get('export_id') != export_id:
        raise ValueError('Cloud export identity mismatch')
    result = collect_experiment(submission['volume_name'], export_id, path / 'artifacts',
                                environment=ENVIRONMENT)
    result.update(phase=phase, attempt_id=submission['attempt_id'])
    write_json(path / 'transfer.json', result, replace=True)
    return result


def _verify(session, phase):
    """核对传输绑定、输入和每个节点的实际产物，然后追加阶段账目。
    Check transfer binding, inputs, and actual node artifacts before recording closure.

    传输成功不等于语义正确；云端证明必须与本地 CPU 核验完全一致。
    Transport success does not prove semantics; cloud proof must equal local CPU
    verification. Include predecessors, but charge only this phase's increments.
    """
    from .admission import phase_jobs, record_phase

    _check_preparation(session.root, session.directory)
    path = session._phase(phase)
    transfer = read_json(path / 'transfer.json')
    submission = read_json(path / 'submission.json')
    root = _check_local(path / 'artifacts').resolve(strict=True)
    export_id = session.workflow['experiment_id'] + '-' + phase
    base = root / 'experiments' / export_id
    if (transfer.get('transport_complete') is not True
            or transfer.get('attempt_id') != submission['attempt_id']
            or transfer.get('experiment_id') != export_id
            or Path(transfer['destination']).resolve() != root):
        raise ValueError('Exact complete phase transfer required')
    state = read_json(base / 'state.json')
    if (state.get('status') != 'cloud_verified' or state.get('experiment_id') != export_id
            or read_json(base / 'workflow.json') != session.workflow):
        raise ValueError('Returned workflow/state mismatch')
    marker = read_json(session.directory / 'ablation-workspace.json')
    if read_json(base / 'ablation-workspace.json') != marker:
        raise ValueError('Returned workspace marker mismatch')
    _immutable(root / 'ablation-workspace.json', marker)
    _immutable(root / 'workflow.json', session.workflow)
    risk = (session.directory / 'inputs/risk.pt').read_bytes()
    if (base / 'inputs/risk.pt').read_bytes() != risk:
        raise ValueError('Returned fitted risk input differs from frozen source')
    target = _check_local(root / 'inputs/risk.pt')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != risk:
        raise ValueError('Restored risk input differs')
    if not target.exists():
        target.write_bytes(risk)
    selected = [row for row in session.workflow['nodes'] if PHASES.index(row['phase']) <= PHASES.index(phase)]
    snapshot = {'schema': 'ablation_phase_v1', 'experiment_id': export_id,
                'ablation_experiment_id': session.workflow['experiment_id'], 'phase': phase,
                'nodes': [{'node_id': row['spec']['run_id'], 'spec': row['spec']} for row in selected]}
    if read_json(base / 'snapshot.json') != snapshot:
        raise ValueError('Returned phase snapshot differs from the frozen specs')
    proofs = {}
    for row in selected:
        run_id = row['spec']['run_id']
        spec_path = path / 'controller' / f'verify-{run_id}.spec.json'
        _immutable(spec_path, row['spec'])
        proof_path = path / 'controller' / f'verify-{run_id}.proof.json'
        command = [sys.executable, '-c',
                   'import sys;sys.path.insert(0,sys.argv.pop(1));from ablations.cloud import verify_one;'
                   'verify_one(*sys.argv[1:])', str(session.root), str(spec_path), str(root), str(proof_path)]
        run_bounded(command, 90, path / 'controller' / f'verify-{run_id}-{uuid.uuid4().hex}.log')
        proofs[run_id] = read_json(proof_path)
        if (proofs[run_id] != read_json(base / 'nodes' / run_id / 'verification.json')
                or read_json(root / 'runs' / run_id / 'summary.json')
                != read_json(base / 'nodes' / run_id / 'result.json')):
            raise ValueError('Cloud and local semantic proof differ')
    jobs = phase_jobs(admission_view(session.workflow, phase), phase)
    current = [row['spec']['run_id'] for row in selected if row['phase'] == phase]
    result = {'state': 'verified', 'experiment_id': session.workflow['experiment_id'], 'phase': phase,
              'call_id': submission['call_id'], 'app_id': submission['app_id'], 'jobs': jobs,
              'actual_interactions': sum(proofs[key]['actual_interactions'] for key in current),
              'proofs': proofs, 'actual_cost_usd': None, 'official_test_access': False,
              'risk_measurement': 'predicted_risk', 'formal_acceptance': False}
    _immutable(path / 'closure.json', result)
    evidence = {key: result[key] for key in ('state', 'jobs', 'call_id', 'app_id')}
    record_phase(session.directory, phase, {**result, 'verification_source': {
        'path': str(path / 'closure.json'), 'fields': evidence}})
    return result


def verify_one(spec_path, root, proof_path):
    """隔离子进程中的单节点只读核验入口。 / Read-only verification entry point for one isolated subprocess."""
    from .execution import verify_stage

    spec = RunSpec(**read_json(Path(spec_path)))
    root = Path(root)
    proof = verify_stage(spec, root, read_json(root / 'runs' / spec.run_id / 'summary.json'))
    _immutable(Path(proof_path), proof)


def main():
    """分派单个显式动作，将完整结果写入独立操作回执。
    Dispatch one explicit action and write its full result to an operation receipt.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['submit', 'status', 'wait', 'collect', 'verify', 'cancel'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', default=DEFAULT_NAME)
    parser.add_argument('--phase', choices=PHASES, default='pilot')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    session = Session(args.root, name=args.name)
    if args.action == 'submit':
        result = _submit(session, args.phase)
    elif args.action in ('status', 'wait'):
        result = _remote(session, args.phase, wait=args.action == 'wait')
    elif args.action == 'collect':
        result = _collect(session, args.phase)
    elif args.action == 'verify':
        result = _verify(session, args.phase)
    else:
        import modal
        submission = read_json(session._phase(args.phase) / 'submission.json')
        modal.FunctionCall.from_id(submission['call_id']).cancel(terminate_containers=True)
        result = stop_and_verify(submission['app_id'], environment=ENVIRONMENT)
        write_json(session._phase(args.phase) / 'cancellation.json', result, replace=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
