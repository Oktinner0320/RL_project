"""统一面向 Notebook 的实验生命周期。 / The single notebook-facing experiment lifecycle.

规划不执行计算；提交保留准入与预约屏障，计算、传输和本地闭环分别持久化。
下载失败不能触发新的训练调用。
Planning is inert. Submission retains the existing admission and reservation
barriers. Cloud computation, artifact transfer and local closure are separate
durable states: a failed download can never start another training call.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from .config import RunSpec, load_config, scientific_configuration, source_identity
from .experiment_graph import MAX_METADATA, expand, freeze_graph, run_graph, validate_graph
from .transport import read_json, run_bounded, write_json


class RemoteCompletionUnavailable(RuntimeError):
    """保留两种查询失败，不将缺少证据解释为完成。 / Keep both lookup failures without treating missing evidence as completion."""

    def __init__(self, call_error, receipt_error):
        """同时记录原调用和持久回执查询错误。 / Record both original-call and durable-receipt lookup errors."""
        super().__init__('Original call and durable completion unavailable; no resubmission')
        self.details = {
            'call': {'type': type(call_error).__name__, 'message': str(call_error)},
            'receipt': {'type': type(receipt_error).__name__, 'message': str(receipt_error)},
        }


class Experiment:
    """管理冻结计划、云端提交、产物收集和本地核验。 / Manage frozen plans, cloud submission, artifact collection and local verification."""

    def __init__(self, directory, root):
        """打开有大小上限的冻结图并校验其契约。 / Open a size-bounded frozen graph and validate its contract."""
        self.directory, self.root = Path(directory).resolve(), Path(root).resolve()
        self._requested_controls = None
        path = self.directory / 'snapshot.json'
        if path.stat().st_size > MAX_METADATA:
            raise ValueError('Frozen graph exceeds metadata bound')
        self.graph = read_json(path)
        validate_graph(self.graph)

    @classmethod
    def open(cls, directory, root=None):
        """从既有目录打开实验，不规划或提交任务。 / Open an existing experiment without planning or submitting jobs."""
        return cls(directory, Path.cwd() if root is None else root)

    @classmethod
    def prepare(cls, config, root, specs=None, verifiers=None, directory=None):
        """冻结配置与有序任务图，或复用一致的旧快照；不授予执行许可。 / Freeze configuration and ordered jobs, or reopen matching snapshots, without execution permission."""
        root = Path(root).resolve()
        controls = config['workflow']['experiment']
        target = Path(directory) if directory is not None else root / controls['directory']
        target = target.resolve()
        if (target / 'snapshot.json').exists():
            # 可调整执行控制，但科学配置和显式任务不能在重开时改变。 / Reopening keeps scientific configuration and explicit jobs frozen.
            saved = cls(target, root)
            if scientific_configuration(read_json(target / 'configuration.json')) != scientific_configuration(config):
                raise ValueError('Frozen configuration changed; reopen the original experiment')
            if specs is not None:
                from dataclasses import asdict
                actual = [expand(saved.graph['pool'], n['spec_ref']) for n in saved.graph['nodes']]
                if actual != [asdict(s) for s in specs]:
                    raise ValueError('Frozen run specifications changed')
            saved._requested_controls = dict(controls)
            return saved
        if (target / 'submission.json').exists():
            raise ValueError('Missing immutable snapshot after possible execution')
        dependencies = None
        source_snapshot = None
        if specs is None:
            # 各模式复用自己的计划器，统一交给冻结图管理依赖和核验。 / Reuse each mode's planner and freeze its dependencies and verifiers in one graph.
            mode = controls['mode']
            if mode == 'fresh':
                from .fresh_benchmark import build_plan, build_run_specs
                plan = build_plan(config)
                identities = {n['id']: f"{n['phase']}-{uuid.uuid4().hex}" for n in plan['nodes']}
                provenance = source_identity(root)
                source_snapshot = provenance['source_contents']
                provenance['source_contents_path'] = str(target / 'source-contents.json')
                specs = build_run_specs(config, root, identities, provenance)
                dependencies = {identities[n['id']]: [identities[d] for d in n['dependencies']]
                                for n in plan['nodes']}
                verifiers = verifiers or 'cloud_edge_rl.fresh_benchmark:verify_fresh_artifacts'
            elif mode == 'selection':
                from .parameter_selection_pipeline import prepare_parameter_specs
                specs = prepare_parameter_specs(config, root)
            elif mode == 'comparisons':
                from .comparisons.workflow import prepare_comparison_specs
                specs = prepare_comparison_specs(config, root)
            else:
                raise ValueError('Unknown experiment planning mode')
        if len(specs) > controls.get('max_nodes', 256):
            raise ValueError('Configured graph node limit exceeded')
        if verifiers is None:
            raise ValueError('Explicit per-stage semantic verifiers required')
        graph = freeze_graph(specs, experiment_id=f'experiment-{uuid.uuid4().hex}',
                             verifier=verifiers, node_dependencies=dependencies,
                             node_seconds=controls['node_timeout_seconds'],
                             verify_seconds=controls['verification_timeout_seconds'],
                              graph_seconds=controls['graph_timeout_seconds'])
        # 先保存配置和可选源码快照，最后发布图；中断后不覆盖不同内容。 / Save configuration/source before publishing the graph, without overwriting changed partial contents.
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / 'configuration.json'
        if config_path.exists():
            if scientific_configuration(read_json(config_path)) != scientific_configuration(config):
                raise ValueError('Partial preparation configuration changed')
        else:
            write_json(config_path, config)
        if source_snapshot is not None:
            source_path = target / 'source-contents.json'
            if source_path.exists():
                if read_json(source_path) != source_snapshot:
                    raise ValueError('Partial preparation source contents changed')
            else:
                write_json(source_path, source_snapshot)
        write_json(target / 'snapshot.json', graph)
        prepared = cls(target, root)
        prepared._requested_controls = dict(controls)
        return prepared

    def _controls(self):
        """科学配置保持冻结，仅读取当前允许变化的工作流控制。 / Keep scientific configuration frozen while reading current workflow controls."""
        frozen = read_json(self.directory / 'configuration.json')
        live_path = self.root / 'config' / 'experiment.json'
        if live_path.is_file():
            live = load_config(live_path, root=self.root)
            if scientific_configuration(live) != scientific_configuration(frozen):
                raise ValueError('Scientific configuration changed since preparation')
            return live['workflow']['experiment']
        return self._requested_controls or frozen['workflow']['experiment']

    def _command(self, action, timeout):
        """在有界子进程中执行控制动作并保存独立日志。 / Run a control action in a bounded subprocess with a separate log."""
        return run_bounded([sys.executable, '-m', 'cloud_edge_rl.experiment',
                            '--action', action, '--directory', str(self.directory),
                            '--root', str(self.root)], timeout,
                           self.directory / 'controller' / f'{action}-{uuid.uuid4().hex}.log')

    def submit(self):
        """按显式开关提交一次；已有绑定只返回状态。 / Submit once when enabled; return status for an existing binding."""
        controls = self._controls()
        if controls['submission_enabled'] is not True:
            raise ValueError('Submission is disabled in the frozen configuration')
        receipt = self.directory / 'submission.json'
        if receipt.exists():
            saved = read_json(receipt)
            if not saved.get('call_id'):
                raise ValueError('Submission identity unknown; explicit reconciliation required')
            return self.status()
        self._command('submit', controls['controller_timeout_seconds'])
        return self.status()

    def resume(self):
        """仅在原调用安全返回后显式继续。 / Continue explicitly only after the original call returned safely."""
        if self._controls()['submission_enabled'] is not True:
            raise ValueError('Submission disabled')
        self._command('resume', self._controls()['controller_timeout_seconds'])
        return self.status()

    def cancel(self):
        """只停止本实验记录的调用和应用，保留未知消耗。 / Stop only this experiment's recorded call/app; retain unknown consumption."""
        self._command('cancel', 120)
        return self.status()

    def status(self, refresh=False):
        """汇总当前提交的持久状态，可显式刷新云端状态。 / Summarize the current attempt's durable state with optional cloud refresh."""
        if refresh:
            self._command('status', 60)
        result = {'experiment_id': self.graph['experiment_id'],
                  'planned_nodes': len(self.graph['nodes']), 'status': 'prepared',
                  'formal_acceptance': 'not_evaluated', 'actual_cloud_cost_usd': None}
        for name in ('submission', 'remote-status', 'transfer', 'partial-transfer', 'closure', 'cancellation'):
            # 只关联当前 attempt，防止旧收集或闭环回执误报本次完成。 / Bind receipts to the current attempt so stale transfers or closure cannot imply completion.
            path = self.directory / f'{name}.json'
            if path.exists():
                value = read_json(path)
                if name == 'submission' or value.get('attempt_id') == result.get('submission', {}).get('attempt_id'):
                    result[name] = value
        recovery_path = self.directory / 'export-recovery.json'
        if recovery_path.exists() and 'submission' in result:
            recovery = read_json(recovery_path)
            binding = recovery.get('binding', {})
            keys = ('experiment_id', 'attempt_id', 'call_id', 'app_id', 'volume_name', 'environment')
            if all(key in binding and key in result['submission']
                   and binding[key] == result['submission'][key] for key in keys):
                result['export-recovery'] = recovery
        if 'cancellation' in result:
            result['status'] = 'cancelled_consumption_requires_reconciliation'
        elif 'closure' in result:
            result['status'] = result['closure']['status']
        elif 'remote-status' in result:
            result['status'] = result['remote-status']['status']
        elif 'submission' in result:
            result['status'] = result['submission']['status']
        return result

    def collect(self, *, available_only=False):
        """确认调用完成及应用停止后收集产物；部分收集不宣称完整核验。 / Collect artifacts after completion/shutdown checks; partial transfer cannot claim full verification."""
        from .experiment_transfer import collect_experiment
        if type(available_only) is not bool:
            raise ValueError('Explicit boolean available-only transfer required')
        if available_only and self._controls()['collection_enabled'] is not True:
            raise ValueError('Collection disabled')
        submission = read_json(self.directory / 'submission.json')
        self.status(refresh=True)
        remote = read_json(self.directory / 'remote-status.json')
        if remote['status'] == 'running':
            raise ValueError('Cloud graph still running; terminal inventory required')
        if not remote.get('call_completed'):
            raise ValueError('Cloud completion unknown; reconcile original call before transfer')
        if (remote.get('attempt_id') != submission['attempt_id']
                or remote.get('experiment_id') != self.graph['experiment_id']):
            raise ValueError('Stale or unbound remote completion; refresh the current attempt')
        # 必须先确认原应用停止，再开始产物回传。 / Verify exact app shutdown before artifact return.
        shutdown = self.directory / f"shutdown-{submission['attempt_id']}.json"
        if not shutdown.exists():
            if available_only:
                raise ValueError('Original app shutdown must be verified before read-only partial transfer')
            self._command('cleanup', 120)
        stopped = read_json(shutdown)
        if stopped.get('stopped') is not True or stopped.get('app_id') != submission['app_id']:
            raise ValueError('Exact app shutdown not verified')
        if available_only:
            recovery_path = self.directory / 'export-recovery.json'
            if recovery_path.exists():
                recovery = read_json(recovery_path)
                keys = ('experiment_id', 'attempt_id', 'call_id', 'app_id', 'volume_name', 'environment')
                if recovery.get('binding') != {key: submission[key] for key in keys}:
                    raise ValueError('Partial transfer recovery binding mismatch')
                recovery_stop = read_json(self.directory / f"export-shutdown-{recovery['recovery_id']}.json")
                if (recovery_stop.get('stopped') is not True
                        or not recovery.get('app_id')
                        or recovery_stop.get('app_id') != recovery['app_id']):
                    raise ValueError('Exact export app shutdown required before partial transfer')
        elif (remote.get('status') == 'blocked'
                and remote.get('error', {}).get('phase') == 'export'):
            # 导出恢复只打包既有核验产物，不恢复训练计算。 / Export recovery packages existing verified artifacts, never resumes compute.
            self._command('repair-export', 2400)
        destination = self.directory / ('partial-artifacts' if available_only else 'artifacts') / submission['attempt_id']
        options = {'available_only': True, 'expected_graph': self.graph} if available_only else {}
        result = collect_experiment(submission['volume_name'], self.graph['experiment_id'],
                                    destination, environment=submission['environment'], **options)
        if available_only and (result.get('transport_complete') is not False
                               or result.get('semantic_verified') is not False):
            raise ValueError('Partial transfer cannot claim full transport or semantic completion')
        result['attempt_id'] = submission['attempt_id']
        receipt = 'partial-transfer.json' if available_only else 'transfer.json'
        write_json(self.directory / receipt, result, replace=True)
        return result

    def verify(self):
        """仅在本地完成证据闭环，不启动云端应用或计算。 / Close evidence entirely locally, without cloud apps or compute."""
        transfer = read_json(self.directory / 'transfer.json')
        submission = read_json(self.directory / 'submission.json')
        if (transfer.get('transport_complete') is not True
                or transfer.get('experiment_id') != self.graph['experiment_id']
                or transfer.get('attempt_id') != submission['attempt_id']):
            raise ValueError('Unified artifact transfer incomplete or stale')
        artifacts = self.directory / 'artifacts' / submission['attempt_id']
        if Path(transfer['destination']).resolve() != artifacts.resolve():
            raise ValueError('Artifact transfer destination mismatch')
        base = artifacts / 'experiments' / self.graph['experiment_id']
        if read_json(base / 'snapshot.json') != self.graph:
            raise ValueError('Returned graph differs from the frozen plan')
        state = read_json(base / 'state.json')
        if state['status'] != 'cloud_verified':
            raise ValueError('Full experiment has not reached cloud verification')
        closure = self.directory / 'closure-progress.json'
        proofs = []
        for node in self.graph['nodes']:
            if state['nodes'][node['run_id']]['status'] != 'verified':
                raise ValueError('Unverified cloud node')
            # 按回传依赖树重新核验科学证据，不只信任云端完成状态。 / Reverify scientific evidence against the returned dependency tree.
            run_bounded([sys.executable, '-m', 'cloud_edge_rl.experiment_graph',
                         '--snapshot', str(self.directory / 'snapshot.json'),
                         '--workspace', str(artifacts), '--run-id', node['run_id'],
                         '--phase', 'verify'], node['verify_seconds'],
                        self.directory / 'closure-logs' / f"{node['run_id']}-{uuid.uuid4().hex}.log")
            proofs.append({'run_id': node['run_id'], 'verified': True})
            write_json(closure, {'attempt_id': submission['attempt_id'], 'nodes': proofs}, replace=True)
        result = {'experiment_id': self.graph['experiment_id'], 'status': 'completed',
                  'attempt_id': submission['attempt_id'],
                  'nodes': proofs, 'formal_acceptance': 'not_evaluated',
                  'actual_cloud_cost_usd': None}
        write_json(self.directory / 'closure.json', result, replace=True)
        return result


def run_remote_graph(graph, volume_name, attempt_id):
    """在独立云端调用中提交进度，不依赖 Notebook 连接。 / Commit detached-call progress independently of notebook connectivity."""
    import modal
    volume = modal.Volume.from_name(volume_name)
    volume.reload()
    from .experiment_graph import publish_inventory
    from .transport import safe_run_id
    safe_run_id(attempt_id)
    workspace = Path('/workspace')
    result = run_graph(graph, workspace, commit=volume.commit, export_inventory=False)
    receipt = workspace / 'experiments' / graph['experiment_id'] / 'calls' / f'{attempt_id}.json'
    # 先持久化待导出状态；导出失败不构成重新计算的许可。 / Persist export-pending state; export failure never permits recomputation.
    pending = {**result, 'status': 'blocked', 'error': {'phase': 'export', 'message': 'Export pending'}}
    write_json(receipt, {'attempt_id': attempt_id, 'state': pending})
    volume.commit()
    try:
        publish_inventory(graph, workspace, commit=volume.commit)
    except Exception as error:
        result = {**result, 'status': 'blocked',
                  'error': {'phase': 'export', 'type': type(error).__name__, 'message': str(error)}}
    write_json(receipt, {'attempt_id': attempt_id, 'state': result}, replace=True)
    volume.commit()
    return result


def _remote_state(experiment, submission):
    """原调用结果不可用时读取持久 attempt 回执，并核对身份。 / Fall back to a durable attempt receipt when the original result is unavailable, checking identity."""
    import modal
    try:
        state = modal.FunctionCall.from_id(submission['call_id']).get(timeout=0)
    except TimeoutError:
        # 尚在运行不等于结果过期，不能回退到旧的完成回执。 / A running call is not an expired result; do not fall back on timeout.
        raise
    except Exception as original:
        from .transport import _decode_json
        content = bytearray()
        path = f"experiments/{experiment.graph['experiment_id']}/calls/{submission['attempt_id']}.json"
        try:
            volume = modal.Volume.from_name(submission['volume_name'],
                                            environment_name=submission['environment'])
            for chunk in volume.read_file(path):
                if len(content) + len(chunk) > 2 * 1024**2:
                    raise ValueError('Bounded remote completion receipt exceeded')
                content.extend(chunk)
            receipt = _decode_json(content.decode('utf-8'))
            if receipt['attempt_id'] != submission['attempt_id']:
                raise ValueError('Remote completion attempt mismatch')
            state = receipt['state']
        except Exception as receipt_error:
            raise RemoteCompletionUnavailable(original, receipt_error) from original
    if (type(state) is not dict or state.get('experiment_id') != experiment.graph['experiment_id']
            or set(state.get('nodes', {})) != {node['run_id'] for node in experiment.graph['nodes']}
            or state.get('status') not in ('cloud_verified', 'paused_deadline', 'blocked')):
        raise ValueError('Remote experiment completion identity/state mismatch')
    return state


def _submit(experiment, *, continuation=False):
    """持有提交锁后进入统一提交路径。 / Enter the common submission path under the submission lock."""
    from .admission import _locked
    with _locked(experiment.directory / 'submission.json'):
        return _submit_locked(experiment, continuation=continuation)


def _submit_locked(experiment, *, continuation=False):
    """核对继续条件并预约任务，持久化唯一云端提交绑定。 / Validate continuation, reserve jobs and persist the cloud submission binding."""
    import modal

    from .experiment_admission import validate_and_reserve_graph
    from .modal_runner import create_app

    graph, directory = experiment.graph, experiment.directory
    if experiment._controls()['submission_enabled'] is not True:
        raise ValueError('Submission disabled in the frozen configuration')
    submission = directory / 'submission.json'
    specs = [RunSpec(**expand(graph['pool'], node['spec_ref'])) for node in graph['nodes']]
    if continuation:
        # 只允许已观测到的暂停或核验阻断；未知或失败计算不能自动重试。 / Continue only observed pauses or verification blocks, never ambiguous or failed compute.
        prior = read_json(submission)
        result = _remote_state(experiment, prior)
        if result.get('status') not in ('paused_deadline', 'blocked'):
            raise ValueError('Only an observed paused or verification-blocked graph can continue')
        if any(row['status'] not in ('pending', 'verified', 'computed', 'verification_failed')
               for row in result['nodes'].values()):
            raise ValueError('Ambiguous compute cannot be automatically retried')
        from .transport import stop_and_verify
        stopped = stop_and_verify(prior['app_id'], environment=prior['environment'])
        if stopped.get('stopped') is not True or stopped.get('app_id') != prior['app_id']:
            raise ValueError('Prior graph app shutdown not verified')
        write_json(directory / f"prior-shutdown-{uuid.uuid4().hex}.json", stopped)
        write_json(directory / f"prior-submission-{uuid.uuid4().hex}.json", prior)
    elif submission.exists():
        raise ValueError('Existing submission cannot be repeated')
    # 保留历史许可、配额、失败记录与源码检查，不因继续而清零。 / Preserve historical approvals, allocations, failures and source checks on continuation.
    validate_and_reserve_graph(graph, experiment.root, directory)
    settings = specs[0].config['modal']
    record = {'experiment_id': graph['experiment_id'], 'status': 'reserved',
              'attempt_id': uuid.uuid4().hex,
              'call_id': None, 'app_id': None, 'volume_name': settings['volume_name'],
              'environment': settings.get('environment', 'main'),
              'reserved_run_ids': [s.run_id for s in specs],
              'actual_cloud_cost_usd': None, 'historical_consumption_reset': False}
    write_json(submission, record, replace=continuation)
    app, function, _ = create_app(experiment.root, settings, specs,
                                  graph_timeout_seconds=graph['graph_seconds'],
                                  graph_controller_directory=directory)
    with modal.enable_output(), app.run(detach=True):
        # spawn 前保存应用身份；未知调用绑定需人工核对而不是重新提交。 / Persist app identity before spawn; reconcile unknown call bindings instead of resubmitting.
        record.update(status='submitting', app_id=app.app_id)
        write_json(submission, record, replace=True)
        call = function.spawn(graph, settings['volume_name'], record['attempt_id'])
        record.update(status='submitted', call_id=call.object_id)
        write_json(submission, record, replace=True)


def _export_recovery_result(record, *, timeout):
    """读取导出恢复结果或有界持久回执，核对原提交绑定。 / Read an export-recovery result or bounded durable receipt and validate the original binding."""
    import modal
    from modal.exception import FunctionTimeoutError

    try:
        result = modal.FunctionCall.from_id(record['call_id']).get(timeout=timeout)
    except (TimeoutError, FunctionTimeoutError):
        raise
    except Exception:
        from .transport import _decode_json
        binding = record['binding']
        volume = modal.Volume.from_name(binding['volume_name'], environment_name=binding['environment'])
        content = bytearray()
        path = f"experiments/{binding['experiment_id']}/calls/export-{record['recovery_id']}.json"
        for chunk in volume.read_file(path):
            if len(content) + len(chunk) > 2 * 1024**2:
                raise ValueError('Export recovery receipt exceeded byte bound')
            content.extend(chunk)
        result = _decode_json(content.decode('utf-8'))
    if (type(result) is not dict or result.get('recovery_id') != record['recovery_id']
            or any(result.get(key) != value for key, value in record['binding'].items())):
        raise ValueError('Export recovery completion binding mismatch')
    if result.get('status') != 'export_complete':
        raise ValueError(f"Export recovery blocked; no automatic retry: {result.get('error')}")
    return result


def _repair_export(experiment):
    """核对单次 CPU 导出调用，不预约或恢复科学计算。 / Reconcile one CPU export call, never reserve or resume scientific compute."""
    import modal
    from modal.exception import FunctionTimeoutError

    from .admission import _locked
    from .experiment_export_recovery import create_export_recovery_app
    from .transport import stop_and_verify

    if experiment._controls()['collection_enabled'] is not True:
        raise ValueError('Collection permission required for export recovery')
    path = experiment.directory / 'export-recovery.json'
    with _locked(path):
        submission = read_json(experiment.directory / 'submission.json')
        state = _remote_state(experiment, submission)
        if (state['status'] != 'blocked' or state.get('error', {}).get('phase') != 'export'
                or any(row['status'] != 'verified' for row in state['nodes'].values())):
            raise ValueError('Export-only recovery requires the completed original verified graph')
        shutdown = read_json(experiment.directory / f"shutdown-{submission['attempt_id']}.json")
        if shutdown.get('stopped') is not True or shutdown.get('app_id') != submission['app_id']:
            raise ValueError('Original app shutdown required before export recovery')
        binding = {key: submission[key] for key in
                   ('experiment_id', 'attempt_id', 'call_id', 'app_id', 'volume_name', 'environment')}
        existing = path.exists()
        if existing:
            # 未知、超时或失败的恢复记录保留，不自动创建第二个导出调用。 / Retain unknown, timed-out or blocked recovery records without a second automatic call.
            record = read_json(path)
            if record.get('binding') != binding:
                raise ValueError('Export recovery belongs to another original submission')
            if not record.get('call_id') or not record.get('app_id'):
                raise ValueError('Export recovery submission unknown; no automatic retry')
            if record.get('status') in {'timed_out', 'blocked'}:
                raise ValueError('Export recovery stop condition retained; no automatic retry')
        else:
            record = {'binding': binding, 'recovery_id': uuid.uuid4().hex, 'status': 'reserved',
                      'call_id': None, 'app_id': None, 'actual_cloud_cost_usd': None,
                      'new_real_interactions': 0, 'gpu': None, 'cpu': 2, 'memory_mb': 2048,
                      'timeout_seconds': 1800, 'automatic_retries': 0}
            write_json(path, record)
        app = None
        try:
            if not existing:
                app, function = create_export_recovery_app(binding['volume_name'], binding['environment'])
                with modal.enable_output(), app.run(detach=True, environment_name=binding['environment']):
                    record.update(status='submitting', app_id=app.app_id)
                    write_json(path, record, replace=True)
                    call = function.spawn(experiment.graph, binding['volume_name'], binding, record['recovery_id'])
                    record.update(status='submitted', call_id=call.object_id)
                    write_json(path, record, replace=True)
            result = _export_recovery_result(record, timeout=1950)
            record.update(status='export_complete', result=result)
            write_json(path, record, replace=True)
        except (TimeoutError, FunctionTimeoutError) as error:
            if isinstance(error, TimeoutError) and record.get('call_id'):
                try:
                    modal.FunctionCall.from_id(record['call_id']).cancel(terminate_containers=True)
                except Exception as cancel_error:
                    record['cancel_error'] = f'{type(cancel_error).__name__}: {cancel_error}'[:4096]
            record.update(status='timed_out', error={
                'type': type(error).__name__, 'message': str(error)[:4096]})
            write_json(path, record, replace=True)
            raise
        except Exception as error:
            record.update(status='blocked', error={
                'type': type(error).__name__, 'message': str(error)[:4096]})
            write_json(path, record, replace=True)
            raise
        finally:
            # 成功和失败都清理确切恢复应用，且不掩盖原异常。 / Clean up the exact recovery app on either outcome without masking the original exception.
            original_error = sys.exc_info()[1]
            app_id = record.get('app_id') or (getattr(app, 'app_id', None) if app is not None else None)
            if app_id is not None:
                try:
                    stopped = stop_and_verify(app_id, environment=binding['environment'])
                except Exception as cleanup_error:
                    stopped = {'app_id': app_id, 'stopped': False, 'verified': False,
                               'error': f'{type(cleanup_error).__name__}: {cleanup_error}'[:4096]}
                write_json(experiment.directory / f"export-shutdown-{record['recovery_id']}.json",
                           stopped, replace=True)
                if (original_error is None
                        and (stopped.get('stopped') is not True or stopped.get('app_id') != app_id)):
                    raise ValueError('Exact export recovery app shutdown not verified')
        if read_json(experiment.directory / 'submission.json') != submission:
            raise ValueError('Original submission changed during export recovery')
        return result


def cli():
    """分派提交、继续、状态、清理、取消和仅导出恢复动作。 / Dispatch submission, continuation, status, cleanup, cancellation and export-only recovery."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('submit', 'resume', 'status', 'cleanup', 'cancel',
                                            'repair-export'), required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    experiment = Experiment.open(args.directory, args.root)
    if args.action in ('submit', 'resume'):
        _submit(experiment, continuation=args.action == 'resume')
        return
    if args.action == 'repair-export':
        _repair_export(experiment)
        return
    submission = read_json(args.directory / 'submission.json')
    if not submission.get('call_id') or not submission.get('app_id'):
        raise ValueError('Submission identity unknown')
    if args.action in ('cleanup', 'cancel'):
        if args.action == 'cancel':
            import modal
            modal.FunctionCall.from_id(submission['call_id']).cancel(terminate_containers=True)
        from .transport import stop_and_verify
        result = stop_and_verify(submission['app_id'], environment=submission['environment'])
        write_json(args.directory / f"shutdown-{submission['attempt_id']}.json", result, replace=True)
        if args.action == 'cancel':
            write_json(args.directory / 'cancellation.json',
                       {'attempt_id': submission['attempt_id'], 'call_id': submission['call_id'],
                        'shutdown': result, 'actual_interactions': None}, replace=True)
    else:
        binding = {'attempt_id': submission['attempt_id'], 'call_id': submission['call_id'],
                   'experiment_id': experiment.graph['experiment_id']}
        try:
            state = _remote_state(experiment, submission)
            result = {**binding, 'status': state['status'], 'call_completed': True,
                       'nodes': state['nodes']}
            if state.get('error') is not None:
                result['error'] = state['error']
        except TimeoutError:
            result = {**binding, 'status': 'running', 'call_completed': False}
        except RemoteCompletionUnavailable as error:
            result = {**binding, 'status': 'remote_completion_unavailable', 'call_completed': False,
                      'error': error.details, 'reconciliation_required': True,
                      'next_action': 'Reconcile the original call before preparing an explicit continuation.'}
        write_json(args.directory / 'remote-status.json', result, replace=True)


if __name__ == '__main__':
    cli()
