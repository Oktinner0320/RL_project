"""在控制器暂存的独立本地工作区执行有界消融阶段。
Bounded standalone ablation computation on a controller-staged local workspace.

执行许可、源码快照、云盘暂存和导出由控制器负责，本模块不提交云任务。
Admission, source snapshots, Volume staging and export belong to the controller.
This module neither submits cloud work nor grants scientific acceptance.
阶段核验不是科学验收。 / Stage verification is not scientific acceptance.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from ablations.studies import arm_configuration, study_definition
from cloud_edge_rl.config import RunSpec, json_copy
from cloud_edge_rl.experiment_transfer import _check_local, _path
from cloud_edge_rl.transport import read_json, run_bounded, safe_run_id, write_json

PHASES = ['pilot', 'continue', 'replicate']
NODE_SECONDS = 1800
VERIFY_SECONDS = 90
RESERVE_SECONDS = 120


def build_workflow(base: dict, plan: dict, risk_version: str) -> dict:
    """生成已登记的 18 节点图，不授权任何执行。
    Build the registered 18-node graph without authorizing any execution.

    每个训练种子的两臂各执行 500 -> 700 -> 独立验证，共 3*2*3 个节点。
    Each paired training seed has two arms with 500 -> 700 -> independent
    validation, totaling 3*2*3 nodes. Each arm depends on its own predecessor.
    """
    if plan.get('schema') != 'ablation_plan_v1' or plan.get('study_id') != 'N':
        raise ValueError('Registered N ablation plan required')
    experiment = safe_run_id(plan['experiment_id'])
    _path(experiment)
    if type(risk_version) is not str or not risk_version:
        raise ValueError('Explicit frozen risk version required')
    definition = study_definition('N')
    if plan.get('definition') != definition:
        raise ValueError('Registered study definition changed')
    nodes = []
    for index, seed in enumerate(definition['training_seeds']):
        for arm in definition['arms']:
            previous = None
            for stage, steps, suffix in [('train_policy', 500, 'train500'),
                                         ('train_policy', 700, 'train700'),
                                         ('validate', 700, 'validate')]:
                phase = ('pilot' if suffix == 'train500' else 'continue') if index == 0 else 'replicate'
                run_id = f'{experiment}-{arm}-{seed}-{suffix}'
                evaluation_seed = seed if stage == 'train_policy' else definition['validation_seeds'][index]
                cfg = arm_configuration(base, experiment, arm, evaluation_seed,
                                        stage=stage, stage_steps=steps)
                inputs = {'risk_checkpoint': 'inputs/risk.pt'}
                inputs['previous_training_checkpoint' if stage == 'train_policy' else 'agent_checkpoint'] = (
                    f'runs/{previous}/training.pt' if previous else None)
                spec = RunSpec(run_id, stage, 'dqn', evaluation_seed, risk_version, cfg, inputs,
                               {'ablation_experiment_id': experiment})
                nodes.append({'phase': phase, 'spec': asdict(spec),
                              'dependencies': [previous] if previous else []})
                previous = run_id
    # 按阶段排序，确保所有依赖先出现，便于逐阶段持久化和导出。
    # Sort by phase so dependencies precede consumers and phase export stays ordered.
    nodes.sort(key=lambda row: PHASES.index(row['phase']))
    return json_copy({'schema': 'ablation_workflow_v1', 'experiment_id': experiment,
                      'phases': PHASES, 'nodes': nodes})


def _validate_workflow(workflow):
    """拒绝危险身份、错误依赖和未登记的阶段结构。
    Reject unsafe IDs, malformed graph edges and unregistered phase shapes.
    """
    if (workflow.get('schema') != 'ablation_workflow_v1'
            or workflow.get('phases') != PHASES or len(workflow.get('nodes', [])) != 18):
        raise ValueError('Invalid ablation workflow schema/shape')
    experiment = safe_run_id(workflow['experiment_id'])
    _path(experiment)
    seen = {}
    counts = dict.fromkeys(PHASES, 0)
    for row in workflow['nodes']:
        if set(row) != {'phase', 'spec', 'dependencies'} or row['phase'] not in PHASES:
            raise ValueError('Invalid workflow node')
        spec = RunSpec(**row['spec'])
        run_id = safe_run_id(spec.run_id)
        _path(run_id)
        if run_id.casefold() in {key.casefold() for key in seen}:
            raise ValueError('Duplicate workflow run identity')
        if (spec.provenance.get('ablation_experiment_id') != experiment
                or spec.method != 'dqn' or spec.stage not in ('train_policy', 'validate')
                or type(spec.risk_version) is not str or not spec.risk_version):
            raise ValueError('Workflow spec identity mismatch')
        cfg = spec.config
        arm = cfg['experiment']['variant']
        seed = cfg['rl']['seed']
        definition = study_definition('N')
        if arm not in definition['arms'] or seed not in definition['training_seeds']:
            raise ValueError('Unregistered workflow arm/seed')
        index = definition['training_seeds'].index(seed)
        steps = cfg['rl']['stage_interaction_limit']
        suffix = 'validate' if spec.stage == 'validate' else f'train{steps}'
        expected_phase = ('pilot' if suffix == 'train500' else 'continue') if index == 0 else 'replicate'
        expected_seed = seed if spec.stage == 'train_policy' else definition['validation_seeds'][index]
        if (run_id != f'{experiment}-{arm}-{seed}-{suffix}' or row['phase'] != expected_phase
                or spec.seed != expected_seed or steps not in (500, 700)
                or cfg['rl']['n_step'] != definition['arms'][arm]['n_step']):
            raise ValueError('Unregistered workflow progression')
        deps = row['dependencies']
        if type(deps) is not list or len(deps) > 1 or any(dep not in seen for dep in deps):
            raise ValueError('Workflow dependencies must refer to earlier nodes')
        key = 'previous_training_checkpoint' if spec.stage == 'train_policy' else 'agent_checkpoint'
        expected = f'runs/{deps[0]}/training.pt' if deps else None
        if (spec.inputs != {'risk_checkpoint': 'inputs/risk.pt', key: expected}
                or (spec.stage == 'validate' and not deps)):
            raise ValueError('Workflow dependency/input mismatch')
        expected_deps = ([] if suffix == 'train500' else
                         [f'{experiment}-{arm}-{seed}-' + ('train700' if suffix == 'validate' else 'train500')])
        if deps != expected_deps:
            raise ValueError('Unregistered workflow dependency')
        if deps:
            prior = RunSpec(**seen[deps[0]]['spec'])
            if (prior.stage != 'train_policy' or prior.risk_version != spec.risk_version
                    or prior.config['experiment']['variant'] != spec.config['experiment']['variant']
                    or prior.seed != spec.config['rl']['seed']):
                raise ValueError('Cross-arm/seed/risk dependency forbidden')
        seen[run_id] = row
        counts[row['phase']] += 1
    if counts != {'pilot': 2, 'continue': 4, 'replicate': 12}:
        raise ValueError('Invalid registered phase counts')
    return seen


def _workspace(workflow, workspace):
    """要求控制器建立的专属身份标记。 / Require the controller-staged workspace identity marker."""
    root = _check_local(Path(workspace)).resolve(strict=True)
    marker = read_json(_check_local(root / 'ablation-workspace.json'))
    if marker != {'schema': 'ablation_workspace_v1', 'experiment_id': workflow['experiment_id']}:
        raise ValueError('A controller-staged isolated workspace is required')
    return root


def _proof(value, run_id):
    """核对节点证明和非正式风险标签。 / Check node proof and explicitly non-formal risk labeling."""
    if (type(value) is not dict or value.get('verified') is not True
            or value.get('run_id') != run_id or type(value.get('actual_interactions')) is not int
            or value['actual_interactions'] <= 0 or value.get('risk_measurement') != 'predicted_risk'
            or value.get('formal_acceptance') is not False):
        raise ValueError('Invalid actual stage verification proof')
    return value


def run_phase(workflow: dict, workspace: Path, phase: str, *, commit=lambda: None,
              worker=run_bounded) -> dict:
    """在本地执行一个已获准阶段；每次计算前持久化执行意图。
    Execute one admitted phase locally; commit durable state before each compute.

    未知、正在运行或已失败的计算不能重放；已计算节点只能恢复核验。
    Unknown/running/failed compute is never replayed. Computed or verification-failed
    nodes can resume verification only. The caller must supply a LOCAL ephemeral disk,
    copy prior artifacts here first, and persist state through the commit callback.
    调用方先恢复前序产物，再通过回调将本地盘状态持久化到云盘。
    The caller restores prior artifacts and persists local state through the callback.
    """
    workflow = json_copy(workflow)
    graph = _validate_workflow(workflow)
    if phase not in PHASES:
        raise ValueError('Unknown ablation phase')
    root = _workspace(workflow, workspace)
    lock = _check_local(root / '.workflow-lock')
    # 遗留锁需要显式核查，不能自动移除后重跑随机计算。
    # Reconcile stale locks explicitly; never clear them to replay stochastic work.
    lock.mkdir()
    try:
        manifest = _check_local(root / 'workflow.json')
        if manifest.exists():
            if read_json(manifest) != workflow:
                raise ValueError('Immutable workflow snapshot changed')
        else:
            write_json(manifest, workflow)
        state_path = _check_local(root / 'workflow-state.json')
        if state_path.exists():
            state = read_json(state_path)
            if (state.get('schema') != 'ablation_workflow_state_v1'
                    or state.get('experiment_id') != workflow['experiment_id']
                    or set(state.get('nodes', {})) != set(graph)
                    or state.get('workflow') != workflow):
                raise ValueError('Workflow state snapshot/identity mismatch')
        else:
            state = {'schema': 'ablation_workflow_state_v1',
                     'experiment_id': workflow['experiment_id'], 'workflow': workflow,
                     'nodes': {run_id: {'status': 'pending'} for run_id in graph}}

        def persist():
            """先写本地状态，再由回调持久化。 / Write local state, then invoke durable persistence."""
            write_json(state_path, state, replace=state_path.exists())
            commit()

        selected = [row for row in workflow['nodes'] if row['phase'] == phase]

        def report(reason=None):
            """汇总已核验计数，缺失证明不伪造零风险。 / Sum verified counts without inventing missing risk evidence."""
            nodes = {row['spec']['run_id']: json_copy(state['nodes'][row['spec']['run_id']])
                     for row in selected}
            verified = all(value['status'] == 'verified' for value in nodes.values())
            status = 'cloud_verified' if verified and reason is None else 'blocked'
            state.update(status=status, phase=phase)
            persist()
            return {'status': status,
                    'experiment_id': workflow['experiment_id'], 'phase': phase, 'nodes': nodes,
                    'actual_interactions': sum(value['proof']['actual_interactions']
                                               for value in nodes.values() if value['status'] == 'verified'),
                    'failures': {key: value.get('error') for key, value in nodes.items()
                                 if value['status'] not in ('pending', 'computed', 'verified')} or None,
                    'reason': reason, 'risk_measurement': 'predicted_risk', 'formal_acceptance': False}

        state.update(status='running', phase=phase)
        persist()
        for row in workflow['nodes']:
            if PHASES.index(row['phase']) < PHASES.index(phase):
                if state['nodes'][row['spec']['run_id']]['status'] != 'verified':
                    return report('Earlier phase is not fully verified')
        deadline = time.monotonic() + len(selected) * (NODE_SECONDS + VERIFY_SECONDS) + RESERVE_SECONDS
        for row in selected:
            run_id = row['spec']['run_id']
            node = state['nodes'][run_id]
            if node['status'] == 'verified':
                _proof(node['proof'], run_id)
                continue
            if node['status'] not in ('pending', 'computed', 'verification_failed'):
                return report('Compute outcome unknown or failed; explicit reconciliation required')
            if any(state['nodes'][dep]['status'] != 'verified' for dep in row['dependencies']):
                return report('Node dependency is not verified')
            actions = ['compute', 'verify'] if node['status'] == 'pending' else ['verify']
            # pending 才允许计算；computed/verification_failed 只重做不耗交互的核验。
            # Only pending nodes compute; computed/verification_failed nodes verify only.
            for action in actions:
                remaining = deadline - time.monotonic() - RESERVE_SECONDS
                if remaining <= 0:
                    return report('Phase deadline exhausted')
                if action == 'compute':
                    node['status'] = 'running'
                    # 先提交意图再启动，断线后才能区分“未开始”与“结果未知”。
                    # Persist intent first to distinguish unstarted from unknown outcomes.
                    persist()
                command = [sys.executable, '-m', 'ablations.workflow', '--action', action,
                           '--workspace', str(root), '--run-id', run_id]
                log = _check_local(root / 'workflow-logs' / f'{run_id}-{action}-{uuid.uuid4().hex}.log')
                try:
                    record = worker(command, min(NODE_SECONDS if action == 'compute' else VERIFY_SECONDS,
                                                 remaining), log)
                    if (record.get('exit_code') != 0 or record.get('timed_out') is not False
                            or record.get('process_tree_clean') is not True):
                        raise RuntimeError('Bounded stage did not exit cleanly')
                    if action == 'compute':
                        node['result'] = read_json(_check_local(root / 'runs' / run_id / 'summary.json'))
                        node['status'] = 'computed'
                    else:
                        node['proof'] = _proof(read_json(_check_local(root / 'runs' / run_id / 'verification.json')), run_id)
                        node['status'] = 'verified'
                    node.pop('error', None)
                    node[action + '_process'] = record
                except Exception as error:
                    node['status'] = 'failed' if action == 'compute' else 'verification_failed'
                    node['error'] = f'{type(error).__name__}: {error}'
                    persist()
                    return report(node['error'])
                # 计算成功独立落盘；核验失败不得让训练被自动重复。
                # Persist computation separately so failed verification never repeats training.
                persist()
        return report()
    finally:
        lock.rmdir()


def main(argv=None) -> int:
    """执行一次有界子进程动作，不调度、不重试、不提交云任务。
    One bounded subprocess action; no scheduling, retries or cloud submission.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['compute', 'verify'], required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    root = _check_local(args.workspace).resolve(strict=True)
    workflow = read_json(_check_local(root / 'workflow.json'))
    graph = _validate_workflow(workflow)
    _workspace(workflow, root)
    spec = RunSpec(**graph[safe_run_id(args.run_id)]['spec'])
    state = read_json(_check_local(root / 'workflow-state.json'))
    if state.get('workflow') != workflow or state.get('experiment_id') != workflow['experiment_id']:
        raise ValueError('Worker state snapshot mismatch')
    node = state['nodes'][spec.run_id]
    allowed = ('running',) if args.action == 'compute' else ('computed', 'verification_failed')
    if node['status'] not in allowed:
        raise ValueError('Worker action is not durably scheduled')
    if any(state['nodes'][dep]['status'] != 'verified' for dep in graph[spec.run_id]['dependencies']):
        raise ValueError('Worker dependency is not verified')
    from ablations.execution import execute_stage, verify_stage

    if args.action == 'compute':
        execute_stage(spec, root)
    else:
        result = read_json(_check_local(root / 'runs' / spec.run_id / 'summary.json'))
        if result != node.get('result'):
            raise ValueError('Computed summary changed before verification')
        proof = _proof(verify_stage(spec, root, result), spec.run_id)
        target = _check_local(root / 'runs' / spec.run_id / 'verification.json')
        if target.exists():
            if read_json(target) != proof:
                raise ValueError('Immutable verification proof changed')
        else:
            write_json(target, proof)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
