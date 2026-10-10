"""有界执行所有者冻结的新开发实验阶段，不生成任务定义或执行许可。
Drive owner-frozen fresh stages within bounds; never generate job definitions or approval.
"""

import argparse
import importlib
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from .admission import check_admission
from .config import RunSpec
from .fresh_benchmark import build_plan
from .parameter_selection_pipeline import load_notebook_config
from .training_pipeline import _controller_lock, run_training_pipeline
from .transport import safe_run_id, write_json

EXPERIMENT = 'fresh-seven-methods-20261010-v1'
PURPOSE = 'fresh_seven_method_development_benchmark_v1'
VERIFIER = 'cloud_edge_rl.fresh_benchmark:verify_fresh_artifacts'


def _strict_json(path, budget):
    """在累计大小和数量预算内读取 JSON，拒绝重复键及非有限数值。
    Read JSON within cumulative size/count limits, rejecting duplicate keys and nonfinite numbers.
    """
    path = Path(path)
    size = path.stat().st_size
    # 单文件最多 64 MiB，累计最多 128 MiB、256 个文件；budget 在多次读取间共享。 / Share limits of 64 MiB per file, 128 MiB total, and 256 files across reads.
    if size > 64 * 1024 * 1024 or budget[0] + size > 128 * 1024 * 1024 or budget[1] >= 256:
        raise ValueError('Fresh stage JSON size/count limit exceeded')
    budget[0] += size
    budget[1] += 1

    def pairs(items):
        """构造 JSON 对象，拒绝重复键而不是静默覆盖。 / Build a JSON object, rejecting duplicate keys instead of overwriting them."""
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def constant(value):
        """拒绝 JSON 中的 NaN 和无穷常量。 / Reject NaN and infinity constants in JSON."""
        raise ValueError('Nonfinite JSON value: ' + value)

    result = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                        parse_constant=constant, parse_float=_finite_float)
    return result


def _finite_float(value):
    """解析浮点数字并拒绝溢出产生的无穷值。 / Parse a float and reject infinite values produced by overflow."""
    import math
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite JSON number')
    return result


def _absolute(value):
    """要求冻结产物路径为绝对路径字符串，并返回解析后的路径。 / Require an absolute frozen artifact path string and return its resolved path."""
    if type(value) is not str or not Path(value).is_absolute():
        raise ValueError('Absolute frozen artifact path required')
    return Path(value).resolve()


def _gate(proof, run_id):
    """核对本实验的运行核验证据，返回非负真实交互数。 / Check this experiment's run verification proof and return nonnegative physical interactions."""
    if (type(proof) is not dict or proof.get('verified') is not True
            or proof.get('experiment_id') != EXPERIMENT or proof.get('run_id') != run_id
            or type(proof.get('physical_interactions')) is not int
            or proof['physical_interactions'] < 0):
        raise ValueError('Fresh semantic gate with physical interaction counter required')
    return proof['physical_interactions']


def _journal_identity(descriptor, directory, specs, *, historical=False, budget=None):
    """将精简日志身份绑定到独占保存的阶段描述副本，返回身份记录。
    Bind a compact journal to an exclusively saved descriptor copy; return its identity record.

    旧版日志保留原身份格式，不创建新版绑定文件。 / Preserve the legacy journal identity without creating a new binding file.
    """
    identity = {'schema': 'fresh_benchmark_journal_v2',
            'descriptor_ref': str(directory / 'stage.json'),
            'freeze_id': descriptor['freeze_id'],
            'spec_refs': list(descriptor['spec_paths']),
            'run_ids': [spec.run_id for spec in specs]}
    if historical:
        return identity
    binding_path = directory / 'stage-binding.json'
    if binding_path.exists():
        binding = _strict_json(binding_path, [0, 0] if budget is None else budget)
    else:
        binding = {'descriptor_id': str(uuid.uuid4()), 'descriptor_snapshot': descriptor}
        # 独占创建避免并发时替换已有身份，随后刷新到磁盘。 / Create exclusively to prevent identity replacement during races, then flush to disk.
        with binding_path.open('x', encoding='utf-8') as stream:
            json.dump(binding, stream, allow_nan=False)
            stream.flush()
            import os
            os.fsync(stream.fileno())
    if (type(binding) is not dict or set(binding) != {'descriptor_id', 'descriptor_snapshot'}
            or binding['descriptor_snapshot'] != descriptor
            or type(binding['descriptor_id']) is not str):
        raise ValueError('Fresh immutable stage descriptor changed')
    uuid.UUID(binding['descriptor_id'])
    identity.update(descriptor_id=binding['descriptor_id'], descriptor_binding_ref=str(binding_path))
    return identity


def run_fresh_stage(config: dict, root: Path, directory: Path, *, max_new_jobs: int,
                    runner=run_training_pipeline) -> dict:
    """串行执行最多三个已获准的冻结任务，不重试失败调用，返回阶段日志。
    Execute at most three exact approved specs serially, without failed-call retry; return the stage journal.

    读取 directory/stage.json；核验证据须包含运行身份及含预热的真实环境调用数，不能用样本处理数替代。
    Read directory/stage.json; proofs must identify the run and count actual environment calls including warmup, not processed samples.
    资源预留、调用绑定、清理和下载由共享控制器负责，本驱动不管理费用台账。
    The shared controller owns reservations, call binding, cleanup, and downloads; this driver owns no cost ledger.
    """
    if type(max_new_jobs) is not int or not 1 <= max_new_jobs <= 3:
        raise ValueError('Explicit max_new_jobs exact integer 1..3 required')
    root, directory = Path(root).resolve(), Path(directory).resolve()
    budget = [0, 0]
    descriptor = _strict_json(directory / 'stage.json', budget)
    # 描述文件必须精确匹配冻结字段和本次任务上限，不能仅靠 approved 标志放行。 / Match frozen descriptor fields and the job limit; approved alone cannot authorize execution.
    fields = {'schema', 'stage_id', 'freeze_id', 'experiment_id', 'spec_paths',
              'max_new_jobs', 'approved', 'purpose', 'dependencies', 'output_root'}
    if (type(descriptor) is not dict or set(descriptor) != fields
            or descriptor['schema'] != 'fresh_benchmark_stage_v1'
            or descriptor['experiment_id'] != EXPERIMENT or descriptor['purpose'] != PURPOSE
            or descriptor['approved'] is not True
            or type(descriptor['max_new_jobs']) is not int
            or descriptor['max_new_jobs'] != max_new_jobs
            or type(descriptor['freeze_id']) is not str or not descriptor['freeze_id']):
        raise ValueError('Exact approved fresh stage descriptor required')
    safe_run_id(descriptor['stage_id'])
    output = _absolute(descriptor['output_root'])
    settings = config.get('native_acceptance', {}).get('fresh_benchmark', {})
    if (settings.get('experiment_id') != EXPERIMENT
            or not directory.is_relative_to(output) or directory == output):
        raise ValueError('Registered fresh experiment and exclusive stage output required')
    build_plan(config)
    paths = descriptor['spec_paths']
    # 只读取输出根目录内已冻结的唯一任务定义，不在此生成新任务。 / Read unique frozen specs inside the output root; do not generate new jobs here.
    if (type(paths) is not list or not paths or len(paths) > max_new_jobs
            or len(set(paths)) != len(paths)):
        raise ValueError('Nonempty unique bounded frozen specs required')
    specs = []
    for path in paths:
        absolute = _absolute(path)
        if not absolute.is_relative_to(output):
            raise ValueError('Specs must belong to fresh output root')
        specs.append(RunSpec(**_strict_json(absolute, budget)))
    if len({spec.run_id for spec in specs}) != len(specs):
        raise ValueError('Duplicate frozen run ID')
    for spec in specs:
        safe_run_id(spec.run_id)
        # 限定登记阶段、单容器和零重试，避免执行范围或并发悄然扩大。 / Restrict registered stages, one container, and zero retries to keep execution bounded.
        if (spec.config.get('experiment', {}).get('tag') != EXPERIMENT
                or spec.config.get('native_acceptance', {}).get('fresh_benchmark') != settings
                or spec.stage not in {'collect', 'benchmark_attack', 'audit', 'fit_risk',
                                      'train_policy', 'validate'}
                or spec.config.get('modal', {}).get('max_containers') != 1
                or type(spec.config['modal']['max_containers']) is not int
                or type(spec.config['modal'].get('retries')) is not int
                or spec.config['modal']['retries'] != 0):
            raise ValueError('Fresh registered scope and single-container no-retry specs required')
        if (spec.config.get('data', {}).get('split_seed') != 8501
                or type(spec.config['data']['split_seed']) is not int
                or spec.config['data'].get('partition_seed') != 8502
                or type(spec.config['data']['partition_seed']) is not int
                or spec.config['modal'].get('startup_timeout_seconds') != 180
                or type(spec.config['modal']['startup_timeout_seconds']) is not int
                or spec.config['modal'].get('volume_name') != settings.get('volume_name')
                or not settings.get('volume_name')):
            raise ValueError('Exact fresh split, partition, startup deadline and volume required')
    dependencies = descriptor['dependencies']
    if type(dependencies) is not list:
        raise ValueError('Explicit declared dependencies required')
    seen = set()
    for dependency in dependencies:
        # 依赖必须属于本实验输出，且核验快照与持久化云端调用身份一致。 / Require dependencies in this output with matching proof snapshots and durable call identities.
        if type(dependency) is not dict or set(dependency) != {'run_id', 'gate_path', 'gate_snapshot'}:
            raise ValueError('Exact dependency snapshot required')
        run_id = safe_run_id(dependency['run_id'])
        if run_id in seen or run_id in {spec.run_id for spec in specs}:
            raise ValueError('Duplicate or self dependency')
        seen.add(run_id)
        path = _absolute(dependency['gate_path'])
        if not path.is_relative_to(output):
            raise ValueError('Historical dependency forbidden')
        proof = _strict_json(path, budget)
        if proof != dependency['gate_snapshot']:
            raise ValueError('Dependency gate snapshot changed')
        _gate(proof, run_id)
        if not all(type(proof.get(key)) is str and proof[key] for key in ('call_id', 'app_id')):
            raise ValueError('Durable dependency call binding required')
        binding_path = _absolute(proof.get('call_binding_path'))
        if not binding_path.is_relative_to(output):
            raise ValueError('Historical dependency call binding forbidden')
        binding = _strict_json(binding_path, budget)
        if any(binding.get(key) != proof[key] for key in ('run_id', 'call_id', 'app_id')):
            raise ValueError('Dependency durable call binding mismatch')
    # 强制检查已批准快照及当前源码，描述文件的 approved 标志不能独立授权启动。 / Check approved snapshots and current source; descriptor.approved alone cannot authorize an app.
    approvals = check_admission(specs, root)
    if any(witness.get('purpose') != PURPOSE or witness.get('freeze_id') != descriptor['freeze_id']
           for witness, *_ in approvals) or len(approvals) != len(specs):
        raise ValueError('Fresh stage admission binding mismatch')
    module, name = VERIFIER.split(':')
    if not callable(getattr(importlib.import_module(module), name, None)):
        raise ValueError('Fresh semantic verifier unavailable; execution blocked')
    with _controller_lock(directory):
        # 锁内核对已有日志身份；新版使用独立绑定，旧版直接比较描述和任务快照。 / Under the lock, check new bindings or compare legacy descriptor/spec snapshots.
        journal = directory / 'fresh-journal.json'
        old = _strict_json(journal, budget) if journal.exists() else None
        historical = old is not None and 'snapshot' in old
        if old is not None and not historical and not (directory / 'stage-binding.json').exists():
            raise ValueError('Missing retained stage descriptor identity')
        identity = _journal_identity(descriptor, directory, specs, historical=historical, budget=budget)
        if journal.exists():
            matches = (old.get('snapshot') == {'descriptor': descriptor,
                       'specs': [asdict(spec) for spec in specs]} if historical else
                       all(old.get(key) == value for key, value in identity.items()))
            if not matches:
                raise ValueError('Fresh stage snapshot changed; never blindly retry')
        if any((directory / item).exists() for item in ('state.json', 'manifest.json', 'jobs')):
            # 不接管旧控制器布局；允许的调用恢复由下方 transport 子目录负责。 / Reject the old controller layout; supported call recovery belongs to transport below.
            raise ValueError('Historical controller phase recovery forbidden')
        state = {**identity, 'status': 'started', 'verification_refs': {}}
        if not historical:
            write_json(journal, state, replace=journal.exists())
        try:
            # 共享控制器核对缓存和准入，重连已知调用；新调用仍须通过资源预留。 / Validate caches/admission, reattach known calls, and reserve resources for new calls.
            result = runner(specs, root, directory / 'transport', verify=VERIFIER,
                            artifact_destination=directory / 'artifacts', local_only=False)
            if result.get('status') != 'completed' or set(result.get('jobs', {})) != {s.run_id for s in specs}:
                raise ValueError('All fresh jobs must complete semantic verification')
            count = 0
            for spec in specs:
                # 只有语义核验通过、清理已确认且调用绑定一致的任务才计入完成。 / Count completion only with verified proof, confirmed cleanup, and matching call binding.
                row = result['jobs'][spec.run_id]
                if row.get('phase') != 'verified' or row.get('shutdown_verified') is not True:
                    raise ValueError('Verified semantic gate and exact cleanup required')
                count += _gate(row.get('verification'), spec.run_id)
                binding = _strict_json(directory / 'transport' / 'jobs' / spec.run_id / 'call.json', budget)
                if (binding.get('run_id') != spec.run_id or not row.get('call_id') or not row.get('app_id')
                        or binding.get('call_id') != row['call_id'] or binding.get('app_id') != row['app_id']):
                    raise ValueError('Durable current-call binding mismatch')
                # 用 JSON 指针引用控制器的权威证据，避免复制另一份可能失同步的结果。 / Reference authoritative controller proof with a JSON pointer, rather than copying it.
                state['verification_refs'][spec.run_id] = (
                    str(directory / 'transport' / 'state.json') +
                    '#/jobs/' + spec.run_id + '/verification')
            state.update(status='completed', physical_interactions=count, actual_cloud_cost=None,
                         controller_ref=str(directory / 'transport' / 'state.json'))
            if not historical:
                write_json(journal, state, replace=True)
            return state
        except BaseException as error:
            # 记录阻断原因后继续抛出，不能把失败当作完成，也不自动重试。 / Record the blocker and re-raise; never mark failure completed or retry automatically.
            state.update(status='blocked', error={'type': type(error).__name__, 'message': str(error)})
            if not historical:
                write_json(journal, state, replace=True)
            raise


def main():
    """读取 Notebook 配置和显式阶段上限，执行已冻结阶段。 / Read notebook configuration and an explicit job limit, then execute the frozen stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--stage-directory', type=Path, required=True)
    parser.add_argument('--max-new-jobs', type=int, required=True)
    args = parser.parse_args()
    config = load_notebook_config(args.root / 'experiment.ipynb')
    run_fresh_stage(config, args.root, args.stage_directory, max_new_jobs=args.max_new_jobs)


if __name__ == '__main__':
    main()
