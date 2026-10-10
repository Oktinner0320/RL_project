"""显式限定执行范围，并用只追加账目保留历史消耗。
Explicit scoped admission and conservative, append-only execution accounting.

本模块不提交计算；失败或结果未知时不退还保留额度。
No operation submits compute; reservations survive failures and unknown outcomes.
证据由调用方明确提供，不通过无界遍历旧产物推断授权。
Evidence is supplied by the caller, never inferred by crawling legacy outputs.
"""

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

from cloud_edge_rl import transport


def _integer(value, low=0, high=None):
    """严格检查整数范围，布尔值不能冒充计数。 / Check exact integers; reject booleans as counters."""
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError('Invalid exact integer')
    return value


def phase_jobs(workflow: dict, phase: str) -> list:
    """核对固定执行图，返回按臂、种子和划分聚合的阶段真实交互额度。
    Validate the fixed graph and return aggregated physical phase counters.

    完整图仍是 18 个节点；复制阶段的 500/700 仅在账目视图中累加。
    Full views retain their original 18 nodes; replication's 500/700 stages are
    summed only in this accounting projection, never in the executable graph.
    """
    if phase not in ('pilot', 'continue', 'replicate'):
        raise ValueError('Unknown phase')
    _integer(workflow['full_workflow_jobs'], 18, 18)
    expected = []
    for index, seed in enumerate((96101, 96102, 96103)):
        for arm in ('N1', 'N3'):
            for partition, interactions in (('train', 500), ('train', 700), ('validation', 30)):
                node_phase = ('pilot' if interactions == 500 else 'continue') if index == 0 else 'replicate'
                expected.append({'phase': node_phase, 'arm': arm,
                                 'seed': seed if partition == 'train' else 96201 + index,
                                 'partition': partition, 'interactions': interactions})
    jobs = workflow['jobs']
    if type(jobs) is not list or not jobs:
        raise ValueError('Explicit jobs required')
    tagged = any('phase' in job for job in jobs)
    selected = jobs
    if tagged:
        for job in jobs:
            _integer(job['seed'])
            _integer(job['interactions'], 1)
        def canonical(rows):
            return sorted(json.dumps(row, sort_keys=True) for row in rows)
        if canonical(jobs) != canonical(expected):
            raise ValueError('Full workflow differs from exact registered 18-node stage increments')
        selected = [job for job in jobs if job['phase'] == phase]
    totals = {}
    for job in selected:
        _integer(job['seed'])
        _integer(job['interactions'], 1)
        key = (job['arm'], job['seed'], job['partition'])
        totals[key] = totals.get(key, 0) + job['interactions']
    required = {}
    for job in expected:
        if job['phase'] == phase:
            key = (job['arm'], job['seed'], job['partition'])
            required[key] = required.get(key, 0) + job['interactions']
    if totals != required or (not tagged and len(jobs) != len(required)):
        raise ValueError('Phase counters differ from registered physical scope')
    return [{'arm': arm, 'seed': seed, 'partition': partition, 'interactions': total}
            for (arm, seed, partition), total in sorted(totals.items())]


def _read_descriptor(value):
    """选择并核对实际 JSON 对象，兼容数组账目及池化运行配置。
    Select and validate an actual JSON object, including array-root ledgers.

    path/pointer 或 graph_run_id 必须指向原证据，再逐字段核对声明。
    Resolve path/pointer or graph_run_id against original evidence before checking
    claimed fields; a copied claim alone is not an original producer witness.
    """
    if type(value) is not dict or type(value.get('fields')) is not dict or not value['fields']:
        raise ValueError('Evidence needs a path and structural fields')
    path = Path(value['path']).resolve(strict=True)
    actual = transport.read_json(path)
    if 'graph_run_id' in value:
        # 原图用内容池保存 RunSpec；解码绑定的原生产者，不相信自行复制的声明。
        # Expand the bound producer from the original RunSpec pool, not a copied claim.
        from cloud_edge_rl.experiment_graph import expand
        run_id = value['graph_run_id']
        if (type(run_id) is not str or not run_id or value.get('pointer')
                or type(actual) is not dict or actual.get('schema') != 'experiment_graph_v1'):
            raise ValueError('Invalid original graph evidence selection')
        matches = [node for node in actual['nodes'] if node['run_id'] == run_id]
        if len(matches) != 1:
            raise ValueError('Original graph run must occur exactly once')
        actual = expand(actual['pool'], matches[0]['spec_ref'])
        if actual.get('run_id') != run_id:
            raise ValueError('Original graph run/spec binding mismatch')
    pointer = value.get('pointer', [])
    if type(pointer) is not list or len(pointer) > 16:
        raise ValueError('Evidence pointer must be a list of at most 16 tokens')
    for token in pointer:
        if type(token) is str and type(actual) is dict and token in actual:
            actual = actual[token]
        elif type(token) is int and type(actual) is list and 0 <= token < len(actual):
            actual = actual[token]
        else:
            raise ValueError('Invalid evidence pointer token or missing selected object')
    if type(actual) is not dict or any(type(k) is not str or k not in actual or json.dumps(actual[k], sort_keys=True, allow_nan=False) != json.dumps(v, sort_keys=True, allow_nan=False) for k, v in value['fields'].items()):
        raise ValueError('Evidence structural fields do not match source')
    return str(path), actual


def _descriptor(value):
    """验证来源描述并返回真实路径。 / Validate an evidence descriptor and return its source path."""
    return _read_descriptor(value)[0]


def _location(preparation):
    """将准备身份绑定到共享只追加账目。 / Bind a preparation identity to the shared append-only ledger."""
    path = Path(preparation).absolute()
    if path != path.resolve(strict=True) or path.parent.name != 'output' or path.parent.parent.name != 'ablations':
        raise ValueError('Preparation must be a real ablations/output child')
    identity = transport.read_json(path / 'preparation.json')['experiment_id']
    if type(identity) is not str or not identity:
        raise ValueError('Missing preparation identity')
    return path.parent / 'execution-ledger.jsonl', identity


@contextmanager
def _lock(ledger):
    """串行化账目更新，遗留锁不自动清理。 / Serialize ledger changes; never auto-clear stale locks."""
    # 旧传输版本没有 submission_lock；退回目录锁，遗留锁必须显式核查。
    # Fall back to a directory guard on older transport; stale guards need reconciliation.
    shared = getattr(transport, 'submission_lock', None)
    if shared is not None:
        with shared(ledger):
            yield
    else:
        guard = ledger.with_name('.' + ledger.name + '.submission-lock')
        guard.mkdir()
        try:
            yield
        finally:
            guard.rmdir()


def _rows(ledger):
    """严格读取原始事件，不把缺失或坏行解释为已退款。 / Read strict events without inventing refunds."""
    if ledger.is_symlink():
        raise ValueError('Refuse symbolic ledger')
    if not ledger.exists():
        return []
    return [transport._decode_json(line) for line in ledger.read_text(encoding='utf-8').splitlines()]


def _append(ledger, rows, row):
    """追加并落盘一个事件，不覆盖历史行。 / Append and fsync one event without rewriting history."""
    # 写入前验证序列化；损坏的部分写入会阻断后续执行，而不是被跳过。
    # Validate serialization first; partial writes remain fail-closed rather than skipped.
    text = json.dumps(row, allow_nan=False, sort_keys=True) + '\n'
    with ledger.open('a', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    return row


def _event(preparation, kind, phase, payload, validate):
    """在锁内核验并追加；重复事件只接受完全相同内容。
    Validate and append under lock; repeated events must have identical content.
    """
    ledger, identity = _location(preparation)
    row = {'schema': 'ablation_execution_v1', 'experiment_id': identity,
           'kind': kind, 'phase': phase, 'payload': payload}
    with _lock(ledger):
        rows = _rows(ledger)
        old = [r for r in rows if (r['experiment_id'], r['kind'], r['phase']) == (identity, kind, phase)]
        if old:
            if json.dumps(old, sort_keys=True) != json.dumps([row], sort_keys=True):
                raise ValueError('Immutable admission event differs')
            return old[0]
        validate(rows, identity)
        return _append(ledger, rows, row)


def _one(rows, identity, kind, phase):
    """要求唯一前序事件，拒绝缺失或歧义授权。 / Require one prior event; reject missing or ambiguous approval."""
    found = [r['payload'] for r in rows if (r['experiment_id'], r['kind'], r['phase']) == (identity, kind, phase)]
    if len(found) != 1:
        raise ValueError('Required prior admission event missing')
    return found[0]


def authorize_pilot(preparation: Path, *, evidence: dict) -> dict:
    """只登记 m14610 的 N1/N3 各 500 步预检，不授权后续扩样。
    Register only the m14610 N1/N3 500+500 pilot, not future execution.

    新种子、一个 N3 配置名额、采集分摊和资金范围必须绑定实际来源；
    历史未知费用与交互仍保留未知，不能因新增实验名称而清零。
    Bind new seeds, one N3 slot, collection allocation, and funding scope to actual
    sources; a new experiment name never resets unknown historical consumption.
    """
    def validate(rows, identity):
        if evidence.get('request_ref') != 'm14610':
            raise ValueError('Explicit pilot request required')
        history = evidence['history']
        paths = [_descriptor(source) for source in history['sources']]
        if not paths or history.get('global_cost_usd') is not None:
            raise ValueError('Historical global cost must remain unknown')
        _integer(history['risk_collection_interactions'], 90, 90)
        for group in history['groups']:
            _integer(group['seed'])
            if group['interactions'] is not None:
                _integer(group['interactions'])
            if group['partition'] not in ('train', 'validation') or group['method'] != 'dqn':
                raise ValueError('Invalid historical group')
            original = group['original_source']
            _, fields = _read_descriptor(original)
            _integer(fields.get('seed'))
            if fields.get('seed') != group['seed'] or fields.get('method') != group['method'] or not isinstance(fields.get('run_id'), str) or not fields['run_id']:
                raise ValueError('Historical group must bind original run identity/method/seed')
            if group['interactions'] is not None:
                counter = group['counter_source']
                _, values = _read_descriptor(counter)
                count = values.get('actual_interactions', values.get('interactions', values.get('environment_steps')))
                _integer(count)
                if count != group['interactions']:
                    raise ValueError('Historical counters must bind actual source values')
        scope = evidence['seed_scope']
        if scope['training_seeds'] != [96101, 96102, 96103] or scope['validation_seeds'] != [96201, 96202, 96203]:
            raise ValueError('Only registered paired seeds allowed')
        for seed in scope['training_seeds'] + scope['validation_seeds']:
            _integer(seed)
        if set(scope['inspected_sources']) != set(paths):
            raise ValueError('Unused seed scope must bind inspected history sources')
        closed = scope['closed_scope']
        _descriptor(closed)
        if closed['fields'] != {'sources': paths, 'groups': history['groups'],
                                'training_seeds': scope['training_seeds'],
                                'validation_seeds': scope['validation_seeds']}:
            raise ValueError('Unused seed evidence must bind complete closed history scope')
        parsed = []
        for source in history['sources']:
            _, actual = _read_descriptor(source)
            if 'groups' not in actual or not isinstance(actual['groups'], list):
                raise ValueError('History sources need parsed manifest interaction groups')
            parsed.extend(actual['groups'])
        if parsed != history['groups']:
            raise ValueError('Historical groups differ from parsed source manifests')
        if any(g['seed'] in scope['training_seeds'] + scope['validation_seeds'] for g in history['groups']):
            raise ValueError('Registered seeds already used historically')
        slots = evidence['configuration_slots']
        _descriptor(slots['source'])
        permission = {'additional': 1, 'candidate': 'N3', 'control': 'N1', 'control_is_existing': True}
        if {k: v for k, v in slots.items() if k != 'source'} != permission or slots['source']['fields'] != permission:
            raise ValueError('Exactly one new N3 slot with existing N1 control required')
        _integer(slots['additional'], 1, 1)
        if slots['control_is_existing'] is not True:
            raise ValueError('Existing control must be explicit')
        funding = evidence['funding_authorization']
        if (type(funding.get('request_ref')) is not str or not funding['request_ref'] or funding.get('scope') != 'pilot'
                or funding.get('reservation_policy') != 'estimated_not_bill'):
            raise ValueError('Explicit funding authorization required')
        _descriptor(funding['source'])
        if funding['source']['fields'] != {k: v for k, v in funding.items() if k != 'source'}:
            raise ValueError('Funding must bind actual current scoped policy evidence')
        removal = funding['money_cap_removal']
        if (type(removal.get('request_ref')) is not str or not removal['request_ref']
                or type(removal.get('context')) is not str or not removal['context']):
            raise ValueError('Historical money-cap removal context required')
        limits = evidence['cloud_limits']
        expected = {'gpu': 'H100', 'cpu': 16, 'memory': 65536, 'max_containers': 1, 'retries': 0}
        if any(type(limits.get(k)) is not type(v) or limits[k] != v for k, v in expected.items()):
            raise ValueError('Cloud resource limits differ')
        _integer(limits['deadline_seconds'], 4800, 4800)
        for row in rows:
            if row['kind'] == 'authorize' and row['phase'] == 'pilot':
                other = row['payload']['seed_scope']
                if set(other['training_seeds'] + other['validation_seeds']) & set(scope['training_seeds'] + scope['validation_seeds']):
                    raise ValueError('Seeds already admitted in another preparation')
    return _event(preparation, 'authorize', 'pilot', evidence, validate)


def authorize_phase(preparation: Path, phase: str, *, evidence: dict, workflow: dict) -> dict:
    """前序结果核验后，由调用方显式批准准确的下一阶段范围。
    Caller explicitly approves exact future scope after verified prior results.
    """
    if phase not in ('continue', 'replicate'):
        raise ValueError('Unknown subsequent phase')
    def validate(rows, identity):
        prior = _one(rows, identity, 'result', 'pilot' if phase == 'continue' else 'continue')
        if prior.get('state') != 'verified' or evidence.get('prior_result') != prior:
            raise ValueError('Verified exact prior result required')
        if not evidence.get('request_ref') or evidence.get('scope') != phase:
            raise ValueError('Explicit exact phase approval required')
        _descriptor(evidence['decision_source'])
    return _event(preparation, 'authorize', phase, {'evidence': evidence, 'workflow': workflow}, validate)


def reserve_phase(preparation: Path, phase: str, *, workflow: dict) -> dict:
    """云调用前保留真实交互额度；未知结果不自动退款。
    Reserve physical interactions before any call; unknown results never refund.

    两臂按同一方法/种子/划分合并计数，再计入 90 次共享采集分摊。
    Combine both arms by method/seed/partition, then include the 90-interaction
    shared collection allocation. Phase names never create fresh quota.
    """
    if phase not in ('pilot', 'continue', 'replicate'):
        raise ValueError('Unknown phase')
    original = workflow
    workflow = {**workflow, 'planned_jobs': workflow['jobs'], 'jobs': phase_jobs(workflow, phase)}
    def validate(rows, identity):
        approval = _one(rows, identity, 'authorize', phase)
        pilot = _one(rows, identity, 'authorize', 'pilot')
        if phase != 'pilot' and approval['workflow'] != original:
            raise ValueError('Workflow differs from explicitly approved scope')
        jobs = workflow['jobs']
        _integer(workflow['full_workflow_jobs'], 18, 18)
        if not isinstance(jobs, list) or not jobs:
            raise ValueError('Explicit jobs required')
        if phase == 'pilot' and jobs != [
                {'arm': arm, 'seed': 96101, 'partition': 'train', 'interactions': 500}
                for arm in ('N1', 'N3')]:
            raise ValueError('Pilot must be exactly two arms and 1000 physical interactions')
        totals = {}
        # 累加全部已有保留记录，包括尚未核清的失败；不能只统计成功调用。
        # Sum all prior reservations, including unresolved failures, not just successes.
        for row in rows:
            if row['kind'] == 'reserve':
                for job in row['payload']['jobs']:
                    key = (job['seed'], job['partition'])
                    totals[key] = totals.get(key, 0) + job['interactions']
        seen = set()
        for job in jobs:
            _integer(job['seed'])
            _integer(job['interactions'], 1)
            key = (job['seed'], job['partition'])
            unique = (job['arm'], *key)
            seeds = pilot['seed_scope']['training_seeds' if job['partition'] == 'train' else 'validation_seeds']
            if job['arm'] not in ('N1', 'N3') or job['partition'] not in ('train', 'validation') or job['seed'] not in seeds or unique in seen:
                raise ValueError('Invalid or duplicate job scope')
            seen.add(unique)
            if phase == 'continue' and job['partition'] == 'train' and (job['seed'] != 96101 or job['interactions'] != 700):
                raise ValueError('Continuation is only the remaining 700 interactions')
            if phase == 'replicate' and job['partition'] == 'train' and (job['seed'] == 96101 or job['interactions'] != 1200):
                raise ValueError('Replication is only new paired seeds at 1200')
            totals[key] = totals.get(key, 0) + job['interactions']
            if totals[key] + 90 > 3000:
                raise ValueError('Joint method/seed/partition cap exceeded')
        train = sum(v for (s, p), v in totals.items() if p == 'train')
        evaluation = sum(v for (s, p), v in totals.items() if p == 'validation')
        if train > 7200 or evaluation > 180:
            raise ValueError('Physical workflow maximum exceeded')
        limits = {**pilot['cloud_limits'], 'deadline_seconds': {'pilot': 4800, 'continue': 9000, 'replicate': 27000}[phase]}
        if json.dumps(workflow.get('cloud_limits'), sort_keys=True) != json.dumps(limits, sort_keys=True):
            raise ValueError('Workflow cloud limits differ from admission')
        _integer(workflow['cloud_limits']['deadline_seconds'], limits['deadline_seconds'], limits['deadline_seconds'])
    return _event(preparation, 'reserve', phase, workflow, validate)


def bind_call(preparation: Path, phase: str, *, call_id: str, app_id: str) -> dict:
    """将已保留阶段绑定到唯一云调用和应用。 / Bind a reserved phase to one exact cloud call and app."""
    if any(type(v) is not str or not v.strip() for v in (call_id, app_id)):
        raise ValueError('Exact cloud call and app IDs required')
    def validate(rows, identity):
        _one(rows, identity, 'reserve', phase)
        if any(r['kind'] == 'bind' and r['payload']['call_id'] == call_id for r in rows):
            raise ValueError('Cloud call already bound elsewhere')
    return _event(preparation, 'bind', phase, {'call_id': call_id, 'app_id': app_id}, validate)


def record_phase(preparation: Path, phase: str, result: dict) -> dict:
    """记录实际结果与来源；未知消耗不得填写零。
    Record actual outcomes with evidence; unknown consumption must not become zero.
    """
    def validate(rows, identity):
        reservation = _one(rows, identity, 'reserve', phase)
        binding = _one(rows, identity, 'bind', phase)
        if any(result.get(k) != binding[k] for k in ('call_id', 'app_id')):
            raise ValueError('Result cloud identity mismatch')
        if result.get('state') not in ('verified', 'failed', 'unknown'):
            raise ValueError('Invalid result state')
        cost = result.get('actual_cost_usd')
        if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
            raise ValueError('Invalid observed cloud cost')
        if result['state'] == 'verified':
            _descriptor(result['verification_source'])
            if result.get('jobs') != reservation['jobs']:
                raise ValueError('Verified actual counters must equal reserved jobs including warmup')
            fields = result['verification_source']['fields']
            if any(fields.get(k) != result[k] for k in ('state', 'jobs', 'call_id', 'app_id')):
                raise ValueError('Verification source must bind state, physical counters and cloud identity')
        elif result.get('jobs') is not None or cost == 0:
            raise ValueError('Unknown/failed consumption cannot be reported as zero')
    return _event(preparation, 'result', phase, result, validate)
