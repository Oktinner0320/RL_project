"""Notebook 配置快照与源码身份。 / Notebook snapshots and source identity."""

import json
import math
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


_SECTIONS = frozenset('experiment data model environment resources reward baseline risk rl attack audit execution seeds modal variants tuning analysis workflow reference_benchmarks components comparisons native_acceptance'.split())
_CORE_FIELDS = {
    'experiment': {'tag': str, 'variant': str},
    'data': {'url': str, 'validation_fraction': float, 'split_seed': int, 'partition_seed': int, 'num_clients': int, 'partition': str, 'label_concentration': float},
    'model': {'hidden_sizes': list, 'activation': str, 'output_size': int, 'loss': str, 'learning_rate': float, 'optimizer': str, 'device': str, 'dtype': str},
    'environment': {'client_sets': list, 'batch_sizes': list, 'max_rounds': int, 'time_budget_seconds': float, 'sample_budget': int, 'target_validation_loss': float, 'utility_prior': list, 'utility_ema_decay': float},
    'resources': {'interval_seconds': float, 'bandwidth_mbps': list, 'download_mbps': list, 'samples_per_second': list, 'compute_watts': list, 'transmit_watts': list, 'receive_watts': list, 'bandwidth_jitter': float, 'load_range': list, 'aggregation_seconds': float, 'server_watts': float},
    'reward': {'protocol': str, 'accuracy_weight': float, 'time_weight': float, 'energy_weight': float, 'risk_weight': float, 'uncertainty_weight': float, 'time_scale': float, 'energy_scale': float},
    'baseline': {'fixed_action': int, 'round_robin_actions': list},
    'risk': {'mode': str, 'ensemble_size': int, 'hidden_sizes': list, 'learning_rate': float, 'epochs': int, 'batch_size': int, 'device': str, 'validation_group_fraction': float, 'validation_fraction': float},
    'rl': {'algorithm': str, 'device': str, 'hidden_sizes': list, 'total_steps': int, 'learning_rate': float, 'gamma': float, 'batch_size': int, 'buffer_size': int, 'learning_starts': int, 'train_frequency': int, 'target_update_interval': int, 'epsilon_start': float, 'epsilon_end': float, 'epsilon_decay_steps': int, 'checkpoint_interval': int, 'ppo': dict},
    'attack': {'method': str, 'iterations': int, 'restarts': int, 'learning_rate': float, 'labels_known': bool, 'device': str},
    'audit': {'policy': str, 'max_calls': int, 'max_seconds': float, 'period': int, 'device': str},
    'execution': {'split': str, 'save_updates': bool, 'warmup_actions': list},
    'modal': {'app_name': str, 'volume_name': str, 'gpu': str, 'cpu': float, 'memory_mb': int, 'timeout_seconds': int, 'max_containers': int, 'retries': int},
    'workflow': {'reuse_completed': bool, 'jobs': dict, 'training_pipeline': dict},
    'components': {'backend': str, 'risk': str, 'attack': str},
}


def _config_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate configuration key: {key}')
        result[key] = value
    return result


def _config_constant(value):
    raise ValueError(f'Nonfinite configuration number: {value}')


def _validate_config(config):
    """Validate inputs, not execution authorization; admission remains separate."""
    if type(config) is not dict or set(config) != _SECTIONS:
        raise ValueError('Configuration requires the complete known top-level schema')
    for section, obj in config.items():
        if type(obj) is not dict:
            raise ValueError(f'{section} must be an object')
        for key, kind in _CORE_FIELDS.get(section, {}).items():
            value = obj.get(key)
            valid = type(value) in (int, float) if kind is float else type(value) is kind
            if not valid:
                raise ValueError(f'{section}.{key} requires {kind.__name__}')

    positive_ints = {'num_clients', 'output_size', 'max_rounds', 'sample_budget', 'ensemble_size', 'epochs', 'batch_size', 'total_steps', 'buffer_size', 'train_frequency', 'target_update_interval', 'epsilon_decay_steps', 'checkpoint_interval', 'rollout_steps', 'ppo_epochs', 'iterations', 'restarts', 'period', 'memory_mb', 'max_containers', 'interaction_limit', 'group_limit', 'max_tuning_configs', 'required_paired_seeds'}
    nonnegative_ints = {'retries', 'learning_starts', 'max_calls', 'new_interactions', 'new_fits', 'new_cloud_calls', 'new_cloud_call_limit', 'new_real_interaction_limit'}
    probabilities = {'gamma', 'gae_lambda', 'epsilon_start', 'epsilon_end', 'utility_ema_decay', 'bandwidth_jitter', 'dropout', 'clip_range', 'clip_ratio', 'ratio', 'calibration_strength'}
    fractions = {'validation_fraction', 'validation_group_fraction'}
    positive_numbers = {'learning_rate', 'label_concentration', 'time_scale', 'energy_scale', 'time_budget_seconds', 'interval_seconds', 'max_grad_norm', 'dual_step_size', 'cost_multiplier', 'target_multiplier'}
    bool_keys = {'enabled', 'labels_known', 'save_updates', 'double_dqn', 'normalize_advantage', 'mean_calibration', 'calibration_nonnegative', 'require_complete_audit', 'paid_stage_approved', 'paid_execution_approved', 'official_test_allowed', 'official_test_access', 'formal_confirmation', 'automatic_retry', 'unknown_consumption_is_zero', 'historical_consumption_retained'}

    def visit(value, path):
        if type(value) is dict:
            for key, child in value.items():
                location = f'{path}.{key}'
                if type(child) is float and not math.isfinite(child):
                    raise ValueError(f'{location} must be finite')
                if key in positive_ints or key in nonnegative_ints:
                    if child is not None and (type(child) is not int or child < (1 if key in positive_ints else 0)):
                        raise ValueError(f'{location} requires a bounded integer')
                if key in probabilities | fractions | positive_numbers and not (path == 'config.tuning' and type(child) is list):
                    if child is not None:
                        if type(child) not in (int, float) or not math.isfinite(child):
                            raise ValueError(f'{location} requires a finite number')
                        if key in probabilities and not 0 <= child <= 1:
                            raise ValueError(f'{location} must be in [0, 1]')
                        if key in fractions and not 0 < child < 1:
                            raise ValueError(f'{location} must be in (0, 1)')
                        if key in positive_numbers and child <= 0:
                            raise ValueError(f'{location} must be positive')
                if key in bool_keys and type(child) is not bool and not (key == 'enabled' and type(child) is dict):
                    raise ValueError(f'{location} requires bool')
                if key == 'hidden_sizes' and (type(child) is not list or not child or any(type(x) is not int or x <= 0 for x in child)):
                    raise ValueError(f'{location} requires positive integer widths')
                if (key.endswith('_seed') or key == 'seed') and child is not None and type(child) is not int:
                    raise ValueError(f'{location} requires an integer seed')
                visit(child, location)
        elif type(value) is list:
            for i, child in enumerate(value):
                visit(child, f'{path}[{i}]')
        elif type(value) not in (str, int, float, bool, type(None)) or (type(value) is float and not math.isfinite(value)):
            raise ValueError(f'{path} is not strict JSON')

    visit(config, 'config')
    for group, seeds in config['seeds'].items():
        if type(seeds) is not list or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f'seeds.{group} requires distinct integer seeds')
    if config['rl']['algorithm'] not in ('dqn', 'ppo'):
        raise ValueError('rl.algorithm must be dqn or ppo')
    if config['data']['partition'] not in ('iid', 'label_skew'):
        raise ValueError('Unsupported data.partition')
    if config['execution']['split'] not in ('train', 'validation', 'test'):
        raise ValueError('Unsupported execution.split')
    if config['risk']['mode'] not in ('unavailable', 'fitted', 'learned'):
        raise ValueError('Unsupported risk.mode')
    clients = config['data']['num_clients']
    env = config['environment']
    sets, batches = env['client_sets'], env['batch_sizes']
    if not sets or any(type(row) is not list or not row or any(type(i) is not int or not 0 <= i < clients for i in row) or len(set(row)) != len(row) for row in sets):
        raise ValueError('environment.client_sets requires valid distinct client IDs')
    if not batches or any(type(n) is not int or n <= 0 for n in batches) or len(set(batches)) != len(batches):
        raise ValueError('environment.batch_sizes requires distinct positive integers')
    actions = len(sets) * len(batches)
    if len(env['utility_prior']) != actions or any(type(x) not in (int, float) for x in env['utility_prior']):
        raise ValueError('environment.utility_prior must match the action space')
    for key in ('bandwidth_mbps', 'download_mbps', 'samples_per_second', 'compute_watts', 'transmit_watts', 'receive_watts'):
        values = config['resources'][key]
        if len(values) != clients or any(type(x) not in (int, float) or x <= 0 for x in values):
            raise ValueError(f'resources.{key} must match clients with positive values')
    for values in (config['execution']['warmup_actions'], config['baseline']['round_robin_actions'], [config['baseline']['fixed_action']]):
        if not values or any(type(x) is not int or not 0 <= x < actions for x in values):
            raise ValueError('Action IDs must be within the configured action space')
    for name, job in config['workflow']['jobs'].items():
        if type(job) is not dict or type(job.get('enabled')) is not bool or job.get('backend') not in ('local', 'modal') or job.get('seed_group') not in config['seeds']:
            raise ValueError(f'Invalid workflow job: {name}')
        if type(job.get('variants')) is not list or any(x not in config['variants'] for x in job['variants']):
            raise ValueError(f'Invalid workflow variants: {name}')


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> dict:
    """Read the sole editable JSON config without imports, jobs, or authorization.

    Relative paths are resolved against root (the working directory by default).
    Existing RunSpec/evidence payloads remain readable without this new schema.
    """
    base = Path.cwd() if root is None else Path(root)
    target = Path('config/experiment.json') if path is None else Path(path)
    if not target.is_absolute():
        target = base / target
    config = json.loads(target.read_text(encoding='utf-8'), object_pairs_hook=_config_pairs, parse_constant=_config_constant)
    _validate_config(config)
    return config


@dataclass(frozen=True)
class RunSpec:
    """运行的完整输入；没有隐含实验默认值。 / Complete run input with no hidden defaults."""

    run_id: str
    stage: str
    method: str
    seed: int
    risk_version: str
    config: dict
    inputs: dict
    provenance: dict


def json_copy(value):
    """断开 Notebook 可变对象引用并检查可序列化性。 / Detach values via strict JSON encoding."""
    from .comparisons.acceptance import _first_json

    _first_json(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def source_identity(project_root: Path) -> dict:
    """Freeze source contents without computing file hashes."""
    # New snapshots bind scientific JSON, not the read-only notebook frontend.
    # Legacy roots retain their original notebook-based source identity.
    config_path = project_root / 'config/experiment.json'
    names = ["pyproject.toml", "uv.lock", ".python-version"]
    names.append('config/experiment.json' if config_path.is_file() else 'experiment.ipynb')
    paths = [project_root / name for name in names]
    paths += sorted((project_root / "src").rglob("*.py"))
    paths += sorted((project_root / "reference_algorithms/provenance").glob("*.json"))
    paths += sorted((project_root / "reference_algorithms/repos").rglob("*.py"))
    contents = {}
    for path in paths:
        # Notebook outputs/execution counts are results, not executable source.
        # Notebook 输出与执行次数是产物，不应改变源码身份。
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            cells = json.loads(content)["cells"]
            content = [[c["cell_type"], c["source"]] for c in cells]
        contents[path.relative_to(project_root).as_posix()] = content
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "identity_kind": "source_contents_no_hash",
        "source_contents": contents,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_status": status.stdout.splitlines(),
    }


def make_run_spec(
    config: dict,
    stage: str,
    method: str,
    seed: int,
    risk_version: str,
    inputs: dict,
    project_root: Path,
    provenance: dict | None = None,
) -> RunSpec:
    """Freeze strict JSON inputs with a non-content-derived run identity."""
    if (type(seed) is not int or any(type(value) is not str or not value
                                    for value in (stage, method, risk_version))
            or type(config) is not dict or type(inputs) is not dict):
        raise ValueError('RunSpec fields require explicit exact types')
    provenance = source_identity(project_root) if provenance is None else json_copy(provenance)
    payload = json_copy(
        {
            "stage": stage,
            "method": method,
            "seed": seed,
            "risk_version": risk_version,
            "config": config,
            "inputs": inputs,
        }
    )
    suffix = uuid.uuid4().hex
    run_id = f"{stage}-{method}-s{seed}-{suffix}"
    return RunSpec(
        run_id, stage, method, seed, risk_version, payload["config"], payload["inputs"], provenance
    )


def spec_to_dict(spec: RunSpec) -> dict:
    """本地与远端使用同一个 JSON 契约。 / Share the same JSON contract locally and remotely."""
    return json_copy(asdict(spec))
