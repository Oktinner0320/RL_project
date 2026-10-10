"""实验阶段编排；学习逻辑保留给研究者。 / Stage orchestration, not learning implementation."""

import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from . import core
from .artifacts import RunStore
from .audit import read_audits, read_candidates, risk_training_rows, run_audits, save_upload
from .baselines import FixedPolicy, RiskGreedyPolicy, RoundRobinPolicy, TimeGreedyPolicy, run_warmup
from .comparisons.dispatch import (
    make_attack,
    make_backend,
    make_risk,
    policy_snapshot_steps,
    required_sources,
    train_policy,
)
from .comparisons.sources import source_records, source_root
from .config import RunSpec
from .data import prepare_adult
from .environment import FEATURE_NAMES, CloudEdgeEnv


def runtime_info() -> dict:
    """记录实际版本和加速器，不运行训练。 / Report the actual runtime without training."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "gymnasium", "modal", "numpy", "pandas"]
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class RecordingEnv(gym.Wrapper):
    """记录每个真实步骤并隔离审计真值。 / Record real transitions through a trusted side channel."""

    def __init__(self, env: CloudEdgeEnv, store: RunStore, root: Path, data, spec: RunSpec):
        """绑定运行记录器；不改变观测与奖励。 / Bind the recorder without changing observations."""
        super().__init__(env)
        self.store, self.root, self.data, self.run_spec = store, root, data, spec
        self.episode = -1
        self.decision_diagnostics = None

    def set_decision_diagnostics(self, diagnostics: dict):
        self.decision_diagnostics = deepcopy(diagnostics)

    def reset(self, *, seed=None, options=None):
        """新回合只清理环境，不混合日志。 / Reset the environment while retaining the run journal."""
        self.episode += 1
        self.observation, info = self.env.reset(seed=seed, options=options)
        return self.observation, info

    def step(self, action):
        """先固定决策前预测，再执行和保存更新。 / Freeze pre-decision predictions before execution."""
        values = self.observation["candidates"]
        weights = self.unwrapped.config["reward"]
        scores = (
            weights.get('accuracy_weight', 1.) * values[:, 0]
            - weights["time_weight"] * values[:, 1] / weights["time_scale"]
            - weights["energy_weight"] * values[:, 2] / weights["energy_scale"]
            - weights["risk_weight"] * values[:, 3]
        )
        ordered = np.sort(scores)
        margin = float(np.ptp(ordered[-2:]))
        decision_kind = 'heuristic_scores'
        if self.decision_diagnostics is not None:
            margin = self.decision_diagnostics['margin']
            decision_kind = self.decision_diagnostics['kind']
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.store.append_metrics(
            {
                "episode": self.episode,
                **info,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                'decision_kind': decision_kind,
                'decision_margin': margin,
            }
        )
        updates = self.unwrapped.pop_updates()
        if self.run_spec.config["execution"]["save_updates"]:
            for update, features, mean, uncertainty in updates:
                save_upload(
                    self.store,
                    self.root,
                    update,
                    features,
                    mean,
                    uncertainty,
                    margin,
                    self.episode,
                    info["round"],
                    int(action),
                    self.run_spec.config["execution"]["split"],
                    self.data,
                    info["risk_version"],
                    info['is_warmup'],
                    info['reward_protocol'],
                    decision_kind,
                )
        if self.decision_diagnostics is not None:
            with (self.store.path / 'policy_decisions.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'episode': self.episode, 'round': info['round'],
                                         **self.decision_diagnostics}) + '\n')
        self.decision_diagnostics = None
        self.observation = observation
        return observation, reward, terminated, truncated, info

    def snapshot(self) -> dict:
        """同时保存包装器和环境状态。 / Preserve recorder position along with environment state."""
        return {"environment": self.unwrapped.snapshot(), "episode": self.episode}

    def load_snapshot(self, state: dict) -> dict:
        """恢复后第一步使用恢复的观测，不能先重置。 / Resume from restored observations without reset."""
        self.episode = state["episode"]
        self.observation = self.unwrapped.load_snapshot(state["environment"])
        return self.observation


def _make_environment(spec: RunSpec, data, root: Path, store: RunStore) -> RecordingEnv:
    """派生客户端大小，载入显式指定的风险版本。 / Derive sizes and load the requested risk version."""
    config = deepcopy(spec.config)
    config["environment"]["client_sizes"] = [len(rows) for rows in data.client_indices]
    if min(config["environment"]["client_sizes"]) < max(config["environment"]["batch_sizes"]):
        raise ValueError("客户端记录不足以支持动作批量 / A client is smaller than an action batch")
    backend = make_backend(spec, root, data)
    mode = config["risk"]["mode"]
    if mode == "unavailable":
        risk = None
    elif mode == "fitted":
        checkpoint = torch.load(
            root / spec.inputs["risk_checkpoint"], map_location="cpu", weights_only=False
        )
        if checkpoint["risk_version"] != spec.risk_version:
            raise ValueError("风险检查点版本不匹配 / Risk checkpoint version mismatch")
        if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
            raise ValueError("风险特征契约不匹配 / Risk feature schema mismatch")
        if checkpoint["implementation"] != spec.config["components"]["risk"]:
            raise ValueError("风险实现与检查点不匹配 / Risk implementation differs from checkpoint")
        risk = make_risk(spec, root)
        risk.load_checkpoint(checkpoint["state"])
    else:
        raise ValueError(f"未知风险模式 / Unknown risk mode: {mode}")
    return RecordingEnv(CloudEdgeEnv(config, backend, risk), store, root, data, spec)


def _policy(spec: RunSpec, root: Path, actions=None):
    """构造透明基线或载入用户策略。 / Build a transparent baseline or load the user's agent."""
    match spec.method:
        case "fixed":
            return FixedPolicy(spec.config["baseline"]["fixed_action"])
        case "round_robin":
            return RoundRobinPolicy(spec.config["baseline"]["round_robin_actions"])
        case "time_greedy":
            return TimeGreedyPolicy()
        case "risk_greedy":
            return RiskGreedyPolicy(spec.config["reward"])
        case "metacs":
            from .comparisons.scheduling import MetaCSPolicy

            return MetaCSPolicy(
                actions,
                {
                    **spec.config["comparisons"]["metacs"],
                    "risk_weight": spec.config["reward"]["risk_weight"],
                },
                source_root(spec.config, root),
            )
        case "pqn":
            from .comparisons.dispatch import load_policy

            state = torch.load(
                root / spec.inputs["agent_checkpoint"], map_location="cpu", weights_only=False
            )
            return load_policy(spec, state["agent"])
        case "dqn" | "ppo":
            state = torch.load(
                root / spec.inputs["agent_checkpoint"], map_location="cpu", weights_only=False
            )
            return core.load_agent(
                state["agent"],
                {
                    **spec.config["rl"],
                    "algorithm": spec.method,
                    "seed": spec.seed,
                    "risk_version": spec.risk_version,
                },
            )
        case _:
            raise ValueError(f"未知调度器 / Unknown policy: {spec.method}")


def _run_episode(env: RecordingEnv, policy, seed: int) -> dict:
    """仅评估动作；不替用户更新策略。 / Evaluate decisions without implementing learning."""
    observation, info = env.reset(seed=seed)
    action_ids = env.run_spec.config["execution"]["warmup_actions"]
    observation, warmup_info, terminated, truncated, total_reward = run_warmup(
        env, observation, action_ids
    )
    if warmup_info:
        info = warmup_info
    if getattr(env.unwrapped, 'protocol', 'legacy_v1') == 'exposure_v1':
        total_reward = 0.
    finished = terminated or truncated
    while not finished:
        action = policy.act(observation, explore=False)
        if hasattr(policy, 'last_diagnostics'):
            env.set_decision_diagnostics(policy.last_diagnostics)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        finished = terminated or truncated
    return {**info, "total_reward": total_reward}


def _include_official_test(spec: RunSpec) -> bool:
    """Development never touches official test; test access needs a verified formal freeze."""
    if spec.stage == 'runtime_check':
        return False
    split = spec.config['execution']['split']
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Explicit train/validation/test split required')
    if spec.stage == 'train_policy' and split != 'train':
        raise ValueError('Policy training requires the training split')
    if spec.stage == 'validate' and split != 'validation':
        raise ValueError('Tuning evaluation requires validation runs')
    if spec.stage == 'evaluate' and split != 'test':
        raise ValueError('Final evaluation requires test runs')
    if split != 'test':
        return False
    from .comparisons.acceptance import _first_artifact, _first_json_equal, _first_time

    confirmation = spec.config['native_acceptance']['confirmation']
    if (confirmation.get('frozen') is not True
            or type(confirmation.get('freeze_id')) is not str
            or not confirmation['freeze_id'].strip()):
        raise ValueError('Official test requires frozen formal confirmation')
    freeze = _first_artifact(confirmation['freeze_manifest'])
    for key, value in confirmation.items():
        if key not in ('freeze_manifest', 'evidence_manifest'):
            if key not in freeze or not _first_json_equal(freeze[key], value):
                raise ValueError(f'Frozen confirmation mismatch: {key}')
    if _first_time(freeze['registered_at']) > datetime.now(timezone.utc):
        raise ValueError('Formal confirmation registration is in the future')
    return True


def run_experiment(spec: RunSpec, root: Path) -> dict:
    """统一的本地/Modal入口；所有输入来自 Notebook 快照。 / One local/Modal entry point."""
    root = Path(root)
    if spec.stage == 'train_policy':
        policy_snapshot_steps(spec)
    include_official_test = _include_official_test(spec)
    with RunStore(root, spec) as store:
        store.write_json("runtime.json", runtime_info())
        sources = required_sources(spec.config, spec.method, spec.stage)
        if sources:
            store.write_json(
                "reference_sources.json", source_records(source_root(spec.config, root), sources)
            )
        if spec.stage == "runtime_check":
            result = {"run_id": spec.run_id, "stage": spec.stage, **runtime_info()}
        else:
            data = prepare_adult(spec.config["data"], root / "data",
                                 include_official_test=include_official_test)
            store.write_json("data_metadata.json", data.metadata)
            if spec.stage == "native_acceptance_backend":
                from .comparisons.native_acceptance import backend_gate

                result = backend_gate(data, spec.config, source_root(spec.config, root), store)
            elif spec.stage == "native_acceptance_sample":
                from .comparisons.native_acceptance import sample_gate

                result = sample_gate(data, spec.config, source_root(spec.config, root), store)
            elif spec.stage == "prepare_data":
                result = {
                    "train_rows": len(data.train_y),
                    "validation_rows": len(data.validation_y),
                    "test_rows": len(data.test_y),
                    "encoded_features": data.train_x.shape[1],
                    "client_sizes": [len(rows) for rows in data.client_indices],
                }
            elif spec.stage in ("audit", "evaluate_audits", "benchmark_attack"):
                candidates = read_candidates(root, spec.inputs["source_run_ids"])
                if "target_ids" in spec.inputs:
                    # 显式分片，不改变单目标攻击随机性。 / Explicit shards keep per-target randomness unchanged.
                    target_ids = spec.inputs["target_ids"]
                    if not isinstance(target_ids, (list, tuple)):
                        raise ValueError("Audit target IDs must be nonempty unique (run_id, upload_id) pairs")
                    if any(isinstance(target, str) for target in target_ids):
                        fresh = spec.config.get('native_acceptance', {}).get('fresh_benchmark', {})
                        source_ids = spec.inputs["source_run_ids"]
                        if (spec.stage not in ('benchmark_attack', 'audit')
                                or fresh.get('schema') != 'fresh_seven_method_development_benchmark_v1'
                                or not isinstance(target_ids, (list, tuple))
                                or not all(isinstance(target, str) and target for target in target_ids)
                                or not isinstance(source_ids, (list, tuple))
                                or len(source_ids) != 1
                                or not isinstance(source_ids[0], str) or not source_ids[0]):
                            raise ValueError("Audit target IDs must be nonempty unique (run_id, upload_id) pairs")
                        targets = [(source_ids[0], target) for target in target_ids]
                    else:
                        targets = [tuple(pair) for pair in target_ids]
                    if not targets or any(len(pair) != 2 for pair in targets) or len(set(targets)) != len(targets):
                        raise ValueError("Audit target IDs must be nonempty unique (run_id, upload_id) pairs")
                    available = {(item.run_id, item.upload_id): item for item in candidates}
                    if any(pair not in available for pair in targets):
                        raise ValueError("Audit shard contains unknown target IDs")
                    candidates = [available[pair] for pair in targets]
                expected = spec.config["execution"]["split"]
                if any(candidate.split != expected for candidate in candidates):
                    raise ValueError("审计来源的 split 不匹配 / Audit source split mismatch")
                if spec.stage == "evaluate_audits" and expected != "test":
                    raise ValueError(
                        "独立审计必须使用测试运行 / Final audits require held-out runs"
                    )
                attack = make_attack(spec, root, store.path / "attack_details")
                audit_config = deepcopy(spec.config["audit"])
                if spec.stage == "benchmark_attack":
                    policy = "benchmark"
                else:
                    benchmark = load_summary(root, spec.inputs["benchmark_run_id"])
                    audit_config["estimated_seconds_by_batch"] = benchmark["seconds_by_batch"]
                    policy = audit_config["policy"]
                store.write_json("resolved_audit_config.json", audit_config)
                result = run_audits(
                    root,
                    candidates,
                    attack,
                    audit_config,
                    policy,
                    spec.seed,
                    data.train_records,
                    store,
                )
            elif spec.stage in ("fit_risk", "calibrate_risk"):
                frame = risk_training_rows(root, spec.inputs["audit_run_ids"])
                risk = make_risk(spec, root)
                risk.fit(
                    np.stack(frame["features"]),
                    frame["feature_recovery"].to_numpy(),
                    frame["run_id"].to_numpy(),
                )
                store.save_checkpoint(
                    {
                        "risk_version": spec.risk_version,
                        "feature_names": FEATURE_NAMES,
                        "implementation": spec.config["components"]["risk"],
                        "state": risk.checkpoint(),
                    },
                    "risk.pt",
                )
                result = {
                    "risk_version": spec.risk_version,
                    "training_uploads": len(frame),
                    "training_runs": int(frame["run_id"].nunique()),
                }
                if hasattr(risk, 'fit_metadata'):
                    result["fit_metadata"] = risk.fit_metadata
            elif spec.stage == "evaluate_risk":
                frame = read_audits(root, spec.inputs["audit_run_ids"])
                if (
                    not frame["split"].eq(spec.config["execution"]["split"]).all()
                    or frame["split"].eq("train").any()
                ):
                    raise ValueError(
                        "风险评价需独立验证/测试审计 / Risk evaluation requires held-out audits"
                    )
                saved = torch.load(
                    root / spec.inputs["risk_checkpoint"], map_location="cpu", weights_only=False
                )
                if (
                    saved["risk_version"] != spec.risk_version
                    or saved["implementation"] != spec.config["components"]["risk"]
                ):
                    raise ValueError("风险评价检查点不匹配 / Risk evaluation checkpoint mismatch")
                risk = make_risk(spec, root)
                risk.load_checkpoint(saved["state"])
                prediction = risk.predict(np.stack(frame["features"]))
                residual = frame["feature_recovery"].to_numpy() - prediction.mean
                frame = frame.assign(
                    reference_mean=prediction.mean,
                    reference_uncertainty=prediction.uncertainty,
                    signed_error=residual,
                    underestimation=np.maximum(residual, 0),
                )
                frame.to_csv(store.path / "risk_predictions.csv", index=False)
                exposure_weights = frame['batch_size'].to_numpy() if 'batch_size' in frame else np.ones(len(frame))
                result = {
                    "uploads": len(frame),
                    "mae": float(np.abs(residual).mean()),
                    "mean_underestimation": float(np.maximum(residual, 0).mean()),
                    "worst_underestimation": float(np.maximum(residual, 0).max()),
                    "risk_version": risk.version,
                    "implementation": spec.config["components"]["risk"],
                    'exposure_weighted_mae': float(np.average(np.abs(residual), weights=exposure_weights)),
                    'exposure_weighted_underestimation': float(np.average(np.maximum(residual, 0), weights=exposure_weights)),
                }
            elif spec.stage == "compare_calibration":
                from .comparisons.calibration import compare_calibration

                frame = read_audits(root, spec.inputs["audit_run_ids"])
                summaries, timeline, metadata = compare_calibration(
                    source_root(spec.config, root), frame, spec.config["comparisons"]["calibration"]
                )
                timeline.to_csv(store.path / "calibration_timeline.csv", index=False)
                store.write_json("calibration_metadata.json", metadata)
                result = {
                    "methods": summaries,
                    "axis": "audited uploads only",
                    "metadata": metadata,
                }
            elif spec.stage in ("collect", "train_policy", "validate", "evaluate"):
                if spec.stage == "validate" and spec.config["execution"]["split"] != "validation":
                    raise ValueError(
                        "调参评价只使用验证运行 / Tuning evaluation requires validation runs"
                    )
                if spec.stage == "evaluate" and spec.config["execution"]["split"] != "test":
                    raise ValueError("最终评估只使用测试运行 / Final evaluation requires test runs")
                env = _make_environment(spec, data, root, store)
                if spec.stage == "train_policy":
                    previous_path = spec.inputs["previous_training_checkpoint"]
                    previous = (
                        torch.load(root / previous_path, map_location="cpu", weights_only=False)
                        if previous_path
                        else None
                    )

                    def save_training(state):
                        """每个训练断点记录风险版本。 / Attach the risk version to every training state."""
                        store.save_checkpoint(
                            {
                                "state": state,
                                "risk_version": spec.risk_version,
                                "run_id": spec.run_id,
                            },
                            "training.pt",
                        )

                    fresh = spec.config.get('native_acceptance', {}).get('fresh_benchmark', {})
                    snapshot_paths = []
                    requested_steps = policy_snapshot_steps(spec)
                    emitted_steps = set()

                    def same_snapshot(left, right):
                        if isinstance(left, dict):
                            return (isinstance(right, dict) and left.keys() == right.keys()
                                    and all(same_snapshot(value, right[key])
                                            for key, value in left.items()))
                        if isinstance(left, (list, tuple)):
                            return (type(left) is type(right) and len(left) == len(right)
                                    and all(same_snapshot(a, b) for a, b in zip(left, right)))
                        if isinstance(left, torch.Tensor):
                            return isinstance(right, torch.Tensor) and torch.equal(left, right)
                        if isinstance(left, np.ndarray):
                            return isinstance(right, np.ndarray) and np.array_equal(left, right)
                        return left == right

                    def save_policy_snapshot(snapshot):
                        step = snapshot['environment_steps']
                        if step not in requested_steps or step in emitted_steps:
                            return
                        envelope = {'format': 'policy_snapshot_v1', 'run_id': spec.run_id,
                                    **snapshot}
                        name = f'agent-step{step:05d}.pt'
                        path = store.path / name
                        if path.exists():
                            if not same_snapshot(store.load_checkpoint(name), envelope):
                                raise ValueError(f'Conflicting policy snapshot: {name}')
                        else:
                            path = store.save_checkpoint(envelope, name)
                        snapshot_paths.append(str(path.relative_to(root)))
                        emitted_steps.add(step)

                    training = train_policy(
                        spec, root, env, previous, save_training,
                        on_policy_snapshot=save_policy_snapshot
                        if fresh.get('policy_snapshots_enabled', False) else None)
                    save_training(training.checkpoint)
                    store.save_checkpoint(
                        {
                            "agent": training.agent.checkpoint(),
                            "training": training.checkpoint,
                            "risk_version": spec.risk_version,
                        },
                        "agent.pt",
                    )
                    result = {"trained": True, "risk_version": spec.risk_version}
                    if fresh.get('policy_snapshots_enabled', False):
                        result['policy_snapshot_paths'] = snapshot_paths
                    if 'learned_steps' in training.checkpoint:
                        result.update(
                            learned_steps=training.checkpoint["learned_steps"],
                            warmup_steps=training.checkpoint["warmup_steps"],
                            updates=training.checkpoint["updates"],
                        )
                        store.write_json("policy_training.json", training.checkpoint["history"])
                else:
                    policy = _policy(spec, root, env.unwrapped.actions)
                    result = _run_episode(env, policy, spec.seed)
                    store.save_checkpoint(env.snapshot(), "environment.pt")
                    if spec.stage == "evaluate":
                        result["test"] = env.unwrapped.backend.evaluate_test()
            else:
                raise ValueError(f"未知实验阶段 / Unknown experiment stage: {spec.stage}")
        result = {"run_id": spec.run_id, "stage": spec.stage, **result}
        store.write_json("summary.json", result)
        return result


def load_summary(root: Path, run_id: str) -> dict:
    """只读加载结果，不重新运行实验。 / Read results without rerunning an experiment."""
    return json.loads((Path(root) / "runs" / run_id / "summary.json").read_text(encoding="utf-8"))
