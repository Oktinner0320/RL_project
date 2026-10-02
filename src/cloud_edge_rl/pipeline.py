"""实验阶段编排；学习逻辑保留给研究者。 / Stage orchestration, not learning implementation."""

import importlib.metadata
import json
import platform
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from . import core
from .artifacts import RunStore
from .audit import read_candidates, risk_training_rows, run_audits, save_upload
from .baselines import FixedPolicy, RiskGreedyPolicy, RoundRobinPolicy, TimeGreedyPolicy, run_warmup
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
            values[:, 0]
            - weights["time_weight"] * values[:, 1] / weights["time_scale"]
            - weights["energy_weight"] * values[:, 2] / weights["energy_scale"]
            - weights["risk_weight"] * values[:, 3]
        )
        ordered = np.sort(scores)
        margin = float(np.ptp(ordered[-2:]))
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.store.append_metrics(
            {
                "episode": self.episode,
                **info,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
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
                )
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
    backend = core.FederatedBackend(data, {**config["model"], "seed": spec.seed})
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
        risk = core.RiskEstimator({**config["risk"], "seed": spec.seed}, spec.risk_version)
        risk.load_checkpoint(checkpoint["state"])
    else:
        raise ValueError(f"未知风险模式 / Unknown risk mode: {mode}")
    return RecordingEnv(CloudEdgeEnv(config, backend, risk), store, root, data, spec)


def _policy(spec: RunSpec, root: Path):
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
    finished = terminated or truncated
    while not finished:
        action = policy.act(observation, explore=False)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        finished = terminated or truncated
    return {**info, "total_reward": total_reward}


def run_experiment(spec: RunSpec, root: Path) -> dict:
    """统一的本地/Modal入口；所有输入来自 Notebook 快照。 / One local/Modal entry point."""
    root = Path(root)
    with RunStore(root, spec) as store:
        store.write_json("runtime.json", runtime_info())
        if spec.stage == "runtime_check":
            result = {"run_id": spec.run_id, "stage": spec.stage, **runtime_info()}
        else:
            data = prepare_adult(spec.config["data"], root / "data")
            store.write_json("data_metadata.json", data.metadata)
            if spec.stage == "prepare_data":
                result = {
                    "train_rows": len(data.train_y),
                    "validation_rows": len(data.validation_y),
                    "test_rows": len(data.test_y),
                    "encoded_features": data.train_x.shape[1],
                    "client_sizes": [len(rows) for rows in data.client_indices],
                }
            elif spec.stage in ("audit", "evaluate_audits", "benchmark_attack"):
                candidates = read_candidates(root, spec.inputs["source_run_ids"])
                expected = spec.config["execution"]["split"]
                if any(candidate.split != expected for candidate in candidates):
                    raise ValueError("审计来源的 split 不匹配 / Audit source split mismatch")
                if spec.stage == "evaluate_audits" and expected != "test":
                    raise ValueError(
                        "独立审计必须使用测试运行 / Final audits require held-out runs"
                    )
                attack_config = {**spec.config["attack"], "seed": spec.seed}
                attack = core.GradientAttack(attack_config)
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
                risk = core.RiskEstimator(
                    {**spec.config["risk"], "seed": spec.seed}, spec.risk_version
                )
                risk.fit(
                    np.stack(frame["features"]),
                    frame["feature_recovery"].to_numpy(),
                    frame["run_id"].to_numpy(),
                )
                store.save_checkpoint(
                    {
                        "risk_version": spec.risk_version,
                        "feature_names": FEATURE_NAMES,
                        "state": risk.checkpoint(),
                    },
                    "risk.pt",
                )
                result = {
                    "risk_version": spec.risk_version,
                    "training_uploads": len(frame),
                    "training_runs": int(frame["run_id"].nunique()),
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
                    rl_config = {
                        **spec.config["rl"],
                        "seed": spec.seed,
                        "algorithm": spec.method,
                        "risk_version": spec.risk_version,
                        "warmup_actions": spec.config["execution"]["warmup_actions"],
                    }

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

                    training = core.train_agent(env, rl_config, previous, save_training)
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
                else:
                    policy = _policy(spec, root)
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
