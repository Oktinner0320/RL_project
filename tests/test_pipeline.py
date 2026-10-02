"""用测试替身验证编排，不实现学习算法 / Test orchestration with doubles, never learning algorithms."""

import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch

from cloud_edge_rl import pipeline
from cloud_edge_rl.config import RunSpec
from cloud_edge_rl.types import TrainingResult


def make_spec(stage, config, inputs, method="dqn"):
    """创建显式测试运行 / Create an explicit test run."""
    return RunSpec(f"test-{stage}", stage, method, 17, "risk-v2", config, inputs, {})


@pytest.fixture
def isolated_pipeline(monkeypatch):
    """不下载数据、不请求云资源 / Isolate tests from downloads and cloud resources."""
    data = SimpleNamespace(metadata={"synthetic": True}, train_records=pd.DataFrame({"x": [0.0]}))
    monkeypatch.setattr(pipeline, "runtime_info", lambda: {"runtime": "test"})
    monkeypatch.setattr(pipeline, "prepare_adult", lambda config, cache: data)
    return data


def training_rows():
    """构造按来源运行分组的审计标签 / Build audit labels grouped by source run."""
    return pd.DataFrame(
        {
            "features": [[0.0, 1.0], [1.0, 2.0]],
            "feature_recovery": [0.2, 0.4],
            "run_id": ["source-a", "source-b"],
        }
    )


def test_unimplemented_core_is_logged_and_rethrown(tmp_path, monkeypatch, isolated_pipeline):
    """用户核心空缺导致显式失败，不生成伪结果 / Log and propagate the real unimplemented core."""
    monkeypatch.setattr(pipeline, "risk_training_rows", lambda root, run_ids: training_rows())
    spec = make_spec("fit_risk", {"data": {}, "risk": {}}, {"audit_run_ids": ["source"]})
    with pytest.raises(NotImplementedError, match="RiskEstimator.fit"):
        pipeline.run_experiment(spec, tmp_path)
    run = tmp_path / "runs" / spec.run_id
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert "RiskEstimator.fit" in (run / "error.txt").read_text(encoding="utf-8")
    assert (run / "config.json").exists()
    assert not (run / "summary.json").exists()
    assert not (run / "risk.pt").exists()


def test_risk_fit_receives_run_seed_and_group_ids(tmp_path, monkeypatch, isolated_pipeline):
    """运行种子进入拟合，完整运行组交给用户核心 / Pass the run seed and complete groups to the user core."""
    captured = {}

    class FakeRisk:
        """仅记录调用的拟合替身 / Fit double that only records its inputs."""

        def __init__(self, config, version):
            captured.update(config=config, version=version)

        def fit(self, features, targets, groups):
            """记录输入，不执行学习 / Record inputs without learning."""
            captured.update(features=features, targets=targets, groups=groups)

        def checkpoint(self):
            """返回合成状态 / Return synthetic state."""
            return {"fixture": True}

    monkeypatch.setattr(pipeline.core, "RiskEstimator", FakeRisk)
    monkeypatch.setattr(pipeline, "risk_training_rows", lambda root, run_ids: training_rows())
    spec = make_spec("fit_risk", {"data": {}, "risk": {}}, {"audit_run_ids": ["source"]})
    result = pipeline.run_experiment(spec, tmp_path)
    assert captured["config"]["seed"] == spec.seed
    assert captured["version"] == spec.risk_version
    assert captured["groups"].tolist() == ["source-a", "source-b"]
    assert result["training_runs"] == 2
    checkpoint = torch.load(tmp_path / "runs" / spec.run_id / "risk.pt", weights_only=True)
    assert checkpoint["risk_version"] == spec.risk_version
    assert checkpoint["state"] == {"fixture": True}
    assert "seed" not in spec.config["risk"]


def test_policy_training_receives_run_identity_and_saves_user_state(
    tmp_path, monkeypatch, isolated_pipeline
):
    """派生训练种子与算法标识且保存回调状态 / Derive seed/algorithm identity and save user checkpoints."""
    captured = {}
    env = SimpleNamespace(action_space=SimpleNamespace(n=12))
    env.unwrapped = env
    env.actions = list(range(12))
    monkeypatch.setattr(pipeline, "_make_environment", lambda *args: env)

    def train_agent(environment, config, previous, on_checkpoint):
        """回调只模拟状态保存 / Exercise checkpoint wiring without implementing training."""
        captured.update(config=config, previous=previous, environment=environment)
        on_checkpoint({"step": 1})
        return TrainingResult(
            SimpleNamespace(checkpoint=lambda: {"policy_fixture": True}), {"step": 1}
        )

    monkeypatch.setattr(pipeline.core, "train_agent", train_agent)
    config = {
        "data": {},
        "rl": {"learning_rate": 0.01},
        "execution": {"warmup_actions": list(range(12))},
    }
    spec = make_spec("train_policy", config, {"previous_training_checkpoint": ""})
    result = pipeline.run_experiment(spec, tmp_path)
    assert captured["config"]["seed"] == spec.seed
    assert captured["config"]["algorithm"] == spec.method
    assert captured["config"]["risk_version"] == spec.risk_version
    assert captured["config"]["warmup_actions"] == list(range(12))
    assert captured["previous"] is None
    assert result["trained"] is True
    run = tmp_path / "runs" / spec.run_id
    assert torch.load(run / "training.pt", weights_only=True) == {
        "state": {"step": 1},
        "risk_version": spec.risk_version,
        "run_id": spec.run_id,
    }
    assert torch.load(run / "agent.pt", weights_only=True)["agent"] == {"policy_fixture": True}
    assert spec.config["rl"] == {"learning_rate": 0.01}


@pytest.mark.parametrize("method", ["dqn", "ppo"])
def test_policy_loading_receives_the_same_algorithm_contract(tmp_path, monkeypatch, method):
    """策略加载与训练共享算法身份 / Loading receives the algorithm identity used during training."""
    captured = {}
    torch.save({"agent": {"fixture": True}}, tmp_path / "agent.pt")

    def load_agent(state, config):
        """记录加载参数，不构造网络 / Record load arguments without constructing a network."""
        captured.update(state=state, config=config)
        return "test-agent"

    monkeypatch.setattr(pipeline.core, "load_agent", load_agent)
    spec = make_spec(
        "evaluate", {"rl": {"learning_rate": 0.01}}, {"agent_checkpoint": "agent.pt"}, method=method
    )
    assert pipeline._policy(spec, tmp_path) == "test-agent"
    assert captured["state"] == {"fixture": True}
    assert captured["config"]["algorithm"] == method
    assert captured["config"]["seed"] == spec.seed
    assert captured["config"]["risk_version"] == spec.risk_version
    assert spec.config["rl"] == {"learning_rate": 0.01}


@pytest.mark.parametrize(
    "stage,expected,source_split",
    [
        ("audit", "test", "train"),
        ("evaluate_audits", "train", "train"),
    ],
)
def test_audit_split_mismatch_fails_before_attack(
    tmp_path,
    monkeypatch,
    isolated_pipeline,
    stage,
    expected,
    source_split,
):
    """错误来源在攻击前被拒绝 / Reject mismatched or non-held-out sources before any attack."""
    monkeypatch.setattr(
        pipeline, "read_candidates", lambda *args: [SimpleNamespace(split=source_split)]
    )

    def forbidden_attack(config):
        """错误分组不应消耗攻击预算 / Invalid splits must not consume attack budget."""
        raise AssertionError("attack must not start")

    monkeypatch.setattr(pipeline.core, "GradientAttack", forbidden_attack)
    spec = make_spec(
        stage, {"data": {}, "execution": {"split": expected}}, {"source_run_ids": ["source"]}
    )
    with pytest.raises(ValueError):
        pipeline.run_experiment(spec, tmp_path)


def test_final_audits_only_evaluate_without_fitting(tmp_path, monkeypatch, isolated_pipeline):
    """最终标签只能落盘评价，不能触发风险拟合 / Final audit labels are evaluated without risk fitting."""
    monkeypatch.setattr(pipeline, "read_candidates", lambda *args: [SimpleNamespace(split="test")])
    monkeypatch.setattr(pipeline, "run_audits", lambda *args: {"calls": 1})
    monkeypatch.setattr(pipeline, "load_summary", lambda *args: {"seconds_by_batch": {"8": 2.0}})

    def forbidden_fit(*args):
        """测试阶段禁止调用拟合 / Fitting is forbidden during held-out evaluation."""
        raise AssertionError("final labels must not enter fitting")

    monkeypatch.setattr(pipeline, "risk_training_rows", forbidden_fit)
    monkeypatch.setattr(pipeline.core.RiskEstimator, "fit", forbidden_fit)
    config = {
        "data": {},
        "execution": {"split": "test"},
        "attack": {},
        "audit": {"policy": "panel"},
    }
    spec = make_spec(
        "evaluate_audits", config, {"source_run_ids": ["held-out"], "benchmark_run_id": "benchmark"}
    )
    assert pipeline.run_experiment(spec, tmp_path)["calls"] == 1
    assert not (tmp_path / "runs" / spec.run_id / "risk.pt").exists()


def test_recorder_uses_gym_wrapper_and_restores_decision_observation(tmp_path):
    """真实 Gym 包装器可初始化且恢复同一决策状态 / Construct a real Gym wrapper and restore its decision state."""

    class TinyEnv(gym.Env):
        """仅用于包装器测试的确定环境 / Deterministic environment used only to test recording."""

        def __init__(self):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Dict(
                {
                    "candidates": gym.spaces.Box(-np.inf, np.inf, shape=(2, 5), dtype=np.float32),
                }
            )
            self.config = {
                "reward": {
                    "time_weight": 1,
                    "energy_weight": 1,
                    "risk_weight": 1,
                    "time_scale": 1,
                    "energy_scale": 1,
                }
            }
            self.position = 0

        def observe(self):
            """以位置生成观测 / Derive observations from the current position."""
            return {"candidates": np.full((2, 5), self.position, dtype=np.float32)}

        def reset(self, *, seed=None, options=None):
            """重置测试位置 / Reset the test position."""
            super().reset(seed=seed)
            self.position = 0
            return self.observe(), {"round": 0}

        def step(self, action):
            """前进一步，不生成梯度 / Advance without generating gradients."""
            self.position += 1
            return self.observe(), 0.0, False, False, {"round": self.position}

        def pop_updates(self):
            """此测试没有更新包 / This fixture has no update packets."""
            return []

        def snapshot(self):
            """保存当前位置 / Save the current position."""
            return {"position": self.position}

        def load_snapshot(self, state):
            """恢复位置并返回正确观测 / Restore the position and its correct observation."""
            self.position = state["position"]
            return self.observe()

    journal = []
    store = SimpleNamespace(append_metrics=journal.append)
    spec = make_spec("collect", {"execution": {"save_updates": False}}, {})
    wrapped = pipeline.RecordingEnv(TinyEnv(), store, tmp_path, None, spec)
    assert wrapped.spec is None
    wrapped.reset(seed=17)
    first, *_ = wrapped.step(0)
    snapshot = wrapped.snapshot()
    wrapped.step(0)
    restored = wrapped.load_snapshot(snapshot)
    np.testing.assert_array_equal(first["candidates"], restored["candidates"])
    np.testing.assert_array_equal(wrapped.observation["candidates"], restored["candidates"])
    assert wrapped.episode == 0
    _, _, _, _, info = wrapped.step(0)
    assert info["round"] == 2
    assert len(journal) == 3
