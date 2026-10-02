"""以测试专用后端检查环境边界和审计隔离。 / Check environment contracts and audit isolation using a test-only backend."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from cloud_edge_rl.environment import FEATURE_NAMES, CloudEdgeEnv
from cloud_edge_rl.types import AttackInput, ClientUpdate, ModelSummary, RiskPrediction, RoundResult


@pytest.fixture
def config() -> dict:
    """所有测试数值独立于实验默认值。 / Test values are independent from experiment settings."""
    return {
        "environment": {
            "client_sizes": [8, 8],
            "client_sets": [[0], [1], [0, 1]],
            "batch_sizes": [2, 4],
            "time_budget_seconds": 100,
            "sample_budget": 100,
            "max_rounds": 10,
            "target_validation_loss": 0.01,
            "utility_prior": [0.2] * 6,
            "utility_ema_decay": 0.5,
        },
        "resources": {
            "interval_seconds": 10,
            "bandwidth_mbps": [8, 4],
            "download_mbps": [16, 8],
            "samples_per_second": [100, 50],
            "compute_watts": [20, 30],
            "transmit_watts": [5, 6],
            "receive_watts": [3, 4],
            "bandwidth_jitter": 0.1,
            "load_range": [0.1, 0.3],
            "aggregation_seconds": 0.5,
            "server_watts": 40,
        },
        "reward": {
            "time_weight": 0.1,
            "energy_weight": 0.2,
            "risk_weight": 2,
            "uncertainty_weight": 2,
            "time_scale": 10,
            "energy_scale": 100,
        },
    }


class TestBackend:
    """无学习能力的测试替身，不进入生产实验。 / A non-learning test double excluded from production experiments."""

    __test__ = False

    def reset(self, seed):
        self.round = 0
        return ModelSummary(1.0, 0.2, 1_000_000)

    def step_round(self, action):
        self.round += 1
        updates = []
        for client_id in action.client_ids:
            attack = AttackInput(
                model_state={},
                gradients={"w": torch.ones(1)},
                labels=torch.zeros(action.batch_size, dtype=torch.long),
                batch_size=action.batch_size,
                feature_metadata={},
                model_config={"activation": "relu", "loss": "mse", "dtype": "float32"},
            )
            updates.append(ClientUpdate(client_id, np.arange(action.batch_size), attack))
        return RoundResult(
            ModelSummary(1 - 0.1 * self.round, 0.2 + 0.05 * self.round, 1_000_000), updates
        )

    def checkpoint(self):
        return {"round": self.round}

    def load_checkpoint(self, state):
        self.round = state["round"]


class TestRisk:
    """按客户端区分风险以验证聚合次序。 / Distinguish clients to verify risk aggregation order."""

    __test__ = False
    version = "test-risk-v1"

    def predict(self, features):
        return RiskPrediction(np.where(features[:, 0] == 0, 0.9, 0.1), np.full(len(features), 0.2))


def make_env(config):
    """构造独立测试环境。 / Construct an independent test environment."""
    return CloudEdgeEnv(config, TestBackend(), TestRisk())


def test_gymnasium_contract(config):
    check_env(make_env(config), skip_render_check=True)


def test_observation_and_audit_expose_only_intended_data(config):
    env = make_env(config)
    observation, info = env.reset(seed=17)
    assert env.observation_space.contains(observation)
    assert len(FEATURE_NAMES) == 13
    assert len(env.actions) == 6
    assert observation["candidates"][4, 3] == pytest.approx(0.75)
    observation, reward, terminated, truncated, info = env.step(4)
    assert not terminated and not truncated
    assert set(observation) == {"resources", "progress", "history", "candidates", "risk_available"}
    assert not {
        "updates",
        "gradients",
        "labels",
        "sample_ids",
        "true_risk",
        "attack_input",
    }.intersection(info)
    audits = env.pop_updates()
    assert len(audits) == 2
    assert all(len(features) == 13 for _, features, _, _ in audits)
    assert all(features[9] == pytest.approx(1.0) for _, features, _, _ in audits)
    assert env.pop_updates() == []
    assert info["round_upload_bytes"] == info["round_download_bytes"] == 2_000_000
    assert info["upload_bytes"] == info["download_bytes"] == 2_000_000
    assert info["c0_samples"] == 2
    assert info["c0_unique"] == 2
    assert info["c0_participations"] == 1
    assert info["worst_predicted_risk"] == pytest.approx(1.0)
    assert observation["progress"][4] == pytest.approx(0.96)
    expected = (
        0.1
        - 0.1 * info["round_time_seconds"] / 10
        - 0.2 * info["round_energy_joules"] / 100
        - 2 * 0.75
    )
    assert reward == pytest.approx(expected)


def test_history_counts_repeated_samples_and_updates_only_executed_action(config):
    env = make_env(config)
    env.reset(seed=17)
    env.step(0)
    observation, *_ = env.step(0)
    np.testing.assert_array_equal(observation["history"], [[2, 4, 2], [0, 0, 0]])
    assert env.processed_samples == 4
    assert env.utility[0] == pytest.approx(0.125)
    np.testing.assert_allclose(env.utility[1:], 0.2)


@pytest.mark.parametrize("limit", ["time_budget_seconds", "sample_budget"])
def test_budget_boundary_accounts_for_whole_round(config, limit):
    config["environment"][limit] = 1
    env = make_env(config)
    env.reset(seed=17)
    _, _, terminated, truncated, info = env.step(4)
    assert terminated and not truncated
    assert info["processed_samples"] == 4
    assert info["elapsed_seconds"] > 1


def test_round_cap_is_truncation_and_loss_target_is_termination(config):
    config["environment"]["max_rounds"] = 1
    env = make_env(config)
    env.reset(seed=17)
    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated and truncated
    config["environment"]["target_validation_loss"] = 0.95
    env = make_env(config)
    env.reset(seed=17)
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated


def test_checkpoint_reproduces_next_step(config):
    env = make_env(config)
    env.reset(seed=17)
    env.step(4)
    snapshot = env.snapshot()
    expected = env.step(1)
    restored = make_env(deepcopy(config))
    restored.load_snapshot(snapshot)
    actual = restored.step(1)
    for name in expected[0]:
        np.testing.assert_array_equal(actual[0][name], expected[0][name])
    assert actual[1:] == expected[1:]
    assert restored.snapshot()["rng"] == env.snapshot()["rng"]
    assert len(restored.pop_updates()) == 3


def test_checkpoint_rejects_different_risk_version(config):
    env = make_env(config)
    env.reset(seed=17)
    state = env.snapshot()
    env.risk.version = "test-risk-v2"
    with pytest.raises(ValueError, match="Risk model version mismatch"):
        env.load_snapshot(state)


def test_bootstrap_marks_risk_unavailable_and_audit_prediction_missing(config):
    config["reward"]["risk_weight"] = 0
    env = CloudEdgeEnv(config, TestBackend(), None)
    observation, info = env.reset(seed=17)
    assert env.observation_space.contains(observation)
    assert observation["risk_available"].tolist() == [0]
    assert info["risk_available"] is False
    assert info["risk_version"] == "unavailable"
    observation, _, _, _, info = env.step(4)
    assert info["predicted_risk"] is None
    assert info["worst_predicted_risk"] is None
    assert all(
        mean is None and uncertainty is None for _, _, mean, uncertainty in env.pop_updates()
    )
    assert env.snapshot()["risk_version"] == "unavailable"


def test_bootstrap_rejects_nonzero_risk_penalty(config):
    with pytest.raises(ValueError, match="risk_weight=0"):
        CloudEdgeEnv(config, TestBackend(), None)


def test_sample_budget_is_explicit_in_observation_including_full_round_overrun(config):
    config["environment"]["sample_budget"] = 3
    env = make_env(config)
    observation, _ = env.reset(seed=17)
    assert observation["progress"].shape == (5,)
    assert observation["progress"][4] == 1
    observation, _, terminated, _, _ = env.step(4)
    assert terminated
    assert observation["progress"][4] == pytest.approx(-1 / 3)
    assert env.observation_space.contains(observation)
