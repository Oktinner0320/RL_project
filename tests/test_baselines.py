"""验证基线仅按公开候选矩阵决策。 / Verify baseline decisions depend only on public candidates."""

import numpy as np

from cloud_edge_rl.baselines import (
    FixedPolicy,
    RiskGreedyPolicy,
    RoundRobinPolicy,
    TimeGreedyPolicy,
    run_warmup,
)


def test_fixed_and_round_robin_actions():
    assert FixedPolicy(3).act({}, explore=False) == 3
    policy = RoundRobinPolicy([2, 0, 1])
    assert [policy.act({}, explore=False) for _ in range(5)] == [2, 0, 1, 2, 0]


def test_time_greedy_uses_current_latency_with_stable_ties():
    observation = {"candidates": np.asarray([[4, 3, 1, 0, 0], [0, 1, 1, 1, 0], [9, 1, 1, 0, 0]])}
    assert TimeGreedyPolicy().act(observation, explore=False) == 1


def test_risk_greedy_uses_already_clipped_client_risk_without_double_penalty():
    config = {
        "time_weight": 0.1,
        "energy_weight": 0.2,
        "risk_weight": 2,
        "uncertainty_weight": 10,
        "time_scale": 10,
        "energy_scale": 100,
    }
    observation = {"candidates": np.asarray([[1, 10, 100, 0.2, 0.1], [1, 10, 100, 0.1, 0.8]])}
    # uncertainty 已合入 risk 列，不能再次扣除。 / Uncertainty is included in risk and must not be penalized twice.
    assert RiskGreedyPolicy(config).act(observation, explore=False) == 1


class WarmupEnv:
    """记录调用次数并模拟预算边界。 / Track calls and simulate an episode budget boundary."""

    def __init__(self, stop_after, truncate=False):
        self.actions = []
        self.stop_after = stop_after
        self.truncate = truncate

    def step(self, action_id):
        self.actions.append(action_id)
        finished = len(self.actions) == self.stop_after
        return (
            {"action": action_id},
            action_id + 1,
            finished and not self.truncate,
            finished and self.truncate,
            {"steps": len(self.actions)},
        )


def test_warmup_executes_each_given_action_once():
    env = WarmupEnv(stop_after=10)
    observation, info, terminated, truncated, reward = run_warmup(env, {}, [2, 0, 1])
    assert env.actions == [2, 0, 1]
    assert observation == {"action": 1}
    assert info == {"steps": 3}
    assert not terminated and not truncated
    assert reward == 6


def test_warmup_stops_at_budget_termination():
    env = WarmupEnv(stop_after=2)
    _, info, terminated, truncated, reward = run_warmup(env, {}, [2, 0, 1])
    assert env.actions == [2, 0]
    assert info == {"steps": 2}
    assert terminated and not truncated
    assert reward == 4


def test_warmup_stops_at_round_truncation():
    env = WarmupEnv(stop_after=1, truncate=True)
    _, _, terminated, truncated, _ = run_warmup(env, {}, [2, 0, 1])
    assert env.actions == [2]
    assert not terminated and truncated


def test_empty_warmup_preserves_observation_and_does_not_step():
    env = WarmupEnv(stop_after=2)
    observation = {"initial": True}
    assert run_warmup(env, observation, []) == (observation, {}, False, False, 0.0)
    assert env.actions == []
