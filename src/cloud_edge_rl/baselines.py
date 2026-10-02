"""共享公开观测的确定性策略。 / Deterministic policies using the same public observation."""

import numpy as np


def run_warmup(
    env, observation: dict, action_ids: list[int]
) -> tuple[dict, dict, bool, bool, float]:
    """依序执行一次预热动作，预算终止立即停止。 / Execute each warmup action once and stop immediately at episode end."""
    info = {}
    terminated = truncated = False
    total_reward = 0.0
    for action_id in action_ids:
        observation, reward, terminated, truncated, info = env.step(action_id)
        total_reward += float(reward)
        if terminated or truncated:
            break
    return observation, info, terminated, truncated, total_reward


class FixedPolicy:
    """始终选定一个动作。 / Always choose one configured action."""

    def __init__(self, action_id: int):
        """保存 Notebook 指定的动作。 / Store the action selected in the Notebook."""
        self.action_id = action_id

    def act(self, observation: dict, explore: bool) -> int:
        """返回固定动作编号。 / Return the fixed action index."""
        return self.action_id


class RoundRobinPolicy:
    """按给定动作顺序循环选择。 / Cycle through the configured action order."""

    def __init__(self, action_ids: list[int]):
        """保存动作顺序并从首项开始。 / Store the action order and begin at its first item."""
        self.action_ids = action_ids
        self.position = 0

    def act(self, observation: dict, explore: bool) -> int:
        """按给定顺序选择动作。 / Select actions in the configured order."""
        action = self.action_ids[self.position % len(self.action_ids)]
        self.position += 1
        return action


class TimeGreedyPolicy:
    """选择预测轮时延最短的动作。 / Choose the action with the lowest predicted round latency."""

    def act(self, observation: dict, explore: bool) -> int:
        """只比较当前公开候选时延。 / Compare only the currently visible candidate latency."""
        return int(np.argmin(observation["candidates"][:, 1]))


class RiskGreedyPolicy:
    """按共享奖励权重选择预测净收益最高的动作。 / Maximize predicted net utility with shared reward weights."""

    def __init__(self, reward_config: dict):
        """复用环境的奖励权重与归一化尺度。 / Reuse environment reward weights and normalization scales."""
        self.reward_config = reward_config

    def act(self, observation: dict, explore: bool) -> int:
        """以风险代理而非真实攻击结果评分。 / Score proxy risk without accessing attack ground truth."""
        utility, seconds, energy, risk, _ = observation["candidates"].T
        config = self.reward_config
        scores = (
            utility
            - config["time_weight"] * seconds / config["time_scale"]
            - config["energy_weight"] * energy / config["energy_scale"]
            - config["risk_weight"] * risk
        )
        return int(np.argmax(scores))
