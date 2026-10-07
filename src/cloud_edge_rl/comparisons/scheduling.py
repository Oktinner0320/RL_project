"""MetaCS 官方目标在项目动作表上的适配。 / Adapt the official MetaCS objective to project actions.

Source: https://github.com/alan-lira/metacs-fl (CeCILL-C 1.0).
直接枚举现有动作；不称为原版固定工作量 LNS。 / Enumerate project actions, not native fixed-workload LNS.
"""

from pathlib import Path

import numpy as np

from cloud_edge_rl.types import Action


class MetaCSPolicy:
    """官方成本、历史参与熵和效用组成的透明对比策略。 / Compare official cost, historical participation entropy, and utility."""

    def __init__(self, actions: list[Action], config: dict, root: Path):
        """参数全部来自 Notebook；不读取私有类别计数。 / Use Notebook settings without private class histograms."""
        from .sources import load_module

        if config["weights"]["K_weight"] != 0:
            raise ValueError(
                "无公开类别直方图，K_weight 必须为 0 / Class-distribution weight must be zero"
            )
        self.actions = actions
        self.config = config
        self.utility = load_module(
            root,
            "metacs_fl",
            "metacs_fl/utils/task_scheduler_util.py",
            "metacs_fl.utils.task_scheduler_util",
        )
        lns = load_module(
            root, "metacs_fl", "metacs_fl/metaheuristic/lns.py", "metacs_fl.metaheuristic.lns"
        )
        self.objective = lns.lns_F

    def score_actions(self, observation: dict) -> tuple[np.ndarray, list[dict]]:
        """同一公开状态评价全部候选，分数越小越好。 / Score every candidate from one public state; lower is better.

        原版类别项置零，轮时延/能耗使用共享资源模型；单客户端效用不可观测，复用
        每个动作的历史验证增益。 / Disable unavailable class statistics, reuse shared resource
        costs, and use historical action validation gains because individual utilities are unobserved.
        """
        count = len(observation["history"])
        window = np.asarray(observation["participation_window"])
        q = self.config["history_window"]
        if window.shape != (q - 1, count):
            raise ValueError(
                "参与历史窗口与配置不一致 / Participation window differs from configuration"
            )
        selected_history = {
            round_index: np.flatnonzero(selection).tolist()
            for round_index, selection in enumerate(window, start=1)
        }
        scores, components = [], []
        for action, candidate in zip(self.actions, observation["candidates"]):
            gain, seconds, energy, risk, _ = candidate
            allocation = [
                action.batch_size if client in action.client_ids else 0 for client in range(count)
            ]
            action_gain = float(gain) / self.config["utility_scale"]
            costs = {
                "M_X": float(seconds),
                "E_X": float(energy),
                "D_X": self.utility.calculate_client_diversity_score(
                    allocation, list(range(count)), q, q, {}, selected_history
                ),
                "K_X": 0.0,
                "U_X": self.utility.estimate_utility(
                    allocation, sum(allocation), [action_gain] * count, [action_gain] * count, 1.0
                ),
            }
            normalized = self.utility.normalize_costs(
                {"action": costs}, self.config["normalization_bounds"]
            )["action"]
            native_score = float(self.objective(normalized, self.config["weights"]))
            risk_cost = self.config["risk_weight"] * float(risk)
            scores.append(native_score + risk_cost)
            components.append(
                {
                    "action_id": action.action_id,
                    "allocation": allocation,
                    **costs,
                    "normalized_time": normalized["M_X"],
                    "normalized_energy": normalized["E_X"],
                    "metacs_objective": native_score,
                    "risk_cost": risk_cost,
                }
            )
        return np.asarray(scores), components

    def act(self, observation: dict, explore: bool) -> int:
        """只返回已有动作，稳定选择首个最小分数。 / Return an existing action with stable first-minimum tie breaking."""
        scores, _ = self.score_actions(observation)
        return self.actions[int(np.argmin(scores))].action_id
