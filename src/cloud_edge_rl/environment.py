"""公开状态驱动的联合调度环境。 / Joint scheduling environment driven by public state."""

from copy import deepcopy
from dataclasses import asdict
from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import FederatedBackend, RiskEstimator
from .resources import ResourceModel
from .types import Action, ClientUpdate, ModelSummary

# 风险训练与在线预测共享列顺序。 / Risk training and online prediction share this column contract.
FEATURE_NAMES = (
    "client_id_fraction",
    "batch_size",
    "selected_client_fraction",
    "upload_mbps",
    "download_mbps",
    "load",
    "participation_count",
    "processed_samples",
    "unique_samples",
    "validation_loss",
    "validation_accuracy",
    "elapsed_fraction",
    "round_fraction",
)


class CloudEdgeEnv(gym.Env):
    """每步完成一个联邦轮，动作联合选择客户端集合和 batch。 / Each action selects a client set and batch for one full round."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict, backend: FederatedBackend, risk: RiskEstimator | None):
        """由 Notebook 配置建立动作表和公开观察空间。 / Build actions and public observation spaces from Notebook settings."""
        self.config = config
        self.protocol = config['reward'].get('protocol', 'legacy_v1')
        if self.protocol not in ('legacy_v1', 'exposure_v1'):
            raise ValueError('Unknown reward protocol')
        self.warmup = False
        self.backend = backend
        self.risk = risk
        if risk is None and config["reward"]["risk_weight"] != 0:
            raise ValueError(
                "无风险模型时风险权重必须为 0 / Unavailable risk requires risk_weight=0"
            )
        environment = config["environment"]
        if environment["history_window"] < 2:
            raise ValueError(
                "公开参与窗口至少包含上一轮 / Public participation window must include a past round"
            )
        self.client_sizes = np.asarray(environment["client_sizes"], dtype=int)
        self.actions = [
            Action(index, tuple(client_ids), batch_size)
            for index, (client_ids, batch_size) in enumerate(
                product(environment["client_sets"], environment["batch_sizes"])
            )
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        count = len(self.client_sizes)
        self.observation_space = spaces.Dict(
            {
                "resources": spaces.Box(0, np.inf, (count, 3), dtype=np.float32),
                "progress": spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32),
                "history": spaces.Box(0, np.inf, (count, 3), dtype=np.float32),
                "participation_window": spaces.MultiBinary(
                    (environment["history_window"] - 1, count)
                ),
                "candidates": spaces.Box(-np.inf, np.inf, (len(self.actions), 5), dtype=np.float32),
                "risk_available": spaces.MultiBinary(1),
            }
        )
        if self.protocol == 'exposure_v1':
            self.observation_space['risk_ledger'] = spaces.Box(0, np.inf, (2,), dtype=np.float32)

    @property
    def risk_version(self) -> str:
        """无代理时显式标记未提供风险，而非假定零风险。 / Mark unavailable estimates explicitly instead of assuming zero risk."""
        return "unavailable" if self.risk is None else self.risk.version

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置模型、资源轨迹与采样历史。 / Reset the model, resource trace, and sampling history."""
        super().reset(seed=seed)
        self.resource_seed = int(self.np_random.integers(0, 2**32))
        self.resources = ResourceModel(self.config["resources"], self.resource_seed)
        self.summary = self.backend.reset(seed)
        self.round_index = 0
        self.elapsed_seconds = 0.0
        self.energy_joules = 0.0
        self.upload_bytes = 0
        self.download_bytes = 0
        self.processed_samples = 0
        self.risk_exposures = 0
        self.risk_numerator = 0.0
        self.finished = False
        self.history = np.zeros((len(self.client_sizes), 3), dtype=np.float32)
        self.participation_window = np.zeros(
            (self.config["environment"]["history_window"] - 1, len(self.client_sizes)),
            dtype=np.int8,
        )
        self.seen_samples = [set() for _ in self.client_sizes]
        self.utility = np.asarray(
            self.config["environment"]["utility_prior"], dtype=np.float64
        ).copy()
        self.pending_updates = []
        return self._observe(), self._info()

    def _features(self, action: Action, client_id: int, resources: np.ndarray) -> np.ndarray:
        """特征仅来自当前公开资源、模型指标和历史计数。 / Features contain current public resources, model metrics, and history only."""
        environment = self.config["environment"]
        return np.asarray(
            [
                client_id / len(self.client_sizes),
                action.batch_size,
                len(action.client_ids) / len(self.client_sizes),
                *resources[client_id],
                *self.history[client_id],
                self.summary.validation_loss,
                self.summary.validation_accuracy,
                self.elapsed_seconds / environment["time_budget_seconds"],
                self.round_index / environment["max_rounds"],
            ],
            dtype=np.float32,
        )

    def _observe(self) -> dict[str, np.ndarray]:
        """为所有候选动作计算同一状态下的代价与风险代理。 / Estimate every candidate from the same current state."""
        resources = self.resources.observe(self.elapsed_seconds)
        rows = [
            self._features(action, client_id, resources)
            for action in self.actions
            for client_id in action.client_ids
        ]
        prediction = None if self.risk is None else self.risk.predict(np.stack(rows))
        candidates = []
        self.candidate_audits = {}
        offset = 0
        for action in self.actions:
            end = offset + len(action.client_ids)
            if prediction is None:
                # 标志位说明这两列不可用；它们不是零风险估计。 / The availability flag marks these as encoding placeholders.
                risk = 0.0
                mean_uncertainty = 0.0
                audit_predictions = [(None, None)] * len(action.client_ids)
            else:
                means = np.asarray(prediction.mean[offset:end])
                uncertainty = np.asarray(prediction.uncertainty[offset:end])
                # 先按客户端截断再平均，防止聚合掩盖高风险客户端。 / Clip each client's upper risk before averaging.
                risk = np.clip(
                    means + self.config["reward"]["uncertainty_weight"] * uncertainty, 0, 1
                ).mean()
                mean_uncertainty = float(uncertainty.mean())
                audit_predictions = [
                    (float(mean), float(std)) for mean, std in zip(means, uncertainty)
                ]
            cost = self.resources.estimate(
                action.client_ids, action.batch_size, self.summary.model_bytes, self.elapsed_seconds
            )
            if self.protocol == 'exposure_v1':
                exposure = action.batch_size * len(action.client_ids)
                previous_risk = self.risk_numerator / self.risk_exposures if self.risk_exposures else 0.
                risk = (self.risk_numerator + exposure * risk) / (self.risk_exposures + exposure) - previous_risk
            candidates.append(
                [
                    self.utility[action.action_id],
                    cost.time_seconds,
                    cost.energy_joules,
                    risk,
                    mean_uncertainty,
                ]
            )
            self.candidate_audits[action.action_id] = {
                client_id: (rows[offset + index].copy(), *audit_predictions[index])
                for index, client_id in enumerate(action.client_ids)
            }
            offset = end
        environment = self.config["environment"]
        self.candidates = np.asarray(candidates, dtype=np.float32)
        observation = {
            "resources": resources,
            "progress": np.asarray(
                [
                    self.summary.validation_loss,
                    self.summary.validation_accuracy,
                    self.elapsed_seconds / environment["time_budget_seconds"],
                    self.round_index / environment["max_rounds"],
                    (environment["sample_budget"] - self.processed_samples)
                    / environment["sample_budget"],
                ],
                dtype=np.float32,
            ),
            "history": self.history.copy(),
            "participation_window": self.participation_window.copy(),
            "candidates": self.candidates.copy(),
            "risk_available": np.asarray([self.risk is not None], dtype=np.int8),
        }
        if self.protocol == 'exposure_v1':
            observation['risk_ledger'] = np.asarray([self.risk_exposures, self.risk_numerator], dtype=np.float32)
        return observation

    def step(self, action_id: int):
        """执行整轮并结算真实资源开销，攻击输入仅交审计侧通道。 / Account for a whole round; route attack inputs only to audit."""
        if self.protocol == 'exposure_v1' and self.finished:
            raise RuntimeError('Cannot step a finished finite-horizon task')
        if not self.action_space.contains(action_id):
            raise ValueError('Invalid environment action')
        action = self.actions[action_id]
        audits = self.candidate_audits[action_id]
        old_summary = self.summary
        risk_delta = float(self.candidates[action_id, 3])
        risk = risk_delta
        if self.protocol == 'exposure_v1' and self.risk is not None:
            risk = float(np.mean([np.clip(mean + self.config['reward']['uncertainty_weight'] * std, 0, 1)
                                  for _, mean, std in audits.values()]))
        worst_risk = (
            None
            if self.risk is None
            else float(
                max(
                    np.clip(mean + self.config["reward"]["uncertainty_weight"] * std, 0, 1)
                    for _, mean, std in audits.values()
                )
            )
        )
        result = self.backend.step_round(action)
        # 最近参与信息向所有策略公开。 / The same recent participation history is public to every policy.
        self.participation_window = np.roll(self.participation_window, -1, axis=0)
        self.participation_window[-1] = 0
        self.participation_window[-1, list(action.client_ids)] = 1
        counts = {update.client_id: len(update.sample_ids) for update in result.updates}
        cost = self.resources.estimate(
            action.client_ids,
            action.batch_size,
            old_summary.model_bytes,
            self.elapsed_seconds,
            np.asarray([counts[client_id] for client_id in action.client_ids]),
        )
        for update in result.updates:
            client_id = update.client_id
            self.history[client_id, 0] += 1
            self.history[client_id, 1] += len(update.sample_ids)
            self.seen_samples[client_id].update(update.sample_ids.tolist())
            self.history[client_id, 2] = len(self.seen_samples[client_id])
            self.pending_updates.append((update, *audits[client_id]))
        self.summary = result.summary
        self.round_index += 1
        self.elapsed_seconds += cost.time_seconds
        self.energy_joules += cost.energy_joules
        round_bytes = old_summary.model_bytes * len(action.client_ids)
        self.upload_bytes += round_bytes
        self.download_bytes += round_bytes
        self.processed_samples += sum(counts.values())
        loss_gain = old_summary.validation_loss - self.summary.validation_loss
        utility = (self.summary.validation_accuracy - old_summary.validation_accuracy
                   if self.protocol == 'exposure_v1' else loss_gain)
        decay = self.config["environment"]["utility_ema_decay"]
        self.utility[action_id] = decay * self.utility[action_id] + (1 - decay) * utility
        weights = self.config["reward"]
        reward = (
            utility
            - weights["time_weight"] * cost.time_seconds / weights["time_scale"]
            - weights["energy_weight"] * cost.energy_joules / weights["energy_scale"]
            - weights["risk_weight"] * risk
        )
        if self.protocol == 'exposure_v1':
            # 只按准确率增量更新一次效用。 / One accuracy EMA, never mixed with loss gain.
            gain = utility
            previous_risk = self.risk_numerator / self.risk_exposures if self.risk_exposures else 0.
            if not self.warmup and self.risk is not None:
                for client_id, count in counts.items():
                    _, mean, std = audits[client_id]
                    self.risk_numerator += count * float(np.clip(mean + weights['uncertainty_weight'] * std, 0, 1))
                    self.risk_exposures += count
            current_risk = self.risk_numerator / self.risk_exposures if self.risk_exposures else 0.
            reward = (weights['accuracy_weight'] * gain
                      - weights['time_weight'] * cost.time_seconds / weights['time_scale']
                      - weights['energy_weight'] * cost.energy_joules / weights['energy_scale']
                      - weights['risk_weight'] * (current_risk - previous_risk))
        environment = self.config["environment"]
        terminated = bool(
            self.summary.validation_loss <= environment["target_validation_loss"]
            or self.elapsed_seconds >= environment["time_budget_seconds"]
            or self.processed_samples >= environment["sample_budget"]
        )
        if self.protocol == 'exposure_v1':
            terminated = terminated or self.round_index >= environment['max_rounds']
        truncated = bool(self.round_index >= environment["max_rounds"] and not terminated)
        self.finished = terminated or truncated
        info = self._info()
        info.update(
            action_id=int(action_id),
            loss_gain=loss_gain,
            round_time_seconds=cost.time_seconds,
            round_energy_joules=cost.energy_joules,
            predicted_risk=None if self.risk is None else risk,
            worst_predicted_risk=worst_risk,
            round_upload_bytes=round_bytes,
            round_download_bytes=round_bytes,
        )
        return self._observe(), float(reward), terminated, truncated, info

    def _info(self) -> dict:
        """公开摘要不包含梯度、样本标识或攻击真值。 / Public summaries exclude gradients, sample IDs, and attack truth."""
        info = {
            "round": self.round_index,
            "elapsed_seconds": self.elapsed_seconds,
            "energy_joules": self.energy_joules,
            "upload_bytes": self.upload_bytes,
            "download_bytes": self.download_bytes,
            "processed_samples": self.processed_samples,
            "unique_samples": sum(len(samples) for samples in self.seen_samples),
            "validation_loss": self.summary.validation_loss,
            "validation_accuracy": self.summary.validation_accuracy,
            "risk_version": self.risk_version,
            "risk_available": self.risk is not None,
            'reward_protocol': self.protocol,
            'is_warmup': self.warmup,
            'risk_exposures': self.risk_exposures,
            'predicted_exposure_risk': None if self.risk is None or not self.risk_exposures else self.risk_numerator / self.risk_exposures,
        }
        for client_id, (participations, samples, unique) in enumerate(self.history):
            info.update(
                {
                    f"c{client_id}_participations": int(participations),
                    f"c{client_id}_samples": int(samples),
                    f"c{client_id}_unique": int(unique),
                }
            )
        return info

    def pop_updates(self) -> list[tuple[ClientUpdate, np.ndarray, float | None, float | None]]:
        """取出并清空离线审计侧通道。 / Drain the offline audit side channel."""
        updates = self.pending_updates
        self.pending_updates = []
        return updates

    def snapshot(self) -> dict:
        """保存模型、资源种子、随机状态与完整历史。 / Save the model, resource seed, RNG, and complete history."""
        return deepcopy(
            {
                "backend": self.backend.checkpoint(),
                'format': 'environment_v3' if self.protocol == 'exposure_v1' else 'environment_v2',
                'config': self.config,
                'risk_exposures': self.risk_exposures,
                'risk_numerator': self.risk_numerator,
                'finished': self.finished,
                "risk_version": self.risk_version,
                "summary": asdict(self.summary),
                "resource_seed": self.resource_seed,
                "rng": self.np_random.bit_generator.state,
                "round_index": self.round_index,
                "elapsed_seconds": self.elapsed_seconds,
                "energy_joules": self.energy_joules,
                "upload_bytes": self.upload_bytes,
                "download_bytes": self.download_bytes,
                "processed_samples": self.processed_samples,
                "history": self.history,
                "participation_window": self.participation_window,
                "seen_samples": self.seen_samples,
                "utility": self.utility,
                "pending_updates": self.pending_updates,
            }
        )

    def load_snapshot(self, state: dict) -> dict[str, np.ndarray]:
        """恢复同一风险版本下的训练轨迹。 / Restore a trajectory using the identical risk model version."""
        expected_format = 'environment_v3' if self.protocol == 'exposure_v1' else 'environment_v2'
        if state.get('format') != expected_format or state['config'] != self.config:
            raise ValueError('Environment protocol/configuration mismatch')
        if state["risk_version"] != self.risk_version:
            raise ValueError("风险模型版本不匹配 / Risk model version mismatch")
        state = deepcopy(state)
        self.backend.load_checkpoint(state["backend"])
        self.summary = ModelSummary(**state["summary"])
        self.resource_seed = state["resource_seed"]
        self.resources = ResourceModel(self.config["resources"], self.resource_seed)
        self.np_random.bit_generator.state = state["rng"]
        for name in (
            "round_index",
            'risk_exposures',
            'risk_numerator',
            'finished',
            "elapsed_seconds",
            "energy_joules",
            "upload_bytes",
            "download_bytes",
            "processed_samples",
            "history",
            "participation_window",
            "seen_samples",
            "utility",
            "pending_updates",
        ):
            setattr(self, name, state[name])
        return self._observe()
