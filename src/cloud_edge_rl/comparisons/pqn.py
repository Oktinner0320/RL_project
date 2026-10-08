"""PQN 的 PyTorch 对照移植；用户核心独立保留。 / A separate PyTorch PQN comparison port.

Derived from purejaxql/pqn_gymnax.py, commit 47af6d7b35c8 (Apache-2.0):
https://github.com/mttga/purejaxql
LayerNorm, online Q(lambda), recent rollouts and RAdam follow the reference.
The costly stateful FedSGD environment is sampled serially, rather than through
JAX vmap. Gymnasium truncations bootstrap the final observation and stop the
cross-episode trace. This is a reference port, not an exact paper reproduction.
移植保留归一化、在线多步目标和优化器；串行采样及截断处理与原版不同。
All tunable values are supplied by the Notebook. / 可调参数全部由 Notebook 提供。
"""

from collections.abc import Callable
from copy import deepcopy
from math import sqrt

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.utils import flatdim, flatten
from torch import nn

from ..baselines import run_warmup
from ..types import TrainingResult


class QNetwork(nn.Module):
    """逐层归一化的在线 Q 网络。 / An online Q network with normalization in every hidden layer."""

    def __init__(self, observation_size: int, action_count: int, config: dict):
        """复用原版 Dense → LayerNorm → ReLU 顺序。 / Preserve the upstream block order."""
        super().__init__()
        layers = []
        width = observation_size
        for _ in range(config["num_layers"]):
            layers.extend(
                [
                    nn.Linear(width, config["hidden_width"]),
                    nn.LayerNorm(config["hidden_width"], eps=config["layer_norm_epsilon"]),
                    nn.ReLU(),
                ]
            )
            width = config["hidden_width"]
        layers.append(nn.Linear(width, action_count))
        self.layers = nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Flax LeCun normal truncates at ±2σ; correct the resulting variance.
                # Flax 初始化截断正态并修正截断造成的方差变化。
                deviation = 1 / sqrt(layer.in_features) / 0.8796256610342398
                nn.init.trunc_normal_(
                    layer.weight, std=deviation, a=-2 * deviation, b=2 * deviation
                )
                nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """返回每个离散动作的 Q 值。 / Return one Q value per discrete action."""
        return self.layers(observations)


def lambda_targets(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    trace_lambda: float,
) -> torch.Tensor:
    """冻结在线 Q(lambda) 目标，截断不跨重置传播。 / Freeze online Q(lambda) targets at episode boundaries.

    Termination has no bootstrap; truncation uses Q(final observation).
    真终止目标只有奖励；截断仍自举，但不能接入重置后回合的奖励。
    """
    targets = torch.empty_like(rewards)
    continuation = next_values[-1]
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = rewards[index] + gamma * (~terminated[index]) * next_values[index]
        if index == len(rewards) - 1 or terminated[index] or truncated[index]:
            continuation = bootstrap
        else:
            continuation = rewards[index] + gamma * (
                (1 - trace_lambda) * next_values[index] + trace_lambda * continuation
            )
        targets[index] = continuation
    return targets.detach()


class PQNAgent:
    """只读取公共观测的 ε 贪心策略。 / An epsilon-greedy policy reading public observations only."""

    def __init__(self, observation_space: gym.Space, action_count: int, config: dict):
        """构建显式配置的策略及独立探索 RNG。 / Build the configured network and exploration RNG."""
        self.observation_space = observation_space
        self.action_count = action_count
        self.config = deepcopy(config)
        self.device = torch.device(config["device"])
        self.network = QNetwork(flatdim(observation_space), action_count, config).to(self.device)
        self.rng = np.random.default_rng(config["seed"])
        self.epsilon = config["epsilon_start"]

    def encode(self, observation: dict) -> np.ndarray:
        """按 Gymnasium 空间顺序展开字典。 / Flatten the dictionary using Gymnasium's space order."""
        return np.asarray(flatten(self.observation_space, observation), dtype=np.float32)

    def act(self, observation: dict, explore: bool) -> int:
        """评估不消耗探索随机数。 / Greedy evaluation does not consume exploration randomness."""
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))
        with torch.no_grad():
            values = self.network(torch.as_tensor(self.encode(observation), device=self.device))
        return int(values.argmax().item())

    def checkpoint(self) -> dict:
        """策略含空间契约和探索状态。 / Persist the space contract and exploration state."""
        return deepcopy(
            {
                "algorithm": "pqn_torch",
                "observation_space": self.observation_space,
                "action_count": self.action_count,
                "network_config": {
                    name: self.config[name]
                    for name in ("hidden_width", "num_layers", "layer_norm_epsilon")
                },
                "model_state": {
                    name: value.detach().cpu().clone()
                    for name, value in self.network.state_dict().items()
                },
                "rng": self.rng.bit_generator.state,
                "epsilon": self.epsilon,
            }
        )


def load_agent(state: dict, config: dict) -> PQNAgent:
    """载入对照策略，不触碰用户 Agent。 / Load this comparison without using the user Agent."""
    architecture = {
        name: config[name] for name in ("hidden_width", "num_layers", "layer_norm_epsilon")
    }
    if state["algorithm"] != "pqn_torch" or state["network_config"] != architecture:
        raise ValueError("PQN 策略结构不匹配 / PQN policy architecture mismatch")
    # Restored initialization is overwritten; it must not consume a caller's RNG.
    # 恢复时初始权重会被覆盖，不应消耗调用方的随机数。
    with torch.random.fork_rng(devices=[]):
        agent = PQNAgent(state["observation_space"], state["action_count"], config)
    agent.network.load_state_dict(state["model_state"])
    agent.rng.bit_generator.state = deepcopy(state["rng"])
    agent.epsilon = state["epsilon"]
    return agent


def _fresh_episode(env, seed: int, actions: list[int]):
    """共享预热真实消耗环境预算。 / Shared warmup consumes actual environment resources."""
    observation, _ = env.reset(seed=seed)
    reward = 0.0
    for action in actions:
        observation, _, terminated, truncated, value = run_warmup(env, observation, [action])
        reward += value
        if terminated or truncated:
            raise ValueError(
                "预热已耗尽回合预算 / Warmup exhausts the episode before policy training"
            )
    return observation, reward, len(actions)


def _resume_settings(config: dict) -> dict:
    """同阶段只允许延长总步数或改变存储频率。 / Same-stage resumes may extend steps or storage frequency."""
    return {
        name: value
        for name, value in config.items()
        if name not in ("total_steps", "checkpoint_interval_updates", "device")
    }


def train_agent(
    env, config: dict, previous: dict | None, on_checkpoint: Callable[[dict], None]
) -> TrainingResult:
    """训练第三方 PQN 对照，保存可精确续跑的状态。 / Train a separate PQN comparison with resumable state.

    Checkpoints occur after complete rollouts, with no replay or stale targets.
    New risk versions retain weights but start a fresh stage and optimizer.
    在完整 rollout 后保存；风险变更保留权重，重新初始化阶段与优化器。
    """
    if (
        min(
            config["total_steps"],
            config["rollout_steps"],
            config["num_minibatches"],
            config["epochs"],
            config["epsilon_decay_steps"],
            config["lr_decay_steps"],
            config["checkpoint_interval_updates"],
        )
        <= 0
    ):
        raise ValueError("PQN 迭代预算必须为正 / PQN iteration budgets must be positive")
    actual_budget = config.get('budget_includes_warmup', False)
    if config["rollout_steps"] % config["num_minibatches"] or (
        not actual_budget and config["total_steps"] % config["rollout_steps"]
    ):
        raise ValueError("PQN 步数不能整除分批数 / PQN rollout and stage sizes must divide exactly")
    if not 0 <= config["gamma"] <= 1 or not 0 <= config["lambda"] <= 1:
        raise ValueError("PQN 折扣与轨迹权重必须在 [0,1] / PQN gamma and lambda must be in [0,1]")
    torch.manual_seed(config["seed"])
    if previous is None:
        agent = PQNAgent(env.observation_space, env.action_space.n, config)
    else:
        state = previous["state"]
        agent = load_agent(state["agent"], config)
        if (
            state["agent"]["observation_space"] != env.observation_space
            or state["agent"]["action_count"] != env.action_space.n
        ):
            raise ValueError("PQN 环境空间不匹配 / PQN environment spaces mismatch")
    optimizer = torch.optim.RAdam(
        agent.network.parameters(),
        lr=config["learning_rate"],
        betas=tuple(config["adam_betas"]),
        eps=config["adam_epsilon"],
        weight_decay=config["weight_decay"],
    )
    learned_steps = updates = gradient_steps = episode = warmup_steps = warmup_position = 0
    history, episode_returns = [], []
    finished = False
    same_stage = previous is not None and previous["risk_version"] == config["risk_version"]
    if same_stage:
        if state["settings"] != _resume_settings(config):
            raise ValueError("同阶段 PQN 参数已改变 / Same-stage PQN settings changed")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if agent.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        observation = env.load_snapshot(state["environment"])
        learned_steps = state["learned_steps"]
        updates = state["updates"]
        gradient_steps = state["gradient_steps"]
        episode = state["episode"]
        warmup_steps = state["warmup_steps"]
        warmup_position = state.get('warmup_position', len(config['warmup_actions']))
        episode_return = state["episode_return"]
        finished = state["finished"]
        history = deepcopy(state["history"])
        episode_returns = deepcopy(state["episode_returns"])
        if learned_steps + (warmup_steps if actual_budget else 0) > config["total_steps"]:
            raise ValueError("恢复点超过训练预算 / Restored PQN steps exceed the stage budget")
    else:
        # Stage changes preserve only weights; no old actions or targets enter the new estimator.
        # 阶段切换只保留策略权重，旧行为与目标不能混入新风险版本。
        agent.rng = np.random.default_rng(config["seed"])
        agent.epsilon = config["epsilon_start"]
        if actual_budget:
            observation, _ = env.reset(seed=config['seed'])
            episode_return = 0.
        else:
            observation, episode_return, warmup_steps = _fresh_episode(
                env, config["seed"], config["warmup_actions"]
            )

    def checkpoint() -> dict:
        """保存优化器、所有 RNG 与真实环境断点。 / Persist optimizer, all RNGs and the actual environment."""
        return deepcopy(
            {
                "agent": agent.checkpoint(),
                "optimizer": optimizer.state_dict(),
                "environment": env.snapshot(),
                "risk_version": config["risk_version"],
                "settings": _resume_settings(config),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if agent.device.type == "cuda" else [],
                "learned_steps": learned_steps,
                "updates": updates,
                "gradient_steps": gradient_steps,
                "episode": episode,
                "warmup_steps": warmup_steps,
                'warmup_position': warmup_position,
                'environment_steps': learned_steps + warmup_steps,
                "episode_return": episode_return,
                "finished": finished,
                "history": history,
                "episode_returns": episode_returns,
                "source": "purejaxql/pqn_gymnax.py@47af6d7b35c8 (Apache-2.0)",
                "adaptation": "Serial Gymnasium rollout, PyTorch RAdam, final-observation truncation bootstrap",
            }
        )

    def consumed():
        return learned_steps + (warmup_steps if actual_budget else 0)

    while consumed() < config["total_steps"]:
        fraction = min(learned_steps / config["epsilon_decay_steps"], 1.0)
        agent.epsilon = config["epsilon_start"] + fraction * (
            config["epsilon_finish"] - config["epsilon_start"]
        )
        observations, actions, rewards, next_values, terminations, truncations = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for _ in range(config["rollout_steps"]):
            if actual_budget and consumed() >= config['total_steps']:
                break
            if finished:
                episode += 1
                if actual_budget:
                    observation, _ = env.reset(seed=config['seed'] + episode)
                    warmup_position, episode_return, finished = 0, 0., False
                else:
                    observation, episode_return, count = _fresh_episode(
                        env, config["seed"] + episode, config["warmup_actions"]
                    )
                    warmup_steps += count
            if actual_budget:
                # 每次真实交互均计费；预算可在预热中间结束。 / Count every interaction, even mid-warmup.
                while warmup_position < len(config['warmup_actions']) and consumed() < config['total_steps']:
                    observation, _, terminated, truncated, _ = run_warmup(
                        env, observation, [config['warmup_actions'][warmup_position]])
                    warmup_position += 1
                    warmup_steps += 1
                    if terminated or truncated:
                        raise ValueError('Warmup exhausts the episode before policy training')
                if consumed() >= config['total_steps']:
                    break
            encoded = agent.encode(observation)
            action = agent.act(observation, explore=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            with torch.no_grad():
                next_value = agent.network(
                    torch.as_tensor(agent.encode(observation), device=agent.device)
                ).max()
            observations.append(encoded)
            actions.append(action)
            rewards.append(reward * config["reward_scale"])
            next_values.append(next_value)
            terminations.append(terminated)
            truncations.append(truncated)
            learned_steps += 1
            episode_return += float(reward)
            finished = terminated or truncated
            if finished:
                episode_returns.append(episode_return)
        if not observations:
            break
        observations = torch.as_tensor(np.stack(observations), device=agent.device)
        actions = torch.tensor(actions, dtype=torch.long, device=agent.device)
        targets = lambda_targets(
            torch.tensor(rewards, dtype=torch.float32, device=agent.device),
            torch.stack(next_values),
            torch.tensor(terminations, dtype=torch.bool, device=agent.device),
            torch.tensor(truncations, dtype=torch.bool, device=agent.device),
            config["gamma"],
            config["lambda"],
        )
        losses, q_means = [], []
        for _ in range(config["epochs"]):
            order = agent.rng.permutation(len(observations))
            for indices in np.array_split(order, min(len(order), config["num_minibatches"])):
                indices = torch.as_tensor(indices, device=agent.device)
                if config["linear_lr_decay"]:
                    fraction = min(gradient_steps / config["lr_decay_steps"], 1.0)
                    optimizer.param_groups[0]["lr"] = config["learning_rate"] + fraction * (
                        config["min_learning_rate"] - config["learning_rate"]
                    )
                values = agent.network(observations[indices]).gather(1, actions[indices, None])[
                    :, 0
                ]
                loss = 0.5 * (values - targets[indices]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.network.parameters(), config["max_grad_norm"])
                optimizer.step()
                gradient_steps += 1
                losses.append(float(loss.detach()))
                q_means.append(float(values.detach().mean()))
        updates += 1
        history.append(
            {
                "learned_steps": learned_steps,
                "warmup_steps": warmup_steps,
                "environment_steps": learned_steps + warmup_steps,
                "updates": updates,
                "gradient_steps": gradient_steps,
                "td_loss": float(np.mean(losses)),
                "q_mean": float(np.mean(q_means)),
                "epsilon": agent.epsilon,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "risk_version": config["risk_version"],
            }
        )
        if updates % config["checkpoint_interval_updates"] == 0:
            on_checkpoint(checkpoint())
    return TrainingResult(agent, checkpoint())
