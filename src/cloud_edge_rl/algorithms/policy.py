"""DQN/PPO 共享公开观测、交互预算与可恢复训练。 / Native policy lifecycle and exact resumes."""

from collections import deque
from copy import deepcopy

import numpy as np
import torch
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import flatdim, flatten

from ..baselines import run_warmup
from ..types import TrainingResult
from . import dqn, ppo
from .risk import mlp


class Agent:
    def __init__(self, observation_space, action_count: int, config: dict):
        if config['algorithm'] not in ('dqn', 'ppo'):
            raise ValueError('Native policy algorithm must be dqn or ppo')
        self.observation_space, self.action_count = observation_space, action_count
        self.config = deepcopy(config)
        # 只用公开预算缩放，不拟合未来轨迹。 / Fixed public scales, never future-data statistics.
        self.observation_divisors = deepcopy(config.get('observation_divisors', {}))
        if not isinstance(self.observation_divisors, dict):
            raise ValueError('Observation divisors must be a component mapping')
        if self.observation_divisors and not isinstance(observation_space, Dict):
            raise ValueError('Observation divisors require a Dict observation space')
        for name, divisor in self.observation_divisors.items():
            if name not in observation_space.spaces:
                raise ValueError('Unknown observation component divisor')
            scale = np.asarray(divisor, dtype=np.float32)
            np.broadcast_to(scale, observation_space.spaces[name].shape)
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('Observation divisors must be finite and positive')
        self.device = torch.device(config['device'])
        self.rng = np.random.default_rng(config['seed'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'])
            self.network = mlp(flatdim(observation_space), config['hidden_sizes'],
                               action_count + int(config['algorithm'] == 'ppo')).to(self.device)
        self.epsilon = config['epsilon_start']

    def encode(self, observation: dict) -> np.ndarray:
        if self.observation_divisors:
            observation = {name: np.asarray(value, dtype=np.float32) / np.asarray(
                self.observation_divisors[name], dtype=np.float32)
                if name in self.observation_divisors else value
                for name, value in observation.items()}
        return np.asarray(flatten(self.observation_space, observation), dtype=np.float32)

    def output(self, observation: dict) -> np.ndarray:
        with torch.no_grad():
            return self.network(torch.tensor(self.encode(observation), device=self.device)).cpu().numpy()

    def act(self, observation: dict, explore: bool) -> int:
        values = self.output(observation)
        if self.config['algorithm'] == 'dqn':
            self.last_diagnostics = {'kind': 'q_values', 'values': values.tolist(),
                                     'margin': float(np.sort(values)[-1] - np.sort(values)[-2])}
            if explore and self.rng.random() < self.epsilon:
                return int(self.rng.integers(self.action_count))
            return int(values.argmax())
        logits = values[:self.action_count]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        self.last_diagnostics = {'kind': 'action_probabilities', 'values': probabilities.tolist(),
                                 'entropy': float(-(probabilities * np.log(probabilities.clip(1e-30))).sum()),
                                 'state_value': float(values[-1]),
                                 'margin': float(np.sort(probabilities)[-1] - np.sort(probabilities)[-2])}
        return int(self.rng.choice(self.action_count, p=probabilities)) if explore else int(probabilities.argmax())

    def checkpoint(self) -> dict:
        return deepcopy({'format': 'native_agent_v1', 'algorithm': self.config['algorithm'],
                         'hidden_sizes': self.config['hidden_sizes'],
                         'observation_space': self.observation_space, 'action_count': self.action_count,
                         'observation_divisors': self.observation_divisors,
                         'model': {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()},
                         'rng': self.rng.bit_generator.state, 'epsilon': self.epsilon})


def load_agent(state: dict, config: dict) -> Agent:
    if state.get('observation_divisors', {}) != config.get('observation_divisors', {}):
        raise ValueError('Native policy checkpoint observation scaling mismatch')
    if state['format'] != 'native_agent_v1' or state['algorithm'] != config['algorithm'] or state['hidden_sizes'] != config['hidden_sizes']:
        raise ValueError('Native policy checkpoint architecture mismatch')
    agent = Agent(state['observation_space'], state['action_count'], config)
    agent.network.load_state_dict(state['model'])
    agent.rng.bit_generator.state = deepcopy(state['rng'])
    agent.epsilon = state['epsilon']
    return agent


def _settings(config):
    return {k: v for k, v in config.items()
            if k not in ('device', 'total_steps', 'checkpoint_interval', 'stage_interaction_limit')}


def train_agent(env, config: dict, previous: dict | None, on_checkpoint) -> TrainingResult:
    stage_limit = config.get('stage_interaction_limit')
    if stage_limit is not None and (type(stage_limit) is not int or stage_limit <= 0):
        raise ValueError('Stage interaction limit must be a positive integer')
    positive = ('total_steps', 'batch_size', 'checkpoint_interval', 'max_grad_norm')
    positive += ('buffer_size', 'target_update_interval', 'train_frequency', 'epsilon_decay_steps') if config['algorithm'] == 'dqn' else ('rollout_steps', 'ppo_epochs')
    if any(config[k] <= 0 for k in positive) or not 0 <= config['gamma'] <= 1:
        raise ValueError('Invalid native policy budgets or discount')
    if config['algorithm'] == 'ppo' and not 0 <= config['gae_lambda'] <= 1:
        raise ValueError('GAE lambda must be in [0,1]')
    base = env.unwrapped
    if base.config['reward'].get('protocol', 'legacy_v1') == 'exposure_v1' and config['gamma'] != 1:
        raise ValueError('Exposure finite-horizon protocol requires gamma=1')
    state = None if previous is None else previous['state']
    agent = Agent(env.observation_space, env.action_space.n, config) if state is None else load_agent(state['agent'], config)
    if agent.observation_space != env.observation_space or agent.action_count != env.action_space.n:
        raise ValueError('Native policy environment space mismatch')
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=config['learning_rate'])
    target = deepcopy(agent.network).eval().requires_grad_(False)
    replay = deque(maxlen=config['buffer_size'])
    rollout, history = [], []
    learned_steps = warmup_steps = updates = episode = warmup_position = 0
    finished = False
    same_stage = previous is not None and previous['risk_version'] == config['risk_version']
    if same_stage:
        if state['format'] != 'native_training_v1' or state['settings'] != _settings(config):
            raise ValueError('Same-stage native training settings mismatch')
        observation = env.load_snapshot(state['environment'])
        optimizer.load_state_dict(deepcopy(state['optimizer']))
        target.load_state_dict(state['target'])
        replay.extend(deepcopy(state['replay']))
        rollout, history = deepcopy(state['rollout']), deepcopy(state['history'])
        learned_steps, warmup_steps, updates, episode, warmup_position = (state[k] for k in
            ('learned_steps', 'warmup_steps', 'updates', 'episode', 'warmup_position'))
        finished = state['finished']
        torch.set_rng_state(state['torch_rng'].cpu())
        if agent.device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        agent.rng = np.random.default_rng(config['seed'])
        agent.epsilon = config['epsilon_start']
        observation, _ = env.reset(seed=config['seed'])
    if learned_steps + warmup_steps > config['total_steps']:
        raise ValueError('Restored interaction count exceeds stage budget')
    # 单次调用有界，未完成的 PPO 轨迹不提前更新。 / Bound this call without flushing a pending rollout.
    stage_end = config['total_steps'] if stage_limit is None else min(
        config['total_steps'], learned_steps + warmup_steps + stage_limit)

    def checkpoint():
        return deepcopy({'format': 'native_training_v1', 'settings': _settings(config),
                         'risk_version': config['risk_version'], 'agent': agent.checkpoint(),
                         'optimizer': optimizer.state_dict(), 'target': target.state_dict(),
                         'replay': list(replay), 'rollout': rollout, 'history': history,
                         'learned_steps': learned_steps, 'warmup_steps': warmup_steps,
                         'environment_steps': learned_steps + warmup_steps, 'updates': updates,
                         'episode': episode, 'warmup_position': warmup_position, 'finished': finished,
                         'environment': env.snapshot(), 'torch_rng': torch.get_rng_state(),
                         'cuda_rng': torch.cuda.get_rng_state_all() if agent.device.type == 'cuda' else []})

    while learned_steps + warmup_steps < stage_end:
        if finished:
            episode += 1
            observation, _ = env.reset(seed=config['seed'] + episode)
            warmup_position, finished = 0, False
        if warmup_position < len(config['warmup_actions']):
            observation, _, terminated, truncated, _ = run_warmup(env, observation, [config['warmup_actions'][warmup_position]])
            warmup_position += 1
            warmup_steps += 1
            finished = terminated or truncated
            if finished:
                raise ValueError('Warmup exhausted the episode before policy training')
        else:
            fraction = min(learned_steps / config['epsilon_decay_steps'], 1.)
            agent.epsilon = config['epsilon_start'] + fraction * (config['epsilon_end'] - config['epsilon_start'])
            encoded = agent.encode(observation)
            action = agent.act(observation, explore=True)
            if hasattr(env, 'set_decision_diagnostics'):
                env.set_decision_diagnostics(agent.last_diagnostics)
            before = agent.output(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            learned_steps += 1
            finished = terminated or truncated
            diagnostics = None
            if config['algorithm'] == 'dqn':
                replay.append((encoded, action, reward, agent.encode(observation), terminated))
                if learned_steps >= config['learning_starts'] and len(replay) >= config['batch_size'] and learned_steps % config['train_frequency'] == 0:
                    diagnostics = dqn.update(agent, optimizer, target, replay, config)
                    updates += 1
                if learned_steps % config['target_update_interval'] == 0:
                    target.load_state_dict(agent.network.state_dict())
            else:
                logits = torch.tensor(before[:agent.action_count])
                log_prob = torch.log_softmax(logits, dim=0)[action].item()
                rollout.append((encoded, action, reward, float(before[-1]),
                                float(agent.output(observation)[-1]), log_prob, terminated, truncated))
                if len(rollout) >= config['rollout_steps'] or learned_steps + warmup_steps == config['total_steps']:
                    diagnostics = ppo.update(agent, optimizer, rollout, config)
                    rollout.clear()
                    updates += 1
            if diagnostics is not None:
                history.append({'learned_steps': learned_steps, 'warmup_steps': warmup_steps,
                                'environment_steps': learned_steps + warmup_steps,
                                'updates': updates, 'risk_version': config['risk_version'], **diagnostics})
        if (learned_steps + warmup_steps) % config['checkpoint_interval'] == 0:
            on_checkpoint(checkpoint())
    return TrainingResult(agent, checkpoint())
