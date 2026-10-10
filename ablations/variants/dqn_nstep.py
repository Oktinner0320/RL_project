"""独立的 1-step/3-step DQN；预算边界可恢复，不视为终止。
Isolated one/three-step DQN with resumable, non-terminal budget boundaries.
"""

from collections import deque
from copy import deepcopy

import numpy as np
import torch

from cloud_edge_rl.algorithms import dqn
from cloud_edge_rl.algorithms.policy import Agent, _settings, load_agent
from cloud_edge_rl.baselines import run_warmup
from cloud_edge_rl.types import TrainingResult


def _emit(pending, replay, gamma):
    """写入最早状态的多步回报，使用实际长度和末状态。 / Store the oldest state's return with its actual length and endpoint."""
    first, last = pending[0], pending[-1]
    # 回报按步数折扣；记录实际长度，供后续自举使用。 / Discount rewards and save the length for bootstrap.
    reward = sum(gamma ** i * row[2] for i, row in enumerate(pending))
    replay.append((first[0], first[1], reward, last[3], last[4], len(pending)))
    pending.popleft()


def append_transition(pending, replay, transition, n_step, gamma, truncated=False):
    """收集单步记录，形成多步回放，并在回合边界清空队列。
    Collect transitions, create multi-step replay, and drain the queue at episode boundaries.
    """
    pending.append(transition)
    # 收齐 n 步后写入一条回放，其余记录等待下一步。 / Emit one replay row after collecting n steps.
    if len(pending) == n_step:
        _emit(pending, replay, gamma)
    # 终止和截断都排空队列，避免跨回合；只有终止标志禁止自举。 / Drain at both boundaries; only termination stops bootstrap.
    if transition[4] or truncated:
        while pending:
            _emit(pending, replay, gamma)


def update(agent, optimizer, target, replay, config):
    """采样多步回放并更新 Q 网络一次。 / Sample multi-step replay and update the Q network once.

    沿用原 DQN 的采样、损失和优化器，只改变回报计算。
    Use the native DQN sampler, loss, and optimizer; only the return differs.
    """
    ids = agent.rng.choice(len(replay), config['batch_size'], replace=False)
    x, actions, rewards, next_x, done, lengths = zip(*(replay[i] for i in ids))
    x = torch.tensor(np.stack(x), device=agent.device)
    next_x = torch.tensor(np.stack(next_x), device=agent.device)
    actions = torch.tensor(actions, device=agent.device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=agent.device)
    done = torch.tensor(done, device=agent.device)
    # 回合末尾可能不足 n 步，因此按各条记录的实际长度折扣。 / Use each row's actual length, including short final returns.
    discounts = torch.tensor([config['gamma'] ** m for m in lengths],
                             dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        next_values = target(next_x)
        if config['double_dqn']:
            # 在线网络选动作，目标网络估值。 / The online network selects; the target network evaluates.
            next_values = next_values.gather(1, agent.network(next_x).argmax(1)[:, None])[:, 0]
        else:
            next_values = next_values.max(1).values
        labels = dqn.td_targets(rewards, next_values, done, discounts)
    # 只拟合选中动作的 Q 值，目标不参与反向传播。 / Fit Q for chosen actions without gradients through targets.
    values = agent.network(x).gather(1, actions[:, None])[:, 0]
    loss = torch.nn.functional.smooth_l1_loss(values, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.network.parameters(), config['max_grad_norm'])
    optimizer.step()
    return {'td_loss': loss.item(), 'q_mean': values.detach().mean().item()}


def _validate(config, previous):
    """检查消融配置和恢复状态，返回固定训练设置。 / Check ablation settings and saved state; return fixed training settings."""
    if config['algorithm'] != 'dqn' or config.get('resource_feedback') is not None:
        raise ValueError('N-step ablation requires DQN without resource feedback')
    if type(config.get('n_step')) is not int or config['n_step'] not in (1, 3):
        raise ValueError('n_step must be 1 or 3')
    if config['gamma'] != 1:
        raise ValueError('Exposure ablation requires gamma=1')
    for key in ('total_steps', 'batch_size', 'checkpoint_interval', 'buffer_size',
                'target_update_interval', 'train_frequency', 'epsilon_decay_steps'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError('Invalid integer interaction budget or cadence: ' + key)
    if config['total_steps'] > 3000:
        raise ValueError('Total interaction budget must be in [1,3000]')
    if type(config['learning_starts']) is not int or config['learning_starts'] < 0:
        raise ValueError('Invalid learning_starts')
    if not np.isfinite(config['max_grad_norm']) or config['max_grad_norm'] <= 0:
        raise ValueError('Invalid max_grad_norm')
    if not np.isfinite(config['learning_rate']) or config['learning_rate'] <= 0:
        raise ValueError('Invalid learning_rate')
    if not 0 <= config['epsilon_end'] <= config['epsilon_start'] <= 1:
        raise ValueError('Invalid exploration settings')
    limit = config.get('stage_interaction_limit')
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('Stage interaction limit must be a positive integer')
    # 恢复可以改变单次调用长度，但不能改变已注册的总预算。 / A call limit may change, but the total budget must stay fixed.
    settings = {**_settings(config), 'total_steps': config['total_steps']}
    if previous is not None:
        state = previous['state']
        if (previous['risk_version'] != config['risk_version']
                or state['risk_version'] != config['risk_version']):
            raise ValueError('Cannot reset risk stage or interaction counters')
        if state['format'] != 'ablation_dqn_training_v1' or state['settings'] != settings:
            raise ValueError('Ablation training settings mismatch')
        for key in ('learned_steps', 'warmup_steps', 'environment_steps', 'updates',
                    'episode', 'warmup_position'):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError('Invalid restored counter: ' + key)
        # 核对计数、回放数量和未完成队列，防止恢复时丢步或跨回合。 / Check counters, replay size, and pending transitions.
        if (state['environment_steps'] != state['learned_steps'] + state['warmup_steps']
                or state['environment_steps'] > config['total_steps']
                or state['updates'] > state['learned_steps']
                or state['episode'] > state['environment_steps']
                or state['warmup_position'] > len(config['warmup_actions'])
                or state['warmup_steps'] != state['episode'] * len(config['warmup_actions']) + state['warmup_position']
                or len(state['history']) != state['updates']
                or len(state['replay']) != min(config['buffer_size'], state['learned_steps'] - len(state['pending_n_step']))
                or len(state['pending_n_step']) >= config['n_step']
                or len(state['pending_n_step']) > state['learned_steps']
                or (state['finished'] and state['pending_n_step'])
                or state.get('resource_feedback') is not None):
            raise ValueError('Inconsistent restored counters or pending queue')
        if type(state['finished']) is not bool or state['rollout']:
            raise ValueError('Invalid DQN lifecycle state')
        last_step = 0
        # 更新历史必须连续编号，学习步数递增，并符合更新频率。 / Check update numbering, step order, and training cadence.
        for number, record in enumerate(state['history'], 1):
            if (record['updates'] != number or type(record['learned_steps']) is not int
                    or not last_step < record['learned_steps'] <= state['learned_steps']
                    or record['learned_steps'] < config['learning_starts']
                    or record['learned_steps'] % config['train_frequency']
                    or record['risk_version'] != config['risk_version']
                    or type(record['warmup_steps']) is not int
                    or not 0 <= record['warmup_steps'] <= state['warmup_steps']
                    or record['environment_steps'] != record['learned_steps'] + record['warmup_steps']):
                raise ValueError('Invalid restored training history')
            last_step = record['learned_steps']
        # 待处理记录是单步格式；回放另含实际回报长度。 / Pending rows are single-step; replay rows also store return length.
        for row in state['pending_n_step']:
            if len(row) != 5 or row[4]:
                raise ValueError('Invalid pending transition')
        for row in state['replay']:
            if len(row) != 6 or type(row[5]) is not int or not 1 <= row[5] <= config['n_step']:
                raise ValueError('Invalid replay return length')
    return settings


def train_agent(env, config: dict, previous: dict | None, on_checkpoint,
                on_policy_snapshot=None, policy_snapshot_steps=()) -> TrainingResult:
    """新建或恢复独立多步 DQN 训练，返回策略和完整状态。
    Start or resume isolated multi-step DQN training; return the policy and full state.

    总预算包含预热。单次调用结束时保留未完成的多步队列。
    The total budget includes warmup. Keep pending transitions when a bounded call ends.
    """
    settings = _validate(config, previous)
    requested_snapshots, emitted_snapshots = frozenset(policy_snapshot_steps), set()
    state = None if previous is None else previous['state']
    agent = Agent(env.observation_space, env.action_space.n, config) if state is None else load_agent(state['agent'], config)
    if agent.observation_space != env.observation_space or agent.action_count != env.action_space.n:
        raise ValueError('Native policy environment space mismatch')
    optimizer = torch.optim.Adam(agent.network.parameters(), lr=config['learning_rate'])
    target = deepcopy(agent.network).eval().requires_grad_(False)
    replay, pending, history = deque(maxlen=config['buffer_size']), deque(), []
    learned_steps = warmup_steps = updates = episode = warmup_position = 0
    finished = False
    # 连同未完成队列和随机状态一起恢复，避免重算或丢失回报。 / Restore pending returns and RNG state with training progress.
    if state is not None:
        observation = env.load_snapshot(state['environment'])
        optimizer.load_state_dict(deepcopy(state['optimizer']))
        target.load_state_dict(state['target'])
        replay.extend(deepcopy(state['replay']))
        pending.extend(deepcopy(state['pending_n_step']))
        history = deepcopy(state['history'])
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
    stage_limit = config.get('stage_interaction_limit')
    # 预算边界只暂停调用，不生成终止标志，也不提前排空队列。 / A call boundary pauses training without ending the episode.
    stage_end = config['total_steps'] if stage_limit is None else min(
        config['total_steps'], learned_steps + warmup_steps + stage_limit)

    def checkpoint():
        """复制环境、优化器、回放及未完成队列，供完整恢复。 / Copy environment, optimizer, replay, and pending queue for resume."""
        return deepcopy({'format': 'ablation_dqn_training_v1', 'settings': settings,
                         'risk_version': config['risk_version'], 'agent': agent.checkpoint(),
                         'optimizer': optimizer.state_dict(), 'target': target.state_dict(),
                         'replay': list(replay), 'pending_n_step': list(pending),
                         'rollout': [], 'history': history, 'learned_steps': learned_steps,
                         'warmup_steps': warmup_steps, 'environment_steps': learned_steps + warmup_steps,
                         'updates': updates, 'episode': episode, 'warmup_position': warmup_position,
                         'finished': finished, 'environment': env.snapshot(),
                         'torch_rng': torch.get_rng_state(),
                         'cuda_rng': torch.cuda.get_rng_state_all() if agent.device.type == 'cuda' else []})

    while learned_steps + warmup_steps < stage_end:
        if finished:
            episode += 1
            observation, _ = env.reset(seed=config['seed'] + episode)
            warmup_position, finished = 0, False
        if warmup_position < len(config['warmup_actions']):
            # 预热计入交互预算，但不放入策略学习回放。 / Count warmup in the budget, but not in policy replay.
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
            agent.output(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            learned_steps += 1
            finished = terminated or truncated
            # 先形成多步回放，样本足够且达到更新时点后再训练。 / Build multi-step replay before checking update conditions.
            append_transition(pending, replay,
                              (encoded, action, reward, agent.encode(observation), terminated),
                              config['n_step'], config['gamma'], truncated)
            diagnostics = None
            if learned_steps >= config['learning_starts'] and len(replay) >= config['batch_size'] and learned_steps % config['train_frequency'] == 0:
                diagnostics = update(agent, optimizer, target, replay, config)
                updates += 1
            # 目标网络同步频率按学习步数计算，不包含预热。 / Sync target weights by learning steps, not warmup steps.
            if learned_steps % config['target_update_interval'] == 0:
                target.load_state_dict(agent.network.state_dict())
            if diagnostics is not None:
                history.append({'learned_steps': learned_steps, 'warmup_steps': warmup_steps,
                                'environment_steps': learned_steps + warmup_steps,
                                'updates': updates, 'risk_version': config['risk_version'], **diagnostics})
        snapshot_step = learned_steps + warmup_steps
        # 保存和策略快照按总交互步数触发；保存状态不消费队列。 / Save by total interaction steps without draining the queue.
        if snapshot_step % config['checkpoint_interval'] == 0:
            on_checkpoint(checkpoint())
        if on_policy_snapshot is not None and snapshot_step in requested_snapshots and snapshot_step not in emitted_snapshots:
            on_policy_snapshot({'agent': agent.checkpoint(), 'algorithm': config['algorithm'],
                                'settings': settings, 'risk_version': config['risk_version'],
                                'environment_steps': snapshot_step, 'learned_steps': learned_steps,
                                'warmup_steps': warmup_steps, 'updates': updates})
            emitted_snapshots.add(snapshot_step)
    return TrainingResult(agent, checkpoint())
