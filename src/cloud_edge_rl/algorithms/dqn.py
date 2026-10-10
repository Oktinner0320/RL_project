"""普通 DQN：回放与硬同步目标网络。 / Vanilla DQN, replay and hard target synchronization."""

import torch


def td_targets(rewards, next_values, terminated, gamma: float):
    """计算一步 TD 目标，不保留梯度。 / Compute one-step TD targets without gradients."""
    # 真正终止后只使用当前奖励；截断可保留下一状态估值。 / Only termination removes the next value.
    return (rewards + gamma * (~terminated) * next_values).detach()


def update(agent, optimizer, target, replay, config):
    """从回放中采样并更新一次 Q 网络。 / Sample replay and update the Q network once.

    每条记录为状态、动作、奖励、下一状态、终止标志。
    Each row holds state, action, reward, next state, and a termination flag.
    返回 TD 损失与平均 Q 值。 / Return TD loss and mean Q.
    """
    # 无放回采样，随后把记录转为同一设备上的张量。 / Sample without replacement, then build tensors.
    ids = agent.rng.choice(len(replay), config['batch_size'], replace=False)
    rows = [replay[i] for i in ids]
    import numpy as np

    x, actions, rewards, next_x, done = zip(*rows)
    x = torch.tensor(np.stack(x), device=agent.device)
    next_x = torch.tensor(np.stack(next_x), device=agent.device)
    actions = torch.tensor(actions, device=agent.device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=agent.device)
    done = torch.tensor(done, device=agent.device)
    # 目标值不参与反向传播。 / Do not backpropagate through target values.
    with torch.no_grad():
        next_values = target(next_x)
        if config['double_dqn']:
            # 在线网络选动作，目标网络估值。 / The online network selects; the target network evaluates.
            next_values = next_values.gather(1, agent.network(next_x).argmax(1)[:, None])[:, 0]
        else:
            # 普通 DQN 直接使用目标网络的最大值。 / Plain DQN uses the maximum target value.
            next_values = next_values.max(1).values
        labels = td_targets(rewards, next_values, done, config['gamma'])
    # 只拟合实际选择动作的 Q 值，并裁剪梯度。 / Fit Q for chosen actions and clip gradients.
    values = agent.network(x).gather(1, actions[:, None])[:, 0]
    loss = torch.nn.functional.smooth_l1_loss(values, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.network.parameters(), config['max_grad_norm'])
    optimizer.step()
    return {'td_loss': loss.item(), 'q_mean': values.detach().mean().item()}
