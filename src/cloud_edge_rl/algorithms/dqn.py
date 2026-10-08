"""普通 DQN：回放与硬同步目标网络。 / Vanilla DQN, replay and hard target synchronization."""

import torch


def td_targets(rewards, next_values, terminated, gamma: float):
    return (rewards + gamma * (~terminated) * next_values).detach()


def update(agent, optimizer, target, replay, config):
    ids = agent.rng.choice(len(replay), config['batch_size'], replace=False)
    rows = [replay[i] for i in ids]
    import numpy as np

    x, actions, rewards, next_x, done = zip(*rows)
    x = torch.tensor(np.stack(x), device=agent.device)
    next_x = torch.tensor(np.stack(next_x), device=agent.device)
    actions = torch.tensor(actions, device=agent.device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=agent.device)
    done = torch.tensor(done, device=agent.device)
    with torch.no_grad():
        next_values = target(next_x)
        if config['double_dqn']:
            next_values = next_values.gather(1, agent.network(next_x).argmax(1)[:, None])[:, 0]
        else:
            next_values = next_values.max(1).values
        labels = td_targets(rewards, next_values, done, config['gamma'])
    values = agent.network(x).gather(1, actions[:, None])[:, 0]
    loss = torch.nn.functional.smooth_l1_loss(values, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.network.parameters(), config['max_grad_norm'])
    optimizer.step()
    return {'td_loss': loss.item(), 'q_mean': values.detach().mean().item()}
