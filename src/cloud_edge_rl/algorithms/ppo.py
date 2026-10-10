"""离散 PPO 与 GAE，终局不自举，采样边界停止轨迹。 / Discrete clipped PPO and GAE."""

import numpy as np
import torch


def gae(rewards, values, next_values, terminated, truncated, gamma: float, trace_lambda: float):
    """逆序计算 GAE 优势和价值目标。 / Compute GAE advantages and value targets in reverse.

    输入是一段按时间排列的轨迹。返回值不保留梯度。 / Inputs form a time-ordered rollout.
    Return targets without gradients.
    """
    advantages = torch.empty_like(rewards)
    tail = torch.zeros((), device=rewards.device)
    for i in range(len(rewards) - 1, -1, -1):
        # 终止不自举；截断仍使用下一状态价值。 / Termination stops bootstrap; truncation keeps it.
        delta = rewards[i] + gamma * (~terminated[i]) * next_values[i] - values[i]
        # 两种边界都切断递推，避免混入下一回合。 / Both boundaries stop the trace across episodes.
        tail = delta + gamma * trace_lambda * (~(terminated[i] | truncated[i])) * tail
        advantages[i] = tail
    return advantages.detach(), (advantages + values).detach()


def update(agent, optimizer, rollout, config):
    """用一段已采样轨迹更新 PPO。 / Update PPO from a collected rollout.

    轨迹保存旧策略的价值和动作对数概率。 / The rollout stores old values and action log probabilities.
    返回损失、熵、近似 KL 和裁剪比例。 / Return loss, entropy, approximate KL, and clip fraction.
    """
    # 旧概率保持不变，用于比较新旧策略。 / Keep old probabilities fixed to compare the policies.
    x, actions, rewards, values, next_values, log_probs, terminated, truncated = zip(*rollout)
    tensor = lambda v, dtype=torch.float32: torch.tensor(v, dtype=dtype, device=agent.device)
    inputs = tensor(np.stack(x))
    actions = tensor(actions, torch.long)
    old_log = tensor(log_probs)
    advantages, returns = gae(tensor(rewards), tensor(values), tensor(next_values),
                             tensor(terminated, torch.bool), tensor(truncated, torch.bool),
                             config['gamma'], config['gae_lambda'])
    if config['normalize_advantage'] and len(advantages) > 1:
        # 只缩放优势，不改变价值目标。 / Normalize advantages, not value targets.
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    losses, entropies, kls, clips = [], [], [], []
    # 多次使用同一段轨迹，每次打乱小批次顺序。 / Reuse this rollout with shuffled minibatches.
    for _ in range(config['ppo_epochs']):
        order = agent.rng.permutation(len(rollout))
        for start in range(0, len(order), config['batch_size']):
            ids = tensor(order[start:start + config['batch_size']], torch.long)
            output = agent.network(inputs[ids])
            distribution = torch.distributions.Categorical(logits=output[:, :agent.action_count])
            log_prob = distribution.log_prob(actions[ids])
            ratio = (log_prob - old_log[ids]).exp()
            # 裁剪概率比，限制策略目标的变化。 / Clip the probability ratio in the policy objective.
            policy_loss = -torch.minimum(ratio * advantages[ids],
                                        ratio.clamp(1 - config['clip_range'], 1 + config['clip_range']) * advantages[ids]).mean()
            value_loss = .5 * (output[:, -1] - returns[ids]).square().mean()
            entropy = distribution.entropy().mean()
            # 拟合价值，并用熵项鼓励探索。 / Fit values and use entropy to encourage exploration.
            loss = policy_loss + config['value_coefficient'] * value_loss - config['entropy_coefficient'] * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), config['max_grad_norm'])
            optimizer.step()
            losses.append(loss.item())
            entropies.append(entropy.item())
            kls.append(((ratio - 1) - (log_prob - old_log[ids])).detach().mean().item())
            clips.append(((ratio - 1).abs() > config['clip_range']).float().mean().item())
    return {'ppo_loss': float(np.mean(losses)), 'entropy': float(np.mean(entropies)),
            'approx_kl': float(np.mean(kls)), 'clip_fraction': float(np.mean(clips))}
