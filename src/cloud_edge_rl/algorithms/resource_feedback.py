"""完整轨迹阶段的公开资源反馈；不是安全界。 / Stage-frozen resource feedback, not a safety bound."""

from copy import deepcopy
from math import fsum, isfinite
from numbers import Real


def _number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


class ResourceFeedback:
    """仅在完整阶段、PPO buffer 清空后更新乘子。 / Advance only after the stage rollout is consumed."""

    def __init__(self, config: dict, reward: dict):
        keys = {'mode', 'stage_episodes', 'time_limit', 'energy_limit', 'dual_step_size'}
        if not isinstance(config, dict) or set(config) != keys:
            raise ValueError('Resource feedback requires exactly the registered settings')
        if config['mode'] not in ('fixed', 'dual'):
            raise ValueError('Resource feedback mode must be fixed or dual')
        count = config['stage_episodes']
        if type(count) is not int or not 1 <= count <= 3000:
            raise ValueError('Resource stage episodes must be an integer in [1,3000]')
        self.config = deepcopy(config)
        self.reward = {key: _number(reward[key], key, positive=True) for key in
                       ('time_weight', 'time_scale', 'energy_weight', 'energy_scale')}
        self.limits = [_number(config[key], key, positive=True)
                       for key in ('time_limit', 'energy_limit')]
        self.step_size = _number(config['dual_step_size'], 'dual_step_size', positive=True)
        self.coefficients = [self.reward['time_weight'] / self.reward['time_scale'],
                             self.reward['energy_weight'] / self.reward['energy_scale']]
        self.initial = [weight * limit for weight, limit in zip(self.coefficients, self.limits)]
        if not all(isfinite(value) and value > 0 for value in self.initial):
            raise ValueError('Initial resource multipliers must be finite and positive')
        self.multipliers = self.initial.copy()
        self.pending = []
        self.history = []

    @property
    def ready(self) -> bool:
        return len(self.pending) == self.config['stage_episodes']

    @property
    def completed_episodes(self) -> int:
        return len(self.history) * self.config['stage_episodes'] + len(self.pending)

    def adjust_reward(self, reward: float, time_seconds: float, energy_joules: float) -> float:
        """只替换资源罚项；不改变准确率、风险或公开观察。 / Replace only the two resource coefficients."""
        if isinstance(reward, bool) or not isinstance(reward, Real) or not isfinite(reward):
            raise ValueError('Training reward must be finite')
        costs = [_number(time_seconds, 'round time'), _number(energy_joules, 'round energy')]
        # 初始/固定对照严格返回原奖励，避免无意义的浮点抵消。 / Exact identity for the fixed control.
        if self.multipliers == self.initial:
            return float(reward)
        result = float(reward) - fsum((weight / limit - original) * cost
                                     for weight, limit, original, cost in
                                     zip(self.multipliers, self.limits, self.coefficients, costs))
        if not isfinite(result):
            raise ValueError('Resource-adjusted reward must be finite')
        return result

    def record_episode(self, time_seconds: float, energy_joules: float) -> None:
        """传入完整任务的累计 T/E，包含预热。 / Record total task costs including warmup."""
        if self.ready:
            raise ValueError('Consume the completed resource stage before another episode')
        self.pending.append([_number(time_seconds, 'episode time'),
                             _number(energy_joules, 'episode energy')])

    def advance_stage(self) -> dict:
        if not self.ready:
            raise ValueError('Cannot advance an incomplete resource stage')
        means = [fsum(row[index] for row in self.pending) / len(self.pending) for index in (0, 1)]
        excess = [mean / limit - 1. for mean, limit in zip(means, self.limits)]
        before = self.multipliers.copy()
        after = before.copy()
        if self.config['mode'] == 'dual':
            after = [max(0., weight + self.step_size * initial * violation)
                     for weight, initial, violation in zip(before, self.initial, excess)]
        if not all(isfinite(value) for value in after):
            raise ValueError('Updated resource multipliers must be finite')
        record = {'stage': len(self.history), 'episodes': deepcopy(self.pending),
                  'mean_time_seconds': means[0], 'mean_energy_joules': means[1],
                  'normalized_excess': excess, 'multipliers_before': before,
                  'multipliers_after': after}
        self.history.append(record)
        self.pending = []
        self.multipliers = after.copy()
        return deepcopy(record)

    def snapshot(self) -> dict:
        return deepcopy({'format': 'resource_feedback_v1', 'config': self.config,
                         'reward': self.reward, 'multipliers': self.multipliers,
                         'pending': self.pending, 'history': self.history})

    def load_snapshot(self, state: dict) -> None:
        if (state.get('format') != 'resource_feedback_v1' or state.get('config') != self.config
                or state.get('reward') != self.reward):
            raise ValueError('Resource feedback snapshot settings mismatch')
        history, pending = state['history'], state['pending']
        if (not isinstance(history, list) or not isinstance(pending, list)
                or len(history) * self.config['stage_episodes'] + len(pending) > 3000
                or len(pending) > self.config['stage_episodes']):
            raise ValueError('Invalid resource feedback snapshot episode count')
        restored = ResourceFeedback(self.config, self.reward)
        for record in history:
            rows = record['episodes']
            if not isinstance(rows, list) or len(rows) != self.config['stage_episodes']:
                raise ValueError('Invalid completed resource stage')
            for row in rows:
                restored.record_episode(*row)
            if restored.advance_stage() != record:
                raise ValueError('Resource feedback snapshot stage history mismatch')
        for row in pending:
            restored.record_episode(*row)
        multipliers = [_number(value, 'restored multiplier') for value in state['multipliers']]
        if multipliers != restored.multipliers:
            raise ValueError('Resource feedback snapshot multiplier mismatch')
        self.multipliers, self.pending, self.history = (
            restored.multipliers, restored.pending, restored.history)
