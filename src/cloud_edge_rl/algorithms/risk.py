"""独立 MLP 集成，运行分组验证，阶段内冻结。 / Run-grouped frozen MLP ensemble."""

from copy import deepcopy
from math import ceil, isfinite
from numbers import Real

import numpy as np
import torch
from torch import nn

from ..types import RiskPrediction


def mlp(input_size: int, widths: list[int], output_size: int) -> nn.Sequential:
    """创建隐藏层使用 ReLU 的全连接网络。 / Build a dense network with ReLU hidden layers."""
    layers = []
    for width in widths:
        layers.extend([nn.Linear(input_size, width), nn.ReLU()])
        input_size = width
    return nn.Sequential(*layers, nn.Linear(input_size, output_size))


def _calibration_strength(config: dict) -> float:
    """检查校准强度，并返回 [0, 1] 内的数值。 / Check calibration strength and return a value in [0, 1]."""
    strength = config.get('calibration_strength', 1.)
    if (isinstance(strength, (bool, np.bool_)) or not isinstance(strength, Real)
            or not 0 <= strength <= 1 or not isfinite(strength)):
        raise ValueError('Calibration strength must be a finite real scalar in [0, 1]')
    if strength != 1 and not config.get('mean_calibration', False):
        raise ValueError('Non-unit calibration strength requires mean calibration')
    return float(strength)


class RiskEstimator:
    """成员标准差是分歧，不是概率置信界。 / Member dispersion is not a confidence bound."""

    def __init__(self, config: dict, version: str):
        """保存风险模型配置和版本，检查校准选项。 / Store risk settings and version; check calibration options."""
        self.config, self.version = deepcopy(config), version
        self.device = torch.device(config['device'])
        if not isinstance(config.get('mean_calibration', False), bool):
            raise ValueError('Mean calibration must be boolean')
        _calibration_strength(config)
        if not isinstance(config.get('calibration_nonnegative', False), bool):
            raise ValueError('Nonnegative calibration must be boolean')
        if config.get('calibration_nonnegative', False) and not config.get('mean_calibration', False):
            raise ValueError('Nonnegative calibration requires mean calibration')
        index = config.get('calibration_feature_index')
        if index is not None and (type(index) is not int or index < 0 or not config.get('mean_calibration', False)):
            raise ValueError('Feature calibration requires mean calibration and a nonnegative integer index')

    def _build(self):
        """创建多个结构相同、参数独立的风险网络。 / Build risk networks with the same shape and separate weights."""
        return nn.ModuleList([mlp(self.feature_count, self.config['hidden_sizes'], 1)
                              for _ in range(self.config['ensemble_size'])]).to(self.device)

    def _members(self, inputs):
        """返回各成员的风险估计，形状为样本数乘成员数。 / Return risk estimates as a sample-by-member matrix."""
        return torch.cat([member(inputs).sigmoid() for member in self.model], dim=1)

    def fit(self, features: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> None:
        """按运行组划分数据，训练风险集成模型。 / Split data by run group and train the risk ensemble.

        特征形状为 [N, F]，标签和组编号形状为 [N]。标签范围为 [0, 1]。
        Features have shape [N, F]. Targets and group IDs have shape [N]. Targets are in [0, 1].
        使用内部验证误差选权重；可选校准也只使用内部验证组。
        Select weights by internal validation error. Optional calibration uses the same internal groups.
        """
        features, targets, groups = np.asarray(features), np.asarray(targets), np.asarray(groups)
        if features.ndim != 2 or targets.shape != (len(features),) or groups.shape != targets.shape:
            raise ValueError('Risk training shapes must be [N,F], [N], [N]')
        if not np.isfinite(features).all() or not np.isfinite(targets).all() or not ((targets >= 0) & (targets <= 1)).all():
            raise ValueError('Risk training data must be finite and labels in [0,1]')
        index = self.config.get('calibration_feature_index')
        if index is not None and index >= features.shape[1]:
            raise ValueError('Calibration feature index is outside the feature contract')
        config = self.config
        if min(config['ensemble_size'], config['epochs'], config['batch_size']) <= 0:
            raise ValueError('Risk training budgets must be positive')
        runs = np.unique(groups)
        count = ceil(len(runs) * config['validation_fraction'])
        if not 0 < count < len(runs):
            raise ValueError('Risk model requires disjoint training and validation runs')
        rng = np.random.default_rng(config['seed'])
        # 整组划分，避免同一次运行的样本同时进入训练集和验证集。 / Keep each run in only one split.
        validation_groups = rng.permutation(runs)[:count]
        validation = np.isin(groups, validation_groups)
        self.feature_count = features.shape[1]
        # 缩放参数只从训练样本计算；常量特征使用单位尺度。 / Fit scales on training data; use one for constants.
        self.mean = features[~validation].mean(axis=0)
        self.scale = features[~validation].std(axis=0)
        self.scale = np.where(self.scale == 0, 1., self.scale)
        x = torch.tensor((features - self.mean) / self.scale, dtype=torch.float32, device=self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)
        train_ids = np.flatnonzero(~validation)
        valid_ids = torch.tensor(np.flatnonzero(validation), device=self.device)
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        history, best = [], float('inf')
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config['seed'])
            self.model = self._build()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'],
                                          weight_decay=config['weight_decay'])
            for epoch in range(config['epochs']):
                self.model.train()
                order = rng.permutation(train_ids)
                for start in range(0, len(order), config['batch_size']):
                    ids = torch.tensor(order[start:start + config['batch_size']], device=self.device)
                    predictions = self._members(x[ids])
                    # 每个成员拟合同一批标签，不先平均预测。 / Fit each member to the same batch targets.
                    loss = nn.functional.mse_loss(predictions, y[ids, None].expand_as(predictions))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                self.model.eval()
                with torch.no_grad():
                    error = nn.functional.mse_loss(self._members(x[valid_ids]).mean(1), y[valid_ids]).item()
                if not np.isfinite(error):
                    raise RuntimeError('Risk model produced nonfinite validation loss')
                history.append({'epoch': epoch + 1, 'validation_mse': error})
                # 用集成均值的验证误差选择权重，不直接使用最后一轮。 / Select weights by ensemble validation error.
                if error < best:
                    best, best_epoch = error, epoch + 1
                    weights = deepcopy(self.model.state_dict())
        self.model.load_state_dict(weights)
        # 拟合完成后冻结网络，校准另存为偏移量。 / Freeze fitted weights; store calibration as offsets.
        self.model.eval().requires_grad_(False)
        self.calibration_offset = 0.
        self.calibration_values, self.calibration_offsets = np.empty(0), np.empty(0)
        if config.get('mean_calibration', False):
            # 只在已有内部验证组拟合截距；不使用独立面板或测试标签。
            # Fit an intercept on the existing internal group, never on held-out panel/test labels.
            with torch.inference_mode():
                residual = y[valid_ids] - self._members(x[valid_ids]).mean(1)
                self.calibration_offset = residual.mean().item()
            if index is not None:
                # 按指定特征值分别计算残差均值。 / Compute a mean residual for each chosen feature value.
                keys = features[validation, index]
                self.calibration_values = np.unique(keys)
                errors = residual.cpu().numpy()
                self.calibration_offsets = np.asarray([errors[keys == key].mean() for key in self.calibration_values])
        self.fit_metadata = {'train_groups': np.unique(groups[~validation]).tolist(),
                             'validation_groups': validation_groups.tolist(), 'best_epoch': best_epoch,
                             'validation_mse': best, 'history': history,
                             'uncertainty': 'Empirical member standard deviation; not a confidence bound'}
        if config.get('mean_calibration', False):
            self.fit_metadata['mean_calibration'] = {
                'method': 'internal_validation_mean_residual', 'offset': self.calibration_offset,
                'groups': validation_groups.tolist(), 'confidence_guarantee': False}
            if index is not None:
                self.fit_metadata['mean_calibration'].update(
                    method='internal_validation_feature_residual', feature_index=index,
                    values=self.calibration_values.tolist(), offsets=self.calibration_offsets.tolist(),
                    unseen_value_fallback='global_mean_residual')

    def predict(self, features: np.ndarray) -> RiskPrediction:
        """返回校准后的风险均值和原始成员标准差。 / Return calibrated mean risk and raw member standard deviation.

        必须先拟合或加载模型。标准差表示成员分歧，不是置信界。
        Fit or load the model first. Standard deviation measures member disagreement, not a confidence bound.
        """
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != self.feature_count or not np.isfinite(features).all():
            raise ValueError('Risk prediction feature contract mismatch')
        if not len(features):
            return RiskPrediction(np.empty(0), np.empty(0))
        batches = []
        # 使用训练时的缩放参数，分批推断且不计算梯度。 / Reuse training scales and predict batches without gradients.
        with torch.inference_mode():
            for start in range(0, len(features), self.config['batch_size']):
                x = (features[start:start + self.config['batch_size']] - self.mean) / self.scale
                batches.append(self._members(torch.tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy())
        members = np.concatenate(batches)
        raw = members.mean(1)
        # 偏差修正仅作用于点估计；成员分歧仍未经置信校准。 / Correct the mean; dispersion remains uncalibrated.
        correction = self.calibration_offset
        index = self.config.get('calibration_feature_index')
        if index is not None:
            # 只使用内部验证组的公开特征分层；未知值退回已有全局截距。
            # Public-feature strata use internal groups only; unseen values keep the global intercept.
            correction = np.full(len(features), self.calibration_offset)
            for value, offset in zip(self.calibration_values, self.calibration_offsets):
                correction[features[:, index] == value] = offset
        if self.config.get('calibration_nonnegative', False):
            # c⁺ = max(0, c)，仅修正预测均值。 / Project the correction, not fitted offsets.
            correction = np.maximum(correction, 0.)
        calibrated = np.clip(raw + correction, 0., 1.)
        strength = float(self.config.get('calibration_strength', 1.))
        # 裁剪后插值，保留凸组合端点。 / Interpolate after clipping, not the offset.
        mean = calibrated if strength == 1. else raw + strength * (calibrated - raw)
        return RiskPrediction(mean, members.std(1))

    def checkpoint(self) -> dict:
        """复制风险权重、缩放参数、校准值和训练记录。 / Copy risk weights, scales, calibration, and fit records."""
        return deepcopy({'format': 'native_risk_v1', 'version': self.version, 'config': self.config,
                         'feature_count': self.feature_count, 'mean': self.mean, 'scale': self.scale,
                          'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                          'fit_metadata': self.fit_metadata, 'calibration_offset': self.calibration_offset,
                          'calibration_values': self.calibration_values, 'calibration_offsets': self.calibration_offsets})

    def load_checkpoint(self, state: dict) -> None:
        """检查版本和配置，恢复可用于预测的风险模型。 / Check version and settings; restore the risk model for prediction."""
        if state['format'] != 'native_risk_v1' or state['version'] != self.version:
            raise ValueError('Risk checkpoint version mismatch')
        def settings(config):
            """排除运行选项，并补全校准默认值。 / Exclude runtime options and fill calibration defaults."""
            return {**{k: v for k, v in config.items() if k not in ('device', 'mode', 'seed')},
                    'mean_calibration': config.get('mean_calibration', False),
                    'calibration_strength': _calibration_strength(config),
                    'calibration_nonnegative': config.get('calibration_nonnegative', False),
                    'calibration_feature_index': config.get('calibration_feature_index')}
        if settings(state['config']) != settings(self.config):
            raise ValueError('Risk checkpoint configuration mismatch')
        if self.config.get('mean_calibration', False) and 'calibration_offset' not in state:
            raise ValueError('Calibrated risk checkpoint is missing its offset')
        offset = state.get('calibration_offset', 0.)
        if (not np.isfinite(offset) or not -1 <= offset <= 1
                or (not self.config.get('mean_calibration', False) and offset != 0)):
            raise ValueError('Invalid risk mean calibration offset')
        self.calibration_offset = float(offset)
        self.feature_count = state['feature_count']
        index = self.config.get('calibration_feature_index')
        values = np.asarray(state.get('calibration_values', []), dtype=float)
        offsets = np.asarray(state.get('calibration_offsets', []), dtype=float)
        # 分层特征值与偏移量必须一一对应，且全部有效。 / Check that feature values and offsets form valid pairs.
        if index is not None:
            if (index >= self.feature_count or values.ndim != 1 or not len(values) or offsets.shape != values.shape
                    or not np.isfinite(values).all() or not np.isfinite(offsets).all()
                    or len(np.unique(values)) != len(values) or (np.abs(offsets) > 1).any()):
                raise ValueError('Invalid feature calibration checkpoint')
        elif values.size or offsets.size:
            raise ValueError('Unexpected feature calibration checkpoint')
        self.calibration_values, self.calibration_offsets = values.copy(), offsets.copy()
        self.mean, self.scale = state['mean'].copy(), state['scale'].copy()
        # 先按特征数建立网络，再加载权重并冻结。 / Build with the saved feature count, then load and freeze weights.
        with torch.random.fork_rng(devices=[]):
            self.model = self._build()
        self.model.load_state_dict(state['model'])
        self.model.eval().requires_grad_(False)
        self.fit_metadata = deepcopy(state['fit_metadata'])
