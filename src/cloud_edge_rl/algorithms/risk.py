"""独立 MLP 集成，运行分组验证，阶段内冻结。 / Run-grouped frozen MLP ensemble."""

from copy import deepcopy
from math import ceil

import numpy as np
import torch
from torch import nn

from ..types import RiskPrediction


def mlp(input_size: int, widths: list[int], output_size: int) -> nn.Sequential:
    layers = []
    for width in widths:
        layers.extend([nn.Linear(input_size, width), nn.ReLU()])
        input_size = width
    return nn.Sequential(*layers, nn.Linear(input_size, output_size))


class RiskEstimator:
    """成员标准差是分歧，不是概率置信界。 / Member dispersion is not a confidence bound."""

    def __init__(self, config: dict, version: str):
        self.config, self.version = deepcopy(config), version
        self.device = torch.device(config['device'])
        if not isinstance(config.get('mean_calibration', False), bool):
            raise ValueError('Mean calibration must be boolean')

    def _build(self):
        return nn.ModuleList([mlp(self.feature_count, self.config['hidden_sizes'], 1)
                              for _ in range(self.config['ensemble_size'])]).to(self.device)

    def _members(self, inputs):
        return torch.cat([member(inputs).sigmoid() for member in self.model], dim=1)

    def fit(self, features: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> None:
        features, targets, groups = np.asarray(features), np.asarray(targets), np.asarray(groups)
        if features.ndim != 2 or targets.shape != (len(features),) or groups.shape != targets.shape:
            raise ValueError('Risk training shapes must be [N,F], [N], [N]')
        if not np.isfinite(features).all() or not np.isfinite(targets).all() or not ((targets >= 0) & (targets <= 1)).all():
            raise ValueError('Risk training data must be finite and labels in [0,1]')
        config = self.config
        if min(config['ensemble_size'], config['epochs'], config['batch_size']) <= 0:
            raise ValueError('Risk training budgets must be positive')
        runs = np.unique(groups)
        count = ceil(len(runs) * config['validation_fraction'])
        if not 0 < count < len(runs):
            raise ValueError('Risk model requires disjoint training and validation runs')
        rng = np.random.default_rng(config['seed'])
        validation_groups = rng.permutation(runs)[:count]
        validation = np.isin(groups, validation_groups)
        self.feature_count = features.shape[1]
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
                if error < best:
                    best, best_epoch = error, epoch + 1
                    weights = deepcopy(self.model.state_dict())
        self.model.load_state_dict(weights)
        self.model.eval().requires_grad_(False)
        self.calibration_offset = 0.
        if config.get('mean_calibration', False):
            # 只在已有内部验证组拟合截距；不使用独立面板或测试标签。
            # Fit an intercept on the existing internal group, never on held-out panel/test labels.
            with torch.inference_mode():
                self.calibration_offset = (y[valid_ids] - self._members(x[valid_ids]).mean(1)).mean().item()
        self.fit_metadata = {'train_groups': np.unique(groups[~validation]).tolist(),
                             'validation_groups': validation_groups.tolist(), 'best_epoch': best_epoch,
                             'validation_mse': best, 'history': history,
                             'uncertainty': 'Empirical member standard deviation; not a confidence bound'}
        if config.get('mean_calibration', False):
            self.fit_metadata['mean_calibration'] = {
                'method': 'internal_validation_mean_residual', 'offset': self.calibration_offset,
                'groups': validation_groups.tolist(), 'confidence_guarantee': False}

    def predict(self, features: np.ndarray) -> RiskPrediction:
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != self.feature_count or not np.isfinite(features).all():
            raise ValueError('Risk prediction feature contract mismatch')
        if not len(features):
            return RiskPrediction(np.empty(0), np.empty(0))
        batches = []
        with torch.inference_mode():
            for start in range(0, len(features), self.config['batch_size']):
                x = (features[start:start + self.config['batch_size']] - self.mean) / self.scale
                batches.append(self._members(torch.tensor(x, dtype=torch.float32, device=self.device)).cpu().numpy())
        members = np.concatenate(batches)
        # 偏差修正仅作用于点估计；成员分歧仍未经置信校准。 / Correct the mean; dispersion remains uncalibrated.
        return RiskPrediction(np.clip(members.mean(1) + self.calibration_offset, 0., 1.), members.std(1))

    def checkpoint(self) -> dict:
        return deepcopy({'format': 'native_risk_v1', 'version': self.version, 'config': self.config,
                         'feature_count': self.feature_count, 'mean': self.mean, 'scale': self.scale,
                          'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                          'fit_metadata': self.fit_metadata, 'calibration_offset': self.calibration_offset})

    def load_checkpoint(self, state: dict) -> None:
        if state['format'] != 'native_risk_v1' or state['version'] != self.version:
            raise ValueError('Risk checkpoint version mismatch')
        settings = lambda c: {**{k: v for k, v in c.items() if k not in ('device', 'mode', 'seed')},
                              'mean_calibration': c.get('mean_calibration', False)}
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
        self.mean, self.scale = state['mean'].copy(), state['scale'].copy()
        with torch.random.fork_rng(devices=[]):
            self.model = self._build()
        self.model.load_state_dict(state['model'])
        self.model.eval().requires_grad_(False)
        self.fit_metadata = deepcopy(state['fit_metadata'])
