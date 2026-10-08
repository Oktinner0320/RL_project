"""已知标签下的原生余弦梯度反演。 / Native known-label cosine gradient matching."""

import numpy as np
import torch

from ..data import decode_features
from ..types import AttackInput
from .federated import make_model


class GradientAttack:
    def __init__(self, config: dict):
        self.config = dict(config)
        if not config['labels_known'] or config['method'] != 'gradient_matching':
            raise ValueError('Native attack supports known-label gradient_matching only')
        if min(config['iterations'], config['restarts']) <= 0 or config['learning_rate'] <= 0:
            raise ValueError('Attack budgets and learning rate must be positive')
        if config.get('learning_rate_schedule', 'constant') not in ('constant', 'linear'):
            raise ValueError('Attack learning rate schedule must be constant or linear')
        if config.get('learning_rate_schedule', 'constant') == 'linear' and not 0 < config['learning_rate_end_factor'] <= 1:
            raise ValueError('Linear attack learning rate end factor must be in (0,1]')

    def reconstruct(self, observed: AttackInput):
        config = self.config
        device = torch.device(config['device'])
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            model = make_model({**observed.model_config, 'device': str(device)})
            model.load_state_dict(observed.model_state)
            model.eval()
            named = list(model.named_parameters())
            if set(observed.gradients) != {name for name, _ in named}:
                raise ValueError('Gradient names differ from source model')
            dtype = next(model.parameters()).dtype
            target = torch.cat([observed.gradients[n].to(device=device, dtype=dtype).flatten()
                                for n, _ in named])
            if not torch.isfinite(target).all() or target.norm() == 0:
                raise ValueError('Cannot invert zero or nonfinite gradients')
            labels = observed.labels.to(device=device, dtype=dtype).reshape(-1, 1)
            if len(labels) != observed.batch_size:
                raise ValueError('Known labels and batch size differ')
            best, reconstruction, initial = float('inf'), None, []
            for restart in range(config['restarts']):
                torch.manual_seed((config['seed'] + restart) % 2**32)
                inputs = torch.randn(observed.batch_size, observed.model_config['input_size'],
                                     device=device, dtype=dtype, requires_grad=True)
                optimizer = torch.optim.Adam([inputs], lr=config['learning_rate'])
                # 与参考的表格优化协议一致；保留旧版恒定步长。 / Match reference; retain legacy constant LR.
                scheduler = None
                if config.get('learning_rate_schedule', 'constant') == 'linear':
                    scheduler = torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=1., end_factor=config['learning_rate_end_factor'],
                        total_iters=config['iterations'])
                for iteration in range(config['iterations'] + 1):
                    loss = model.criterion(model(inputs), labels)
                    gradient = torch.autograd.grad(loss, [p for _, p in named], create_graph=True)
                    flat = torch.cat([g.flatten() for g in gradient])
                    objective = 1 - torch.nn.functional.cosine_similarity(flat, target, dim=0)
                    value = objective.item()
                    if not np.isfinite(value):
                        raise RuntimeError('Nonfinite gradient inversion objective')
                    if iteration == 0:
                        initial.append(value)
                    if value < best:
                        best, reconstruction = value, inputs.detach().cpu().numpy().copy()
                    if iteration == config['iterations']:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    inputs.grad = torch.autograd.grad(objective, inputs)[0]
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
        self.last_diagnostics = {'gradient_objective': best, 'initial_objectives': initial,
                                 'iterations': config['iterations'], 'restarts': config['restarts'],
                                 'learning_rate_schedule': config.get('learning_rate_schedule', 'constant'),
                                 'final_learning_rate': optimizer.param_groups[0]['lr']}
        return decode_features(reconstruction, observed.feature_metadata)
