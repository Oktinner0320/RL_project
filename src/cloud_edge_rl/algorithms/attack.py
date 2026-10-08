"""已知标签下的原生余弦梯度反演。 / Native known-label cosine gradient matching."""

from functools import lru_cache

import numpy as np
import torch

from ..data import decode_features
from ..types import AttackInput
from .attack_gradients import relu_parameter_gradients
from .federated import make_model


def categorical_relaxation(inputs: torch.Tensor, metadata: dict, temperature: float) -> torch.Tensor:
    """只用公开布局，将类别块映射至概率单纯形。 / Relax public categorical blocks onto simplices."""
    blocks, cursor = [], 0
    for layout in sorted(metadata['encoding_layout'].values(), key=lambda item: item['start']):
        start, stop = layout['start'], layout['stop']
        if start != cursor or not start < stop <= inputs.shape[-1]:
            raise ValueError('Encoding layout must cover inputs exactly without overlaps')
        block = inputs[..., start:stop]
        blocks.append(block if layout['kind'] == 'numeric' else torch.softmax(block / temperature, dim=-1))
        cursor = stop
    if cursor != inputs.shape[-1]:
        raise ValueError('Encoding layout must cover inputs exactly without overlaps')
    return torch.cat(blocks, dim=-1)


def parallel_gradient_objective(model, inputs, labels, target, metadata, parameterization, temperature):
    """同一可微目标；显式梯度允许编译一阶反传。 / Compile the exact differentiable objective."""
    candidate = (categorical_relaxation(inputs, metadata, temperature)
                 if parameterization == 'categorical_softmax' else inputs)
    gradients = relu_parameter_gradients(model, candidate, labels)
    flat = torch.cat([gradients[name].flatten(1) for name, _ in model.named_parameters()], dim=1)
    objective = 1 - torch.nn.functional.cosine_similarity(flat, target.unsqueeze(0), dim=1)
    return objective, candidate


@lru_cache(maxsize=1)
def compiled_parallel_objective():
    # 不允许静默图断裂回退，也不编译含二阶 autograd 的历史路径。
    # No silent graph-break fallback; the historic nested-autograd path stays untouched.
    return torch.compile(parallel_gradient_objective, fullgraph=True, dynamic=False)


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
        if config.get('input_parameterization', 'unconstrained') not in ('unconstrained', 'categorical_softmax'):
            raise ValueError('Unknown attack input parameterization')
        if not np.isfinite(config.get('categorical_temperature', 1.)) or config.get('categorical_temperature', 1.) <= 0:
            raise ValueError('Categorical temperature must be finite and positive')
        if config.get('restart_execution', 'serial') not in ('serial', 'vectorized_analytic'):
            raise ValueError('Unknown attack restart execution')
        compiled = config.get('compile_gradient_objective', False)
        if not isinstance(compiled, bool) or (compiled and config.get('restart_execution') != 'vectorized_analytic'):
            raise ValueError('Compiled objective requires vectorized_analytic execution and a boolean flag')

    def reconstruct(self, observed: AttackInput):
        config = self.config
        device = torch.device(config['device'])
        if config.get('compile_gradient_objective', False) and device.type != 'cuda':
            raise ValueError('Compiled inversion requires CUDA; use eager vectorized execution on CPU')
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
            optimize = (self._parallel_reconstruct
                        if config.get('restart_execution', 'serial') == 'vectorized_analytic'
                        else self._serial_reconstruct)
            best, reconstruction, initial, final_lr = optimize(model, observed, named, target, labels)
        self.last_diagnostics = {'gradient_objective': best, 'initial_objectives': initial,
                                 'iterations': config['iterations'], 'restarts': config['restarts'],
                                  'learning_rate_schedule': config.get('learning_rate_schedule', 'constant'),
                                  'input_parameterization': config.get('input_parameterization', 'unconstrained'),
                                  'categorical_temperature': config.get('categorical_temperature', 1.),
                                  'restart_execution': config.get('restart_execution', 'serial'),
                                  'objective_compilation': 'inductor' if config.get('compile_gradient_objective', False) else 'eager',
                                 'final_learning_rate': final_lr}
        return decode_features(reconstruction, observed.feature_metadata)

    def _serial_reconstruct(self, model, observed, named, target, labels):
        """保留历史优化顺序。 / Retain the original independent sequential restarts."""
        config, device, dtype = self.config, target.device, target.dtype
        best, reconstruction, initial = float('inf'), None, []
        for restart in range(config['restarts']):
            torch.manual_seed((config['seed'] + restart) % 2**32)
            inputs = torch.randn(observed.batch_size, observed.model_config['input_size'],
                                 device=device, dtype=dtype, requires_grad=True)
            optimizer = torch.optim.Adam([inputs], lr=config['learning_rate'])
            scheduler = None
            if config.get('learning_rate_schedule', 'constant') == 'linear':
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1., end_factor=config['learning_rate_end_factor'],
                    total_iters=config['iterations'])
            for iteration in range(config['iterations'] + 1):
                candidate = (categorical_relaxation(inputs, observed.feature_metadata,
                             config.get('categorical_temperature', 1.))
                             if config.get('input_parameterization', 'unconstrained') == 'categorical_softmax'
                             else inputs)
                loss = model.criterion(model(candidate), labels)
                gradient = torch.autograd.grad(loss, [p for _, p in named], create_graph=True)
                flat = torch.cat([g.flatten() for g in gradient])
                objective = 1 - torch.nn.functional.cosine_similarity(flat, target, dim=0)
                value = objective.item()
                if not np.isfinite(value):
                    raise RuntimeError('Nonfinite gradient inversion objective')
                if iteration == 0:
                    initial.append(value)
                if value < best:
                    best, reconstruction = value, candidate.detach().cpu().numpy().copy()
                if iteration == config['iterations']:
                    break
                optimizer.zero_grad(set_to_none=True)
                inputs.grad = torch.autograd.grad(objective, inputs)[0]
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        return best, reconstruction, initial, optimizer.param_groups[0]['lr']

    def _parallel_reconstruct(self, model, observed, named, target, labels):
        """重启梯度、Adam 矩及最优值互相独立。 / Vectorize without mixing restarts."""
        config, device, dtype = self.config, target.device, target.dtype
        starts = []
        for restart in range(config['restarts']):
            torch.manual_seed((config['seed'] + restart) % 2**32)
            starts.append(torch.randn(observed.batch_size, observed.model_config['input_size'],
                                      device=device, dtype=dtype))
        inputs = torch.stack(starts).requires_grad_()
        optimizer = torch.optim.Adam([inputs], lr=config['learning_rate'])
        scheduler = None
        if config.get('learning_rate_schedule', 'constant') == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1., end_factor=config['learning_rate_end_factor'],
                total_iters=config['iterations'])
        best_values = torch.full((config['restarts'],), float('inf'), device=device, dtype=dtype)
        best_inputs, initial = inputs.detach().clone(), None
        finite = torch.ones((), dtype=torch.bool, device=device)
        objective_fn = (compiled_parallel_objective() if config.get('compile_gradient_objective', False)
                        else parallel_gradient_objective)
        for iteration in range(config['iterations'] + 1):
            objective, candidate = objective_fn(model, inputs, labels, target, observed.feature_metadata,
                                                config.get('input_parameterization', 'unconstrained'),
                                                config.get('categorical_temperature', 1.))
            # 累积设备端标志，避免每步同步 GPU；任何非有限值都禁止返回重建。
            # Keep a device-side flag, avoiding a GPU fence per step; never return after a nonfinite value.
            finite = finite & torch.isfinite(objective).all()
            if iteration == 0:
                initial = objective.detach().cpu().tolist()
            improved = objective.detach() < best_values
            best_inputs = torch.where(improved[:, None, None], candidate.detach(), best_inputs)
            best_values = torch.minimum(best_values, objective.detach())
            if iteration == config['iterations']:
                break
            optimizer.zero_grad(set_to_none=True)
            # sum 保持每个重启的原梯度尺度。 / A sum preserves each independent gradient scale.
            inputs.grad = torch.autograd.grad(objective.sum(), inputs)[0]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if not finite:
            raise RuntimeError('Nonfinite gradient inversion objective')
        restart = int(best_values.argmin().item())
        return (best_values[restart].item(), best_inputs[restart].cpu().numpy().copy(),
                initial, optimizer.param_groups[0]['lr'])
