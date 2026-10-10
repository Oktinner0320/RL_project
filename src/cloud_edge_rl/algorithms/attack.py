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
    # 按列顺序检查布局；数值列不变，类别块转为概率。 / Keep numeric columns; turn categorical blocks into probabilities.
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


def gradient_guided_initialization(inputs: torch.Tensor, observed: AttackInput) -> torch.Tensor:
    """只用公开第一层梯度作为软初始化，不断言类别不存在。 / A public-gradient soft prior, not a support mask."""
    name = 'hidden.0.weight' if observed.model_config['hidden_sizes'] else 'out.weight'
    gradient = observed.gradients[name].to(device=inputs.device, dtype=inputs.dtype)
    guided = inputs.clone()
    for layout in observed.feature_metadata['encoding_layout'].values():
        if layout['kind'] != 'categorical':
            continue
        start, stop = layout['start'], layout['stop']
        masses = gradient[:, start:stop].norm(dim=0)
        if masses.sum() > 0:
            # 梯度强度转为初始偏好，不屏蔽任何类别。 / Use gradient strength as an initial preference, not a mask.
            prior = (masses / masses.sum()).clamp_min(1e-6).log()
            guided[..., start:stop] += prior
    return guided


def parallel_gradient_objective(model, inputs, labels, target, metadata, parameterization, temperature):
    """同一可微目标；显式梯度允许编译一阶反传。 / Compile the exact differentiable objective."""
    candidate = (categorical_relaxation(inputs, metadata, temperature)
                 if parameterization == 'categorical_softmax' else inputs)
    gradients = relu_parameter_gradients(model, candidate, labels)
    # 按模型参数顺序拼接；每个重启单独比较梯度方向。 / Flatten in parameter order; compare each restart separately.
    flat = torch.cat([gradients[name].flatten(1) for name, _ in model.named_parameters()], dim=1)
    objective = 1 - torch.nn.functional.cosine_similarity(flat, target.unsqueeze(0), dim=1)
    return objective, candidate


@lru_cache(maxsize=2)
def compiled_parallel_objective(mode):
    """按编译模式缓存并返回并行梯度目标。 / Cache and return the compiled parallel objective by mode."""
    # 不允许静默图断裂回退，也不编译含二阶 autograd 的历史路径。
    # No silent graph-break fallback; the historic nested-autograd path stays untouched.
    return torch.compile(parallel_gradient_objective, fullgraph=True, dynamic=False, mode=mode)


class GradientAttack:
    """使用已知标签，通过梯度匹配重建输入。 / Reconstruct inputs by gradient matching with known labels."""

    def __init__(self, config: dict):
        """保存攻击配置，检查预算和执行选项。 / Store attack settings; check budgets and execution options."""
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
        if config.get('compile_mode', 'default') not in ('default', 'reduce-overhead'):
            raise ValueError('Unknown objective compilation mode')
        if config.get('initialization', 'gaussian') not in ('gaussian', 'public_gradient_prior'):
            raise ValueError('Unknown attack initialization')
        if (config.get('initialization') == 'public_gradient_prior'
                and config.get('input_parameterization') != 'categorical_softmax'):
            raise ValueError('Public gradient prior requires categorical_softmax logits')

    def reconstruct(self, observed: AttackInput):
        """从上传梯度和已知标签重建样本，并解码特征。 / Reconstruct samples from uploaded gradients and known labels.

        使用产生梯度时的模型状态。运行记录保存在 last_diagnostics。
        Use the model state that produced the gradients. Store run details in last_diagnostics.
        """
        config = self.config
        device = torch.device(config['device'])
        if config.get('compile_gradient_objective', False) and device.type != 'cuda':
            raise ValueError('Compiled inversion requires CUDA; use eager vectorized execution on CPU')
        devices = [device.index or 0] if device.type == 'cuda' else []
        # 在局部随机状态中加载源模型，避免攻击改变调用方随机序列。 / Keep attack RNG changes local.
        with torch.random.fork_rng(devices=devices):
            model = make_model({**observed.model_config, 'device': str(device)})
            model.load_state_dict(observed.model_state)
            model.eval()
            named = list(model.named_parameters())
            if set(observed.gradients) != {name for name, _ in named}:
                raise ValueError('Gradient names differ from source model')
            dtype = next(model.parameters()).dtype
            # 目标梯度顺序必须与候选梯度的参数顺序一致。 / Use the same parameter order for target and candidate gradients.
            target = torch.cat([observed.gradients[n].to(device=device, dtype=dtype).flatten()
                                for n, _ in named])
            if not torch.isfinite(target).all() or target.norm() == 0:
                raise ValueError('Cannot invert zero or nonfinite gradients')
            labels = observed.labels.to(device=device, dtype=dtype).reshape(-1, 1)
            if len(labels) != observed.batch_size:
                raise ValueError('Known labels and batch size differ')
            # 串行和并行路径都返回目标最小的候选样本。 / Both paths return the candidate with the lowest objective.
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
                                  'compile_mode': config.get('compile_mode', 'default'),
                                  'initialization': config.get('initialization', 'gaussian'),
                                 'final_learning_rate': final_lr}
        # 按公开特征布局解码，不读取评价侧的真实样本。 / Decode with public metadata, not evaluation samples.
        return decode_features(reconstruction, observed.feature_metadata)

    def _serial_reconstruct(self, model, observed, named, target, labels):
        """保留历史优化顺序。 / Retain the original independent sequential restarts."""
        config, device, dtype = self.config, target.device, target.dtype
        best, reconstruction, initial = float('inf'), None, []
        for restart in range(config['restarts']):
            # 每个重启使用自己的种子和优化器，只优化候选输入。 / Seed each restart and optimize only its inputs.
            torch.manual_seed((config['seed'] + restart) % 2**32)
            inputs = torch.randn(observed.batch_size, observed.model_config['input_size'],
                                 device=device, dtype=dtype)
            if config.get('initialization') == 'public_gradient_prior':
                inputs = gradient_guided_initialization(inputs, observed)
            inputs.requires_grad_()
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
                # 保留梯度计算图，才能继续求目标对候选输入的导数。 / Keep the gradient graph to differentiate inputs.
                gradient = torch.autograd.grad(loss, [p for _, p in named], create_graph=True)
                flat = torch.cat([g.flatten() for g in gradient])
                objective = 1 - torch.nn.functional.cosine_similarity(flat, target, dim=0)
                value = objective.item()
                if not np.isfinite(value):
                    raise RuntimeError('Nonfinite gradient inversion objective')
                if iteration == 0:
                    initial.append(value)
                # 保存所有重启、所有评估步中的最佳候选，不只保存最后一步。 / Keep the best candidate across all steps and restarts.
                if value < best:
                    best, reconstruction = value, candidate.detach().cpu().numpy().copy()
                # 最后只评价，确保恰好执行配置次数的优化更新。 / Evaluate the final step without another update.
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
            start = torch.randn(observed.batch_size, observed.model_config['input_size'],
                                device=device, dtype=dtype)
            if config.get('initialization') == 'public_gradient_prior':
                start = gradient_guided_initialization(start, observed)
            starts.append(start)
        # 首维区分重启；每个元素保留自己的 Adam 状态。 / The first axis holds restarts with separate Adam state per element.
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
        compiled = config.get('compile_gradient_objective', False)
        objective_fn = (compiled_parallel_objective(config.get('compile_mode', 'default')) if compiled
                        else parallel_gradient_objective)
        for iteration in range(config['iterations'] + 1):
            optimizer.zero_grad(set_to_none=True)
            if compiled and config.get('compile_mode', 'default') == 'reduce-overhead':
                # 上一步的输出只保留拷贝；反传结束后才开启下一张 CUDA 图。
                # Best outputs are copied; mark a new capture step only after the prior backward finishes.
                torch.compiler.cudagraph_mark_step_begin()
            objective, candidate = objective_fn(model, inputs, labels, target, observed.feature_metadata,
                                                config.get('input_parameterization', 'unconstrained'),
                                                config.get('categorical_temperature', 1.))
            # 累积设备端标志，避免每步同步 GPU；任何非有限值都禁止返回重建。
            # Keep a device-side flag, avoiding a GPU fence per step; never return after a nonfinite value.
            finite = finite & torch.isfinite(objective).all()
            if iteration == 0:
                initial = objective.detach().cpu().tolist()
            # 逐个重启保存最佳候选，最后再选全局最优。 / Keep each restart's best candidate, then choose the overall best.
            improved = objective.detach() < best_values
            best_inputs = torch.where(improved[:, None, None], candidate.detach(), best_inputs)
            best_values = torch.minimum(best_values, objective.detach())
            if iteration == config['iterations']:
                break
            # sum 保持每个重启的原梯度尺度。 / A sum preserves each independent gradient scale.
            inputs.grad = torch.autograd.grad(objective.sum(), inputs)[0]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if not finite:
            raise RuntimeError('Nonfinite gradient inversion objective')
        # 返回最佳重启的拷贝，避免后续修改影响结果。 / Return a copy of the best restart's reconstruction.
        restart = int(best_values.argmin().item())
        return (best_values[restart].item(), best_inputs[restart].cpu().numpy().copy(),
                initial, optimizer.param_groups[0]['lr'])
