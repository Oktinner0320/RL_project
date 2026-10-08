"""ReLU/BCE 的可微闭式梯度。 / Differentiable exact gradients, not an attack surrogate."""

import torch

from .federated import BinaryMLP


def relu_parameter_gradients(
    model: BinaryMLP, inputs: torch.Tensor, labels: torch.Tensor
) -> dict[str, torch.Tensor]:
    """并行独立批次，不混合其梯度。 / Separate gradients for inputs shaped [R,B,F]."""
    if not isinstance(model.act, torch.nn.ReLU) or model.criterion.reduction != 'mean':
        raise ValueError('Analytic attack gradients require ReLU and mean BCE')
    if model.criterion.weight is not None:
        raise ValueError('Analytic attack gradients do not support element weights')
    if inputs.ndim != 3 or not inputs.shape[0] or not inputs.shape[1]:
        raise ValueError('Analytic inputs must have nonempty shape [R,B,F]')
    if labels.shape != (inputs.shape[1], 1):
        raise ValueError('Analytic labels must have shape [B,1]')
    activations = [inputs]
    for layer in model.hidden:
        activations.append(torch.relu(activations[-1] @ layer.weight.detach().T + layer.bias.detach()))
    logits = activations[-1] @ model.out.weight.detach().T + model.out.bias.detach()
    positive_weight = model.criterion.pos_weight
    if positive_weight is None:
        positive_weight = 1.
    # 对软标签也成立；mean 只除当前批次 B，不除独立重启数 R。
    # Valid also for soft labels: reduction averages B, never the independent restarts R.
    delta = ((1 + (positive_weight - 1) * labels) * logits.sigmoid()
             - positive_weight * labels) / inputs.shape[1]
    gradients = {
        'out.weight': delta.transpose(-2, -1) @ activations[-1],
        'out.bias': delta.sum(dim=1),
    }
    delta = delta @ model.out.weight.detach()
    for index in reversed(range(len(model.hidden))):
        delta = delta * (activations[index + 1] > 0)
        gradients[f'hidden.{index}.weight'] = delta.transpose(-2, -1) @ activations[index]
        gradients[f'hidden.{index}.bias'] = delta.sum(dim=1)
        delta = delta @ model.hidden[index].weight.detach()
    return gradients
