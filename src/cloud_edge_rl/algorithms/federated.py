"""原生同步 FedSGD。 / Native synchronous sample-weighted FedSGD."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from ..data import public_feature_metadata
from ..types import Action, AttackInput, ClientUpdate, ModelSummary, RoundResult


class BinaryMLP(nn.Module):
    """显式二分类模型，与参考小型 MLP 共享参数命名。 / Explicit binary MLP contract."""

    def __init__(self, config: dict):
        super().__init__()
        if config['loss'] != 'binary_cross_entropy_with_logits' or config['output_size'] != 1:
            raise ValueError('Native FedSGD requires binary logits and BCE loss')
        if config['normalization'] != 'none' or config['dropout'] != 0:
            raise ValueError('Native FedSGD requires deterministic layers')
        activations = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}
        self.act = activations[config['activation']]()
        self.hidden = nn.ModuleList()
        width = config['input_size']
        for size in config['hidden_sizes']:
            self.hidden.append(nn.Linear(width, size))
            width = size
        self.out = nn.Linear(width, 1)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1. if config['binary_pos_weight'] is None else config['binary_pos_weight']], dtype=torch.float32)
        )
        for layer in [*self.hidden, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            features = self.act(layer(features))
        return self.out(features)


def make_model(config: dict) -> BinaryMLP:
    if config['architecture'] not in ('native_binary_mlp_v1', 'tabular_gia_mlp'):
        raise ValueError('Unsupported source model architecture')
    config = deepcopy(config)
    if config['architecture'] == 'tabular_gia_mlp':
        config['hidden_sizes'] = [config['hidden_size']] * config['hidden_layers']
    return BinaryMLP(config).to(device=config['device'], dtype=getattr(torch, config['dtype']))


class FederatedBackend:
    """所有客户端在同一更新前状态计算梯度。 / One common pre-update model per round."""

    def __init__(self, data, config: dict):
        self.data = data
        self.config = deepcopy({**config, 'input_size': data.train_x.shape[1],
                                'architecture': 'native_binary_mlp_v1'})
        if config['optimizer'] != 'sgd':
            raise ValueError('Native FedSGD requires SGD')
        self.device = torch.device(config['device'])
        self.dtype = getattr(torch, config['dtype'])
        self.features = torch.as_tensor(data.train_x, device=self.device, dtype=self.dtype)
        self.labels = torch.as_tensor(data.train_y, device=self.device, dtype=self.dtype)[:, None]
        self.metadata = public_feature_metadata(data.metadata)

    def reset(self, seed: int) -> ModelSummary:
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.model = make_model(self.config)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.config['learning_rate'],
                                        momentum=self.config['momentum'],
                                        weight_decay=self.config['weight_decay'])
        self.rng = np.random.default_rng(seed)
        self.round_index = 0
        return self._summary()

    def step_round(self, action: Action) -> RoundResult:
        ids = action.client_ids
        if not ids or len(set(ids)) != len(ids) or action.batch_size <= 0:
            raise ValueError('Invalid FedSGD action')
        if any(i < 0 or i >= len(self.data.client_indices) or
               len(self.data.client_indices[i]) < action.batch_size for i in ids):
            raise ValueError('Client cannot supply the requested batch')
        self.model.eval()
        named = list(self.model.named_parameters())
        source = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        updates, gradients = [], []
        for client in ids:
            rows = self.rng.choice(self.data.client_indices[client], action.batch_size, replace=False)
            loss = self.model.criterion(self.model(self.features[rows]), self.labels[rows])
            values = torch.autograd.grad(loss, [p for _, p in named])
            gradient = {name: value.detach().clone() for (name, _), value in zip(named, values)}
            gradients.append(gradient)
            observed = AttackInput(deepcopy(source),
                                   {k: v.cpu().clone() for k, v in gradient.items()},
                                   self.labels[rows].detach().cpu().reshape(-1).clone(),
                                   len(rows), deepcopy(self.metadata), deepcopy(self.config))
            updates.append(ClientUpdate(client, rows.copy(), observed))
        total = sum(len(u.sample_ids) for u in updates)
        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in named:
            parameter.grad = sum(g[name] * (len(u.sample_ids) / total)
                                 for u, g in zip(updates, gradients))
        self.optimizer.step()
        self.round_index += 1
        return RoundResult(self._summary(), updates)

    def _evaluate(self, features, labels) -> dict:
        if not len(labels):
            raise ValueError('Evaluation split is empty')
        self.model.eval()
        loss, correct = 0., 0
        with torch.no_grad():
            for start in range(0, len(labels), self.config['evaluation_batch_size']):
                end = start + self.config['evaluation_batch_size']
                x = torch.as_tensor(features[start:end], device=self.device, dtype=self.dtype)
                y = torch.as_tensor(labels[start:end], device=self.device, dtype=self.dtype)[:, None]
                logits = self.model(x)
                loss += self.model.criterion(logits, y).item() * len(y)
                correct += ((logits.sigmoid() >= self.config['classification_threshold']) == y).sum().item()
        return {'loss': loss / len(labels), 'accuracy': correct / len(labels)}

    def _summary(self) -> ModelSummary:
        metrics = self._evaluate(self.data.validation_x, self.data.validation_y)
        size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return ModelSummary(metrics['loss'], metrics['accuracy'], size)

    def evaluate_test(self) -> dict:
        return self._evaluate(self.data.test_x, self.data.test_y)

    def checkpoint(self) -> dict:
        return deepcopy({'format': 'native_fedsgd_v1', 'model_config': self.config,
                         'model': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                         'optimizer': self.optimizer.state_dict(),
                         'sampling_rng': self.rng.bit_generator.state, 'round_index': self.round_index})

    def load_checkpoint(self, state: dict) -> None:
        settings = lambda c: {k: v for k, v in c.items() if k != 'device'}
        if state['format'] != 'native_fedsgd_v1' or settings(state['model_config']) != settings(self.config):
            raise ValueError('FedSGD checkpoint configuration mismatch')
        self.reset(self.config['seed'])
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(deepcopy(state['optimizer']))
        self.rng.bit_generator.state = deepcopy(state['sampling_rng'])
        self.round_index = state['round_index']
