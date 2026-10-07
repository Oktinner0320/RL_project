"""官方小型 MLP 的真实 FedSGD 对照。 / Genuine synchronous FedSGD with the upstream small MLP.

Model source: scaleoutsystems/tabular-GIA, Apache-2.0. Only the orchestration is ours.
模型来自官方仓库；同步聚合与采样适配只用于第三方基线。
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ..data import public_feature_metadata
from ..types import Action, AttackInput, ClientUpdate, ModelSummary, RoundResult
from .sources import load_module


def make_model(root: Path, config: dict):
    """按源模型契约构建，不能根据梯度形状猜测。 / Build from the recorded model contract, not gradient guesses."""
    if (
        config["architecture"] != "tabular_gia_mlp"
        or config["loss"] != "binary_cross_entropy_with_logits"
    ):
        raise ValueError("The upstream attack adapter requires the recorded tabular-GIA binary MLP")
    module = load_module(root, "tabular_gia", "tabular_gia/model/model.py", "model.model")
    return module.TabularMLP(
        d_in=config["input_size"],
        d_hidden=config["hidden_size"],
        d_out=config["output_size"],
        n_hidden_layers=config["hidden_layers"],
        norm=config["normalization"],
        dropout=config["dropout"],
        activation=config["activation"],
        task="binary",
        binary_pos_weight=config["binary_pos_weight"],
    ).to(device=config["device"], dtype=getattr(torch, config["dtype"]))


class TabularFedSGD:
    """所有客户端从同一更新前权重计算均值梯度。 / All clients compute mean gradients at the same pre-update weights."""

    def __init__(self, root: Path, data, config: dict):
        """保留显式模型、评估和优化设置。 / Keep explicit model, evaluation and optimizer settings."""
        self.root, self.data = Path(root), data
        self.config = {**config, "input_size": data.train_x.shape[1]}
        if self.config["optimizer"] != "sgd":
            raise ValueError("This FedSGD baseline uses the configured SGD optimizer")
        if self.config["normalization"] != "none" or self.config["dropout"] != 0:
            raise ValueError("This reference FedSGD uses a deterministic MLP without normalization")
        self.device = torch.device(self.config["device"])
        self.dtype = getattr(torch, self.config["dtype"])
        self.features = torch.as_tensor(data.train_x, dtype=self.dtype, device=self.device)
        self.labels = torch.as_tensor(data.train_y, dtype=self.dtype, device=self.device).reshape(
            -1, 1
        )
        self.feature_metadata = public_feature_metadata(data.metadata)

    def reset(self, seed: int) -> ModelSummary:
        """每个 episode 初始化一次，后续轮次持续更新。 / Initialize once per episode and update continuously thereafter."""
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.model = make_model(self.root, self.config)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            momentum=self.config["momentum"],
            weight_decay=self.config["weight_decay"],
        )
        self.round_index = 0
        return self._summary()

    def step_round(self, action: Action) -> RoundResult:
        """先保存各客户端梯度，再按处理量聚合并更新一次。 / Save individual gradients, then aggregate by sample counts and update once."""
        self.model.eval()
        named = list(self.model.named_parameters())
        source_state = {
            name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()
        }
        updates, gradients = [], []
        for client_id in action.client_ids:
            rows = self.rng.choice(
                self.data.client_indices[client_id], action.batch_size, replace=False
            )
            loss = self.model.criterion(self.model(self.features[rows]), self.labels[rows])
            values = torch.autograd.grad(loss, [parameter for _, parameter in named])
            gradient = {name: value.detach().clone() for (name, _), value in zip(named, values)}
            gradients.append(gradient)
            attack_input = AttackInput(
                model_state=deepcopy(source_state),
                gradients={name: value.cpu().clone() for name, value in gradient.items()},
                labels=self.labels[rows].detach().cpu().reshape(-1).clone(),
                batch_size=len(rows),
                feature_metadata=deepcopy(self.feature_metadata),
                model_config=deepcopy(self.config),
            )
            updates.append(ClientUpdate(client_id, rows.copy(), attack_input))
        self.optimizer.zero_grad(set_to_none=True)
        total = sum(len(update.sample_ids) for update in updates)
        for name, parameter in named:
            parameter.grad = sum(
                gradient[name] * (len(update.sample_ids) / total)
                for update, gradient in zip(updates, gradients)
            )
        self.optimizer.step()
        self.round_index += 1
        return RoundResult(self._summary(), updates)

    def _evaluate(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """分块评价，统一按样本数累计损失。 / Evaluate in chunks and weight losses by sample count."""
        self.model.eval()
        loss_sum, correct = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(labels), self.config["evaluation_batch_size"]):
                stop = start + self.config["evaluation_batch_size"]
                x = torch.as_tensor(features[start:stop], dtype=self.dtype, device=self.device)
                y = torch.as_tensor(
                    labels[start:stop], dtype=self.dtype, device=self.device
                ).reshape(-1, 1)
                logits = self.model(x)
                loss_sum += self.model.criterion(logits, y).item() * len(y)
                correct += (
                    ((logits.sigmoid() >= self.config["classification_threshold"]) == y)
                    .sum()
                    .item()
                )
        return {"loss": loss_sum / len(labels), "accuracy": correct / len(labels)}

    def _summary(self) -> ModelSummary:
        """训练只读取验证进度，测试指标另行获取。 / Training reads validation progress; test metrics have a separate entry."""
        metrics = self._evaluate(self.data.validation_x, self.data.validation_y)
        model_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in self.model.parameters()
        )
        return ModelSummary(metrics["loss"], metrics["accuracy"], model_bytes)

    def evaluate_test(self) -> dict:
        """仅由最终评价阶段调用。 / Called only by the final evaluation stage."""
        return self._evaluate(self.data.test_x, self.data.test_y)

    def checkpoint(self) -> dict:
        """保存模型、优化器和采样 RNG，用于连续状态恢复。 / Save weights, optimizer and sampling RNG for exact continuation."""
        return deepcopy(
            {
                "model_config": self.config,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "sampling_rng": self.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                "round_index": self.round_index,
            }
        )

    def load_checkpoint(self, state: dict) -> None:
        """配置冻结，仅允许迁移运行设备。 / Keep training settings frozen while permitting device relocation."""
        saved = {key: value for key, value in state["model_config"].items() if key != "device"}
        current = {key: value for key, value in self.config.items() if key != "device"}
        if current != saved:
            raise ValueError("FedSGD checkpoint model/training configuration differs")
        self.reset(self.config["seed"])
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.bit_generator.state = deepcopy(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.round_index = state["round_index"]
