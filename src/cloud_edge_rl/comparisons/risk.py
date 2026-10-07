"""官方 TabM 风险基线，独立于用户核心。 / Official TabM risk baseline, separate from user core."""

from copy import deepcopy
from math import ceil
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..types import RiskPrediction
from .sources import load_module


class ReferenceRiskEstimator:
    """用 TabM/MLP 预测 [N] 经验风险。 / Predict [N] empirical risks with TabM or a matched MLP.

    成员标准差是经验不确定性，不是置信上界。 / Member deviation is not a confidence bound.
    """

    def __init__(self, root: Path, config: dict, version: str):
        """全部超参数来自 Notebook。 / Every hyperparameter comes from the Notebook."""
        self.root, self.config, self.version = Path(root), deepcopy(config), version
        self.device = torch.device(config["device"])
        self.scaler = StandardScaler()

    def _make_model(self, feature_count: int) -> nn.Module:
        """比较同宽 MLP 与固定版本 TabM。 / Compare a same-width MLP to pinned upstream TabM."""
        config = self.config
        if config["model_kind"] == "mlp":
            layers = []
            width = feature_count
            for _ in range(config["n_blocks"]):
                layers.extend(
                    [
                        nn.Linear(width, config["d_block"]),
                        getattr(nn, config["activation"])(),
                        nn.Dropout(config["dropout"]),
                    ]
                )
                width = config["d_block"]
            return nn.Sequential(*layers, nn.Linear(width, 1)).to(self.device)
        if config["model_kind"] != "tabm":
            raise ValueError("未知风险基线 / Unknown risk baseline")
        module = load_module(self.root, "tabm", "tabm.py", "tabm")
        return module.TabM(
            n_num_features=feature_count,
            cat_cardinalities=None,
            d_out=1,
            num_embeddings=None,
            arch_type=config["arch_type"],
            activation=config["activation"],
            start_scaling_init=config["start_scaling_init"],
            n_blocks=config["n_blocks"],
            d_block=config["d_block"],
            dropout=config["dropout"],
            k=config["k"],
        ).to(self.device)

    def _members(self, features: torch.Tensor) -> torch.Tensor:
        """每个成员预测同一恢复率。 / Each ensemble member predicts the same recovery target."""
        predictions = self.model(features).reshape(len(features), -1)
        if self.config["output_transform"] == "sigmoid":
            return predictions.sigmoid()
        if self.config["output_transform"] == "identity":
            return predictions
        raise ValueError("未知风险输出变换 / Unknown risk output transform")

    def fit(self, features: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> None:
        """完整运行分组划分；仅训练部分拟合缩放。 / Split whole runs; scale from training rows only."""
        run_ids = np.unique(groups)
        validation_count = ceil(len(run_ids) * self.config["validation_fraction"])
        if not 0 < validation_count < len(run_ids):
            raise ValueError("TabM 需要互不相交的训练/验证运行 / TabM needs disjoint fitting runs")
        rng = np.random.default_rng(self.config["seed"])
        validation_groups = rng.permutation(run_ids)[:validation_count]
        selected = np.isin(groups, validation_groups)
        train_x = self.scaler.fit_transform(features[~selected]).astype(np.float32)
        validation_x = self.scaler.transform(features[selected]).astype(np.float32)
        train_y = torch.as_tensor(targets[~selected], dtype=torch.float32, device=self.device)
        validation_y = torch.as_tensor(targets[selected], dtype=torch.float32, device=self.device)
        train_x = torch.as_tensor(train_x, device=self.device)
        validation_x = torch.as_tensor(validation_x, device=self.device)
        self.feature_count = features.shape[1]
        history, best_loss = [], float("inf")
        old_threads = torch.get_num_threads()
        torch.set_num_threads(self.config["threads"])
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.config["seed"])
                self.model = self._make_model(self.feature_count)
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config["learning_rate"],
                    weight_decay=self.config["weight_decay"],
                )
                for epoch in range(self.config["epochs"]):
                    self.model.train()
                    order = rng.permutation(len(train_x))
                    accumulated = 0.0
                    for start in range(0, len(order), self.config["batch_size"]):
                        indices = torch.as_tensor(
                            order[start : start + self.config["batch_size"]], device=self.device
                        )
                        members = self._members(train_x[indices])
                        labels = train_y[indices, None].expand_as(members)
                        # 独立监督成员，然后平均损失。 / Supervise members before averaging losses.
                        loss = nn.functional.mse_loss(members, labels)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        accumulated += loss.item() * len(indices)
                    self.model.eval()
                    with torch.inference_mode():
                        prediction = self._members(validation_x).mean(dim=1)
                        validation_loss = nn.functional.mse_loss(prediction, validation_y).item()
                    history.append(
                        {
                            "epoch": epoch + 1,
                            "train_member_mse": accumulated / len(train_x),
                            "validation_ensemble_mse": validation_loss,
                        }
                    )
                    if validation_loss < best_loss:
                        best_loss, best_epoch = validation_loss, epoch + 1
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
                        }
                self.model.load_state_dict(best_state)
        finally:
            torch.set_num_threads(old_threads)
        self.model.eval().requires_grad_(False)
        self.fit_metadata = {
            "train_groups": np.unique(groups[~selected]).tolist(),
            "validation_groups": validation_groups.tolist(),
            "train_rows": int((~selected).sum()),
            "validation_rows": int(selected.sum()),
            "best_epoch": best_epoch,
            "validation_mse": best_loss,
            "history": history,
            "model_kind": self.config["model_kind"],
            "uncertainty": "Empirical member standard deviation (MLP: zero); no confidence guarantee.",
        }

    def predict(self, features: np.ndarray) -> RiskPrediction:
        """只读取特征，按显式批量推理。 / Infer from features only using the configured batch size."""
        scaled = self.scaler.transform(features).astype(np.float32)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(scaled), self.config["batch_size"]):
                inputs = torch.as_tensor(
                    scaled[start : start + self.config["batch_size"]], device=self.device
                )
                batches.append(self._members(inputs).clamp(0, 1).cpu().numpy())
        members = np.concatenate(batches)
        return RiskPrediction(members.mean(axis=1), members.std(axis=1))

    def checkpoint(self) -> dict:
        """保存网络、训练缩放与分组证据。 / Save weights, fitting scaler, and group evidence."""
        return {
            "adapter": self.config["model_kind"],
            "version": self.version,
            "config": deepcopy(self.config),
            "feature_count": self.feature_count,
            "model": {key: value.cpu().clone() for key, value in self.model.state_dict().items()},
            "scaler": {
                "mean": self.scaler.mean_.copy(),
                "scale": self.scaler.scale_.copy(),
                "var": self.scaler.var_.copy(),
                "samples_seen": int(self.scaler.n_samples_seen_),
            },
            "fit_metadata": deepcopy(self.fit_metadata),
        }

    def load_checkpoint(self, state: dict) -> None:
        """模型设置随检查点；运行设备由当前 Notebook 决定。 / Restore architecture; use current runtime device."""
        self.config = {
            **state["config"],
            "device": self.config["device"],
            "threads": self.config["threads"],
        }
        self.version, self.feature_count = state["version"], state["feature_count"]
        self.model = self._make_model(self.feature_count)
        self.model.load_state_dict(state["model"])
        self.model.eval().requires_grad_(False)
        self.scaler.mean_ = state["scaler"]["mean"].copy()
        self.scaler.scale_ = state["scaler"]["scale"].copy()
        self.scaler.var_ = state["scaler"]["var"].copy()
        self.scaler.n_features_in_ = self.feature_count
        self.scaler.n_samples_seen_ = state["scaler"]["samples_seen"]
        self.fit_metadata = deepcopy(state["fit_metadata"])


TabMRiskEstimator = ReferenceRiskEstimator
