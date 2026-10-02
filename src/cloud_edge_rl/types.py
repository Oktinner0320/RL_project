"""跨模块数据契约，不含实验默认参数。 / Shared contracts without experiment defaults."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Action:
    """一次客户端选择与统一批量。 / Selected zero-based clients and a common batch size."""

    action_id: int
    client_ids: tuple[int, ...]
    batch_size: int


@dataclass
class ModelSummary:
    """只暴露验证集进度及模型大小。 / Public validation progress and dense model bytes."""

    validation_loss: float
    validation_accuracy: float
    model_bytes: int


@dataclass
class AttackInput:
    """攻击者可见信息；绝不包含原始特征。 / Attacker view, never original features.

    model_state is the PRE-update model; gradients are ONE client's mean batch
    gradients with identical parameter names/shapes. labels has shape [B].
    模型是更新前快照；梯度是一位客户端的批均值；labels 为已知标签 [B]。
    """

    model_state: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor]
    labels: torch.Tensor
    batch_size: int
    feature_metadata: dict
    model_config: dict


@dataclass
class ClientUpdate:
    """受信任的更新记录；索引用于评价而非攻击输入。 / Trusted IDs separate from attack input."""

    client_id: int
    sample_ids: np.ndarray
    attack_input: AttackInput


@dataclass
class RoundResult:
    """一轮真实联邦更新的输出。 / Results after one real federated update."""

    summary: ModelSummary
    updates: list[ClientUpdate]


@dataclass
class RiskPrediction:
    """每个候选客户端更新的预测，形状 [N]。 / Per-query means and uncertainty, shape [N]."""

    mean: np.ndarray
    uncertainty: np.ndarray


@dataclass
class TrainingResult:
    """用户训练器返回策略与可恢复状态。 / User trainer returns agent and resumable state."""

    agent: Any
    checkpoint: dict


@dataclass
class AuditCandidate:
    """已上传更新的审计索引，不含重构真值。 / Uploaded-update manifest without reference rows."""

    upload_id: str
    run_id: str
    episode: int
    round_index: int
    action_id: int
    client_id: int
    batch_size: int
    features: list[float]
    predicted_risk: float | None
    uncertainty: float | None
    decision_margin: float
    risk_version: str
    split: str
    packet_path: str
    reference_path: str
