"""上传后审计与可信评价；结果不直接进入策略。 / Post-upload audit and trusted evaluation."""

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import public_feature_metadata
from .metrics import reconstruction_metrics, sample_marginal_records, sample_random_records
from .types import AttackInput, AuditCandidate, ClientUpdate


def save_upload(
    store,
    root: Path,
    update: ClientUpdate,
    features: np.ndarray,
    prediction: float | None,
    uncertainty: float | None,
    margin: float,
    episode: int,
    round_index: int,
    action_id: int,
    split: str,
    data,
    risk_version: str,
) -> AuditCandidate:
    """分开存攻击者输入和真值；二者仅通过上传 ID 对齐。 / Separate attacker inputs from references."""
    upload_id = f"e{episode}-r{round_index}-c{update.client_id}"
    packet = store.path / "updates" / f"{upload_id}.pt"
    reference = store.path / "references" / f"{upload_id}.json"
    packet.parent.mkdir(exist_ok=True)
    reference.parent.mkdir(exist_ok=True)
    observed = update.attack_input
    # Detach/copy tensors so future global updates cannot change the audit target.
    # 拷贝张量以固定审计对应的更新前状态，去除数据集行号等私有元数据。
    payload = {
        "model_state": {k: v.detach().cpu().clone() for k, v in observed.model_state.items()},
        "gradients": {k: v.detach().cpu().clone() for k, v in observed.gradients.items()},
        "labels": observed.labels.detach().cpu().clone(),
        "batch_size": observed.batch_size,
        "feature_metadata": public_feature_metadata(observed.feature_metadata),
        "model_config": deepcopy(observed.model_config),
    }
    torch.save(payload, packet)
    reference.write_text(
        json.dumps(
            {
                "sample_ids": update.sample_ids.tolist(),
                "records": data.train_records.iloc[update.sample_ids].to_dict(orient="records"),
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    candidate = AuditCandidate(
        upload_id,
        store.path.name,
        episode,
        round_index,
        action_id,
        update.client_id,
        observed.batch_size,
        features.tolist(),
        prediction,
        uncertainty,
        margin,
        risk_version,
        split,
        packet.relative_to(root).as_posix(),
        reference.relative_to(root).as_posix(),
    )
    with (store.path / "uploads.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(candidate), ensure_ascii=False, allow_nan=False) + "\n")
    return candidate


def read_candidates(root: Path, run_ids: list[str]) -> list[AuditCandidate]:
    """仅加载已执行上传的索引。 / Read manifests of already-executed uploads."""
    result = []
    for run_id in run_ids:
        for line in (
            (root / "runs" / run_id / "uploads.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            result.append(AuditCandidate(**json.loads(line)))
    return result


def order_candidates(
    candidates: list[AuditCandidate], policy: str, config: dict, seed: int
) -> list[AuditCandidate]:
    """同一记录池比较审计选点，不读取真实恢复率。 / Rank the same pool without reconstruction labels."""
    if policy == "periodic":
        # Sample within each trajectory, then share the budget across trajectories.
        # 各轨迹分别按周期取样，再交替分配预算，避免前一条轨迹耗尽全部调用。
        groups = {}
        for item in candidates:
            groups.setdefault((item.run_id, item.episode), []).append(item)
        sampled = [items[:: config["period"]] for items in groups.values()]
        return [
            items[index]
            for index in range(max(map(len, sampled), default=0))
            for items in sampled
            if index < len(items)
        ]
    if policy == "random":
        order = np.random.default_rng(seed).permutation(len(candidates))
        return [candidates[i] for i in order]
    if policy == "active":
        if any(item.uncertainty is None for item in candidates):
            raise ValueError(
                "主动审计需要真实风险预测 / Active audits require fitted risk predictions"
            )
        return sorted(
            candidates,
            key=lambda item: item.uncertainty / (item.decision_margin + config["ranking_floor"]),
            reverse=True,
        )
    if policy == "panel":
        # Stratify by action and client without adding counterfactual gradients.
        # 独立面板按动作和客户端轮转抽样，仅使用实际保存的更新。
        groups = {}
        for item in candidates:
            groups.setdefault((item.action_id, item.client_id), []).append(item)
        return [
            items[index]
            for index in range(max(map(len, groups.values())))
            for items in groups.values()
            if index < len(items)
        ]
    if policy == "benchmark":
        by_batch = {}
        for item in candidates:
            by_batch.setdefault(item.batch_size, item)
        return [by_batch[batch] for batch in sorted(by_batch)]
    raise ValueError(f"未知审计策略 / Unknown audit policy: {policy}")


def _synchronize(device: str) -> None:
    """GPU 工作完成后计时才有效。 / Synchronize GPU work for meaningful elapsed timings."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_audits(
    root: Path,
    candidates: list[AuditCandidate],
    attack,
    config: dict,
    policy: str,
    seed: int,
    reference_training: pd.DataFrame,
    store,
) -> dict:
    """执行双预算审计，超时整次计入并停止。 / Enforce call/time budgets; account for a full overrun."""
    ordered = order_candidates(candidates, policy, config, seed)
    elapsed, completed, rows = 0.0, 0, []
    for candidate in ordered:
        estimate = (
            0.0
            if policy == "benchmark"
            else config["estimated_seconds_by_batch"][str(candidate.batch_size)]
        )
        if completed >= config["max_calls"] or elapsed + estimate > config["max_seconds"]:
            break
        observed = AttackInput(**torch.load(root / candidate.packet_path, weights_only=True))
        _synchronize(config["device"])
        start = time.perf_counter()
        # A target gets the same restart seeds under every audit ordering.
        # 每个目标的随机性不随审计顺序变化，确保定期/随机/主动比较公平。
        identity = f"{seed}:{candidate.run_id}:{candidate.upload_id}".encode()
        target_seed = int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")
        target_attack = type(attack)({**attack.config, "seed": target_seed})
        reconstructed = target_attack.reconstruct(observed)
        _synchronize(config["device"])
        seconds = time.perf_counter() - start
        elapsed += seconds
        completed += 1
        # Only the evaluator reads reference rows after reconstruction has finished.
        # 重构结束后评价端才读取真值，绝不将它传给攻击函数。
        reference = json.loads((root / candidate.reference_path).read_text(encoding="utf-8"))
        original = pd.DataFrame(reference["records"])
        metric = reconstruction_metrics(
            original, reconstructed, observed.feature_metadata, config["numeric_tolerance"]
        )
        row = {
            **asdict(candidate),
            **metric,
            "attack_seconds": seconds,
            "gpu_seconds": seconds if config["device"].startswith("cuda") else 0.0,
            "audit_policy": policy,
            "attack_seed": target_seed,
        }
        for name, baseline in (
            (
                "random",
                sample_random_records(
                    reference_training, observed.feature_metadata, candidate.batch_size, target_seed
                ),
            ),
            (
                "marginal",
                sample_marginal_records(reference_training, candidate.batch_size, target_seed),
            ),
        ):
            scores = reconstruction_metrics(
                original, baseline, observed.feature_metadata, config["numeric_tolerance"]
            )
            row[f"{name}_feature_recovery"] = scores["feature_recovery"]
            row[f"{name}_emr"] = scores["emr"]
        rows.append(row)
        # Persist each completed attack, including before a later attack fails.
        # 每次完成就落盘，后续攻击失败也保留已经付出的算力结果。
        with (store.path / "audits.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if elapsed >= config["max_seconds"]:
            break
    summary = {
        "calls": completed,
        "attack_seconds": elapsed,
        "budget_overrun_seconds": max(0.0, elapsed - config["max_seconds"]),
        "available_candidates": len(candidates),
        "covered_actions": sorted({row["action_id"] for row in rows}),
        "covered_clients": sorted({row["client_id"] for row in rows}),
    }
    if policy == "benchmark":
        summary["seconds_by_batch"] = {
            str(row["batch_size"]): row["attack_seconds"] for row in rows
        }
    store.write_json("audit_summary.json", summary)
    return summary


def read_audits(root: Path, audit_run_ids: list[str]) -> pd.DataFrame:
    """读取实际审计记录，是否允许拟合由调用阶段控制。 / Read actual audits; each stage controls permitted splits."""
    rows = []
    for run_id in audit_run_ids:
        rows.extend(
            json.loads(line)
            for line in (root / "runs" / run_id / "audits.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    return pd.DataFrame(rows)


def risk_training_rows(root: Path, audit_run_ids: list[str]) -> pd.DataFrame:
    """最终评价标签不能进入风险拟合。 / Final evaluation labels cannot enter risk fitting."""
    frame = read_audits(root, audit_run_ids)
    if not frame["split"].eq("train").all():
        raise ValueError("只允许训练运行的审计标签 / Risk fitting accepts training runs only")
    return frame
