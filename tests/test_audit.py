"""验证审计的信息边界与预算 / Test audit information boundaries and budgets."""

import json
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cloud_edge_rl import audit
from cloud_edge_rl.types import AttackInput, ClientUpdate


class LocalStore:
    """仅用于测试的结果目录 / Minimal result directory used only by tests."""

    def __init__(self, path):
        self.path = path
        self.path.mkdir(parents=True)

    def write_json(self, name, value):
        """保存测试结果 / Save test results."""
        (self.path / name).write_text(json.dumps(value), encoding="utf-8")


def example_data():
    """构造带私有来源信息的混合数据 / Build mixed data with deliberately private provenance."""
    metadata = {
        "feature_columns": ["value", "kind"],
        "numeric_columns": ["value"],
        "category_columns": ["kind"],
        "categories": {"kind": ["a", "b", "Missing", "Unknown"]},
        "numeric_mean": {"value": 1.5},
        "numeric_scale": {"value": 1.0},
        "numeric_std": {"value": 1.0},
        "encoding_layout": {
            "value": {"start": 0, "stop": 1, "kind": "numeric"},
            "kind": {"start": 1, "stop": 5, "kind": "categorical"},
        },
        "encoded_columns": ["value", "kind=a", "kind=b", "kind=Missing", "kind=Unknown"],
        "split": {"train_source_indices": [100, 101, 102, 103], "client_indices": [[0, 2], [1, 3]]},
        "private_records": "must never reach the attacker",
    }
    return SimpleNamespace(
        metadata=metadata,
        train_records=pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0], "kind": ["a", "b", "a", "b"]}),
    )


def create_uploads(root, count=3, split="train", batch_sizes=None):
    """生成真实文件契约的测试上传 / Write synthetic uploads using the production file contract."""
    data = example_data()
    source = LocalStore(root / "runs" / "source")
    candidates, updates = [], []
    for index in range(count):
        batch_size = 2 if batch_sizes is None else batch_sizes[index]
        observed = AttackInput(
            model_state={"weight": torch.tensor([float(index)])},
            gradients={"weight": torch.tensor([float(index + 1)])},
            labels=torch.arange(batch_size) % 2,
            batch_size=batch_size,
            feature_metadata=data.metadata,
            model_config={
                "hidden_sizes": [2],
                "activation": "relu",
                "loss": "binary_cross_entropy_with_logits",
                "dtype": "float32",
            },
        )
        update = ClientUpdate(index % 2, np.arange(batch_size), observed)
        candidates.append(
            audit.save_upload(
                source,
                root,
                update,
                np.array([float(index)]),
                0.4,
                [0.1, 0.9, 0.4][index % 3],
                0.2,
                0,
                index + 1,
                index,
                split,
                data,
                "risk-v1",
            )
        )
        updates.append(update)
    return data, candidates, updates


class SeededAttack:
    """有种子的测试替身，不能访问真值 / Seeded test double without access to reference rows."""

    calls = []

    def __init__(self, config):
        self.config = config

    def reconstruct(self, observed):
        """仅凭种子生成测试输出 / Produce test output using only the supplied seed."""
        assert set(vars(observed)) == {field.name for field in fields(AttackInput)}
        assert "split" not in observed.feature_metadata
        assert "private_records" not in observed.feature_metadata
        assert not hasattr(observed, "sample_ids")
        assert not hasattr(observed, "reference_path")
        assert observed.model_config["activation"] == "relu"
        assert observed.model_config["dtype"] == "float32"
        type(self).calls.append(self.config["seed"])
        rng = np.random.default_rng(self.config["seed"])
        return pd.DataFrame(
            {
                "value": rng.uniform(0, 3, observed.batch_size),
                "kind": rng.choice(["a", "b"], observed.batch_size),
            }
        )


def audit_config():
    """显式提供审计参数 / Supply explicit audit parameters."""
    return {
        "period": 1,
        "ranking_floor": 0.01,
        "estimated_seconds_by_batch": {"2": 1.0},
        "max_calls": 3,
        "max_seconds": 100.0,
        "device": "cpu",
        "numeric_tolerance": 0.1,
    }


def read_rows(store):
    """读取逐次审计日志 / Read per-target audit records."""
    return [
        json.loads(line)
        for line in (store.path / "audits.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_saved_attack_packet_is_private_id_free_and_immutable(tmp_path):
    """攻击包无样本行号，且原张量后续变更不影响快照 / Strip row identities and freeze tensors."""
    data, candidates, updates = create_uploads(tmp_path, count=1)
    candidate = candidates[0]
    updates[0].attack_input.model_state["weight"].add_(20)
    updates[0].attack_input.gradients["weight"].add_(20)
    updates[0].attack_input.labels.fill_(7)
    updates[0].attack_input.model_config["hidden_sizes"].append(999)
    packet = torch.load(tmp_path / candidate.packet_path, weights_only=True)
    assert set(packet) == {field.name for field in fields(AttackInput)}
    assert "split" not in packet["feature_metadata"]
    assert "private_records" not in packet["feature_metadata"]
    assert packet["model_state"]["weight"].item() == 0
    assert packet["gradients"]["weight"].item() == 1
    assert packet["labels"].tolist() == [0, 1]
    assert packet["model_config"]["hidden_sizes"] == [2]
    assert packet["model_config"]["loss"] == "binary_cross_entropy_with_logits"
    reference = json.loads((tmp_path / candidate.reference_path).read_text(encoding="utf-8"))
    assert reference["sample_ids"] == [0, 1]
    assert reference["records"] == data.train_records.iloc[:2].to_dict(orient="records")
    assert audit.read_candidates(tmp_path, ["source"]) == candidates


def test_target_seed_and_scores_do_not_depend_on_audit_order(tmp_path):
    """审计排序不能改变同一目标的攻击或基线随机性 / Reordering audits must not change target randomness."""
    data, candidates, _ = create_uploads(tmp_path)
    outputs = {}
    for policy in ("periodic", "active"):
        store = LocalStore(tmp_path / "runs" / policy)
        audit.run_audits(
            tmp_path,
            candidates,
            SeededAttack({"seed": 7}),
            audit_config(),
            policy,
            7,
            data.train_records,
            store,
        )
        outputs[policy] = read_rows(store)
    assert [row["upload_id"] for row in outputs["periodic"]] != [
        row["upload_id"] for row in outputs["active"]
    ]
    compared = [
        "attack_seed",
        "feature_recovery",
        "emr",
        "random_feature_recovery",
        "random_emr",
        "marginal_feature_recovery",
        "marginal_emr",
    ]
    left = {row["upload_id"]: {key: row[key] for key in compared} for row in outputs["periodic"]}
    right = {row["upload_id"]: {key: row[key] for key in compared} for row in outputs["active"]}
    assert left == right
    assert len({row["attack_seed"] for row in outputs["periodic"]}) == len(candidates)


@pytest.mark.parametrize(
    "max_calls,max_seconds,estimate,duration,expected_calls,expected_overrun",
    [(1, 100, 1, 2, 1, 0), (3, 1, 2, 2, 0, 0), (3, 3, 1, 4, 1, 1), (3, 3, 2, 2, 1, 0)],
)
def test_call_and_time_budgets_account_for_whole_attacks(
    tmp_path,
    monkeypatch,
    max_calls,
    max_seconds,
    estimate,
    duration,
    expected_calls,
    expected_overrun,
):
    """调用上限、预计成本与实际超时均生效 / Honor call caps, projected costs, and actual overruns."""
    data, candidates, _ = create_uploads(tmp_path)
    ticks = iter([0.0, float(duration)])
    monkeypatch.setattr(audit, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    config = audit_config()
    config.update(
        max_calls=max_calls, max_seconds=max_seconds, estimated_seconds_by_batch={"2": estimate}
    )
    store = LocalStore(tmp_path / "runs" / "budget")
    summary = audit.run_audits(
        tmp_path,
        candidates,
        SeededAttack({"seed": 7}),
        config,
        "periodic",
        7,
        data.train_records,
        store,
    )
    assert summary["calls"] == expected_calls
    assert summary["budget_overrun_seconds"] == expected_overrun
    assert summary["attack_seconds"] == duration * expected_calls
    if expected_calls:
        assert len(read_rows(store)) == expected_calls
    else:
        assert not (store.path / "audits.jsonl").exists()


def test_active_audits_require_real_uncertainty(tmp_path):
    """没有风险估计时不伪造主动选点 / Do not invent uncertainty for active audit selection."""
    _, candidates, _ = create_uploads(tmp_path, count=1)
    candidates[0].uncertainty = None
    with pytest.raises(ValueError, match="fitted risk"):
        audit.order_candidates(candidates, "active", audit_config(), seed=7)


def test_benchmark_measures_each_batch_without_guessed_costs(tmp_path, monkeypatch):
    """各批量只测一次，成本来自实测 / Measure each batch once using actual timings, not assumed costs."""
    data, candidates, _ = create_uploads(tmp_path, count=4, batch_sizes=[2, 3, 4, 2])
    ticks = iter([0.0, 2.0, 5.0, 8.0, 10.0, 14.0])
    monkeypatch.setattr(audit, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    config = audit_config()
    del config["estimated_seconds_by_batch"]
    store = LocalStore(tmp_path / "runs" / "benchmark")
    result = audit.run_audits(
        tmp_path,
        candidates,
        SeededAttack({"seed": 7}),
        config,
        "benchmark",
        7,
        data.train_records,
        store,
    )
    assert result["calls"] == 3
    assert result["seconds_by_batch"] == {"2": 2.0, "3": 3.0, "4": 4.0}
    assert result["attack_seconds"] == 9.0


def test_held_out_labels_never_enter_risk_training(tmp_path):
    """混入任一测试标签即拒绝拟合 / Reject a training set containing any held-out audit labels."""
    for run_id, split in (("training-audit", "train"), ("test-audit", "test")):
        store = LocalStore(tmp_path / "runs" / run_id)
        row = {"split": split, "run_id": "source", "features": [0.0], "feature_recovery": 0.5}
        (store.path / "audits.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(audit.risk_training_rows(tmp_path, ["training-audit"])) == 1
    with pytest.raises(ValueError, match="training runs only"):
        audit.risk_training_rows(tmp_path, ["training-audit", "test-audit"])
