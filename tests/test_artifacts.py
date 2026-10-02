"""验证单次实验产物与失败记录。 / Verify isolated artifacts and failure records."""

import csv
import json

import pytest
import torch

from cloud_edge_rl.artifacts import RunStore
from cloud_edge_rl.config import RunSpec


def make_spec(run_id="run-a"):
    """创建测试所需的完整快照。 / Build a complete test snapshot."""
    return RunSpec(
        run_id=run_id,
        stage="evaluate",
        method="dqn",
        seed=7,
        risk_version="v1",
        config={"label": "测试 / test"},
        inputs={},
        provenance={"source_hash": "abc"},
    )


def test_run_records_snapshot_and_completed_status(tmp_path):
    """参数快照与完成状态应可追溯。 / Snapshots and completion must remain traceable."""
    with RunStore(tmp_path, make_spec()) as store:
        snapshot = json.loads((store.path / "config.json").read_text(encoding="utf-8"))
        assert snapshot["config"] == {"label": "测试 / test"}
        assert json.loads((store.path / "status.json").read_text())["state"] == "running"
        store.write_json("summary.json", {"accuracy": 0.75})
    status = json.loads((store.path / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["elapsed_seconds"] >= 0
    metadata = json.loads((store.path / "metadata.json").read_text())
    assert metadata["provenance"]["source_hash"] == "abc"
    assert not list(store.path.glob("*.tmp"))


def test_failed_run_keeps_original_error_and_journal(tmp_path):
    """失败应留下日志且不能被吞掉。 / Failures must be journaled and propagated."""
    error = NotImplementedError("user core / 用户核心")
    with pytest.raises(NotImplementedError) as captured:
        with RunStore(tmp_path, make_spec()) as store:
            raise error
    assert captured.value is error
    status = json.loads((store.path / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert "用户核心" in status["error"]
    assert "NotImplementedError" in (store.path / "error.txt").read_text(encoding="utf-8")


def test_existing_run_is_not_overwritten(tmp_path):
    """重复 run_id 应保持原产物不变。 / Repeated IDs must preserve previous artifacts."""
    with RunStore(tmp_path, make_spec()) as store:
        store.write_json("summary.json", {"value": 1})
    before = (store.path / "status.json").read_bytes()
    with pytest.raises(FileExistsError):
        with RunStore(tmp_path, make_spec()):
            pass
    assert (store.path / "status.json").read_bytes() == before


def test_metrics_follow_header_and_stay_per_run(tmp_path):
    """字典顺序变化不能错位，多个 run 不共写。 / Preserve column alignment and run isolation."""
    with RunStore(tmp_path, make_spec("first")) as first:
        first.append_metrics({"round": 0, "loss": 0.8})
        first.append_metrics({"loss": 0.4, "round": 1})
    with RunStore(tmp_path, make_spec("second")) as second:
        second.append_metrics({"round": 0, "loss": 0.7})
    with (first.path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"round": "0", "loss": "0.8"},
            {"round": "1", "loss": "0.4"},
        ]
    with (second.path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_checkpoint_round_trip_preserves_user_state(tmp_path):
    """框架只序列化状态，不更新训练内容。 / Serialization must preserve user training state."""
    state = {"weights": torch.tensor([1.0, 2.0]), "round": 4, "rng": (1, 2, 3)}
    with RunStore(tmp_path, make_spec()) as store:
        path = store.save_checkpoint(state)
        loaded = store.load_checkpoint()
        assert path == store.path / "checkpoint.pt"
        assert torch.equal(loaded["weights"], state["weights"])
        assert loaded["round"] == 4
        assert loaded["rng"] == (1, 2, 3)
        assert not list(store.path.glob("*.tmp"))
