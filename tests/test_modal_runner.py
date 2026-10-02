"""用替身检验 Modal 编排，不调用云端。 / Test Modal orchestration without cloud calls."""

import sys
from contextlib import nullcontext
from dataclasses import asdict
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from modal.types import FileEntryType

from cloud_edge_rl import modal_runner
from cloud_edge_rl.config import RunSpec


def modal_config(detach=True):
    """显式提供测试资源配置。 / Supply explicit test resource settings."""
    return {
        "app_name": "test-app",
        "volume_name": "test-volume",
        "gpu": "T4",
        "cpu": 2,
        "memory_mb": 4096,
        "timeout_seconds": 600,
        "max_containers": 1,
        "scaledown_window": 2,
        "budget_usd": 20,
        "detach": detach,
    }


def make_spec():
    """生成独立任务输入。 / Build an isolated task input."""
    return RunSpec(
        run_id="run-a",
        stage="evaluate",
        method="dqn",
        seed=7,
        risk_version="v1",
        config={},
        inputs={},
        provenance={},
    )


def test_factory_uses_locked_dependencies_and_notebook_resources(tmp_path, monkeypatch):
    """镜像不申请 GPU，运行资源来自配置。 / Only the runtime receives configured GPU resources."""
    (tmp_path / ".python-version").write_text("3.12.14\n")
    (tmp_path / "pyproject.toml").write_text('[tool.uv]\nrequired-version = "==0.12.11"\n')
    image = MagicMock()
    image.uv_sync.return_value = image
    image.add_local_python_source.return_value = image
    make_image = MagicMock(return_value=image)
    monkeypatch.setattr(modal_runner.modal.Image, "debian_slim", make_image)
    app = MagicMock()
    base = MagicMock()
    variant = MagicMock()
    base.with_options.return_value = variant
    app.function.return_value.return_value = base
    monkeypatch.setattr(modal_runner.modal, "App", MagicMock(return_value=app))
    volume = MagicMock()
    monkeypatch.setattr(modal_runner.modal.Volume, "from_name", MagicMock(return_value=volume))
    result = modal_runner.create_app(tmp_path, modal_config())
    assert result == (app, variant, volume)
    make_image.assert_called_once_with(python_version="3.12.14")
    image.uv_sync.assert_called_once_with(
        str(tmp_path.resolve()), frozen=True, extra_options="--no-dev", uv_version="0.12.11"
    )
    image.add_local_python_source.assert_called_once_with("cloud_edge_rl")
    assert "gpu" not in app.function.call_args.kwargs
    assert app.function.call_args.kwargs["min_containers"] == 0
    assert app.function.call_args.kwargs["buffer_containers"] == 0
    base.with_options.assert_called_once_with(
        gpu="T4", cpu=2, memory=4096, timeout=600, max_containers=1, scaledown_window=2
    )


@pytest.mark.parametrize("detach", [True, False])
def test_submit_preserves_snapshots_and_attached_lifetime(tmp_path, monkeypatch, detach):
    """分离模式返回 ID，附着模式先等待。 / Detach returns IDs; attached mode waits before exit."""
    app = MagicMock()
    app.run.return_value = nullcontext()
    function = MagicMock()
    call = MagicMock()
    call.object_id = "fc-test"
    function.spawn.return_value = call
    monkeypatch.setattr(modal_runner, "create_app", lambda *args: (app, function, None))
    monkeypatch.setattr(modal_runner.modal, "enable_output", nullcontext)
    spec = make_spec()
    assert modal_runner.submit_runs([spec], tmp_path, modal_config(detach)) == ["fc-test"]
    function.spawn.assert_called_once_with(asdict(spec), "test-volume")
    app.run.assert_called_once_with(detach=detach)
    assert call.get.call_count == (0 if detach else 1)


def test_get_run_restores_saved_call_id(monkeypatch):
    """Notebook 重启后可复用调用 ID。 / A saved call ID survives notebook restarts."""
    call = MagicMock()
    call.get.return_value = {"run_id": "run-a"}
    lookup = MagicMock(return_value=call)
    monkeypatch.setattr(modal_runner.modal.FunctionCall, "from_id", lookup)
    assert modal_runner.get_run("fc-test", timeout=0) == {"run_id": "run-a"}
    lookup.assert_called_once_with("fc-test")
    call.get.assert_called_once_with(timeout=0)


def test_download_uses_volume_file_streams(tmp_path, monkeypatch):
    """保留子目录并按块下载。 / Preserve subdirectories while streaming files."""
    volume = MagicMock()
    volume.iterdir.return_value = [
        SimpleNamespace(path="runs/run-a", type=FileEntryType.DIRECTORY),
        SimpleNamespace(path="/runs/run-a/checkpoints/model.pt", type=FileEntryType.FILE),
    ]
    volume.read_file.return_value = iter([b"abc", b"def"])
    monkeypatch.setattr(modal_runner.modal.Volume, "from_name", MagicMock(return_value=volume))
    path = modal_runner.download_run("test-volume", "run-a", tmp_path)
    assert path == tmp_path / "runs" / "run-a"
    assert (path / "checkpoints" / "model.pt").read_bytes() == b"abcdef"
    volume.iterdir.assert_called_once_with("runs/run-a", recursive=True)


def test_remote_failure_commits_journal_and_propagates(monkeypatch):
    """失败退出仍提交日志，但保留原异常。 / Commit failure records without masking the error."""
    error = NotImplementedError("user core")
    runner = MagicMock(side_effect=error)
    pipeline = ModuleType("cloud_edge_rl.pipeline")
    pipeline.run_experiment = runner
    monkeypatch.setitem(sys.modules, "cloud_edge_rl.pipeline", pipeline)
    volume = MagicMock()
    monkeypatch.setattr(modal_runner.modal.Volume, "from_name", MagicMock(return_value=volume))
    with pytest.raises(NotImplementedError) as captured:
        modal_runner._run_remote(asdict(make_spec()), "test-volume")
    assert captured.value is error
    volume.reload.assert_called_once_with()
    volume.commit.assert_called_once_with()
    assert runner.call_args.args[1].as_posix() == "/workspace"
