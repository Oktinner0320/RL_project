"""从 Notebook 提交独立云端实验。 / Submit isolated cloud runs from the notebook."""

from __future__ import annotations

import tomllib
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import modal
from modal.types import FileEntryType

from cloud_edge_rl.config import RunSpec


def _run_remote(spec_data: dict[str, Any], volume_name: str) -> dict[str, Any]:
    """调用共享流水线，并持久化成功或失败记录。 / Run the shared pipeline and persist its journal."""
    from cloud_edge_rl.pipeline import run_experiment

    volume = modal.Volume.from_name(volume_name)
    # 复用容器需刷新其他任务已提交的输入。 / Reused containers need committed inputs from other jobs.
    volume.reload()
    try:
        return run_experiment(RunSpec(**spec_data), Path("/workspace"))
    finally:
        # 失败日志也必须离开临时容器。 / Persist failure logs before the container exits.
        volume.commit()


def create_app(
    project_root: Path, modal_config: dict[str, Any]
) -> tuple[modal.App, modal.Function, modal.Volume]:
    """从工程依赖和 Notebook 配置组装任务。 / Build a job from dependencies and notebook settings."""
    project_root = Path(project_root).resolve()
    python_version = (project_root / ".python-version").read_text(encoding="utf-8").strip()
    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    uv_version = project["tool"]["uv"]["required-version"].removeprefix("==")
    image = (
        modal.Image.debian_slim(python_version=python_version)
        .uv_sync(
            str(project_root),
            frozen=True,
            extra_options="--no-dev",
            uv_version=uv_version,
        )
        .add_local_python_source("cloud_edge_rl")
    )
    app = modal.App(modal_config["app_name"])
    volume = modal.Volume.from_name(modal_config["volume_name"], create_if_missing=True)
    base_function = app.function(
        name="run_experiment",
        image=image,
        volumes={"/workspace": volume},
        min_containers=0,
        buffer_containers=0,
    )(_run_remote)
    function = base_function.with_options(
        gpu=modal_config["gpu"],
        cpu=modal_config["cpu"],
        memory=modal_config["memory_mb"],
        timeout=modal_config["timeout_seconds"],
        max_containers=modal_config["max_containers"],
        scaledown_window=modal_config["scaledown_window"],
    )
    return app, function, volume


def submit_runs(
    specs: list[RunSpec], project_root: Path, modal_config: dict[str, Any]
) -> list[str]:
    """提交快照并返回调用 ID；预算值不是硬额度。 / Submit snapshots; the budget is not a hard cap."""
    app, function, _ = create_app(project_root, modal_config)
    with modal.enable_output(), app.run(detach=modal_config["detach"]):
        calls = [function.spawn(asdict(spec), modal_config["volume_name"]) for spec in specs]
        call_ids = [call.object_id for call in calls]
        if not modal_config["detach"]:
            # 非分离模式在退出 App 前等待，防止取消尚未完成的输入。 / Wait before closing an attached app.
            for call in calls:
                call.get()
    return call_ids


def get_run(call_id: str, timeout: float | None = None) -> dict[str, Any]:
    """恢复调用句柄并读取结果；timeout=0 用于查询。 / Retrieve a run, or poll with timeout=0."""
    return modal.FunctionCall.from_id(call_id).get(timeout=timeout)


def wait_runs(call_ids: list[str]) -> list[dict[str, Any]]:
    """按提交顺序收集结果。 / Collect results in submission order."""
    return [get_run(call_id) for call_id in call_ids]


def download_run(volume_name: str, run_id: str, destination: Path) -> Path:
    """下载一个实验目录；destination 是产物根。 / Download one run under the destination root."""
    volume = modal.Volume.from_name(volume_name)
    remote_root = PurePosixPath("runs") / run_id
    local_root = Path(destination) / "runs" / run_id
    local_root.mkdir(parents=True, exist_ok=True)
    for entry in volume.iterdir(remote_root.as_posix(), recursive=True):
        if entry.type == FileEntryType.FILE:
            relative = PurePosixPath(entry.path.lstrip("/")).relative_to(remote_root)
            path = local_root.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                for chunk in volume.read_file(entry.path):
                    stream.write(chunk)
    return local_root
