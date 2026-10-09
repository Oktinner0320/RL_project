"""从 Notebook 提交独立云端实验。 / Submit isolated cloud runs from the notebook."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from cloud_edge_rl.config import RunSpec
from cloud_edge_rl.admission import check_admission, reserve, bind_call


def _reference_sources(specs: list[RunSpec]) -> tuple[bool, set[str], PurePosixPath | None]:
    """从运行快照推导依赖和源码并集。 / Derive the dependency and source union from run snapshots."""
    names, roots = set(), set()
    enabled = False
    selectors = {
        "backend": {"researcher": None, "tabular_mlp": "tabular_gia"},
        "attack": {"researcher": None, "tabular_gia": "tabular_gia"},
        "risk": {"researcher": None, "mlp": None, "tabm": "tabm"},
    }
    for spec in specs:
        components = spec.config.get("components", {})
        comparison = False
        for component, selected in components.items():
            if component not in selectors or selected not in selectors[component]:
                raise ValueError(f"Unknown comparison component: {component}={selected}")
            if selected != "researcher":
                comparison = True
            repository = selectors[component][selected]
            if repository is not None:
                names.add(repository)
        if spec.method in ("metacs", "pqn"):
            comparison = True
            names.add({"metacs": "metacs_fl", "pqn": "pqn"}[spec.method])
        if spec.stage == "compare_calibration":
            comparison = True
            names.update(("conformalopt", "ocp_unlock"))
        if comparison:
            enabled = True
            roots.add(spec.config["comparisons"]["modal_source_root"])
    if not enabled:
        return False, names, None
    if len(roots) != 1:
        raise ValueError("Comparison runs must use the same Modal source root")
    remote_root = PurePosixPath(roots.pop())
    if not remote_root.is_absolute() or ".." in remote_root.parts:
        raise ValueError("Modal source root must be an absolute container path")
    return True, names, remote_root


def _reference_files(project_root: Path, names: set[str]) -> list[Path]:
    """只选择固定版本源码及归属说明。 / Select pinned source and attribution files only."""
    from .comparisons.sources import require_source

    files = []
    ignored = {".venv", "venv", "__pycache__", ".git"}
    attribution = {"license", "licence", "copying", "notice", "third_party.txt"}
    for name in sorted(names):
        repository = require_source(project_root, name)
        record_path = project_root / "reference_algorithms/provenance" / f"{name}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected_path = repository.relative_to(project_root).as_posix()
        if (
            record["name"] != name
            or record["local_path"] != expected_path
            or re.fullmatch(r"[0-9a-f]{40}", record["commit"]) is None
            or re.fullmatch(r"[0-9a-f]{64}", record["archive_sha256"]) is None
        ):
            raise ValueError(f"Invalid pinned source provenance: {record_path}")
        files.append(record_path)
        files.extend(
            path
            for path in sorted(repository.rglob("*"))
            if path.is_file()
            and not ignored.intersection(path.relative_to(repository).parts)
            and (
                path.suffix == ".py"
                or path.name.lower().split(".")[0] in attribution
                or path.name.lower() in attribution
            )
        )
    return files


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
    project_root: Path, modal_config: dict[str, Any], specs: list[RunSpec] | None = None
) -> tuple[modal.App, modal.Function, modal.Volume]:
    """从锁文件和运行快照组装任务。 / Build from the lockfile and run snapshots.

    specs 可传整个工作流，提前构建同一源码并集镜像。 / Pass a complete workflow to build one union image.
    """
    project_root = Path(project_root).resolve()
    check_admission(specs, project_root)
    from cloud_edge_rl.comparisons.acceptance import _first_json_equal
    if any(not _first_json_equal(modal_config, spec.config['modal']) for spec in specs):
        raise ValueError('Modal execution settings do not match approved specs')
    comparisons, names, remote_root = _reference_sources([] if specs is None else specs)
    files = _reference_files(project_root, names)
    python_version = (project_root / ".python-version").read_text(encoding="utf-8").strip()
    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    uv_version = project["tool"]["uv"]["required-version"].removeprefix("==")
    image = (
        modal.Image.debian_slim(python_version=python_version)
        .uv_sync(
            str(project_root),
            frozen=True,
            extra_options="--no-dev --extra comparisons" if comparisons else "--no-dev",
            uv_version=uv_version,
        )
    )
    if modal_config.get("persistent_compiler_cache", False):
        # 只复用编译器自身校验过的内核，不缓存攻击输入或私人记录。
        # Reuse compiler-keyed kernels, not reconstruction inputs or private records.
        image = image.env({
            "TORCHINDUCTOR_CACHE_DIR": "/workspace/cache/torch_inductor",
            "TRITON_CACHE_DIR": "/workspace/cache/triton",
        })
    image = image.add_local_python_source("cloud_edge_rl")
    for file in files:
        remote = remote_root / file.relative_to(project_root).as_posix()
        image = image.add_local_file(file, remote.as_posix())
    app = modal.App(modal_config["app_name"])
    volume = modal.Volume.from_name(modal_config["volume_name"], create_if_missing=True)
    base_function = app.function(
        name="run_experiment",
        image=image,
        volumes={"/workspace": volume},
        min_containers=0,
        buffer_containers=0,
        startup_timeout=modal_config.get("startup_timeout_seconds"),
        retries=0,
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
    check_admission(specs, project_root)
    app, function, _ = create_app(project_root, modal_config, specs)
    check_admission(specs, project_root)
    with modal.enable_output(), app.run(detach=modal_config["detach"]):
        calls = []
        for spec in specs:
            old_call = reserve(spec, project_root)
            if old_call is not None:
                call = modal.FunctionCall.from_id(old_call)
            else:
                call = function.spawn(asdict(spec), modal_config["volume_name"])
                bind_call(spec, project_root, call.object_id)
            calls.append(call)
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
    """官方 CLI 并发下载目录；失败原样抛出。 / Download concurrently with the supported CLI and propagate failures."""
    remote_root = PurePosixPath("runs") / run_id
    local_parent = Path(destination) / "runs"
    local_parent.mkdir(parents=True, exist_ok=True)
    # CLI 保留远端目录名；UTF-8 支持 Windows 状态符号。 / Keep the remote basename and encode Windows status symbols in UTF-8.
    subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "modal",
            "volume",
            "get",
            volume_name,
            remote_root.as_posix(),
            str(local_parent),
            "--force",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return local_parent / run_id
