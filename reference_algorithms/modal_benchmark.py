"""隔离的官方 TabM 云端测评 / Isolated upstream TabM benchmark on Modal."""

from __future__ import annotations

import json
import platform
import sys
import tomllib
import traceback
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from time import perf_counter

import modal
from modal.types import FileEntryType


def _write_json(path: Path, data: dict) -> None:
    """写入可复核记录 / Save an inspectable benchmark record."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _remote_tabm(packet: dict, volume_name: str) -> dict:
    """调用保存的官方实现并提交产物 / Invoke the saved upstream implementation and commit artifacts."""
    import torch

    sys.path.insert(0, "/study")
    from reference_algorithms.benchmarks.tabular_uncertainty import run_tabm

    volume = modal.Volume.from_name(volume_name)
    volume.reload()
    output = Path("/artifacts") / packet["remote_path"]
    output.mkdir(parents=True, exist_ok=False)
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "rtdl-num-embeddings", "modal")
        },
        "packet": packet,
    }
    _write_json(output / "cloud-runtime.json", runtime)
    started = perf_counter()
    try:
        records = run_tabm(Path("/study"), output, packet["benchmark"], packet["data"])
        result = {
            "status": "completed",
            "remote_path": packet["remote_path"],
            "remote_seconds_including_data_preparation": perf_counter() - started,
            "records": records,
        }
        _write_json(output / "cloud-result.json", result)
        return result
    except Exception:
        # 边界日志保留原错误并抛出。 / Journal the original error at the task boundary and rethrow.
        _write_json(
            output / "cloud-failure.json",
            {
                "status": "failed",
                "traceback": traceback.format_exc(),
                "remote_seconds": perf_counter() - started,
            },
        )
        raise
    finally:
        volume.commit()


def create_tabm_app(project_root: Path, config: dict):
    """从 Notebook 配置和 UV 锁文件构建镜像 / Build from notebook settings and the UV lock."""
    root = Path(project_root).resolve()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    reference = tomllib.loads(
        (root / "reference_algorithms/runtime/pyproject.toml").read_text(encoding="utf-8")
    )
    extras = [
        value
        for value in reference["project"]["dependencies"]
        if value.startswith("rtdl-num-embeddings==")
    ]
    uv_version = project["tool"]["uv"]["required-version"].removeprefix("==")
    image = (
        modal.Image.debian_slim(
            python_version=(root / ".python-version").read_text(encoding="utf-8").strip()
        )
        .uv_sync(str(root), frozen=True, extra_options="--no-dev", uv_version=uv_version)
        .uv_pip_install(extras, uv_version=uv_version, extra_options="--no-deps")
        .add_local_python_source("cloud_edge_rl")
    )
    # 逐文件上传，避免上传下载仓库的数据和虚拟环境。 / Mount only the required files, excluding repository data and virtual environments.
    required = (
        "reference_algorithms/benchmarks/tabular_uncertainty.py",
        "reference_algorithms/repos/tabm/tabm.py",
        "reference_algorithms/runtime/uv.lock",
        "reference_algorithms/catalog.json",
    )
    for relative in required:
        image = image.add_local_file(root / relative, f"/study/{relative}")
    image = image.add_local_file(Path(__file__), "/root/reference_algorithms/modal_benchmark.py")
    settings = config["modal"]
    app = modal.App(settings["app_name"], include_source=False)
    volume = modal.Volume.from_name(settings["volume_name"], create_if_missing=True)
    function = app.function(
        name="reference_tabm",
        image=image,
        volumes={"/artifacts": volume},
        min_containers=0,
        buffer_containers=0,
        include_source=False,
    )(_remote_tabm)
    resource_options = {
        "gpu": settings["gpu"],
        "cpu": settings["cpu"],
        "memory": settings["memory_mb"],
        "timeout": settings["timeout_seconds"],
        "max_containers": settings["max_containers"],
        "scaledown_window": settings["scaledown_window"],
    }
    return (
        app,
        function.with_options(**resource_options),
        volume,
        extras,
        required,
        resource_options,
    )


def download_tabm(volume: modal.Volume, remote_path: str, destination: Path) -> None:
    """回收 CSV、检查点及失败日志 / Download CSV files, checkpoints and failure journals."""
    remote = PurePosixPath(remote_path)
    for entry in volume.iterdir(remote_path, recursive=True):
        if entry.type == FileEntryType.FILE:
            relative = PurePosixPath(entry.path.lstrip("/")).relative_to(remote)
            path = Path(destination).joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                for chunk in volume.read_file(entry.path):
                    stream.write(chunk)


def run_tabm_modal(project_root: Path, config: dict) -> dict:
    """提交一个小规模测评；所有超参数来自 Notebook / Submit one small benchmark with notebook-only parameters."""
    root = Path(project_root).resolve()
    settings = config["reference_benchmarks"]
    benchmark = settings["tabm_modal"]
    destination = root / "reference_algorithms/results" / settings["tag"] / "tabm_modal"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "executed-modal_benchmark.py").write_bytes(Path(__file__).read_bytes())
    app, function, volume, extras, required, resources = create_tabm_app(root, config)
    identity_files = (
        *required,
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "reference_algorithms/modal_benchmark.py",
        *(
            path.relative_to(root).as_posix()
            for path in sorted((root / "src/cloud_edge_rl").rglob("*.py"))
        ),
    )
    packet = {
        "notebook_configuration": config,
        "benchmark": benchmark,
        "data": config["data"],
        "resource_options": resources,
        "image_extra_packages": extras,
        "source_sha256": {
            relative: sha256((root / relative).read_bytes()).hexdigest()
            for relative in identity_files
        },
        "upstream": next(
            item
            for item in json.loads(
                (root / "reference_algorithms/catalog.json").read_text(encoding="utf-8")
            )
            if item["name"] == "tabm"
        ),
    }
    fingerprint = sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()[:12]
    remote_path = f"reference-screen/{settings['tag']}/tabm_modal-{fingerprint}"
    packet["remote_path"] = remote_path
    _write_json(destination / "submission-config.json", packet)
    started = perf_counter()
    with modal.enable_output(), app.run(detach=config["modal"]["detach"]):
        call = function.spawn(packet, config["modal"]["volume_name"])
        journal = {
            "app_id": app.app_id,
            "app_url": app.get_dashboard_url(),
            "call_id": call.object_id,
            "remote_path": remote_path,
        }
        _write_json(destination / "submission.json", journal)
        try:
            result = call.get()
        finally:
            journal["client_wall_seconds_including_build_startup_and_call"] = (
                perf_counter() - started
            )
            _write_json(destination / "submission.json", journal)
            download_tabm(volume, remote_path, destination)
    return result
