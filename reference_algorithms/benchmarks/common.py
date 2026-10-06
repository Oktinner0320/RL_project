"""独立实测的配置和来源记录。 / Configuration and provenance for isolated benchmarks."""

import importlib.metadata
import json
import platform
from hashlib import sha256
from pathlib import Path


def notebook_config(root: Path) -> dict:
    """只读取唯一参数单元。 / Read only the sole parameters cell."""
    notebook = json.loads((root / "experiment.ipynb").read_text(encoding="utf-8"))
    (cell,) = [
        cell
        for cell in notebook["cells"]
        if "parameters" in cell.get("metadata", {}).get("tags", [])
    ]
    scope = {}
    exec("".join(cell["source"]), scope)
    return scope["CONFIG"]


def provenance(root: Path, name: str, source: Path, lock: Path, packages: list[str]) -> dict:
    """记录实际代码、依赖和运行版本。 / Record actual source, dependencies and runtime versions."""
    record = json.loads((root / f"reference_algorithms/provenance/{name}.json").read_text())
    return {
        "repository": record,
        "harness_sha256": sha256(source.read_bytes()).hexdigest(),
        "runtime_lock_sha256": sha256(lock.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }
