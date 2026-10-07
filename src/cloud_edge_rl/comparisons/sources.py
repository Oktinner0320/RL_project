"""加载固定版本的官方源码。 / Load the pinned official sources without copying algorithm code."""

import importlib
import json
import sys
from hashlib import sha256
from pathlib import Path


def require_source(root: Path, name: str) -> Path:
    """缺少源码时明确停止。 / Stop explicitly when the recorded source is missing."""
    repository = Path(root) / "reference_algorithms" / "repos" / name
    if not repository.is_dir():
        raise FileNotFoundError(
            f"Missing {name}: run uv run python reference_algorithms/download.py"
        )
    return repository


def load_module(root: Path, name: str, relative: str, module_name: str):
    """保持官方包导入关系并核对来源。 / Preserve upstream imports and verify the loaded file."""
    repository = require_source(root, name)
    file = repository / relative
    base = file.parents[len(module_name.split(".")) - 1]
    if file.name == "__init__.py":
        base = base.parent
    for path in (repository, base):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != file.resolve():
        raise ImportError(f"{module_name} was loaded from another repository: {module.__file__}")
    return module


def source_root(config: dict, project_root: Path) -> Path:
    """源文件和持久化结果分别定位。 / Locate source assets independently of persisted artifacts."""
    if config["execution"]["backend"] == "modal":
        return Path(config["comparisons"]["modal_source_root"])
    return Path(project_root)


def source_records(root: Path, names: list[str]) -> dict:
    """记录实际源码摘要，避免把修改后的源码当原版。 / Record actual files alongside upstream pins."""
    result = {}
    for name in names:
        repository = require_source(root, name)
        record = json.loads(
            (Path(root) / "reference_algorithms/provenance" / f"{name}.json").read_text()
        )
        files = {
            path.relative_to(repository).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(repository.rglob("*.py"))
        }
        result[name] = {"pin": record, "files_sha256": files}
    return result
