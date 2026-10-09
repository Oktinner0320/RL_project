"""Notebook 配置快照与源码身份。 / Notebook snapshots and source identity."""

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    """运行的完整输入；没有隐含实验默认值。 / Complete run input with no hidden defaults."""

    run_id: str
    stage: str
    method: str
    seed: int
    risk_version: str
    config: dict
    inputs: dict
    provenance: dict


def json_copy(value):
    """断开 Notebook 可变对象引用并检查可序列化性。 / Detach values via strict JSON encoding."""
    from .comparisons.acceptance import _first_json

    _first_json(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def source_identity(project_root: Path) -> dict:
    """Freeze source contents without computing file hashes."""
    names = ["pyproject.toml", "uv.lock", ".python-version", "experiment.ipynb"]
    paths = [project_root / name for name in names]
    paths += sorted((project_root / "src").rglob("*.py"))
    paths += sorted((project_root / "reference_algorithms/provenance").glob("*.json"))
    paths += sorted((project_root / "reference_algorithms/repos").rglob("*.py"))
    contents = {}
    for path in paths:
        # Notebook outputs/execution counts are results, not executable source.
        # Notebook 输出与执行次数是产物，不应改变源码身份。
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            cells = json.loads(content)["cells"]
            content = [[c["cell_type"], c["source"]] for c in cells]
        contents[path.relative_to(project_root).as_posix()] = content
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "identity_kind": "source_contents_no_hash",
        "source_contents": contents,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_status": status.stdout.splitlines(),
    }


def make_run_spec(
    config: dict,
    stage: str,
    method: str,
    seed: int,
    risk_version: str,
    inputs: dict,
    project_root: Path,
    provenance: dict | None = None,
) -> RunSpec:
    """Freeze strict JSON inputs with a non-content-derived run identity."""
    if (type(seed) is not int or any(type(value) is not str or not value
                                    for value in (stage, method, risk_version))
            or type(config) is not dict or type(inputs) is not dict):
        raise ValueError('RunSpec fields require explicit exact types')
    provenance = source_identity(project_root) if provenance is None else json_copy(provenance)
    payload = json_copy(
        {
            "stage": stage,
            "method": method,
            "seed": seed,
            "risk_version": risk_version,
            "config": config,
            "inputs": inputs,
        }
    )
    suffix = uuid.uuid4().hex
    run_id = f"{stage}-{method}-s{seed}-{suffix}"
    return RunSpec(
        run_id, stage, method, seed, risk_version, payload["config"], payload["inputs"], provenance
    )


def spec_to_dict(spec: RunSpec) -> dict:
    """本地与远端使用同一个 JSON 契约。 / Share the same JSON contract locally and remotely."""
    return json_copy(asdict(spec))
