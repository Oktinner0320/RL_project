"""Notebook 配置快照与源码身份。 / Notebook snapshots and source identity."""

import hashlib
import json
import subprocess
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
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def source_identity(project_root: Path) -> dict:
    """内容摘要覆盖工作区修改，Git 信息仅用于溯源。 / Hash working sources, including dirty edits."""
    names = ["pyproject.toml", "uv.lock", ".python-version", "experiment.ipynb"]
    paths = [project_root / name for name in names]
    paths += sorted((project_root / "src").rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(project_root).as_posix().encode())
        # Notebook outputs/execution counts are results, not executable source.
        # Notebook 输出与执行次数是产物，不应改变源码身份。
        content = path.read_bytes()
        if path.suffix == ".ipynb":
            cells = json.loads(content)["cells"]
            content = json.dumps([(c["cell_type"], c["source"]) for c in cells]).encode()
        digest.update(content)
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
        "source_sha256": digest.hexdigest(),
        "lock_sha256": hashlib.sha256((project_root / "uv.lock").read_bytes()).hexdigest(),
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
) -> RunSpec:
    """从唯一配置入口生成稳定运行标识。 / Build a stable ID from notebook config and sources."""
    provenance = source_identity(project_root)
    payload = json_copy(
        {
            "stage": stage,
            "method": method,
            "seed": seed,
            "risk_version": risk_version,
            "config": config,
            "inputs": inputs,
            "source_sha256": provenance["source_sha256"],
        }
    )
    suffix = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    run_id = f"{stage}-{method}-s{seed}-{suffix}"
    return RunSpec(
        run_id, stage, method, seed, risk_version, payload["config"], payload["inputs"], provenance
    )


def spec_to_dict(spec: RunSpec) -> dict:
    """本地与远端使用同一个 JSON 契约。 / Share the same JSON contract locally and remotely."""
    return json_copy(asdict(spec))
