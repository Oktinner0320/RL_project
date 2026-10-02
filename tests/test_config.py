"""检验真实参数与源码身份 / Test real notebook configuration and source identity."""

import json
import shutil
import subprocess
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import nbformat
import pytest

from cloud_edge_rl.config import make_run_spec, source_identity

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def actual_config():
    """只执行真实参数单元 / Execute only the real parameter cell."""
    document = nbformat.read(PROJECT_ROOT / "experiment.ipynb", as_version=4)
    cells = [cell for cell in document.cells if "parameters" in cell.metadata.get("tags", [])]
    assert len(cells) == 1
    namespace = {}
    exec(compile(cells[0].source, "experiment.ipynb:parameters", "exec"), namespace)
    return namespace["CONFIG"]


@pytest.fixture
def project_copy(tmp_path, monkeypatch):
    """隔离 Git 和文件修改，不改全局配置 / Isolate Git and file edits without changing global configuration."""
    for name in ("pyproject.toml", "uv.lock", ".python-version", "experiment.ipynb"):
        shutil.copy2(PROJECT_ROOT / name, tmp_path / name)
    shutil.copytree(
        PROJECT_ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build_spec(config, inputs, seed, project_root):
    """从 Notebook 作业定义构造输入 / Build inputs from the notebook's actual job definition."""
    job = config["workflow"]["jobs"]["train"]
    method = config["variants"][job["variants"][0]]["method"]
    return make_run_spec(
        config, job["stage"], method, seed, job["risk_version"], inputs, project_root
    )


def test_real_repository_source_identity_reads_owned_git():
    """实际仓库的 Git 所有权不能阻断 Notebook 集成 / Read real repository identity without Git ownership failures."""
    identity = source_identity(PROJECT_ROOT)
    assert len(identity["source_sha256"]) == 64
    assert identity["lock_sha256"] == sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest()
    assert isinstance(identity["git_status"], list)


def test_run_identity_tracks_inputs_and_code_but_ignores_notebook_outputs(
    actual_config, project_copy
):
    """参数、种子和源码改变身份；结果输出不改变 / Inputs and source affect IDs, notebook outputs do not."""
    config = actual_config
    job = config["workflow"]["jobs"]["train"]
    inputs = deepcopy(job["inputs"])
    seed = config["seeds"][job["seed_group"]][0]
    baseline = build_spec(config, inputs, seed, project_copy)
    assert build_spec(config, inputs, seed, project_copy).run_id == baseline.run_id
    assert build_spec(config, inputs, seed + 1, project_copy).run_id != baseline.run_id
    changed = deepcopy(config)
    changed["rl"]["learning_rate"] *= 2
    assert build_spec(changed, inputs, seed, project_copy).run_id != baseline.run_id
    notebook_path = project_copy / "experiment.ipynb"
    document = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = next(cell for cell in document["cells"] if cell["cell_type"] == "code")
    cell["execution_count"] = 999
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["new output\n"]}]
    notebook_path.write_text(json.dumps(document), encoding="utf-8")
    assert build_spec(config, inputs, seed, project_copy).run_id == baseline.run_id
    source = project_copy / "src" / "cloud_edge_rl" / "__init__.py"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# Changed source / 修改源码\n", encoding="utf-8"
    )
    assert build_spec(config, inputs, seed, project_copy).run_id != baseline.run_id
    before = deepcopy(config)
    config["model"]["hidden_sizes"].append(999)
    inputs["risk_checkpoint"] = "changed-after-snapshot"
    assert baseline.config == before
    assert baseline.inputs == before["workflow"]["jobs"]["train"]["inputs"]
