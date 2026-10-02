"""读取真实 Notebook 参数检验工作流 / Test workflows using the actual notebook parameter cell."""

import json
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import nbformat
import pytest

from cloud_edge_rl import modal_runner, notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def actual_config():
    """仅执行唯一参数单元，不运行任何任务 / Execute only the unique parameter cell, never jobs."""
    document = nbformat.read(PROJECT_ROOT / "experiment.ipynb", as_version=4)
    cells = [cell for cell in document.cells if "parameters" in cell.metadata.get("tags", [])]
    assert len(cells) == 1
    namespace = {}
    exec(compile(cells[0].source, "experiment.ipynb:parameters", "exec"), namespace)
    return namespace["CONFIG"]


@pytest.fixture
def project_copy(tmp_path, monkeypatch):
    """临时 Git 仓库避免修改真实实验 / Use a temporary Git repository without modifying actual experiments."""
    for name in ("pyproject.toml", "uv.lock", ".python-version", "experiment.ipynb"):
        shutil.copy2(PROJECT_ROOT / name, tmp_path / name)
    shutil.copytree(
        PROJECT_ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_notebook_has_one_parameter_cell_and_compiles():
    """验证真实文件格式与全部代码，不执行任务单元 / Validate the real notebook and compile every code cell."""
    document = nbformat.read(PROJECT_ROOT / "experiment.ipynb", as_version=4)
    nbformat.validate(document)
    parameters = [cell for cell in document.cells if "parameters" in cell.metadata.get("tags", [])]
    assert len(parameters) == 1
    assert parameters[0].cell_type == "code"
    for index, cell in enumerate(document.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"experiment.ipynb:cell-{index}", "exec")


def test_job_variants_are_isolated_materialized_snapshots(actual_config, project_copy):
    """每个方法和种子拥有独立快照 / Give each requested variant and seed its own detached snapshot."""
    config = actual_config
    job = config["workflow"]["jobs"]["evaluate"]
    job["variants"] = [
        name for name, variant in config["variants"].items() if variant["method"] == "dqn"
    ]
    before = deepcopy(config)
    specs = notebook.make_job_specs(config, "evaluate", project_copy)
    seeds = config["seeds"][job["seed_group"]]
    assert len(specs) == len(job["variants"]) * len(seeds)
    assert len({spec.run_id for spec in specs}) == len(specs)
    assert config == before
    for spec in specs:
        variant_name = spec.config["experiment"]["variant"]
        variant = config["variants"][variant_name]
        assert variant_name in job["variants"]
        assert spec.seed in seeds
        assert spec.method == variant["method"]
        assert spec.config["risk"]["mode"] == variant["risk_mode"]
        assert spec.config["reward"]["risk_weight"] == variant["risk_weight"]
        assert spec.config["reward"]["uncertainty_weight"] == variant["uncertainty_weight"]
        assert spec.config["execution"]["split"] == job["split"]
    specs[0].config["model"]["hidden_sizes"].append(999)
    specs[0].inputs["agent_checkpoint"] = "changed-only-in-one-snapshot"
    assert specs[1].config["model"] == before["model"]
    assert specs[1].inputs == before["workflow"]["jobs"]["evaluate"]["inputs"]
    assert config == before


def test_local_runtime_result_cannot_hide_modal_validation(actual_config, project_copy):
    """本地检查不能复用为云端验证 / A cached local runtime check must not satisfy a Modal check."""
    local = notebook.make_job_specs(actual_config, "runtime", project_copy)[0]
    remote = notebook.make_job_specs(actual_config, "modal_check", project_copy)[0]
    assert (local.stage, local.method, local.seed) == (remote.stage, remote.method, remote.seed)
    assert local.run_id != remote.run_id
    assert local.config["execution"]["backend"] == "local"
    assert remote.config["execution"]["backend"] == "modal"
    assert local.config["execution"]["job_name"] == "runtime"
    assert remote.config["execution"]["job_name"] == "modal_check"


def test_tuning_specs_use_selected_variants_seeds_and_unique_trials(actual_config, project_copy):
    """搜索仅展开所选方法与训练种子，候选不重复 / Expand selected training trials without duplicates."""
    config = actual_config
    job = config["workflow"]["jobs"]["tune"]
    specs = notebook.make_tuning_specs(config, project_copy)
    seeds = config["seeds"][job["seed_group"]]
    assert len(specs) == config["tuning"]["trials"] * len(job["variants"]) * len(seeds)
    assert len({spec.run_id for spec in specs}) == len(specs)
    assert {spec.seed for spec in specs} == set(seeds)
    assert {spec.seed for spec in specs}.isdisjoint(config["seeds"]["test"])
    assert {spec.method for spec in specs} == {
        config["variants"][name]["method"] for name in job["variants"]
    }
    for variant in job["variants"]:
        for seed in seeds:
            group = [
                spec
                for spec in specs
                if spec.seed == seed
                and spec.config["experiment"]["variant"].startswith(f"{variant}-trial-")
            ]
            values = {
                (
                    spec.config["rl"]["learning_rate"],
                    spec.config["reward"]["risk_weight"],
                    spec.config["reward"]["uncertainty_weight"],
                )
                for spec in group
            }
            assert len(group) == len(values) == config["tuning"]["trials"]
            for learning_rate, risk_weight, uncertainty_weight in values:
                assert learning_rate in config["tuning"]["learning_rate"]
                assert risk_weight in config["tuning"]["risk_weight"]
                assert uncertainty_weight in config["tuning"]["uncertainty_weight"]
    assert [spec.run_id for spec in notebook.make_tuning_specs(config, project_copy)] == [
        spec.run_id for spec in specs
    ]


def test_validation_restores_saved_trial_not_current_tuning_values(actual_config, project_copy):
    """验证沿用训练快照，最终测试种子完全隔离 / Restore trained trial settings and keep final-test seeds out."""
    config = actual_config
    config["workflow"]["jobs"]["tune"]["inputs"]["risk_checkpoint"] = "runs/fitted-risk/risk.pt"
    trials = notebook.make_tuning_specs(config, project_copy)[:2]
    saved = {}
    for spec in trials:
        folder = project_copy / "runs" / spec.run_id
        folder.mkdir(parents=True)
        snapshot = asdict(spec)
        (folder / "config.json").write_text(json.dumps(snapshot), encoding="utf-8")
        saved[spec.run_id] = snapshot
    config["workflow"]["jobs"]["validate"]["inputs"]["training_run_ids"] = list(saved)
    config["workflow"]["jobs"]["validate"]["backend"] = "local"
    config["experiment"]["tag"] = "validation-only-tag"
    config["rl"]["learning_rate"] *= 100
    config["reward"]["risk_weight"] += 100
    config["model"]["hidden_sizes"] = [999]
    before = deepcopy(config)
    validations = notebook.make_validation_specs(config, project_copy)
    validation_job = config["workflow"]["jobs"]["validate"]
    validation_seeds = config["seeds"][validation_job["seed_group"]]
    assert len(validations) == len(trials) * len(validation_seeds)
    for spec in validations:
        training_run_id = Path(spec.inputs["agent_checkpoint"]).parent.name
        original = saved[training_run_id]
        assert spec.stage == "validate"
        assert spec.config["execution"]["split"] == "validation"
        assert spec.config["execution"]["backend"] == validation_job["backend"]
        assert spec.config["execution"]["job_name"] == "validate"
        assert spec.seed in validation_seeds
        assert spec.seed not in config["seeds"]["test"]
        assert spec.method == original["method"]
        assert spec.risk_version == original["risk_version"]
        assert spec.inputs["risk_checkpoint"] == original["inputs"]["risk_checkpoint"]
        for section in ("model", "rl", "reward", "risk"):
            assert spec.config[section] == original["config"][section]
        assert spec.config["experiment"]["tag"] == config["experiment"]["tag"]
    assert config == before


def test_disabled_jobs_do_not_materialize_runs_or_call_cloud(actual_config, tmp_path, monkeypatch):
    """禁用时先返回，不生成目录或调用本地及云端 / Disabled jobs return before any local/cloud work."""

    def forbidden(*args, **kwargs):
        """任何执行尝试都会使测试失败 / Fail on any attempt to execute or materialize a job."""
        raise AssertionError("disabled jobs must have no execution side effects")

    for name in ("make_job_specs", "make_tuning_specs", "make_validation_specs", "run_experiment"):
        monkeypatch.setattr(notebook, name, forbidden)
    monkeypatch.setattr(modal_runner, "submit_runs", forbidden)
    for name, job in actual_config["workflow"]["jobs"].items():
        job["enabled"] = False
        assert notebook.execute_job(actual_config, name, tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_final_evaluation_keeps_selected_trial_configuration(actual_config, project_copy):
    """最终测试不能被当前默认参数覆盖 / Test the chosen trial rather than current defaults."""
    config = actual_config
    config["workflow"]["jobs"]["tune"]["inputs"]["risk_checkpoint"] = "runs/risk/risk.pt"
    trained = notebook.make_tuning_specs(config, project_copy)[0]
    folder = project_copy / "runs" / trained.run_id
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(asdict(trained)), encoding="utf-8")
    config["workflow"]["jobs"]["evaluate"]["inputs"]["training_run_ids"] = [trained.run_id]
    config["reward"]["risk_weight"] = 999
    config["model"]["hidden_sizes"] = [999]
    evaluations = notebook.make_evaluation_specs(config, project_copy)
    assert {item.seed for item in evaluations} == set(config["seeds"]["test"])
    for item in evaluations:
        assert item.stage == "evaluate"
        assert item.config["execution"]["split"] == "test"
        assert item.config["execution"]["job_name"] == "evaluate"
        assert item.config["model"] == trained.config["model"]
        assert item.config["reward"] == trained.config["reward"]
        assert item.inputs["agent_checkpoint"] == f"runs/{trained.run_id}/agent.pt"
