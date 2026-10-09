"""Notebook 的薄调用层，不包含实验数值默认值。 / Thin notebook helpers with no tuning defaults."""

import json
from copy import deepcopy
from pathlib import Path

from .config import make_run_spec, source_identity
from .pipeline import load_summary, run_experiment
from .tuning import sample_trials


def make_job_specs(config: dict, job_name: str, project_root: Path, provenance: dict | None = None):
    """展开显式方法/种子配置，生成完全物化的运行快照。 / Materialize variant/seed run snapshots."""
    job = config["workflow"]["jobs"][job_name]
    provenance = source_identity(project_root) if provenance is None else provenance
    specs = []
    for variant_name in job["variants"]:
        variant = config["variants"][variant_name]
        for seed in config["seeds"][job["seed_group"]]:
            resolved = deepcopy(config)
            if "components" in variant:
                resolved["components"].update(variant["components"])
            resolved["experiment"]["variant"] = variant_name
            resolved["risk"]["mode"] = variant["risk_mode"]
            resolved["reward"]["risk_weight"] = variant["risk_weight"]
            resolved["reward"]["uncertainty_weight"] = variant["uncertainty_weight"]
            resolved["execution"]["split"] = job["split"]
            resolved["execution"]["backend"] = job["backend"]
            resolved["execution"]["job_name"] = job_name
            specs.append(
                make_run_spec(
                    resolved,
                    job["stage"],
                    variant["method"],
                    seed,
                    job["risk_version"],
                    job["inputs"],
                    project_root,
                    provenance=provenance,
                )
            )
    return specs


def make_tuning_specs(config: dict, project_root: Path):
    """展开同预算验证搜索，保留每个 trial 的参数快照。 / Expand equal-budget validation trials."""
    result = []
    for base in make_job_specs(config, "tune", project_root):
        for trial in sample_trials(config["tuning"]):
            resolved = deepcopy(base.config)
            if base.method == "pqn":
                resolved["comparisons"]["pqn"]["learning_rate"] = trial["learning_rate"]
            else:
                resolved["rl"]["learning_rate"] = trial["learning_rate"]
            resolved["reward"]["risk_weight"] = trial["risk_weight"]
            resolved["reward"]["uncertainty_weight"] = trial["uncertainty_weight"]
            resolved["experiment"]["variant"] += f"-trial-{trial['trial']}"
            result.append(
                make_run_spec(
                    resolved,
                    base.stage,
                    base.method,
                    base.seed,
                    base.risk_version,
                    base.inputs,
                    project_root,
                )
            )
    return result


def _saved_policy_specs(config: dict, job_name: str, project_root: Path):
    """冻结训练设置，验证和最终测试均不覆盖选定超参数。 / Preserve the selected training settings."""
    job = config["workflow"]["jobs"][job_name]
    result = []
    for run_id in job["inputs"]["training_run_ids"]:
        saved = json.loads(
            (project_root / "runs" / run_id / "config.json").read_text(encoding="utf-8")
        )
        for seed in config["seeds"][job["seed_group"]]:
            resolved = deepcopy(saved["config"])
            resolved["experiment"]["tag"] = config["experiment"]["tag"]
            resolved["execution"]["split"] = job["split"]
            resolved["execution"]["backend"] = job["backend"]
            resolved["execution"]["job_name"] = job_name
            resolved["modal"] = deepcopy(config["modal"])
            for section in ("model", "risk", "rl"):
                resolved[section]["device"] = config["model"]["device"]
            if saved["method"] == "pqn":
                resolved["comparisons"]["pqn"]["device"] = config["model"]["device"]
            result.append(
                make_run_spec(
                    resolved,
                    job["stage"],
                    saved["method"],
                    seed,
                    saved["risk_version"],
                    {
                        "risk_checkpoint": saved["inputs"]["risk_checkpoint"],
                        "agent_checkpoint": f"runs/{run_id}/agent.pt",
                    },
                    project_root,
                )
            )
    return result


def make_validation_specs(config: dict, project_root: Path):
    """在共同验证种子上比较训练快照。 / Compare trained snapshots on common validation seeds."""
    return _saved_policy_specs(config, "validate", project_root)


def make_evaluation_specs(config: dict, project_root: Path):
    """最终评价使用选定 trial 的参数，避免重新调参。 / Test selected trials without retuning."""
    return _saved_policy_specs(config, "evaluate", project_root)


def _execute_registered_training(config: dict, project_root: Path) -> list[dict]:
    """Use approved immutable specs, never generate a fresh UUID on notebook resume."""
    from .config import RunSpec
    from .training_pipeline import run_training_pipeline
    from .transport import TransportLimits, read_json

    plan = config["workflow"]["training_pipeline"]
    paths = plan["registered_spec_paths"]
    if (type(paths) is not list or not 1 <= len(paths) <= 32
            or any(type(path) is not str or not path for path in paths)):
        raise ValueError("Register 1..32 approved training spec snapshots in the sole CONFIG first")
    root = Path(project_root).resolve()
    directory = (root / plan["output_directory"]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Training controller output must remain inside the project")
    specs = [RunSpec(**read_json(root / path)) for path in paths]
    if any(item.config["execution"]["backend"] != "modal" for item in specs):
        raise ValueError("Registered cloud training specs must explicitly use Modal")
    run_training_pipeline(specs, root, directory, artifact_destination=root,
                          transport_limits=TransportLimits(**config["modal"]["transport"]))
    return [read_json(directory / "jobs" / item.run_id / "compute-result.json")["result"]
            for item in specs]


def execute_job(config: dict, job_name: str, project_root: Path) -> list[dict]:
    """只执行配置区启用的阶段，云任务ID先持久化。 / Run enabled jobs; persist cloud call IDs first."""
    job = config["workflow"]["jobs"][job_name]
    if not job["enabled"]:
        print(f"未启用 / Disabled: {job_name}")
        return []
    if job_name == "train" and job["backend"] == "modal":
        return _execute_registered_training(config, project_root)
    if job_name == "tune":
        specs = make_tuning_specs(config, project_root)
    elif job_name == "validate":
        specs = make_validation_specs(config, project_root)
    elif job_name == "evaluate":
        specs = make_evaluation_specs(config, project_root)
    else:
        specs = make_job_specs(config, job_name, project_root)
    results, pending = [], []
    for spec in specs:
        path = project_root / "runs" / spec.run_id / "status.json"
        if config["workflow"]["reuse_completed"] and path.exists():
            status = json.loads(path.read_text(encoding="utf-8"))
            if status["state"] != "completed":
                raise FileExistsError(
                    f"运行未完成，请检查日志并使用新tag恢复 / Incomplete run: {spec.run_id}"
                )
            results.append(load_summary(project_root, spec.run_id))
        else:
            pending.append(spec)
    if job["backend"] == "local":
        results.extend(run_experiment(spec, project_root) for spec in pending)
    elif job["backend"] == "modal" and pending:
        from .modal_runner import download_run, submit_runs, wait_runs

        call_ids = submit_runs(pending, project_root, config["modal"])
        folder = project_root / "output" / "submissions"
        folder.mkdir(parents=True, exist_ok=True)
        for spec, call_id in zip(pending, call_ids):
            (folder / f"{spec.run_id}.json").write_text(
                json.dumps({"run_id": spec.run_id, "call_id": call_id}, indent=2), encoding="utf-8"
            )
        results.extend(wait_runs(call_ids, timeout=config["modal"]["timeout_seconds"]
                                 + config["modal"]["startup_timeout_seconds"]))
        for spec in pending:
            download_run(config["modal"]["volume_name"], spec.run_id, project_root)
    elif job["backend"] != "modal":
        raise ValueError(f"未知执行平台 / Unknown backend: {job['backend']}")
    return results
