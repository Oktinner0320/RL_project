"""主流水线中的第三方闭环比较。 / Closed-loop comparisons through the main pipeline.

算法参数来自 Notebook；这里仅绑定已产生的产物路径。
Algorithm settings come from the Notebook; this module binds generated artifacts only.
"""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import modal

from ..config import RunSpec, make_run_spec, source_identity, spec_to_dict
from ..admission import check_admission, reserve, bind_call
from ..modal_runner import create_app, download_run
from ..notebook import make_job_specs
from ..pipeline import run_experiment


class _Journal:
    """运行边界记录成功或原始异常。 / Journal completion or the original error at the run boundary."""

    def __init__(self, path: Path, config: dict):
        """每个 tag 独占目录，拒绝覆盖。 / Give each tag a new directory without overwriting prior results."""
        path.mkdir(parents=True, exist_ok=False)
        self.path = path / "manifest.json"
        self.data = {
            "state": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "phases": [],
        }

    def save(self):
        """原子写入任务 ID，Notebook 重启仍可追踪。 / Atomically save IDs for notebook restart recovery."""
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)

    def __enter__(self):
        """任何执行之前保存配置。 / Save configuration before executing anything."""
        self.save()
        return self

    def __exit__(self, error_type, error, traceback):
        """原异常继续抛出，不提供替代结果。 / Propagate the original error without substitute results."""
        self.data["state"] = "completed" if error is None else "failed"
        if error is not None:
            self.data["error"] = {"type": error_type.__name__, "message": str(error)}
            if self.data["phases"] and self.data["phases"][-1]["state"] == "running":
                self.data["phases"][-1].update(state="failed", error=self.data["error"])
        self.save()
        return False


def _one(items, purpose: str):
    """产物来源必须唯一，不能任意挑种子。 / Require an unambiguous artifact source instead of picking a seed."""
    if len(items) != 1:
        raise ValueError(f"产物来源需唯一 / Expected exactly one {purpose}, received {len(items)}")
    return items[0]


def _bind(spec, phase: str, finished: dict, settings: dict, root: Path):
    """绑定训练、验证与测试各自来源，不让最终标签回流。 / Bind distinct split sources without recycling test labels."""
    config, inputs = deepcopy(spec.config), deepcopy(spec.inputs)
    source_phases = {
        "comparison_benchmark": "comparison_collect",
        "comparison_audit": "comparison_collect",
        "comparison_validate_audit": "comparison_validate_collect",
        "comparison_test_audit": "comparison_test_collect",
    }
    if phase in source_phases:
        inputs["source_run_ids"] = [item.run_id for item in finished[source_phases[phase]]]
        if spec.stage != "benchmark_attack":
            inputs["benchmark_run_id"] = _one(
                finished["comparison_benchmark"], "attack benchmark"
            ).run_id
    if phase == "comparison_scheduler_audits":
        # 各方法只审计自身同种子轨迹。 / Audit each method's own trajectory at its paired seed.
        selected = _one(
            [
                item
                for item in finished["comparison_evaluate_schedulers"]
                if item.method == spec.method
                and item.seed == spec.seed
                and item.config["experiment"]["variant"] == config["experiment"]["variant"]
            ],
            f"scheduler trajectory for {config['experiment']['variant']} seed {spec.seed}",
        )
        inputs["source_run_ids"] = [selected.run_id]
        inputs["benchmark_run_id"] = _one(
            finished["comparison_benchmark"], "attack benchmark"
        ).run_id
    if spec.stage == "fit_risk":
        inputs["audit_run_ids"] = [item.run_id for item in finished["comparison_audit"]]
    if spec.stage == "evaluate_risk":
        inputs["audit_run_ids"] = [item.run_id for item in finished["comparison_test_audit"]]
    if (
        spec.stage in ("collect", "train_policy", "validate", "evaluate", "evaluate_risk")
        and config["risk"]["mode"] == "fitted"
    ):
        selected = _one(
            [
                item
                for item in finished["comparison_fit_risk"]
                if item.config["components"]["risk"] == config["components"]["risk"]
                and item.risk_version == spec.risk_version
            ],
            f"{config['components']['risk']} risk fit for {spec.risk_version}",
        )
        inputs["risk_checkpoint"] = f"runs/{selected.run_id}/risk.pt"
    if spec.method == "pqn" and spec.stage in ("validate", "evaluate"):
        pair = _one(
            [
                item
                for item in settings["policy_seed_pairs"]
                if item["evaluation_seed"] == spec.seed
            ],
            f"PQN train/evaluation seed pair for {spec.seed}",
        )
        trained = _one(
            [
                item
                for item in finished["comparison_train"]
                if item.method == "pqn" and item.seed == pair["train_seed"]
            ],
            f"PQN policy for training seed {pair['train_seed']}",
        )
        inputs["agent_checkpoint"] = f"runs/{trained.run_id}/agent.pt"
    if spec.stage == "compare_calibration":
        inputs["audit_run_ids"] = [
            item.run_id
            for phase_name in ("comparison_validate_audit", "comparison_test_audit")
            for item in finished[phase_name]
        ]
        calibration = config["comparisons"]["calibration"]
        calibration["calibration_run_ids"] = [
            item.run_id for item in finished["comparison_validate_collect"]
        ]
        calibration["evaluation_run_ids"] = [
            item.run_id for item in finished["comparison_test_collect"]
        ]
    return make_run_spec(
        config,
        spec.stage,
        spec.method,
        spec.seed,
        spec.risk_version,
        inputs,
        root,
        provenance=spec.provenance,
    )


def _remote_phase(
    specs, function: modal.Function, config: dict, root: Path, entry: dict, journal: _Journal
):
    """先持久化全部调用，再等待并回收成功或失败日志。 / Save calls first, then wait and retrieve success or failure logs."""
    check_admission(specs, root)
    calls = []
    if "calls" in entry:
        # 已付费调用仅重新连接，不重复提交。 / Reattach paid calls without submitting them again.
        records = {record["run_id"]: record for record in entry["calls"]}
        entry["calls"] = [records[spec.run_id] for spec in specs]
        for spec, record in zip(specs, entry['calls']):
            if reserve(spec, root) != record['call_id']:
                raise ValueError('Resume call does not match durable reservation')
            calls.append(modal.FunctionCall.from_id(record['call_id']))
    else:
        entry["calls"] = []
        for spec in specs:
            old_call = reserve(spec, root)
            if old_call is not None:
                call = modal.FunctionCall.from_id(old_call)
            else:
                call = function.spawn(spec_to_dict(spec), config["volume_name"])
                bind_call(spec, root, call.object_id)
            calls.append(call)
            entry["calls"].append(
                {"run_id": spec.run_id, "call_id": call.object_id, "state": "submitted"}
            )
            journal.save()
    errors, results = [], []
    for spec, call, record in zip(specs, calls, entry["calls"]):
        try:
            result = call.get(timeout=config["timeout_seconds"] + config["startup_timeout_seconds"])
            record.update(state="completed", result=result)
            results.append(result)
        except Exception as error:
            # This is the remote task boundary: retain the error and drain already submitted calls.
            # 远端任务边界保留原异常，同时收尾已经提交的任务。
            record.update(
                state="failed", error={"type": type(error).__name__, "message": str(error)}
            )
            errors.append(error)
        if not record.get("downloaded", False):
            try:
                download_run(config["volume_name"], spec.run_id, root)
                record["downloaded"] = True
                record.pop("download_error", None)
            except Exception as error:
                record["download_error"] = {"type": type(error).__name__, "message": str(error)}
                errors.append(error)
        journal.save()
    if errors:
        raise errors[0] if len(errors) == 1 else ExceptionGroup("Comparison phase failures", errors)
    return results


def _prepare(config: dict, root: Path):
    """冻结此次控制器的配置和源码。 / Freeze configuration and source for this controller."""
    resolved, root = deepcopy(config), Path(root).resolve()
    settings = resolved["comparisons"]["workflow"]
    resolved["experiment"]["tag"] = settings["tag"]
    for name in settings["job_order"]:
        resolved["workflow"]["jobs"][name]["backend"] = settings["backend"]
    # 镜像和阶段使用启动时同一源码身份。 / The image and all phases share the source identity frozen at startup.
    provenance = source_identity(root)
    templates = {
        name: make_job_specs(resolved, name, root, provenance=provenance)
        for name in settings["job_order"]
    }
    return resolved, root, templates


def _experimental_settings(config: dict) -> str:
    """恢复只允许修改运行标签与控制开关。 / Recovery may change tags and controller switches only."""
    result = deepcopy(config)
    result["experiment"].pop("tag", None)
    settings = result["comparisons"]["workflow"]
    for field in ("tag", "backend", "enabled", "resume_manifest"):
        settings.pop(field, None)
    result.get("execution", {}).pop("backend", None)
    for job in result["workflow"]["jobs"].values():
        job.pop("backend", None)
        job.pop("enabled", None)
    return json.dumps(result, sort_keys=True, allow_nan=False)


def _run_controller(resolved: dict, root: Path, templates: dict, prior=None, prior_path=None):
    """共享阶段屏障；恢复阶段保留原快照。 / Share phase barriers while retaining recovered snapshots."""
    settings = resolved["comparisons"]["workflow"]
    finished = {}
    with _Journal(root / "output" / f"comparison-{settings['tag']}", resolved) as journal:
        recovered = {}
        if prior is not None:
            journal.data["resumed_from"] = str(prior_path)
            journal.data["prior_app_id"] = prior["app_id"]
            for entry in prior["phases"]:
                if entry["state"] == "completed":
                    journal.data["phases"].append(deepcopy(entry))
                    finished[entry["name"]] = [RunSpec(**item) for item in entry["specs"]]
                else:
                    recovered[entry["name"]] = deepcopy(entry)
            journal.save()

        def run_phases(function=None):
            """阶段屏障保证下游只读取完成产物。 / Phase barriers expose only completed upstream artifacts."""
            for phase in settings["job_order"]:
                if phase in finished:
                    continue
                if phase in recovered:
                    entry = recovered[phase]
                    entry["state"] = "running"
                    entry.pop("error", None)
                    specs = [RunSpec(**item) for item in entry["specs"]]
                else:
                    specs = [
                        _bind(spec, phase, finished, settings, root) for spec in templates[phase]
                    ]
                    entry = {
                        "name": phase,
                        "state": "running",
                        "specs": [spec_to_dict(spec) for spec in specs],
                    }
                journal.data["phases"].append(entry)
                journal.save()
                print(f"{phase}: {len(specs)} tasks / 任务")
                if function is None:
                    results = []
                    for spec in specs:
                        results.append(run_experiment(spec, root))
                        entry["results"] = results
                        journal.save()
                else:
                    results = _remote_phase(
                        specs, function, resolved["modal"], root, entry, journal
                    )
                entry.update(
                    state="completed", run_ids=[spec.run_id for spec in specs], results=results
                )
                finished[phase] = specs
                journal.save()
                print(f"{phase}: completed / 完成")

        if settings["backend"] == "local":
            run_phases()
        elif settings["backend"] == "modal":
            union = [spec for values in templates.values() for spec in values]
            check_admission(union, root)
            app, function, _ = create_app(root, resolved["modal"], union)
            check_admission(union, root)
            with app.run(detach=resolved["modal"]["detach"]):
                journal.data["app_id"] = app.app_id
                journal.save()
                print(f"Modal app: {app.app_id}")
                run_phases(function)
        else:
            raise ValueError(f"未知执行平台 / Unknown comparison backend: {settings['backend']}")
    return {**journal.data, "manifest": str(journal.path)}


def prepare_comparison_specs(config: dict, root: Path) -> list[RunSpec]:
    """Bind graph-ready comparison inputs without executing any phase or SDK call."""
    resolved, root, templates = _prepare(config, root)
    settings = resolved['comparisons']['workflow']
    planned = {}
    for phase in settings['job_order']:
        planned[phase] = [_bind(spec, phase, planned, settings, root)
                          for spec in templates[phase]]
    return [spec for phase in settings['job_order'] for spec in planned[phase]]


def run_comparisons(config: dict, root: Path, *, verifiers=None):
    """Deprecated switch adapter; new runs use the common Experiment controller.

    Explicit semantic verifiers are required rather than inheriting the historical
    remote loop's completion-only gate. Historical manifests use resume_comparisons.
    """
    if not config["comparisons"]["workflow"]["enabled"]:
        return {"state": "disabled", "phases": []}
    if verifiers is None:
        raise ValueError('New comparison execution requires explicit semantic verifiers')
    from ..experiment import Experiment
    return Experiment.prepare(config, root, specs=prepare_comparison_specs(config, root),
                              verifiers=verifiers).submit()


def resume_comparisons(config: dict, root: Path, prior_manifest_path: Path) -> dict:
    """显式恢复中断的 Modal 控制器，不重训已记录任务。 / Explicitly resume an interrupted Modal controller without retraining recorded tasks."""
    if not config["comparisons"]["workflow"]["enabled"]:
        return {"state": "disabled", "phases": []}
    prior_path = Path(prior_manifest_path).resolve()
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    settings = config["comparisons"]["workflow"]
    if (
        settings["backend"] != "modal"
        or prior["config"]["comparisons"]["workflow"]["backend"] != "modal"
    ):
        raise ValueError("恢复只支持 Modal 控制器 / Recovery requires Modal controllers")
    if prior["state"] == "completed":
        raise ValueError("工作流已完成，无需恢复 / The workflow is already complete")
    if _experimental_settings(config) != _experimental_settings(prior["config"]):
        raise ValueError("恢复不能修改实验参数 / Recovery cannot retune experimental settings")
    names = [entry["name"] for entry in prior["phases"]]
    if names != settings["job_order"][: len(names)]:
        raise ValueError("原阶段需为有序前缀 / Prior phases must be an ordered prefix")
    for index, entry in enumerate(prior["phases"]):
        if entry["state"] == "completed":
            continue
        if index != len(prior["phases"]) - 1:
            raise ValueError(
                "仅最后阶段可以未完成 / Only the last recorded phase may be incomplete"
            )
        spec_ids = [item["run_id"] for item in entry["specs"]]
        records = entry.get("calls", [])
        if (
            len(set(spec_ids)) != len(spec_ids)
            or len(records) != len(spec_ids)
            or {record["run_id"] for record in records} != set(spec_ids)
            or len({record["call_id"] for record in records}) != len(spec_ids)
        ):
            raise ValueError(
                "调用 ID 不完整，请核查原任务 / Inspect incomplete durable call IDs before recovery"
            )
    return _run_controller(*_prepare(config, root), prior=prior, prior_path=prior_path)
