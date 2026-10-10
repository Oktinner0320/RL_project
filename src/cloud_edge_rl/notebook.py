"""比较任务的快照规划；执行由 Experiment 统一负责。 / Comparison planning for Experiment."""

from copy import deepcopy
from pathlib import Path

from .config import make_run_spec, source_identity


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
