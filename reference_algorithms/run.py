"""由唯一 Notebook 编排外部算法实测。 / Orchestrate external benchmarks from the sole Notebook."""

import json
import subprocess
import sys
from pathlib import Path


def run_suite(project_root: Path, config: dict) -> dict:
    """配置快照传入隔离 UV 运行时。 / Pass a configuration snapshot to isolated UV runtimes."""
    root = Path(project_root).resolve()
    settings = config["reference_benchmarks"]
    outputs = root / "reference_algorithms/results" / settings["tag"]
    outputs.mkdir(parents=True, exist_ok=True)
    snapshot = outputs / "notebook-config.json"
    snapshot.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    results = {}
    for name, enabled in settings["enabled"].items():
        if not enabled:
            results[name] = {"status": "disabled"}
            continue
        if name == "tabm_modal":
            from reference_algorithms.modal_benchmark import run_tabm_modal

            results[name] = run_tabm_modal(root, config)
            continue
        runtime = "pqn_runtime" if name in ("pqn", "pqn_paper") else "runtime"
        python = root / "reference_algorithms" / runtime / ".venv"
        python = python / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not python.exists():
            raise FileNotFoundError(
                f"Initialize with uv sync --project reference_algorithms/{runtime}"
            )
        target = outputs / name
        target.mkdir(exist_ok=False)
        with (target / "execution.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    str(python),
                    "-X",
                    "utf8",
                    str(Path(__file__).resolve()),
                    name,
                    str(snapshot),
                    str(target),
                ],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        results[name] = {"status": "completed", "output_dir": str(target)}
    return results


def execute(name: str, root: Path, config: dict, output: Path):
    """仅选择任务，所有实验值来自快照。 / Select a task; all experiment values come from the snapshot."""
    sys.path[:0] = [str(root), str(root / "src")]
    settings = config["reference_benchmarks"][name]
    if name == "tabular_gia":
        from reference_algorithms.benchmarks.gradient_attack import run_attack

        return run_attack(root, output, settings, config["data"])
    if name == "tabm":
        from reference_algorithms.benchmarks.tabular_uncertainty import run_tabm

        return run_tabm(root, output, settings, config["data"])
    if name in ("conformalopt", "conformalopt_small_lr"):
        from reference_algorithms.benchmarks.conformal import run_conformal

        return run_conformal(root, output, settings)
    if name == "metacs_fl":
        from reference_algorithms.benchmarks.scheduling import run_benchmark

        return run_benchmark(root / "reference_algorithms/repos/metacs_fl", settings, output)
    if name in ("pqn", "pqn_paper"):
        from reference_algorithms.benchmarks.pqn import run

        return run(settings, root, output_dir=output)
    if name == "ocp_unlock":
        from reference_algorithms.benchmarks.ocp import run_ocp

        return run_ocp(root, output, settings)
    raise ValueError(f"Unknown reference benchmark: {name}")


if __name__ == "__main__":
    name, snapshot, output = sys.argv[1:]
    root = Path(__file__).resolve().parent.parent
    config = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    execute(name, root, config, Path(output))
