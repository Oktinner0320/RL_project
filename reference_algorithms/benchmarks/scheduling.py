"""实测官方 MetaCS-FL 调度器。 / Benchmark the official MetaCS-FL allocation optimizer."""

import importlib
import importlib.metadata
import itertools
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


def _profile(config: dict, seed: int, case: int) -> dict:
    """生成固定工作量的资源与历史画像。 / Generate a seeded resource and history profile."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, case, 0]))
    count = config["num_clients"]
    capacities = config["capacities"]
    speed = rng.uniform(*config["time_per_task_range"], size=count)
    startup = rng.uniform(*config["startup_seconds_range"], size=count)
    power = rng.uniform(*config["power_watts_range"], size=count)
    time_costs = [
        [float(startup[i] + speed[i] * tasks) if tasks else 0.0 for tasks in capacity]
        for i, capacity in enumerate(capacities)
    ]
    energy_costs = [
        [float(seconds * power[i]) for seconds in times] for i, times in enumerate(time_costs)
    ]
    class_counts = [
        rng.multinomial(
            config["class_capacity"],
            rng.dirichlet([config["class_dirichlet_alpha"]] * config["class_count"]),
        ).tolist()
        for _ in range(count)
    ]
    clients = list(range(count))
    rounds = config["history_rounds"]
    candidate_history = {r: clients for r in range(1, rounds + 1)}
    selected_history = {
        r: [i for i in clients if rng.random() < config["history_selection_probability"]]
        for r in range(1, rounds + 1)
    }
    return {
        "n": count,
        "t": config["total_tasks"],
        "A": capacities,
        "Y": class_counts,
        "I": rng.uniform(*config["idle_watts_range"], size=count).tolist(),
        "B": [config["battery_joules"]] * count,
        "G": time_costs,
        "E": energy_costs,
        "phi_list": rng.uniform(*config["utility_range"], size=count).tolist(),
        "psi_list": rng.uniform(*config["utility_range"], size=count).tolist(),
        "alpha": config["utility_alpha"],
        "beta": config["class_beta"],
        "candidate_client_ids": clients,
        "current_round": rounds + 1,
        "candidate_clients_history_ids": candidate_history,
        "selected_clients_history_ids": selected_history,
    }


def run_benchmark(repo_dir: Path, config: dict, output_dir: Path) -> dict:
    """比较原版优化器与相同候选集穷举解。 / Compare upstream optimization with an identical-domain oracle.

    config 的全部实验参数来自 Notebook。 / Every experimental setting comes from the Notebook.
    此处测固定总任务量分配，不产生 Adult/FedSGD 学习结果。 / This measures allocation, not FL learning.
    """
    repo_dir = Path(repo_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo_dir))
    try:
        lns = importlib.import_module("metacs_fl.metaheuristic.lns")
        utility = importlib.import_module("metacs_fl.utils.task_scheduler_util")
        ecmtc = importlib.import_module("metacs_fl.task_scheduler.ecmtc").ecmtc
    finally:
        sys.path.remove(str(repo_dir))

    # 避免上游 random 分配器内部未设种子的采样。 / Keep upstream class allocation deterministic.
    if config["class_distribution"] != "locally_balanced":
        raise ValueError("This benchmark requires deterministic locally_balanced class allocation.")
    source = json.loads(
        (repo_dir.parent.parent / "provenance" / "metacs_fl.json").read_text(encoding="utf-8")
    )
    rows = []
    profiles = []
    trace_rows = []
    for seed in config["seeds"]:
        for case in range(config["cases_per_seed"]):
            profile = _profile(config, seed, case)
            profiles.append({"seed": seed, "case": case, "profile": profile})
            start = perf_counter()
            initial, _, _ = ecmtc(
                profile["n"],
                profile["t"],
                np.asarray(profile["A"], dtype=object),
                np.asarray(profile["G"], dtype=object),
                np.asarray(profile["E"], dtype=object),
                config["deadline_seconds"],
            )
            initial = [int(x) for x in initial]
            initial_ms = (perf_counter() - start) * 1000

            def costs(allocation: list) -> dict:
                """调用原版成本评价。 / Evaluate costs with the upstream implementation."""
                return utility.estimate_solution_costs(
                    **profile,
                    X=allocation,
                    X_dist_approach=config["class_distribution"],
                    q=config["history_rounds"],
                )

            def objective(solution: dict) -> float:
                """使用固定尺度的论文目标。 / Apply the paper objective with fixed scales."""
                normalized = utility.normalize_costs(
                    {"candidate": solution}, config["normalization_bounds"]
                )["candidate"]
                return float(lns.lns_F(normalized, config["weights"]))

            initial_costs = costs(initial)

            def feasible(solution: dict) -> bool:
                """对所有方法施加同一可行约束。 / Apply identical constraints to every method."""
                return bool(
                    lns.lns_accept(
                        {"X_init": initial_costs, "X_rpr": solution},
                        config["accept"],
                        config["deadline_seconds"],
                        profile["t"],
                        profile["A"],
                    )
                )

            start = perf_counter()
            optimized, _, stats = lns.run_lns(
                **profile,
                X_init=initial,
                rng=np.random.default_rng(np.random.SeedSequence([seed, case, 1])),
                destroy_approach=config["destroy"],
                tau=config["deadline_seconds"],
                client_diversity_last_q_rounds=config["history_rounds"],
                stop_criteria={"name": "LNS_Stop_Max_Iterations", "it_max": config["iterations"]},
                accept_criteria=config["accept"],
                obj_func_weights=config["weights"],
                X_dist_approaches={
                    key: config["class_distribution"] for key in ("X_init", "X_best", "X_rpr")
                },
                normalization_mode="fixed",
                normalization_bounds=config["normalization_bounds"],
            )
            optimized = [int(x) for x in optimized]
            lns_ms = (perf_counter() - start) * 1000
            trace_rows.extend(
                {"seed": seed, "case": case, **trace} for trace in stats["lns_trace_rows"]
            )

            # 穷举器使用同一上游代价、权重、归一化与可行约束。 / The oracle uses identical upstream semantics.
            start = perf_counter()
            candidates = [
                costs(list(allocation))
                for allocation in itertools.product(*profile["A"])
                if sum(allocation) == profile["t"]
            ]
            candidates = [candidate for candidate in candidates if feasible(candidate)]
            oracle = min(candidates, key=objective)
            oracle_ms = (perf_counter() - start) * 1000
            best_value = objective(oracle)
            for method, solution, elapsed_ms in (
                ("ECMTC", initial_costs, initial_ms),
                ("MetaCS-FL", costs(optimized), initial_ms + lns_ms),
                ("Exhaustive", oracle, oracle_ms),
            ):
                gap = objective(solution) - best_value
                rows.append(
                    {
                        "seed": seed,
                        "case": case,
                        "method": method,
                        "allocation": json.dumps(solution["X"]),
                        "feasible_candidates": len(candidates),
                        "objective": objective(solution),
                        "objective_gap": gap,
                        "optimal": abs(gap) <= config["optimality_tolerance"],
                        "constraint_violation": not feasible(solution),
                        "decision_ms": elapsed_ms,
                        "lns_ms": lns_ms if method == "MetaCS-FL" else 0.0,
                        "simulated_time_seconds": float(solution["M_X"]),
                        "simulated_energy_joules": float(solution["E_X"]),
                        "diversity": float(solution["D_X"]),
                        "class_score": float(solution["K_X"]),
                        "historical_utility": float(solution["U_X"]),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "scheduling_cases.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(output_dir / "lns_traces.csv", index=False)
    (output_dir / "profiles.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    methods = {}
    for method, group in frame.groupby("method", sort=False):
        methods[method] = {
            "mean_objective_gap": float(group["objective_gap"].mean()),
            "max_objective_gap": float(group["objective_gap"].max()),
            "optimal_fraction": float(group["optimal"].mean()),
            "constraint_violations": int(group["constraint_violation"].sum()),
            "mean_decision_ms": float(group["decision_ms"].mean()),
            "median_decision_ms": float(group["decision_ms"].median()),
        }
    summary = {
        "algorithm": "MetaCS-FL",
        "scope": (
            "Actual upstream optimizer on seeded synthetic resource/class/history profiles; "
            "fixed-workload allocation, not Adult/FedSGD training."
        ),
        "status": "completed",
        "source_repository": source["repository"],
        "source_revision": source["revision"],
        "source_commit": source["commit"],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "psutil": importlib.metadata.version("psutil"),
            "device": "cpu",
        },
        "profile_count": len(profiles),
        "config": config,
        "methods": methods,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
