"""运行原版 PQN CPU 基准 / Run the unchanged upstream PQN CPU benchmark.

All experiment values come from experiment.ipynb or the explicit run argument.
所有实验值均来自唯一 Notebook 配置，外部实现保持原样。
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


def notebook_config(root: Path, profile: str = "pqn") -> dict[str, Any]:
    """读取唯一参数单元 / Read the single Notebook parameter cell."""
    notebook = json.loads((root / "experiment.ipynb").read_text(encoding="utf-8"))
    cells = [
        cell
        for cell in notebook["cells"]
        if "parameters" in cell.get("metadata", {}).get("tags", [])
    ]
    (cell,) = cells
    scope: dict[str, Any] = {}
    exec("".join(cell["source"]), scope)
    return scope["CONFIG"]["reference_benchmarks"][profile]


def evaluate_policy(network, state, config: dict[str, Any], seed: int, random_policy: bool):
    """独立配对评估初始回合 / Evaluate paired initial episodes without auto-reset leakage."""
    import gymnax
    import jax
    import jax.numpy as jnp

    env, env_params = gymnax.make(config["upstream"]["ENV_NAME"])
    episode_count = config["num_eval_episodes"]
    # Folding separates held-out evaluation from training RNG.
    # 折叠独立标识，评估随机数不参与训练。
    key = jax.random.fold_in(jax.random.PRNGKey(seed), config["eval_seed_offset"])

    def evaluate(key):
        key, reset_key = jax.random.split(key)
        observation, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(reset_key, episode_count), env_params
        )

        def step(carry, _):
            observation, env_state, active, returns, lengths, key = carry
            key, action_key, transition_key = jax.random.split(key, 3)
            if random_policy:
                actions = jax.random.randint(
                    action_key, (episode_count,), 0, env.action_space(env_params).n
                )
            else:
                q_values = network.apply(
                    {"params": state.params, "batch_stats": state.batch_stats},
                    observation,
                    train=False,
                )
                actions = jnp.argmax(q_values, axis=-1)
            next_observation, next_state, rewards, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(transition_key, episode_count), env_state, actions, env_params)
            returns = returns + jnp.where(active, rewards, 0.0)
            lengths = lengths + active.astype(jnp.int32)
            active = active & ~done
            return (next_observation, next_state, active, returns, lengths, key), None

        initial = (
            observation,
            env_state,
            jnp.ones(episode_count, dtype=bool),
            jnp.zeros(episode_count),
            jnp.zeros(episode_count, dtype=jnp.int32),
            key,
        )
        final, _ = jax.lax.scan(step, initial, None, config["max_eval_steps"])
        return {"returns": final[3], "lengths": final[4], "truncated": final[2]}

    started = perf_counter()
    values = jax.block_until_ready(jax.jit(evaluate)(key))
    return {name: value.tolist() for name, value in values.items()}, perf_counter() - started


def run(config: dict[str, Any], root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """调用官方训练函数并留档 / Call official training and preserve raw evidence."""
    import gymnax
    import jax
    import numpy as np
    from flax import serialization

    repository = root / "reference_algorithms" / "repos" / "pqn"
    sys.path.insert(0, str(repository))
    from purejaxql.pqn_gymnax import QNetwork, make_train

    output = (
        output_dir if output_dir is not None else root / "reference_algorithms" / "results" / "pqn"
    )
    output.mkdir(parents=True, exist_ok=True)
    upstream = copy.deepcopy(config["upstream"])
    config_snapshot = copy.deepcopy(config)
    (output / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")
    training = jax.jit(make_train(upstream))
    started = perf_counter()
    compiled_training = training.lower(jax.random.PRNGKey(config["seeds"][0])).compile()
    compile_seconds = perf_counter() - started
    network = QNetwork(
        action_dim=gymnax.make(upstream["ENV_NAME"])[0].action_space().n,
        hidden_size=upstream["HIDDEN_SIZE"],
        num_layers=upstream["NUM_LAYERS"],
        norm_type=upstream["NORM_TYPE"],
        norm_input=upstream["NORM_INPUT"],
    )

    rows = []
    for seed in config["seeds"]:
        started = perf_counter()
        result = jax.block_until_ready(compiled_training(jax.random.PRNGKey(seed)))
        training_seconds = perf_counter() - started
        state = result["runner_state"][0]
        metrics = {name: np.asarray(value) for name, value in result["metrics"].items()}
        np.savez_compressed(output / f"training_seed{seed}.npz", **metrics)
        (output / f"checkpoint_seed{seed}.msgpack").write_bytes(serialization.to_bytes(state))
        greedy, greedy_seconds = evaluate_policy(network, state, config, seed, False)
        random, random_seconds = evaluate_policy(network, state, config, seed, True)
        (output / f"evaluation_seed{seed}.json").write_text(
            json.dumps({"seed": seed, "greedy": greedy, "random": random}, indent=2),
            encoding="utf-8",
        )
        row = {
            "seed": seed,
            "training_steps": int(state.timesteps),
            "gradient_steps": int(state.grad_steps),
            "training_seconds": training_seconds,
            "training_steps_per_second": int(state.timesteps) / training_seconds,
            "greedy_return_mean": float(np.mean(greedy["returns"])),
            "greedy_return_std": float(np.std(greedy["returns"], ddof=1)),
            "random_return_mean": float(np.mean(random["returns"])),
            "paired_return_gain": float(
                np.mean(np.asarray(greedy["returns"]) - np.asarray(random["returns"]))
            ),
            "greedy_evaluation_seconds": greedy_seconds,
            "random_evaluation_seconds": random_seconds,
            "final_td_loss": float(metrics["td_loss"][-1]),
            "final_upstream_test_return": float(metrics["test/returned_episode_returns"][-1]),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    with (output / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = np.asarray([row["greedy_return_mean"] for row in rows])
    gains = np.asarray([row["paired_return_gain"] for row in rows])
    summary = {
        "status": "completed",
        "algorithm": "PQN",
        "started_from": "unchanged upstream purejaxql.pqn_gymnax.make_train",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": json.loads(
            (root / "reference_algorithms" / "provenance" / "pqn.json").read_text(encoding="utf-8")
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["jax", "jaxlib", "flax", "optax", "chex", "gymnax", "numpy"]
        },
        "runtime_lock_sha256": hashlib.sha256(
            (root / "reference_algorithms" / "pqn_runtime" / "uv.lock").read_bytes()
        ).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "devices": [str(device) for device in jax.devices()],
        "jit_compile_seconds": compile_seconds,
        "greedy_return_across_seed_mean": float(means.mean()),
        "greedy_return_across_seed_std": float(means.std(ddof=1)),
        "paired_gain_across_seed_mean": float(gains.mean()),
        "random_return_across_seed_mean": float(
            np.mean([row["random_return_mean"] for row in rows])
        ),
        "training_seconds_mean": float(np.mean([row["training_seconds"] for row in rows])),
        "seeds": rows,
        "limitations": [
            "CPU CartPole proxy; does not evaluate Adult privacy or federated scheduling.",
            f"{upstream['TOTAL_TIMESTEPS']:,} transitions per seed; this is a bounded performance test, not a paper reproduction.",
            "JIT compilation is timed separately; evaluation timings include their own JIT compilation.",
            f"{len(rows)} training seeds and {config['num_eval_episodes']} held-out initial episodes per seed; no hyperparameter search.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    run(notebook_config(project_root), project_root)
