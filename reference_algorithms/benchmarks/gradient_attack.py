"""实测官方 tabular-GIA 的单客户端反演。 / Measure upstream single-client tabular inversion."""

import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from cloud_edge_rl.data import decode_features, prepare_adult
from cloud_edge_rl.metrics import reconstruction_metrics, sample_marginal_records

from .common import provenance


def _sync(device: torch.device) -> None:
    """等待真实 GPU 运算完成。 / Wait for actual GPU computation before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_attack(root: Path, output: Path, config: dict, data_config: dict) -> dict:
    """只运行外部代码，真值只用于末端评价。 / Run external code; ground truth is evaluator-only."""
    output.mkdir(parents=True, exist_ok=True)
    original_directory = Path.cwd()
    os.chdir(output)
    try:
        return _run_attack(root, output, config, data_config)
    finally:
        os.chdir(original_directory)


def _run_attack(root: Path, output: Path, config: dict, data_config: dict) -> dict:
    """使用官方优化器并检查其未静默提前停止。 / Use upstream optimization and detect silent stops."""
    repository = root / "reference_algorithms/repos/tabular_gia"
    sys.path[:0] = [str(repository), str(repository / "tabular_gia")]
    from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
    from leakpro.fl_utils.data_utils import GiaTabularExtension
    from leakpro.fl_utils.gia_optimizers import MetaSGD
    from leakpro.fl_utils.gia_train import train_nostep
    from model.model import TabularMLP

    logging.getLogger("leakpro").setLevel(logging.WARNING)
    torch.set_num_threads(config["threads"])
    device = torch.device(config["device"])
    data = prepare_adult(data_config, root / "data")
    features = torch.tensor(data.train_x, dtype=torch.float32, device=device)
    labels = torch.tensor(data.train_y, dtype=torch.float32, device=device).reshape(-1, 1)
    (output / "config.json").write_text(
        json.dumps({"algorithm": config, "data": data_config}, indent=2) + "\n", encoding="utf-8"
    )
    (output / "data_metadata.json").write_text(
        json.dumps(data.metadata, indent=2), encoding="utf-8"
    )
    rows = []
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = TabularMLP(
            d_in=features.shape[1],
            d_hidden=config["hidden_size"],
            d_out=1,
            n_hidden_layers=config["hidden_layers"],
            norm=config["normalization"],
            dropout=config["dropout"],
            activation=config["activation"],
            task="binary",
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["training_learning_rate"])
        criterion = model.criterion
        evaluation_rng = np.random.default_rng(seed + config["evaluation_seed_offset"])
        # 同一批次在两个训练阶段复用。 / Reuse each batch across training snapshots.
        batches = {
            size: evaluation_rng.choice(len(features), size=size, replace=False)
            for size in config["batch_sizes"]
        }
        completed_steps = 0
        for training_step in config["training_steps"]:
            for _ in range(training_step - completed_steps):
                indices = rng.choice(
                    len(features), size=config["training_batch_size"], replace=False
                )
                optimizer.zero_grad()
                criterion(model(features[indices]), labels[indices]).backward()
                optimizer.step()
            completed_steps = training_step
            model.eval()
            for batch_size, indices in batches.items():
                observed = [
                    gradient.detach().clone()
                    for gradient in torch.autograd.grad(
                        criterion(model(features[indices]), labels[indices]), model.parameters()
                    )
                ]
                state_before = deepcopy(model.state_dict())
                stem = f"seed{seed}-step{training_step}-batch{batch_size}"
                (output / "checkpoints").mkdir(exist_ok=True)
                torch.save(
                    {
                        "model": state_before,
                        "gradients": observed,
                        "known_labels": labels[indices].detach().cpu(),
                        "batch_size": batch_size,
                    },
                    output / "checkpoints" / f"{stem}-attack-input.pt",
                )
                candidates = []
                attack_started = perf_counter()
                for restart in range(config["restarts"]):
                    torch.manual_seed(seed + config["evaluation_seed_offset"] + restart)
                    # 形状占位数据不含任何私有特征。 / Shape placeholders contain no private features.
                    shape_loader = DataLoader(
                        TensorDataset(torch.zeros_like(features[indices]), labels[indices]),
                        batch_size=batch_size,
                        shuffle=False,
                    )
                    settings = InvertingConfig(
                        tv_reg=config["tv_reg"],
                        attack_lr=config["attack_learning_rate"],
                        at_iterations=config["attack_iterations"],
                        optimizer=MetaSGD(lr=config["training_learning_rate"]),
                        criterion=criterion,
                        data_extension=GiaTabularExtension(),
                        epochs=config["epochs"],
                        label_known=config["label_known"],
                        median_pooling=config["median_pooling"],
                        top10norms=config["top10norms"],
                        vectorized_client_batch_size=None,
                    )
                    attacker = InvertingGradients(
                        model,
                        shape_loader,
                        None,
                        None,
                        train_fn=train_nostep,
                        configs=settings,
                        observed_client_gradient=observed,
                    )
                    attacker.prepare_attack()
                    last_iteration = -1
                    for last_iteration, _, _ in attacker.run_attack():
                        pass
                    if last_iteration != config["attack_iterations"] - 1:
                        raise RuntimeError(f"Upstream attack stopped at iteration {last_iteration}")
                    if not np.isfinite(attacker.best_loss):
                        raise RuntimeError("Upstream attack produced a non-finite objective")
                    reconstruction = (
                        torch.cat([batch[0] for batch in attacker.best_reconstruction])
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    candidates.append((attacker.best_loss, reconstruction))
                _sync(device)
                elapsed = perf_counter() - attack_started
                loss, reconstruction = min(candidates, key=lambda candidate: candidate[0])
                assert all(
                    torch.equal(state_before[name], value)
                    for name, value in model.state_dict().items()
                )
                original = data.train_records.iloc[indices]
                reconstructed = decode_features(reconstruction, data.metadata)
                marginal = sample_marginal_records(
                    data.train_records, batch_size, seed + config["evaluation_seed_offset"]
                )
                common = {"seed": seed, "training_steps": training_step, "batch_size": batch_size}
                for method, candidate in (
                    ("upstream_IG", reconstructed),
                    ("marginal_prior", marginal),
                ):
                    scores = reconstruction_metrics(
                        original, candidate, data.metadata, config["numeric_tolerance"]
                    )
                    rows.append(
                        {
                            **common,
                            "method": method,
                            **{
                                key: value
                                for key, value in scores.items()
                                if key != "match_indices"
                            },
                            "attack_seconds": elapsed if method == "upstream_IG" else None,
                            "gradient_cosine_loss": loss if method == "upstream_IG" else None,
                        }
                    )
                np.savez_compressed(
                    output / f"{stem}.npz", indices=indices, reconstructed=reconstruction
                )
                reconstructed.to_csv(output / f"{stem}-reconstructed.csv", index=False)
                print(
                    f"GIA seed={seed} step={training_step} batch={batch_size}: "
                    f"recovery={rows[-2]['feature_recovery']:.3f}, prior={rows[-1]['feature_recovery']:.3f}, "
                    f"{elapsed:.2f}s",
                    flush=True,
                )
    table = pd.DataFrame(rows)
    table.to_csv(output / "per_attack.csv", index=False)
    grouped = (
        table.groupby(["method", "training_steps", "batch_size"])
        .agg(
            feature_recovery_mean=("feature_recovery", "mean"),
            feature_recovery_std=("feature_recovery", "std"),
            emr_mean=("emr", "mean"),
            numeric_recovery_mean=("numeric_recovery", "mean"),
            categorical_recovery_mean=("categorical_recovery", "mean"),
            attack_seconds_mean=("attack_seconds", "mean"),
        )
        .reset_index()
    )
    grouped.to_csv(output / "summary.csv", index=False)
    summary = {
        "scope": "Actual known-label Adult single-client gradients; frozen initial/20-SGD-step models. "
        "Bounded attack screening, not a federated scheduling result or full paper reproduction.",
        "attacks": len(table[table.method == "upstream_IG"]),
        "config": config,
        "provenance": provenance(
            root,
            "tabular_gia",
            Path(__file__),
            root / "reference_algorithms/runtime/uv.lock",
            ["torch", "numpy", "pandas", "scipy", "torchvision"],
        ),
        "data_source_sha256": data.metadata["sha256"],
        "results": json.loads(grouped.to_json(orient="records")),
        "private_features_passed_to_attack": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
