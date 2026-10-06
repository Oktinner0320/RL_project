"""隔离测试官方 TabM 与同宽 MLP / Benchmark upstream TabM against a same-width MLP."""

import importlib.util
import json
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from cloud_edge_rl.data import prepare_adult


def _load_tabm(project_root: Path):
    """加载保存的官方源码 / Import the downloaded official implementation."""
    source = Path(project_root) / "reference_algorithms/repos/tabm/tabm.py"
    spec = importlib.util.spec_from_file_location("reference_tabm", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source


def _model(module, model_name: str, width: int, config: dict):
    """匹配层数与宽度 / Match depth and width while retaining TabM's official members."""
    if model_name == "tabm":
        return module.TabM(
            n_num_features=width,
            cat_cardinalities=None,
            d_out=1,
            num_embeddings=None,
            arch_type=config["arch_type"],
            activation=config["activation"],
            start_scaling_init=config["start_scaling_init"],
            n_blocks=config["n_blocks"],
            d_block=config["d_block"],
            dropout=config["dropout"],
            k=config["k"],
        )
    layers = []
    for _ in range(config["n_blocks"]):
        layers.extend(
            [
                nn.Linear(width, config["d_block"]),
                getattr(nn, config["activation"])(),
                nn.Dropout(config["dropout"]),
            ]
        )
        width = config["d_block"]
    return nn.Sequential(*layers, nn.Linear(width, 1))


def _sync(device: torch.device):
    """等待 GPU 以测量真实算时 / Synchronize a requested GPU for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _predict(model, features: torch.Tensor, task: str):
    """分类先转概率再平均成员 / Average probabilities for classification, values for regression."""
    model.eval()
    with torch.inference_mode():
        predictions = model(features).reshape(len(features), -1)
        if task == "classification":
            predictions = predictions.sigmoid()
    return predictions.cpu().numpy()


def _score(labels: np.ndarray, members: np.ndarray, task: str, config: dict) -> dict:
    """计算验证指标与经验分歧 / Score held-out predictions and empirical member disagreement."""
    mean = members.mean(axis=1)
    if task == "classification":
        return {
            "validation_bce": float(log_loss(labels, mean, labels=[0, 1])),
            "validation_accuracy": float(
                accuracy_score(labels, mean >= config["classification_threshold"])
            ),
            "validation_auc": float(roc_auc_score(labels, mean)),
            "mean_member_std": float(members.std(axis=1).mean()),
        }
    uncertainty = members.std(axis=1)
    upper = np.clip(mean + config["risk_kappa"] * uncertainty, 0, 1)
    return {
        "validation_mae": float(mean_absolute_error(labels, mean)),
        "validation_rmse": float(np.sqrt(np.mean((labels - mean) ** 2))),
        "mean_underestimation": float(np.maximum(labels - mean, 0).mean()),
        "upper_exceedance_rate": float(np.mean(labels > upper)),
        "mean_member_std": float(uncertainty.mean()),
        "mean_upper": float(upper.mean()),
    }


def _train_models(project_root, output_dir, config, arrays, task):
    """相同数据顺序与轮数训练 / Train both models with equal epochs and batch order."""
    root, output = Path(project_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    module, source = _load_tabm(root)
    device = torch.device(config["device"])
    train_x, train_y, validation_x, validation_y = arrays
    features = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    labels = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    validation = torch.as_tensor(validation_x, dtype=torch.float32, device=device)
    records, history = [], []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config["threads"])
    try:
        for seed in config["seeds"]:
            for model_name in ("mlp", "tabm"):
                torch.manual_seed(seed)
                model = _model(module, model_name, train_x.shape[1], config).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                )
                batch_rng = np.random.default_rng(seed)
                _sync(device)
                started = perf_counter()
                for epoch in range(config["epochs"]):
                    model.train()
                    order = batch_rng.permutation(len(train_x))
                    accumulated_loss = 0.0
                    for start in range(0, len(order), config["batch_size"]):
                        indices = torch.as_tensor(
                            order[start : start + config["batch_size"]], device=device
                        )
                        predictions = model(features[indices]).reshape(len(indices), -1)
                        targets = labels[indices, None].expand_as(predictions)
                        # 每个成员独立受监督，平均成员损失。 / Supervise each member; average losses, not predictions.
                        if task == "classification":
                            loss = nn.functional.binary_cross_entropy_with_logits(
                                predictions, targets
                            )
                        else:
                            loss = nn.functional.mse_loss(predictions, targets)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        accumulated_loss += loss.item() * len(indices)
                    epoch_metrics = _score(
                        validation_y, _predict(model, validation, task), task, config
                    )
                    history.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "epoch": epoch + 1,
                            "train_loss": accumulated_loss / len(train_x),
                            **epoch_metrics,
                        }
                    )
                _sync(device)
                train_seconds = perf_counter() - started
                started = perf_counter()
                members = _predict(model, validation, task)
                _sync(device)
                inference_seconds = perf_counter() - started
                records.append(
                    {
                        "task": task,
                        "model": model_name,
                        "seed": seed,
                        "train_rows": len(train_x),
                        "validation_rows": len(validation_x),
                        "epochs": config["epochs"],
                        "batch_size": config["batch_size"],
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "train_seconds_including_epoch_validation": train_seconds,
                        "inference_seconds": inference_seconds,
                        **_score(validation_y, members, task, config),
                    }
                )
                np.savez_compressed(
                    output / f"{task}-{model_name}-s{seed}-predictions.npz",
                    labels=validation_y,
                    members=members,
                )
                torch.save(model.state_dict(), output / f"{task}-{model_name}-s{seed}.pt")
    finally:
        torch.set_num_threads(old_threads)
    pd.DataFrame(records).to_csv(output / f"{task}-summary.csv", index=False)
    pd.DataFrame(history).to_csv(output / f"{task}-epochs.csv", index=False)
    metadata = {
        "upstream_source": str(source.relative_to(root)),
        "upstream_sha256": sha256(source.read_bytes()).hexdigest(),
        "upstream_version": module.__version__,
        "torch": torch.__version__,
        "runtime_packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "rtdl-num-embeddings")
        },
        "runtime_lock_sha256": sha256(
            (root / "reference_algorithms/runtime/uv.lock").read_bytes()
        ).hexdigest(),
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_pin": next(
            item
            for item in json.loads(
                (root / "reference_algorithms/catalog.json").read_text(encoding="utf-8")
            )
            if item["name"] == "tabm"
        ),
        "config": config,
        "task": task,
        "comparison_budget": "equal epochs, batches and optimizer settings",
        "limitation": "Member disagreement is empirical, not a confidence bound or privacy guarantee.",
    }
    (output / f"{task}-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return records


def _subset(labels, count, seed):
    """分层抽样并保存行号 / Stratify a subset without using held-out labels for training."""
    selected, _ = train_test_split(
        np.arange(len(labels)), train_size=count, random_state=seed, stratify=labels
    )
    return selected


def run_tabm(project_root: Path, output_dir: Path, config: dict, data_config: dict) -> list[dict]:
    """真实 Adult 上执行小规模分类 / Run genuine Adult classification, separate from the fixed FL model."""
    root, output = Path(project_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = prepare_adult(data_config, root / "data")
    train_rows = _subset(bundle.train_y, config["train_rows"], config["subset_seed"])
    validation_rows = _subset(bundle.validation_y, config["validation_rows"], config["subset_seed"])
    np.savez_compressed(
        output / "adult-subset.npz", train_rows=train_rows, validation_rows=validation_rows
    )
    provenance = {
        "data": data_config,
        "source_sha256": bundle.metadata["sha256"],
        "train_source_indices": np.asarray(bundle.metadata["split"]["train_source_indices"])[
            train_rows
        ].tolist(),
        "validation_source_indices": np.asarray(
            bundle.metadata["split"]["validation_source_indices"]
        )[validation_rows].tolist(),
        "limitation": "Centralized Adult classification benchmark; not measured FL privacy-risk performance.",
    }
    (output / "adult-provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return _train_models(
        root,
        output,
        config,
        (
            bundle.train_x[train_rows],
            bundle.train_y[train_rows],
            bundle.validation_x[validation_rows],
            bundle.validation_y[validation_rows],
        ),
        "classification",
    )


def run_tabm_risk(
    project_root,
    output_dir,
    config,
    train_features,
    train_risks,
    validation_features,
    validation_risks,
):
    """用真实攻击标签回归；调用方按完整轨迹划分 / Regress actual attack labels with caller-supplied trajectory splits."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(train_features)
    train_x = scaler.transform(train_features).astype(np.float32)
    validation_x = scaler.transform(validation_features).astype(np.float32)
    np.savez_compressed(
        output / "risk-data.npz",
        train_features=train_features,
        train_risks=train_risks,
        validation_features=validation_features,
        validation_risks=validation_risks,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    return _train_models(
        project_root,
        output,
        config,
        (train_x, np.asarray(train_risks), validation_x, np.asarray(validation_risks)),
        "regression",
    )
