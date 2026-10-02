"""只分析真实运行产物；不补造缺失结果。 / Analyze recorded runs without inventing results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import risk_metrics

IDENTITY_COLUMNS = [
    "run_id",
    "stage",
    "method",
    "algorithm",
    "seed",
    "risk_version",
    "split",
    "partition",
    "experiment_tag",
]
PAIR_COLUMNS = [
    "method",
    "reference_method",
    "metric",
    "n_pairs",
    "method_mean",
    "method_std",
    "reference_mean",
    "reference_std",
    "difference_mean",
    "difference_std",
]


def _read_json(path: Path) -> dict:
    """读取已保存的 UTF-8 快照。 / Read a saved UTF-8 snapshot."""
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_runs(root: Path, run_ids: list[str]) -> pd.DataFrame:
    """每个运行保留一行；比较标签来自实验 variant。 / Keep one row per run, labeled by variant."""
    rows = []
    for run_id in run_ids:
        path = Path(root) / "runs" / run_id
        spec = _read_json(path / "config.json")
        config = spec["config"]
        summary = pd.json_normalize(_read_json(path / "summary.json"), sep=".").iloc[0].to_dict()
        row = {}
        metrics_path = path / "metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            if not metrics.empty:
                row = metrics.iloc[-1].to_dict()
                row["mean_predicted_risk"] = metrics["predicted_risk"].mean()
                row["max_predicted_risk"] = metrics["worst_predicted_risk"].max()
                row["metric_rows"] = len(metrics)
                row["episodes"] = metrics["episode"].nunique()
        row.update(summary)
        row.update(
            run_id=run_id,
            stage=spec["stage"],
            method=config["experiment"]["variant"],
            algorithm=spec["method"],
            seed=spec["seed"],
            risk_version=spec["risk_version"],
            split=config["execution"]["split"],
            partition=config["data"]["partition"],
            experiment_tag=config["experiment"]["tag"],
        )
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=IDENTITY_COLUMNS)


def pairwise_summary(frame: pd.DataFrame, metric: str, reference_method: str) -> pd.DataFrame:
    """同种子配对；重复检查点必须先明确筛选。 / Pair seeds after explicitly selecting checkpoints."""
    if frame.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    if (frame[["stage", "split", "partition"]].nunique(dropna=False) > 1).any():
        raise ValueError(
            "先筛选相同阶段、数据集和划分 / Select the same stage, split, and partition"
        )
    if frame.duplicated(["method", "seed"]).any():
        raise ValueError("每个方法和种子只选一个检查点 / Select one checkpoint per method and seed")
    values = frame.pivot(index="seed", columns="method", values=metric)
    reference = values[reference_method]
    rows = []
    for method in values.columns:
        if method == reference_method:
            continue
        pairs = pd.concat([values[method], reference], axis=1).dropna()
        difference = pairs[method] - pairs[reference_method]
        rows.append(
            {
                "method": method,
                "reference_method": reference_method,
                "metric": metric,
                "n_pairs": len(pairs),
                "method_mean": pairs[method].mean(),
                "method_std": pairs[method].std(ddof=1),
                "reference_mean": pairs[reference_method].mean(),
                "reference_std": pairs[reference_method].std(ddof=1),
                "difference_mean": difference.mean(),
                "difference_std": difference.std(ddof=1),
            }
        )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _save_figure(figure, output_dir: Path, name: str) -> dict[str, Path]:
    """统一导出位图与矢量图，释放绘图资源。 / Export raster/vector figures and close them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {suffix: output_dir / f"{name}.{suffix}" for suffix in ("png", "pdf")}
    figure.tight_layout()
    for path in paths.values():
        figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_learning_curves(
    root: Path, run_ids: list[str], output_dir: Path, x_axis: str
) -> dict[str, Path]:
    """绘制真实验证与代价轨迹；缺失风险保持缺失。 / Plot observed trajectories without zero-filling risk."""
    labels = {
        "elapsed_seconds": "Modeled elapsed time (s)",
        "processed_samples": "Processed samples",
    }
    x_label = labels[x_axis]
    if not run_ids:
        return {}
    figure, grid = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = grid.ravel()
    for run_id in run_ids:
        path = Path(root) / "runs" / run_id
        spec = _read_json(path / "config.json")
        metrics = pd.read_csv(path / "metrics.csv")
        label = f"{spec['config']['experiment']['variant']} / seed {spec['seed']}"
        for episode, points in metrics.groupby("episode", sort=False):
            # 回合独立连线；NaN 会留下空隙而非被解释成零风险。 / Separate episodes; NaN creates gaps.
            points = points.sort_values(x_axis)
            episode_label = f"{label} / ep {episode}"
            axes[0].plot(points[x_axis], points["validation_loss"], label=episode_label)
            if points["predicted_risk"].notna().any():
                axes[1].plot(points[x_axis], points["predicted_risk"], label=episode_label)
            axes[2].plot(points[x_axis], points["energy_joules"], label=episode_label)
            communication = (points["upload_bytes"] + points["download_bytes"]) / 1_000_000
            axes[3].plot(points[x_axis], communication, label=episode_label)
    for axis, title in zip(
        axes, ["Validation loss", "Predicted risk", "Modeled energy (J)", "Communication (MB)"]
    ):
        axis.set(xlabel=x_label, ylabel=title)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize="small")
    if not axes[1].lines:
        axes[1].text(
            0.5, 0.5, "Risk estimates unavailable", ha="center", transform=axes[1].transAxes
        )
    return _save_figure(figure, output_dir, f"learning_curves_{x_axis}")


def _read_audit_rows(root: Path, audit_run_ids: list[str]) -> list[dict]:
    """读取已完成攻击的记录。 / Read records of completed attacks."""
    rows = []
    for run_id in audit_run_ids:
        path = Path(root) / "runs" / run_id / "audits.jsonl"
        # 零次攻击不会创建记录文件。 / An audit with zero completed attacks has no record file.
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


def analyze_risk_runs(root: Path, audit_run_ids: list[str], threshold: float) -> pd.DataFrame:
    """按来源运行汇总风险，上传记录不充当独立种子。 / Aggregate uploads within each source run."""
    columns = [
        "risk_version",
        "source_run_id",
        "split",
        "n_uploads",
        "mae",
        "mean_underestimation",
        "underestimation_frequency",
        "exceedance_frequency",
    ]
    rows = _read_audit_rows(root, audit_run_ids)
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows).dropna(subset=["predicted_risk", "feature_recovery"])
    summaries = []
    for (version, run_id, split), uploads in frame.groupby(["risk_version", "run_id", "split"]):
        metrics = risk_metrics(
            uploads["predicted_risk"].to_numpy(dtype=float),
            uploads["feature_recovery"].to_numpy(dtype=float),
            threshold,
        )
        summaries.append(
            {
                "risk_version": version,
                "source_run_id": run_id,
                "split": split,
                "n_uploads": len(uploads),
                **metrics,
            }
        )
    return pd.DataFrame(summaries, columns=columns)


def plot_risk_calibration(
    root: Path, audit_run_ids: list[str], output_dir: Path
) -> dict[str, Path]:
    """按风险版本比较真实预测与重建标签。 / Compare genuine predictions with reconstruction labels."""
    rows = _read_audit_rows(root, audit_run_ids)
    frame = pd.DataFrame(rows, columns=["predicted_risk", "feature_recovery", "risk_version"])
    frame = frame.dropna(subset=["predicted_risk", "feature_recovery"])
    if frame.empty:
        return {}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    versions, underestimation = [], []
    for version, points in frame.groupby("risk_version", sort=True):
        predicted = points["predicted_risk"].to_numpy(dtype=float)
        actual = points["feature_recovery"].to_numpy(dtype=float)
        axes[0].scatter(predicted, actual, alpha=0.65, label=f"{version} (n={len(points)})")
        versions.append(version)
        underestimation.append(float(np.maximum(actual - predicted, 0).mean()))
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[0].set(
        xlabel="Predicted risk", ylabel="Observed feature recovery", xlim=(0, 1), ylim=(0, 1)
    )
    axes[0].legend(fontsize="small")
    axes[1].bar(versions, underestimation)
    axes[1].set(xlabel="Risk version", ylabel="Mean underestimation: max(actual - predicted, 0)")
    for axis in axes:
        axis.grid(alpha=0.25)
    return _save_figure(figure, output_dir, "risk_calibration")
