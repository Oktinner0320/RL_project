"""从实际结果绘制独立筛选图。 / Plot a standalone screening figure from actual measurements."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_results(root: Path, config: dict) -> Path:
    """不同任务各自展示，不合成算法排行榜。 / Show each task separately without a cross-task ranking."""
    base = root / "reference_algorithms/results"
    output = base / config["reference_benchmarks"]["tag"]
    tabm = pd.read_csv(output / "tabm/classification-summary.csv")
    gia = pd.read_csv(output / "tabular_gia/summary.csv")
    cp = pd.read_csv(output / "conformalopt/official_googl-summary.csv")
    small_cp = pd.read_csv(output / "conformalopt_small_lr/official_googl-summary.csv")
    ocp = pd.read_csv(output / "ocp_unlock/summary.csv")
    scheduling = json.loads((base / "metacs_fl" / output.name / "summary.json").read_text())
    pqn = [pd.read_csv(base / name / "per_seed.csv") for name in ("pqn", "pqn_paper")]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    ax = axes[0, 0]
    for index, (name, subset) in enumerate(tabm.groupby("model")):
        ax.bar(
            index,
            subset.validation_accuracy.mean() * 100,
            yerr=subset.validation_accuracy.std() * 100,
            capsize=4,
            label=name,
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["MLP", "TabM"],
        ylim=(80, 86),
        title="Adult classification (3 seeds)",
        ylabel="Validation accuracy (%)",
    )
    ax = axes[0, 1]
    methods = scheduling["methods"]
    ax.bar(list(methods), [row["mean_objective_gap"] for row in methods.values()])
    ax.set(title="3-client workload optimization (15 profiles)", ylabel="Gap to exhaustive optimum")
    ax.tick_params(axis="x", labelsize=9)
    ax = axes[0, 2]
    means = [table.greedy_return_mean.mean() for table in pqn]
    errors = [table.greedy_return_mean.std() for table in pqn]
    ax.bar(["Short profile", "Official preset"], means, yerr=errors, capsize=4)
    ax.axhline(pqn[0].random_return_mean.mean(), color="black", ls="--", label="Paired random")
    ax.legend(fontsize=8)
    ax.set(title="PQN: held-out CartPole episodes", ylabel="Return (maximum 500)", ylim=(0, 530))
    ax = axes[1, 0]
    for method, step, label in (
        ("upstream_IG", 0, "IG: initial"),
        ("upstream_IG", 20, "IG: 20 training steps"),
        ("marginal_prior", 0, "Marginal prior"),
    ):
        rows = gia[(gia.method == method) & (gia.training_steps == step)].sort_values("batch_size")
        ax.errorbar(
            rows.batch_size,
            rows.feature_recovery_mean,
            yerr=rows.feature_recovery_std,
            marker="o",
            capsize=3,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set(
        xticks=[1, 8, 32, 64],
        xticklabels=[1, 8, 32, 64],
        ylim=(0, 1.06),
        title="Actual Adult gradient inversion (1000 iterations)",
        xlabel="Batch size",
        ylabel="Original-feature recovery",
    )
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    static = cp[cp.method == "static_quantile"].iloc[0]
    ax.scatter(
        static.all_mean_width,
        static.all_upper_coverage,
        marker="D",
        color="black",
        label="Static quantile",
    )
    for method, color in (("scalar", "tab:blue"), ("linear", "tab:orange")):
        rows = [table[table.method == method].iloc[0] for table in (cp, small_cp)]
        ax.plot(
            [row.all_mean_width for row in rows],
            [row.all_upper_coverage for row in rows],
            color=color,
            label=method,
        )
        for row, marker in zip(rows, ("o", "s")):
            ax.scatter(row.all_mean_width, row.all_upper_coverage, marker=marker, color=color)
    ax.text(0.05, 0.42, "Circle: lr=.05\nSquare: lr=.0005", transform=ax.transAxes, fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(1 - config["reference_benchmarks"]["conformalopt"]["alpha"], ls="--", color="black")
    ax.set_xscale("log")
    ax.set(
        title="ConformalOpt: real GOOGL residuals",
        xlabel="Mean upper threshold (log scale)",
        ylabel="Test upper coverage",
        ylim=(0.5, 1.04),
    )
    ax = axes[1, 2]
    for index, row in ocp.iterrows():
        ax.bar(
            index,
            row.coverage_mean * 100,
            yerr=row.coverage_std * 100 if np.isfinite(row.coverage_std) else None,
            capsize=3,
        )
    ax.axhline(
        (1 - config["reference_benchmarks"]["ocp_unlock"]["alpha"]) * 100, ls="--", color="black"
    )
    ax.set(
        xticks=list(range(len(ocp))),
        xticklabels=ocp.method,
        title="OCP: native feedback protocol (3 seeds)",
        ylabel="Test coverage (%)",
        ylim=(0, 100),
    )
    ax.tick_params(axis="x", rotation=18, labelsize=8)
    figure.suptitle(
        "External algorithm screening — separate tasks, no FL performance claims", fontsize=14
    )
    path = output / "screening.png"
    figure.savefig(path, dpi=180)
    figure.savefig(output / "screening.pdf")
    plt.close(figure)
    return path
