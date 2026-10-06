"""官方半老虎机校准器的隔离实测。 / Isolated benchmark of the official semi-bandit calibrator."""

import importlib
import json
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


class _FeedbackModel:
    """保留官方评分，仅限制反馈与换算宽度。 / Preserve upstream scoring, restrict feedback, and map interval widths."""

    def __init__(self, scorer_class, scores: np.ndarray, pi: np.ndarray, low: float, high: float):
        """误差越小对应越大的归一化分数。 / Smaller residuals correspond to larger normalized scores."""
        self.scorer = scorer_class((high - scores) / (high - low), pi)
        self.thresholds = high - (high - low) * pi
        self.split_size = len(pi)
        self.bit_queries = np.zeros(len(scores), dtype=int)
        self.score_queries = np.zeros(len(scores), dtype=int)
        self.cover_bits = np.zeros(len(scores), dtype=int)

    def get_reward(self, arm: int, t: int) -> float:
        """每步恰好提供一次覆盖位。 / Provide exactly one coverage bit per step."""
        if self.bit_queries[t] != 0:
            raise RuntimeError("Upstream attempted an additional coverage query at the same step.")
        self.bit_queries[t] += 1
        reward = self.scorer.get_reward(arm, t)
        self.cover_bits[t] = int(reward)
        return reward

    def get_efficiency(self, arm: int, t: int) -> float:
        """绝对误差阈值对应对称区间宽度 2τ。 / An absolute-error threshold gives symmetric width 2τ."""
        return float(2 * self.thresholds[arm])

    def get_optimal_arm(self, t: int) -> int:
        """覆盖成功才能暴露分数并解锁其他阈值。 / Reveal the score and unlock thresholds only on coverage success."""
        if self.bit_queries[t] != 1 or self.cover_bits[t] != 1:
            raise RuntimeError("The true score cannot be revealed without successful coverage.")
        self.score_queries[t] += 1
        return self.scorer.get_optimal_arm(t)


def _metrics(scores: np.ndarray, upper: np.ndarray, split_fraction: float) -> dict:
    """以未裁剪真分数计算覆盖与区间宽度。 / Measure coverage and interval width from unchanged true scores."""
    covered = scores <= upper
    split = round(len(scores) * split_fraction)
    return {
        "coverage": float(covered.mean()),
        "miscoverage": float(1 - covered.mean()),
        "mean_upper_threshold": float(upper.mean()),
        "mean_interval_width": float(2 * upper.mean()),
        "early_coverage": float(covered[:split].mean()),
        "late_coverage": float(covered[split:].mean()),
        "early_mean_interval_width": float((2 * upper[:split]).mean()),
        "late_mean_interval_width": float((2 * upper[split:]).mean()),
    }


def run_ocp(project_root: Path, output_dir: Path, config: dict) -> list[dict]:
    """调用官方 EXP3P，使用原生反馈协议。 / Invoke official EXP3P using its native feedback protocol.

    所有覆盖位由评价端真实计算，失败时不给分数。 / The evaluator supplies real bits and withholds failed scores.
    不适用于没有每步覆盖反馈的稀疏攻击审计。 / Sparse attack audits without per-step coverage feedback are outside scope.
    """
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    repository = root / "reference_algorithms/repos/ocp_unlock"
    sys.path.insert(0, str(repository))
    try:
        bandit_class = importlib.import_module("core.model.bandits.exp3p").EXP3P
        scorer_class = importlib.import_module("core.model.scoring.regression").RegThresholdModel
    finally:
        sys.path.remove(str(repository))

    archive_path = root / config["score_archive"]
    archive = np.load(archive_path)
    calibration, test = archive["calibration"], archive["test"]
    low = config["score_lower_bound"]
    high = float(calibration.max() * config["calibration_upper_multiplier"])
    scores = np.concatenate((calibration, test)) if config["warmup_bandit"] else test
    test_start = len(calibration) if config["warmup_bandit"] else 0
    pi = np.linspace(0, 1, config["arms"])
    horizon = len(scores)
    parameters = (
        config["alpha"],
        config["lambda"],
        config["delta"],
        config["function"],
        config["displeasure"],
        config["loss_type"],
    )
    rows, traces = [], []
    for seed in config["seeds"]:
        for name, structured in config["variants"].items():
            feedback = _FeedbackModel(scorer_class, scores, pi, low, high)
            options = (
                config["C"],
                structured,
                config["version"],
                config["ot_power"],
                config["power_const_option"],
            )
            random_state = np.random.get_state()
            np.random.seed(seed)
            started = perf_counter()
            try:
                algorithm = bandit_class(
                    config["arms"], horizon, False, feedback, pi, parameters, options
                )
                algorithm.run()
            finally:
                np.random.set_state(random_state)
            seconds = perf_counter() - started

            # 检查原生访问规则；不把缺失反馈伪造成零。 / Verify native access rules without inventing missing feedback.
            if not np.all(feedback.bit_queries == 1):
                raise RuntimeError("Native feedback must provide one coverage bit at every step.")
            if structured in (3, 4) and not np.array_equal(
                feedback.score_queries, feedback.cover_bits
            ):
                raise RuntimeError("Unlocking must reveal exactly the successfully covered scores.")
            upper = feedback.thresholds[algorithm.armt.astype(int)]
            if not np.array_equal(algorithm.rpt.astype(bool), scores <= upper):
                raise RuntimeError(
                    "Upstream transformed feedback disagrees with true interval coverage."
                )
            row = {
                "method": name,
                "seed": seed,
                "target_coverage": 1 - config["alpha"],
                "test_steps": len(test),
                "warmup_updates": test_start,
                "seconds": seconds,
                "per_update_ms": seconds * 1000 / horizon,
                "coverage_feedback_fraction": float(feedback.bit_queries[test_start:].mean()),
                "score_feedback_fraction": float(feedback.score_queries[test_start:].mean()),
                "beta": float(algorithm.beta),
                "gamma": float(algorithm.gamma),
                "eta": float(algorithm.eta),
                "upstream_loss_constant": float(algorithm.const),
                **_metrics(test, upper[test_start:], config["test_break_fraction"]),
            }
            rows.append(row)
            traces.extend(
                {
                    "method": name,
                    "seed": seed,
                    "step": t,
                    "phase": "calibration" if t < test_start else "test",
                    "true_score_evaluator_only": float(score),
                    "arm": int(algorithm.armt[t]),
                    "upper_threshold": float(upper[t]),
                    "interval_width": float(2 * upper[t]),
                    "covered": bool(algorithm.rpt[t]),
                    "coverage_bit_queries": int(feedback.bit_queries[t]),
                    "revealed_score_queries": int(feedback.score_queries[t]),
                }
                for t, score in enumerate(scores)
            )
            np.savez_compressed(
                output / f"{name}-seed{seed}-state.npz",
                final_weights=algorithm.G,
                final_probabilities=algorithm.probs,
                counts=algorithm.counts,
                arms=algorithm.armt,
                cover_bits=algorithm.rpt,
                widths=algorithm.effpt,
            )

    started = perf_counter()
    threshold = float(
        np.quantile(calibration, 1 - config["alpha"], method=config["quantile_method"])
    )
    static = np.full(len(test), threshold)
    static_seconds = perf_counter() - started
    rows.append(
        {
            "method": "static_quantile",
            "seed": None,
            "target_coverage": 1 - config["alpha"],
            "test_steps": len(test),
            "warmup_updates": len(calibration),
            "seconds": static_seconds,
            "per_update_ms": static_seconds * 1000 / len(test),
            "coverage_feedback_fraction": 0.0,
            "score_feedback_fraction": 0.0,
            **_metrics(test, static, config["test_break_fraction"]),
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "trials.csv", index=False)
    pd.DataFrame(traces).to_csv(output / "feedback_trace.csv", index=False)
    aggregates = (
        frame.groupby("method", sort=False)[
            ["coverage", "mean_interval_width", "score_feedback_fraction", "seconds"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregates.columns = ["_".join(column).rstrip("_") for column in aggregates.columns]
    aggregates.to_csv(output / "summary.csv", index=False)
    source = json.loads(
        (root / "reference_algorithms/provenance/ocp_unlock.json").read_text(encoding="utf-8")
    )
    metadata = {
        "status": "completed",
        "config": config,
        "upstream": source,
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_archive": str(archive_path.relative_to(root)),
        "source_archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
        "data_source": "Published chronological GOOGL forecaster absolute residuals; not FL attack risk.",
        "runtime_packages": {name: version(name) for name in ("numpy", "pandas")},
        "python": sys.version.split()[0],
        "device": "cpu",
        "split": {"calibration": len(calibration), "test": len(test)},
        "threshold_range": [low, high],
        "test_scores_above_threshold_range": int((test > high).sum()),
        "bounds_protocol": "Range derived only from past calibration scores; true scores are never clipped.",
        "feedback_protocol": (
            "Every step supplies the actual coverage bit; successful coverage additionally reveals "
            "the score to Unlock/Unlock+. Failed coverage never reveals the true score."
        ),
        "width_units": (
            "Reported interval width is 2 * absolute-error upper threshold. Compare upper thresholds "
            "to conformalopt's one-sided widths; raw interval-width columns differ by a factor of two."
        ),
        "upstream_parameters": (
            "EXP3P fixes the loss constant at 40 and requires lambda=1 and O(T) exponent=1/4. "
            "Its C and power_const options do not change that hardcoded constant."
        ),
        "limitation": (
            "Native semi-bandit proxy experiment; no empirical or theoretical guarantee for "
            "project sparse selected gradient-inversion audits without a per-step coverage oracle."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return rows
