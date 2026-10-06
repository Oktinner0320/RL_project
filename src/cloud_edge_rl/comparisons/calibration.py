"""已审计上传上的官方校准对照。 / Official calibration comparisons over audited uploads only.

已审计条目轴不代表全部上传；OCP 的原生覆盖反馈不适用于未审计条目。
The audited-item axis excludes unaudited uploads and does not establish a sparse-audit guarantee.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .sources import load_module


def _ordered_runs(frame: pd.DataFrame, run_ids: list[str], split: str) -> pd.DataFrame:
    """运行顺序明确给定，运行内部按上传时间。 / Use explicit run order and within-run upload time."""
    pieces = []
    for run_id in run_ids:
        selected = frame.loc[frame["run_id"].eq(run_id)]
        if selected.empty or not selected["split"].eq(split).all():
            raise ValueError(f"校准运行划分不匹配 / Calibration split mismatch: {run_id}")
        pieces.append(selected.sort_values(["episode", "round_index", "client_id", "upload_id"]))
    return pd.concat(pieces, ignore_index=True)


def _positive_residual(frame: pd.DataFrame) -> np.ndarray:
    """单侧风险低估残差；保留真实评价标签。 / Keep true labels and measure one-sided risk underestimation."""
    return np.maximum(frame["feature_recovery"].to_numpy() - frame["predicted_risk"].to_numpy(), 0)


def _conformal(root: Path, calibration: np.ndarray, scores: np.ndarray, method: str, config: dict):
    """先预测，再提供已到达的审计残差。 / Predict before supplying arrived audit residuals."""
    module = load_module(root, "conformalopt", "conformalopt/__init__.py", "conformalopt")
    settings = config["conformal"]
    tracker = module.ConformalPredictor(
        alpha=config["alpha"],
        lr_type=settings["learning_rate_type"],
        quantile_tracker={"conformal_sqt": "scalar", "conformal_lqt": "linear"}[method],
        scorecaster=settings["scorecaster"],
        hypers={
            "lr": settings["learning_rate"],
            "p_order_qt": settings["tracker_order"],
            "bias": settings["tracker_bias"],
        },
    )
    tracker.init_active_fields()
    for score in calibration:
        prediction = tracker.predict()
        tracker.step(prediction, score)
    upper = np.empty(len(scores))
    feedback_steps = np.full(len(scores), -1, dtype=int)
    for index in range(len(scores)):
        upper[index] = tracker.predict()
        arrived = index - settings["feedback_delay"]
        if arrived >= 0:
            tracker.step(upper[arrived], scores[arrived])
            feedback_steps[index] = arrived
    return upper, {
        "coverage_bit_queries": np.zeros(len(scores), dtype=int),
        "revealed_score_queries": (feedback_steps >= 0).astype(int),
        "feedback_source_step": feedback_steps,
        "arm": np.full(len(scores), -1, dtype=int),
    }


class _NativeFeedback:
    """评价端仅暴露覆盖位与成功分数。 / The evaluator reveals coverage bits and successful scores only."""

    def __init__(self, scorer_class, scores: np.ndarray, pi: np.ndarray, high: float):
        """官方评分输入反向归一化，不裁剪真实残差。 / Reverse-normalize for upstream scoring without clipping truth."""
        self.scorer = scorer_class((high - scores) / high, pi)
        self.thresholds = high * (1 - pi)
        self.split_size = len(pi)
        self.bit_queries = np.zeros(len(scores), dtype=int)
        self.score_queries = np.zeros(len(scores), dtype=int)
        self.covered = np.zeros(len(scores), dtype=bool)

    def get_reward(self, arm: int, t: int) -> float:
        """每个已审计条目恰好提供一次覆盖位。 / Supply exactly one bit for each audited item."""
        if self.bit_queries[t]:
            raise RuntimeError("重复请求审计覆盖位 / Duplicate audited-item coverage query")
        self.bit_queries[t] += 1
        reward = self.scorer.get_reward(arm, t)
        self.covered[t] = bool(reward)
        return reward

    def get_efficiency(self, arm: int, t: int) -> float:
        """原生对称区间宽度；输出另转单侧风险上界。 / Native symmetric width; reports use one-sided risk bounds."""
        return float(2 * self.thresholds[arm])

    def get_optimal_arm(self, t: int) -> int:
        """只有覆盖成功才解锁分数。 / Unlock the score only after successful coverage."""
        if self.bit_queries[t] != 1 or not self.covered[t] or self.score_queries[t]:
            raise RuntimeError("不允许读取本次审计分数 / This audit score cannot be revealed")
        self.score_queries[t] += 1
        return self.scorer.get_optimal_arm(t)


def _ocp(root: Path, calibration: np.ndarray, scores: np.ndarray, method: str, config: dict):
    """对已审计序列运行官方 EXP3P，不补造未审计反馈。 / Run upstream EXP3P without inventing unaudited feedback."""
    bandit = load_module(
        root, "ocp_unlock", "core/model/bandits/exp3p.py", "core.model.bandits.exp3p"
    ).EXP3P
    scorer = load_module(
        root, "ocp_unlock", "core/model/scoring/regression.py", "core.model.scoring.regression"
    ).RegThresholdModel
    settings = config["ocp"]
    sequence = np.concatenate((calibration, scores)) if settings["warmup_bandit"] else scores
    test_start = len(sequence) - len(scores)
    arms, horizon = settings["arms"], len(sequence)
    beta = np.sqrt(np.log(arms) / (arms * horizon))
    gamma = 1.05 * np.sqrt(arms * np.log(arms) / horizon)
    eta = 0.95 * beta
    minimum_horizon = max(
        4 * 1.05**2 * arms * np.log(arms),
        np.log(arms) / (arms * (1.05 / 0.95 - 1) ** 2),
    )
    if gamma >= 0.5 or (1 + beta) * arms * eta > gamma:
        raise ValueError(
            "官方 OCP 需要更多已审计条目，或在 Notebook 减少 arms / "
            f"Upstream OCP requires horizon above approximately {minimum_horizon:.2f}; "
            f"received {horizon}."
        )
    pi = np.linspace(0, 1, arms)
    feedback = _NativeFeedback(scorer, sequence, pi, settings["score_upper_bound"])
    parameters = (
        config["alpha"],
        settings["lambda"],
        settings["delta"],
        settings["function"],
        settings["displeasure"],
        settings["loss_type"],
    )
    structured = settings["variants"][method]
    options = (
        settings["C"],
        structured,
        settings["version"],
        settings["ot_power"],
        settings["power_const_option"],
    )
    random_state = np.random.get_state()
    try:
        np.random.seed(settings["seed"])
        algorithm = bandit(arms, horizon, False, feedback, pi, parameters, options)
        algorithm.run()
    finally:
        np.random.set_state(random_state)
    if not np.all(feedback.bit_queries == 1):
        raise RuntimeError(
            "官方 OCP 必须每条提供覆盖位 / OCP requires coverage feedback for every item"
        )
    if structured in (3, 4) and not np.array_equal(feedback.score_queries, feedback.covered):
        raise RuntimeError("OCP 解锁协议不匹配 / OCP score-unlocking protocol mismatch")
    thresholds = feedback.thresholds[algorithm.armt.astype(int)]
    return thresholds[test_start:], {
        "coverage_bit_queries": feedback.bit_queries[test_start:],
        "revealed_score_queries": feedback.score_queries[test_start:],
        "feedback_source_step": np.arange(len(scores)),
        "arm": algorithm.armt[test_start:].astype(int),
    }


def compare_calibration(
    root: Path, frame: pd.DataFrame, config: dict
) -> tuple[list[dict], pd.DataFrame, dict]:
    """比较校准验证运行与独立测试运行。 / Compare past validation calibration and independent test runs.

    每个测试运行重置到相同验证历史。标签只在预测完成后进入审计反馈。
    Reset each test run to the same validation history; reveal labels only after prediction.
    """
    calibration_ids, evaluation_ids = config["calibration_run_ids"], config["evaluation_run_ids"]
    if set(calibration_ids) & set(evaluation_ids):
        raise ValueError("校准/测试运行必须分离 / Calibration and test runs must be disjoint")
    if frame["predicted_risk"].isna().any():
        raise ValueError(
            "校准需要决策前的风险预测 / Calibration requires pre-decision risk predictions"
        )
    calibration_frame = _ordered_runs(frame, calibration_ids, "validation")
    calibration_scores = _positive_residual(calibration_frame)
    fixed = float(
        np.quantile(calibration_scores, 1 - config["alpha"], method=config["quantile_method"])
    )
    summaries, timelines = [], []
    for run_id in evaluation_ids:
        current = _ordered_runs(frame, [run_id], "test")
        scores = _positive_residual(current)
        means = current["predicted_risk"].to_numpy()
        truth = current["feature_recovery"].to_numpy()
        for method in config["methods"]:
            started = perf_counter()
            if method == "static_quantile":
                residual_upper = np.full(len(scores), fixed)
                protocol = {
                    "coverage_bit_queries": np.zeros(len(scores), dtype=int),
                    "revealed_score_queries": np.zeros(len(scores), dtype=int),
                    "feedback_source_step": np.full(len(scores), -1, dtype=int),
                    "arm": np.full(len(scores), -1, dtype=int),
                }
            elif method in ("conformal_sqt", "conformal_lqt"):
                residual_upper, protocol = _conformal(
                    root, calibration_scores, scores, method, config
                )
            elif method in config["ocp"]["variants"]:
                residual_upper, protocol = _ocp(root, calibration_scores, scores, method, config)
            else:
                raise ValueError(f"未知校准方法 / Unknown calibration method: {method}")
            elapsed = perf_counter() - started
            # 单侧正残差的阈值下限为零。 / A positive-residual threshold cannot be negative.
            upper = np.clip(means + np.maximum(residual_upper, 0), 0, 1)
            covered = truth <= upper
            summaries.append(
                {
                    "method": method,
                    "run_id": run_id,
                    "evaluation_axis": "audited_uploads_only",
                    "target_coverage": 1 - config["alpha"],
                    "audited_items": len(current),
                    "calibration_items": len(calibration_frame),
                    "coverage": float(covered.mean()),
                    "mean_risk_upper": float(upper.mean()),
                    "mean_margin": float((upper - means).mean()),
                    "mean_underestimation": float(np.maximum(truth - upper, 0).mean()),
                    "mean_residual_upper": float(residual_upper.mean()),
                    "coverage_bit_queries": int(protocol["coverage_bit_queries"].sum()),
                    "revealed_score_queries": int(protocol["revealed_score_queries"].sum()),
                    "seconds": elapsed,
                }
            )
            detail = current[
                ["run_id", "upload_id", "episode", "round_index", "client_id", "risk_version"]
            ].copy()
            detail["method"] = method
            detail["audited_step"] = np.arange(len(current))
            detail["predicted_risk"] = means
            detail["true_risk_evaluator_only"] = truth
            detail["positive_residual_evaluator_only"] = scores
            detail["residual_upper"] = residual_upper
            detail["risk_upper"] = upper
            detail["covered"] = covered
            for key, values in protocol.items():
                detail[key] = values
            timelines.append(detail)
    metadata = {
        "calibration_run_ids": calibration_ids,
        "evaluation_run_ids": evaluation_ids,
        "calibration_split": "validation",
        "evaluation_split": "test",
        "evaluation_axis": "audited_uploads_only",
        "residual": "max(feature_recovery - predecision predicted_risk, 0)",
        "test_reset": "Each held-out run starts from the same past validation audits.",
        "feedback": (
            "ConformalOpt receives audited residuals after prediction with the configured audit-index delay. "
            "OCP receives one actual coverage bit per audited item and scores only upon coverage success. "
            "Unaudited uploads never supply labels or coverage bits."
        ),
        "cost_accounting": (
            "Every row already incurred a complete gradient-inversion audit. OCP score-query "
            "counts measure native information revealed and do not represent saved attack calls; "
            "coverage bits are computed from those same completed audits."
        ),
        "limitation": (
            "Empirical comparisons conditional on the audited subset; no privacy guarantee, "
            "strict confidence bound, or all-upload coverage guarantee. OCP's native theoretical "
            "feedback condition does not hold over project unaudited uploads."
        ),
        "risk_versions": sorted(frame["risk_version"].unique().tolist()),
    }
    return summaries, pd.concat(timelines, ignore_index=True), metadata
