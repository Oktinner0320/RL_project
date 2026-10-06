"""隔离测试官方在线分位数跟踪 / Benchmark official online quantile tracking on chronological residuals."""

import importlib
import json
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


def _load_upstream(project_root):
    """从下载目录导入官方包 / Import the unchanged downloaded official package."""
    repository = Path(project_root) / "reference_algorithms/repos/conformalopt"
    sys.path.insert(0, str(repository))
    try:
        module = importlib.import_module("conformalopt")
    finally:
        sys.path.pop(0)
    return module, repository


def _scores(repository: Path, config: dict):
    """保留真实残差，或明确生成合成压力序列 / Preserve real residuals or explicitly generate a synthetic stress stream."""
    if config["source"] == "official_googl":
        source = repository / "conformalopt/data" / f"{config['source_data_name']}.csv"
        scores = np.loadtxt(source)[config["warmup_scores"] :]
        metadata = {
            "source": config["source"],
            "source_file": str(source.relative_to(repository)),
            "source_sha256": sha256(source.read_bytes()).hexdigest(),
            "description": "Chronological published stock-forecaster absolute residuals; not FL risk labels.",
        }
    elif config["source"] == "synthetic_shift":
        rng = np.random.default_rng(config["seed"])
        means = np.where(
            np.arange(config["n_steps"]) < config["shift_index"],
            config["pre_mean"],
            config["post_mean"],
        )
        deviations = np.zeros(config["n_steps"])
        noise = rng.normal(0, config["noise_std"], config["n_steps"])
        for index in range(config["n_steps"]):
            deviations[index] = config["ar_coefficient"] * deviations[index - 1] + noise[index]
        scores = np.maximum(means + deviations, 0)
        metadata = {
            "source": config["source"],
            "description": "Synthetic chronological shift STRESS TEST, not empirical privacy risk.",
        }
    else:
        raise ValueError("未知来源 / Unknown residual source.")
    return scores / config["score_scale"], metadata


def _summarize(scores, upper, config):
    """按时间段报告单侧覆盖与宽度 / Report chronological one-sided coverage and upper interval width."""
    covered = scores <= upper
    width = upper - config["clip_low"]
    difference = scores - upper
    pinball = np.maximum((1 - config["alpha"]) * difference, -config["alpha"] * difference)
    split = round(len(scores) * config["test_break_fraction"])
    metrics = {}
    for name, selected in (
        ("all", slice(None)),
        ("early", slice(0, split)),
        ("late", slice(split, None)),
    ):
        metrics[f"{name}_upper_coverage"] = float(covered[selected].mean())
        metrics[f"{name}_mean_width"] = float(width[selected].mean())
        metrics[f"{name}_pinball_loss"] = float(pinball[selected].mean())
    metrics["scores_above_clip_high"] = int(np.sum(scores > config["clip_high"]))
    return metrics


def run_conformal(project_root: Path, output_dir: Path, config: dict) -> list[dict]:
    """使用官方 SQT/LQT 比较固定阈值 / Compare upstream SQT/LQT to a fixed calibration quantile."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    module, repository = _load_upstream(project_root)
    scores, source_metadata = _scores(repository, config)
    split = round(len(scores) * config["validation_fraction"])
    calibration, test = scores[:split], scores[split:]
    np.savez_compressed(
        output / f"{config['source']}-scores.npz", calibration=calibration, test=test
    )
    started = perf_counter()
    static = np.quantile(calibration, 1 - config["alpha"], method=config["quantile_method"])
    fixed = np.full(len(test), np.clip(static, config["clip_low"], config["clip_high"]))
    rows = [
        {
            "method": "static_quantile",
            "seconds": perf_counter() - started,
            "source": config["source"],
            "feedback_delay": config["feedback_delay"],
            "target_coverage": 1 - config["alpha"],
            **_summarize(test, fixed, config),
        }
    ]
    detail = [
        {
            "step": index,
            "score": score,
            "method": "static_quantile",
            "raw_upper": static,
            "upper": fixed[index],
            "covered": bool(score <= fixed[index]),
            "feedback_source_step": None,
        }
        for index, score in enumerate(test)
    ]
    for name in ("scalar", "linear"):
        started = perf_counter()
        tracker = module.ConformalPredictor(
            alpha=config["alpha"],
            lr_type=config["learning_rate_type"],
            quantile_tracker=name,
            scorecaster=config["scorecaster"],
            hypers={
                "lr": config["learning_rate"],
                "p_order_qt": config["tracker_order"],
                "bias": config["tracker_bias"],
            },
        )
        # 显式学习率避免官方自动调参选优路径。 / Explicit settings bypass upstream automatic tuning selection.
        tracker.init_active_fields()
        for score in calibration:
            predicted = tracker.predict()
            tracker.step(predicted, score)
        raw_upper = np.empty(len(test))
        for index, score in enumerate(test):
            predicted = tracker.predict()
            raw_upper[index] = predicted
            feedback_index = index - config["feedback_delay"]
            # 先预测，后接收已到达标签；没有未来反馈。 / Predict first, then apply arrived labels without future feedback.
            if feedback_index >= 0:
                tracker.step(raw_upper[feedback_index], test[feedback_index])
            upper = np.clip(predicted, config["clip_low"], config["clip_high"])
            detail.append(
                {
                    "step": index,
                    "score": score,
                    "method": name,
                    "raw_upper": predicted,
                    "upper": upper,
                    "covered": bool(score <= upper),
                    "feedback_source_step": feedback_index if feedback_index >= 0 else None,
                }
            )
        upper = np.clip(raw_upper, config["clip_low"], config["clip_high"])
        rows.append(
            {
                "method": name,
                "seconds": perf_counter() - started,
                "source": config["source"],
                "feedback_delay": config["feedback_delay"],
                "target_coverage": 1 - config["alpha"],
                **_summarize(test, upper, config),
            }
        )
    pd.DataFrame(rows).to_csv(output / f"{config['source']}-summary.csv", index=False)
    pd.DataFrame(detail).to_csv(output / f"{config['source']}-sequence.csv", index=False)
    metadata = {
        "config": config,
        "data": source_metadata,
        "upstream_main_sha256": sha256(
            (repository / "conformalopt/main.py").read_bytes()
        ).hexdigest(),
        "runtime_packages": {
            name: version(name) for name in ("numpy", "pandas", "statsmodels", "tqdm")
        },
        "runtime_lock_sha256": sha256(
            (Path(project_root) / "reference_algorithms/runtime/uv.lock").read_bytes()
        ).hexdigest(),
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_pin": next(
            item
            for item in json.loads(
                (Path(project_root) / "reference_algorithms/catalog.json").read_text(
                    encoding="utf-8"
                )
            )
            if item["name"] == "conformalopt"
        ),
        "license": "setup.py declares MIT; repository lacks a license-text file",
        "split": {"calibration": len(calibration), "test": len(test)},
        "limitation": "Full feedback on this stream. Delayed-feedback adapter is empirical; no guarantee for sparse selected FL audits.",
        "bounds": "Only predicted upper bounds are clipped; true scores stay unchanged, including values above one.",
        "time_segments": "Early/late split is descriptive; it is not an identified real-world change point.",
    }
    (output / f"{config['source']}-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return rows
