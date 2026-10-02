"""使用明确的合成测试夹具检验汇总。 / Verify analysis with explicit synthetic test fixtures."""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from cloud_edge_rl import analysis


def write_run(root, run_id, predicted_risk=(None, None)):
    """写入两步测试记录，不代表研究结果。 / Write two fixture steps, not research results."""
    path = root / "runs" / run_id
    path.mkdir(parents=True)
    spec = {
        "stage": "evaluate",
        "method": "dqn",
        "seed": 3,
        "risk_version": "v1",
        "config": {
            "experiment": {"tag": "test-fixture", "variant": "conservative-dqn"},
            "execution": {"split": "test"},
            "data": {"partition": "iid"},
        },
    }
    (path / "config.json").write_text(json.dumps(spec), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps({"test": {"accuracy": 0.75}}), encoding="utf-8")
    pd.DataFrame(
        {
            "episode": [0, 0],
            "round": [1, 2],
            "elapsed_seconds": [2.0, 4.0],
            "processed_samples": [16, 32],
            "validation_loss": [0.8, 0.6],
            "predicted_risk": predicted_risk,
            "worst_predicted_risk": predicted_risk,
            "energy_joules": [10, 20],
            "upload_bytes": [100, 200],
            "download_bytes": [100, 200],
        }
    ).to_csv(path / "metrics.csv", index=False)
    return path


def comparison_frame():
    """准备共同种子和一个无法配对的种子。 / Include matched seeds and one unmatched seed."""
    return pd.DataFrame(
        {
            "method": [
                "fixed",
                "fixed",
                "conservative-dqn",
                "conservative-dqn",
                "conservative-dqn",
            ],
            "seed": [1, 2, 2, 1, 9],
            "loss": [0.8, 0.6, 0.3, 0.6, 0.01],
            "stage": ["evaluate"] * 5,
            "split": ["test"] * 5,
            "partition": ["iid"] * 5,
        }
    )


def test_summary_keeps_variant_identity_and_missing_risk(tmp_path):
    """保守算法标签和缺失值不能被覆盖。 / Preserve variant labels and unavailable risk."""
    write_run(tmp_path, "test-run")
    frame = analysis.summarize_runs(tmp_path, ["test-run"])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["method"] == "conservative-dqn"
    assert row["algorithm"] == "dqn"
    assert row["test.accuracy"] == 0.75
    assert row["validation_loss"] == 0.6
    assert row["metric_rows"] == 2
    assert row["episodes"] == 1
    assert pd.isna(row["mean_predicted_risk"])
    assert pd.isna(row["max_predicted_risk"])
    assert (row["stage"], row["split"], row["partition"]) == ("evaluate", "test", "iid")


def test_paired_summary_uses_only_common_seeds():
    """未配对种子不能改变方法或差值均值。 / Unmatched seeds must not change paired estimates."""
    row = analysis.pairwise_summary(comparison_frame(), "loss", "fixed").iloc[0]
    assert row["n_pairs"] == 2
    assert row["method_mean"] == pytest.approx(0.45)
    assert row["reference_mean"] == pytest.approx(0.7)
    assert row["difference_mean"] == pytest.approx(-0.25)
    assert row["difference_std"] == pytest.approx(np.std([-0.2, -0.3], ddof=1))


def test_repeated_checkpoints_are_not_independent_seeds():
    """重复检查点应要求显式选择。 / Repeated checkpoints require an explicit selection."""
    frame = comparison_frame()
    with pytest.raises(ValueError, match="checkpoint"):
        analysis.pairwise_summary(pd.concat([frame, frame.iloc[[2]]]), "loss", "fixed")


@pytest.mark.parametrize(
    "column,value", [("stage", "collect"), ("split", "train"), ("partition", "label_skew")]
)
def test_paired_summary_rejects_mixed_conditions(column, value):
    """不同研究场景不可混成配对统计。 / Mixed study conditions must not be pooled."""
    frame = comparison_frame()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="stage, split, and partition"):
        analysis.pairwise_summary(frame, "loss", "fixed")


def test_single_pair_has_unknown_sample_std():
    """一个种子的样本标准差未知，而不是零。 / One seed cannot establish sample variability."""
    frame = comparison_frame().query("seed == 1")
    row = analysis.pairwise_summary(frame, "loss", "fixed").iloc[0]
    assert row["n_pairs"] == 1
    assert pd.isna(row["difference_std"])


@pytest.mark.parametrize("x_axis", ["elapsed_seconds", "processed_samples"])
def test_curves_do_not_draw_unavailable_risk(tmp_path, monkeypatch, x_axis):
    """曲线支持两个公平轴，缺失风险不产生零线。 / Support both fair axes without a fake risk line."""
    write_run(tmp_path, "test-run")
    saved = {}
    save = analysis._save_figure

    def inspect_figure(figure, output_dir, name):
        """检查图中真实曲线再写出。 / Inspect real line data before exporting."""
        saved["risk_lines"] = len(figure.axes[1].lines)
        saved["xlabel"] = figure.axes[0].get_xlabel()
        return save(figure, output_dir, name)

    monkeypatch.setattr(analysis, "_save_figure", inspect_figure)
    paths = analysis.plot_learning_curves(tmp_path, ["test-run"], tmp_path / "plots", x_axis)
    assert saved["risk_lines"] == 0
    assert saved["xlabel"]
    assert all(path.stat().st_size > 1000 for path in paths.values())


def test_calibration_filters_null_predictions_and_separates_versions(tmp_path, monkeypatch):
    """只有真实预测与重构标签进入校准。 / Calibration needs both a prediction and observed recovery."""
    path = tmp_path / "runs" / "audit-test"
    path.mkdir(parents=True)
    rows = [
        {"risk_version": "v1", "predicted_risk": None, "feature_recovery": 0.9},
        {"risk_version": "v1", "predicted_risk": 0.2, "feature_recovery": 0.5},
        {"risk_version": "v1", "predicted_risk": 0.0, "feature_recovery": 0.2},
        {"risk_version": "v2", "predicted_risk": 0.8, "feature_recovery": 0.6},
    ]
    (path / "audits.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    recorded = []
    bar_heights = []
    scatter = Axes.scatter
    save = analysis._save_figure

    def record_scatter(axis, x, y, *args, **kwargs):
        """保留测试图的输入以检验分组。 / Record plotted inputs to verify grouping."""
        recorded.append((list(x), list(y), kwargs["label"]))
        return scatter(axis, x, y, *args, **kwargs)

    def inspect_bars(figure, output_dir, name):
        """校验低估量而不改变绘图。 / Verify underestimation amounts without changing the plot."""
        bar_heights.extend(bar.get_height() for bar in figure.axes[1].patches)
        return save(figure, output_dir, name)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(analysis, "_save_figure", inspect_bars)
    paths = analysis.plot_risk_calibration(tmp_path, ["audit-test"], tmp_path / "plots")
    assert recorded == [([0.2, 0.0], [0.5, 0.2], "v1 (n=2)"), ([0.8], [0.6], "v2 (n=1)")]
    assert bar_heights == pytest.approx([0.25, 0.0])
    assert all(path.stat().st_size > 1000 for path in paths.values())


def test_empty_inputs_create_no_fake_outputs(tmp_path):
    """无真实结果时返回空对象。 / Return empty objects when no real results exist."""
    frame = analysis.summarize_runs(tmp_path, [])
    assert frame.empty
    assert analysis.pairwise_summary(frame, "loss", "fixed").empty
    assert analysis.plot_learning_curves(tmp_path, [], tmp_path / "plots", "elapsed_seconds") == {}
    assert analysis.plot_risk_calibration(tmp_path, [], tmp_path / "plots") == {}
    assert not (tmp_path / "plots").exists()


def write_audit_rows(root, rows):
    """写入已标注为测试的审计记录。 / Write explicitly synthetic audit fixtures."""
    path = root / "runs" / "audit-fixture"
    path.mkdir(parents=True)
    (path / "audits.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_risk_analysis_keeps_source_runs_versions_and_splits_separate(tmp_path):
    """多个上传只形成一个来源运行统计。 / Multiple uploads yield one source-run summary."""
    rows = [
        {
            "run_id": "train-a",
            "risk_version": "v1",
            "split": "train",
            "predicted_risk": 0.2,
            "feature_recovery": value,
        }
        for value in (0.3, 0.4, 0.5)
    ]
    rows += [
        {
            "run_id": "test-b",
            "risk_version": "v1",
            "split": "test",
            "predicted_risk": 0.8,
            "feature_recovery": 0.6,
        },
        {
            "run_id": "test-b",
            "risk_version": "v2",
            "split": "test",
            "predicted_risk": 0.4,
            "feature_recovery": 0.6,
        },
    ]
    write_audit_rows(tmp_path, rows)
    frame = analysis.analyze_risk_runs(tmp_path, ["audit-fixture"], threshold=0.5)
    assert len(frame) == 3
    train = frame.query("source_run_id == 'train-a'").iloc[0]
    assert train["n_uploads"] == 3
    assert train["split"] == "train"
    assert train["mae"] == pytest.approx(0.2)
    assert set(frame.query("source_run_id == 'test-b'")["risk_version"]) == {"v1", "v2"}
    assert frame.query("source_run_id == 'test-b'")["split"].eq("test").all()


def test_risk_threshold_controls_exceedance_only(tmp_path):
    """Notebook 阈值应改变真实超阈频率。 / The notebook threshold must control exceedance."""
    write_audit_rows(
        tmp_path,
        [
            {
                "run_id": "source-a",
                "risk_version": "v1",
                "split": "test",
                "predicted_risk": 0.1,
                "feature_recovery": value,
            }
            for value in (0.2, 0.8)
        ],
    )
    low = analysis.analyze_risk_runs(tmp_path, ["audit-fixture"], threshold=0.3).iloc[0]
    high = analysis.analyze_risk_runs(tmp_path, ["audit-fixture"], threshold=0.9).iloc[0]
    assert low["exceedance_frequency"] == 0.5
    assert high["exceedance_frequency"] == 0.0
    for key in ("mae", "mean_underestimation", "underestimation_frequency"):
        assert low[key] == high[key]


def test_risk_analysis_excludes_none_but_retains_zero(tmp_path):
    """缺失预测不可填零，真实零预测必须保留。 / Missing predictions differ from genuine zeros."""
    write_audit_rows(
        tmp_path,
        [
            {
                "run_id": "source-a",
                "risk_version": "v1",
                "split": "test",
                "predicted_risk": prediction,
                "feature_recovery": actual,
            }
            for prediction, actual in ((None, 0.9), (0.0, 0.4), (0.2, None))
        ],
    )
    row = analysis.analyze_risk_runs(tmp_path, ["audit-fixture"], threshold=0.3).iloc[0]
    assert row["n_uploads"] == 1
    assert row["mae"] == 0.4
    assert row["mean_underestimation"] == 0.4
    assert row["underestimation_frequency"] == 1.0
    assert row["exceedance_frequency"] == 1.0
    assert analysis.analyze_risk_runs(tmp_path, [], threshold=0.3).empty
