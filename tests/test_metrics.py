"""验证原字段评估语义 / Verify reconstruction metrics in original feature space."""

import numpy as np
import pandas as pd
import pytest

from cloud_edge_rl.metrics import (
    reconstruction_metrics,
    risk_metrics,
    sample_marginal_records,
    sample_random_records,
)


def metadata():
    """构造最小混合字段元数据 / Describe a minimal mixed-feature schema."""
    return {
        "feature_columns": ["age", "job"],
        "numeric_columns": ["age"],
        "category_columns": ["job"],
        "numeric_scale": {"age": 10.0},
        "numeric_std": {"age": 10.0},
    }


def test_matching_removes_row_order():
    """顺序颠倒不影响恢复率 / Reversing the reconstructed batch must not change recovery."""
    original = pd.DataFrame({"age": [20.0, 60.0], "job": ["A", "B"]})
    result = reconstruction_metrics(
        original, original.iloc[::-1], metadata(), numeric_tolerance=0.01
    )
    assert result["match_indices"] == [1, 0]
    assert result["feature_recovery"] == result["emr"] == 1.0


def test_feature_scores_use_training_std_and_exact_categories():
    """数值容差以训练标准差为单位 / Apply numerical tolerance in training-STD units and exact category equality."""
    original = pd.DataFrame({"age": [20.0, 60.0], "job": ["A", "B"]})
    reconstructed = pd.DataFrame({"age": [21.0, 62.0], "job": ["Wrong", "B"]})
    result = reconstruction_metrics(original, reconstructed, metadata(), numeric_tolerance=0.1)
    assert result["numeric_recovery"] == 0.5
    assert result["categorical_recovery"] == 0.5
    assert result["feature_recovery"] == 0.5
    assert result["emr"] == 0.0


def test_unequal_batches_are_not_silently_scored():
    """不以子集匹配掩盖缺失记录 / Reject partial matching that would conceal missing records."""
    original = pd.DataFrame({"age": [20.0, 60.0], "job": ["A", "B"]})
    with pytest.raises(ValueError, match="equal lengths"):
        reconstruction_metrics(original, original.iloc[:1], metadata(), numeric_tolerance=0.1)


def test_constant_training_column_requires_exact_recovery():
    """常量列不能借编码比例放宽容差 / Do not turn a constant column's encoding scale into tolerance."""
    original = pd.DataFrame({"age": [20.0], "job": ["A"]})
    reconstructed = pd.DataFrame({"age": [20.05], "job": ["A"]})
    schema = metadata()
    schema["numeric_scale"]["age"] = 1.0
    schema["numeric_std"]["age"] = 0.0
    result = reconstruction_metrics(original, reconstructed, schema, numeric_tolerance=0.1)
    assert result["numeric_recovery"] == 0.0
    assert result["emr"] == 0.0


def test_risk_error_and_actual_exceedance():
    """区分风险低估与真实超限 / Separate underestimation from actual threshold exceedance."""
    result = risk_metrics(np.array([0.2, 0.8, 0.5]), np.array([0.4, 0.6, 0.5]), threshold=0.5)
    assert result["mae"] == pytest.approx(0.4 / 3)
    assert result["mean_underestimation"] == pytest.approx(0.2 / 3)
    assert result["underestimation_frequency"] == pytest.approx(1 / 3)
    assert result["exceedance_frequency"] == pytest.approx(1 / 3)


def test_risk_columns_do_not_broadcast_into_all_pairs():
    """列向量应逐条评价 / Score column vectors pairwise instead of broadcasting all pairs."""
    predicted = np.array([0.1, 0.7])
    actual = np.array([[0.1], [0.7]])
    assert risk_metrics(predicted, actual, threshold=0.5)["mae"] == 0.0
    with pytest.raises(ValueError, match="one-to-one"):
        risk_metrics(predicted, np.array([0.1]), threshold=0.5)


def test_seeded_random_and_marginal_baselines():
    """基线可重复且仅使用参考分布 / Keep baselines reproducible and within the reference distribution."""
    reference = pd.DataFrame({"age": [20.0, 30.0, 60.0], "job": ["A", "A", "B"]})
    random_rows = sample_random_records(reference, metadata(), size=30, seed=9)
    pd.testing.assert_frame_equal(
        random_rows, sample_random_records(reference, metadata(), size=30, seed=9)
    )
    assert random_rows["age"].between(20, 60).all()
    assert set(random_rows["job"]) <= {"A", "B"}
    marginal_rows = sample_marginal_records(reference, size=30, seed=9)
    pd.testing.assert_frame_equal(
        marginal_rows, sample_marginal_records(reference, size=30, seed=9)
    )
    assert set(marginal_rows["age"]) <= set(reference["age"])
    assert set(marginal_rows["job"]) <= set(reference["job"])
