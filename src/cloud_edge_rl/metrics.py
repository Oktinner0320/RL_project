"""重建与风险评估 / Evaluate reconstruction and predicted privacy risk."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def reconstruction_metrics(
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
    metadata: dict,
    numeric_tolerance: float,
) -> dict:
    """先匹配记录再评估原字段 / Match unordered records, then score original fields in training-STD units."""
    if len(original) != len(reconstructed):
        raise ValueError(
            "重建批次必须等长 / Original and reconstructed batches must have equal lengths."
        )
    distances = np.zeros((len(original), len(reconstructed)), dtype=float)
    for column in metadata["numeric_columns"]:
        distances += (
            np.abs(original[column].to_numpy()[:, None] - reconstructed[column].to_numpy()[None, :])
            / metadata["numeric_scale"][column]
        )
    for column in metadata["category_columns"]:
        distances += (
            original[column].to_numpy()[:, None] != reconstructed[column].to_numpy()[None, :]
        )
    _, match_indices = linear_sum_assignment(distances)
    aligned = reconstructed.iloc[match_indices]
    numeric = np.column_stack(
        [
            np.abs(original[column].to_numpy() - aligned[column].to_numpy())
            <= numeric_tolerance * metadata["numeric_std"][column]
            for column in metadata["numeric_columns"]
        ]
    )
    categorical = np.column_stack(
        [
            original[column].to_numpy() == aligned[column].to_numpy()
            for column in metadata["category_columns"]
        ]
    )
    recovered = np.concatenate((numeric, categorical), axis=1)
    return {
        "feature_recovery": float(recovered.mean()),
        "numeric_recovery": float(numeric.mean()),
        "categorical_recovery": float(categorical.mean()),
        "emr": float(recovered.all(axis=1).mean()),
        "match_indices": match_indices.tolist(),
    }


def risk_metrics(predicted: np.ndarray, actual: np.ndarray, threshold: float) -> dict:
    """衡量误差、低估与超阈值频率 / Measure error, underestimation, and actual threshold exceedance."""
    predicted = np.asarray(predicted).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
    if predicted.shape != actual.shape:
        raise ValueError(
            "预测与真值必须逐条对应 / Predictions and actual risks must correspond one-to-one."
        )
    error = actual - predicted
    return {
        "mae": float(np.abs(error).mean()),
        "mean_underestimation": float(np.maximum(error, 0).mean()),
        "underestimation_frequency": float((error > 0).mean()),
        "exceedance_frequency": float((actual > threshold).mean()),
    }


def sample_random_records(
    reference: pd.DataFrame, metadata: dict, size: int, seed: int
) -> pd.DataFrame:
    """评估端均匀基线 / Evaluator-only uniform sampling over training ranges and observed categories."""
    rng = np.random.default_rng(seed)
    columns = {}
    for column in metadata["feature_columns"]:
        if column in metadata["numeric_columns"]:
            columns[column] = rng.uniform(
                reference[column].min(), reference[column].max(), size=size
            )
        else:
            columns[column] = rng.choice(reference[column].unique(), size=size)
    return pd.DataFrame(columns, columns=metadata["feature_columns"])


def sample_marginal_records(reference: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """评估端边际先验 / Evaluator-only independent column sampling from empirical training marginals."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {column: rng.choice(reference[column].to_numpy(), size=size) for column in reference}
    )
