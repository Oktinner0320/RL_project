"""合成 Adult 数据测试，不访问网络 / Test synthetic Adult rows without network access."""

import json
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from cloud_edge_rl.data import (
    FEATURE_COLUMNS,
    _partition_clients,
    decode_features,
    prepare_adult,
    public_feature_metadata,
)


def make_source(tmp_path):
    """创建结构匹配的本地 ZIP / Create a local ZIP with the official file layout."""
    rows = [
        [
            20 + i,
            "?" if i % 3 == 0 else "Private",
            100 + i,
            "HS-grad",
            9,
            "Never-married",
            "Sales",
            "Own-child",
            "White",
            "Female",
            0,
            0,
            40,
            "United-States",
            "<=50K" if i % 2 == 0 else ">50K",
        ]
        for i in range(16)
    ]
    test_rows = [
        [
            1000,
            "Test-only",
            5000,
            "HS-grad",
            9,
            "Never-married",
            "Sales",
            "Own-child",
            "White",
            "Female",
            0,
            0,
            40,
            "United-States",
            "<=50K.",
        ],
        [
            30,
            "?",
            100,
            "HS-grad",
            9,
            "Never-married",
            "Sales",
            "Own-child",
            "White",
            "Female",
            0,
            0,
            40,
            "United-States",
            ">50K.",
        ],
    ]
    source = tmp_path / "source.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("adult.data", pd.DataFrame(rows).to_csv(index=False, header=False))
        archive.writestr(
            "adult.test",
            "| official test header\n" + pd.DataFrame(test_rows).to_csv(index=False, header=False),
        )
    return source


def data_config(source):
    """显式提供所有数据参数 / Supply every data parameter explicitly."""
    return {
        "url": source.as_uri(),
        "validation_fraction": 0.25,
        "split_seed": 7,
        "partition_seed": 4,
        "num_clients": 3,
        "partition": "iid",
        "label_concentration": 1.0,
    }


def test_preserves_test_and_fits_only_training(tmp_path):
    """验证官方测试隔离及训练集编码 / Verify official-test isolation and train-only encodings."""
    source = make_source(tmp_path)
    bundle = prepare_adult(data_config(source), tmp_path / "cache")
    assert bundle.train_x.dtype == np.float32
    assert bundle.train_y.dtype == np.int64
    assert len(bundle.train_x) == 12
    assert len(bundle.validation_x) == 4
    assert bundle.test_y.tolist() == [0, 1]
    assert np.bincount(bundle.validation_y).tolist() == [2, 2]
    assert bundle.metadata["numeric_mean"]["age"] == bundle.train_records["age"].mean()
    assert bundle.metadata["numeric_scale"]["age"] == bundle.train_records["age"].std(ddof=0)
    assert bundle.metadata["numeric_scale"]["capital-gain"] == 1
    assert bundle.metadata["numeric_std"]["capital-gain"] == 0
    assert "Test-only" not in bundle.metadata["categories"]["workclass"]
    assert {"Missing", "Unknown"}.issubset(bundle.metadata["categories"]["workclass"])
    decoded_test = decode_features(bundle.test_x, bundle.metadata)
    assert decoded_test["workclass"].tolist() == ["Unknown", "Missing"]
    assert decoded_test.iloc[0]["age"] == pytest.approx(1000)
    decoded_train = decode_features(bundle.train_x, bundle.metadata)
    assert decoded_train.columns.tolist() == FEATURE_COLUMNS
    for column in bundle.metadata["numeric_columns"]:
        np.testing.assert_allclose(decoded_train[column], bundle.train_records[column], rtol=1e-6)
    for column in bundle.metadata["category_columns"]:
        assert decoded_train[column].tolist() == bundle.train_records[column].tolist()
    np.testing.assert_array_equal(np.sort(np.concatenate(bundle.client_indices)), np.arange(12))
    split = bundle.metadata["split"]
    assert set(split["train_source_indices"]).isdisjoint(split["validation_source_indices"])
    assert sorted(split["train_source_indices"] + split["validation_source_indices"]) == list(
        range(16)
    )


def test_cache_hash_and_reproducibility(tmp_path):
    """缓存脱离源文件仍可重用 / Reuse a hashed cache after the local source disappears."""
    source = make_source(tmp_path)
    config = data_config(source)
    expected_hash = sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    first = prepare_adult(config, cache)
    source.unlink()
    second = prepare_adult(config, cache)
    np.testing.assert_array_equal(first.train_x, second.train_x)
    assert first.metadata == second.metadata
    manifest = json.loads(next(cache.glob("*.manifest.json")).read_text(encoding="utf-8"))
    assert manifest["sha256"] == expected_hash == first.metadata["sha256"]
    assert manifest["url"] == config["url"]


def test_label_skew_is_single_seeded_partition(tmp_path):
    """站点划分覆盖所有记录且可重复 / Cover every training row once with a reproducible skew partition."""
    config = data_config(tmp_path / "unused.zip")
    config["partition"] = "label_skew"
    labels = np.tile([0, 1], 60)
    first = _partition_clients(labels, config)
    second = _partition_clients(labels, config)
    assert all(len(rows) > 0 for rows in first)
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(len(labels)))
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_empty_client_fails_without_retry(tmp_path):
    """不静默重抽空站点 / Fail explicitly instead of silently retrying an empty allocation."""
    config = data_config(tmp_path / "unused.zip")
    config["partition"] = "label_skew"
    config["num_clients"] = 5
    with pytest.raises(ValueError, match="empty client"):
        _partition_clients(np.array([0, 0, 1, 1]), config)


def test_public_metadata_excludes_record_identities_and_is_independent(tmp_path):
    """白名单公开编码，且与私有字典独立 / Allowlist encoding fields and detach them from private metadata."""
    source = make_source(tmp_path)
    bundle = prepare_adult(data_config(source), tmp_path / "cache")
    bundle.metadata["private_records"] = bundle.train_records.to_dict(orient="records")
    public = public_feature_metadata(bundle.metadata)
    assert set(public) == {
        "feature_columns",
        "numeric_columns",
        "category_columns",
        "categories",
        "numeric_mean",
        "numeric_scale",
        "numeric_std",
        "encoding_layout",
        "encoded_columns",
    }
    pd.testing.assert_frame_equal(
        decode_features(bundle.train_x, public),
        decode_features(bundle.train_x, bundle.metadata),
    )
    public["categories"]["workclass"].append("Changed")
    public["encoding_layout"]["age"]["start"] = 999
    assert "Changed" not in bundle.metadata["categories"]["workclass"]
    assert bundle.metadata["encoding_layout"]["age"]["start"] == 0
