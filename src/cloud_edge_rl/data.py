"""Adult 数据下载与可复现编码 / Download and reproducibly encode Adult data."""

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORY_COLUMNS = [name for name in FEATURE_COLUMNS if name not in NUMERIC_COLUMNS]


@dataclass
class DataBundle:
    """编码数据及训练记录 / Encoded splits and original training records."""

    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_records: pd.DataFrame
    client_indices: list[np.ndarray]
    metadata: dict


def public_feature_metadata(metadata: dict) -> dict:
    """仅公开特征编码信息，隐藏真实行号 / Expose only encoding information, never private row identities."""
    fields = (
        "feature_columns",
        "numeric_columns",
        "category_columns",
        "categories",
        "numeric_mean",
        "numeric_scale",
        "numeric_std",
        "encoding_layout",
        "encoded_columns",
    )
    return {field: deepcopy(metadata[field]) for field in fields}


def _read_adult(content: bytes) -> tuple[pd.DataFrame, np.ndarray]:
    """读取官方文件并统一缺失值 / Parse official rows and canonicalize missing values."""
    frame = pd.read_csv(
        BytesIO(content),
        header=None,
        names=[*FEATURE_COLUMNS, "income"],
        skipinitialspace=True,
        na_values=["?"],
        comment="|",
    )
    records = frame[FEATURE_COLUMNS].copy()
    for column in CATEGORY_COLUMNS:
        records[column] = records[column].fillna("Missing").astype(str).str.strip()
    for column in NUMERIC_COLUMNS:
        records[column] = pd.to_numeric(records[column])
    labels = frame["income"].str.strip().str.rstrip(".").map({"<=50K": 0, ">50K": 1})
    return records, labels.to_numpy(dtype=np.int64)


def _split_validation(
    labels: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """按类别留出验证记录 / Hold out a seeded fraction within each label class."""
    rng = np.random.default_rng(seed)
    train_parts, validation_parts = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = round(len(indices) * fraction)
        if not 0 < count < len(indices):
            raise ValueError(
                "每类必须保留训练和验证记录 / Each class needs training and validation rows."
            )
        validation_parts.append(indices[:count])
        train_parts.append(indices[count:])
    return rng.permutation(np.concatenate(train_parts)), rng.permutation(
        np.concatenate(validation_parts)
    )


def _partition_clients(labels: np.ndarray, config: dict) -> list[np.ndarray]:
    """一次性划分站点且不重抽 / Partition clients once without resampling empty allocations."""
    rng = np.random.default_rng(config["partition_seed"])
    num_clients = config["num_clients"]
    if config["partition"] == "iid":
        clients = list(np.array_split(rng.permutation(len(labels)), num_clients))
    elif config["partition"] == "label_skew":
        parts = [[] for _ in range(num_clients)]
        for label in np.unique(labels):
            indices = rng.permutation(np.flatnonzero(labels == label))
            proportions = rng.dirichlet(np.full(num_clients, config["label_concentration"]))
            counts = rng.multinomial(len(indices), proportions)
            for client, rows in zip(parts, np.split(indices, np.cumsum(counts)[:-1])):
                client.extend(rows.tolist())
        clients = [rng.permutation(np.asarray(rows, dtype=np.int64)) for rows in parts]
    else:
        raise ValueError(
            "partition 必须为 iid 或 label_skew / partition must be iid or label_skew."
        )
    if any(len(rows) == 0 for rows in clients):
        raise ValueError(
            "一次划分产生空站点；请显式调整参数 / One-pass partition has an empty client; adjust parameters explicitly."
        )
    return clients


def _fit_encoding(records: pd.DataFrame) -> dict:
    """仅从训练记录拟合编码 / Fit scaling and category vocabularies on training rows only."""
    means = records[NUMERIC_COLUMNS].mean()
    standard_deviations = records[NUMERIC_COLUMNS].std(ddof=0)
    # 常量列以 1 编码，但评价时仍使用真实的 0 标准差。 / Encode constants with scale 1; retain true STD 0 for scoring.
    scales = standard_deviations.replace(0, 1)
    categories = {
        column: sorted(set(records[column]) - {"Missing", "Unknown"}) + ["Missing", "Unknown"]
        for column in CATEGORY_COLUMNS
    }
    layout, encoded_columns = {}, []
    for column in FEATURE_COLUMNS:
        start = len(encoded_columns)
        if column in NUMERIC_COLUMNS:
            encoded_columns.append(column)
            kind = "numeric"
        else:
            encoded_columns.extend(f"{column}={value}" for value in categories[column])
            kind = "categorical"
        layout[column] = {"start": start, "stop": len(encoded_columns), "kind": kind}
    return {
        "feature_columns": FEATURE_COLUMNS.copy(),
        "numeric_columns": NUMERIC_COLUMNS.copy(),
        "category_columns": CATEGORY_COLUMNS.copy(),
        "categories": categories,
        "numeric_mean": means.to_dict(),
        "numeric_scale": scales.to_dict(),
        "numeric_std": standard_deviations.to_dict(),
        "encoding_layout": layout,
        "encoded_columns": encoded_columns,
    }


def encode_features(records: pd.DataFrame, metadata: dict) -> np.ndarray:
    """应用训练编码，未知类别单列 / Apply training encodings with an explicit unknown category."""
    encoded = np.zeros((len(records), len(metadata["encoded_columns"])), dtype=np.float32)
    for column in metadata["feature_columns"]:
        layout = metadata["encoding_layout"][column]
        if layout["kind"] == "numeric":
            encoded[:, layout["start"]] = (
                records[column].to_numpy() - metadata["numeric_mean"][column]
            ) / metadata["numeric_scale"][column]
        else:
            categories = metadata["categories"][column]
            values = records[column].fillna("Missing")
            values = values.where(values.isin(categories), "Unknown")
            codes = pd.Categorical(values, categories=categories).codes
            encoded[np.arange(len(records)), layout["start"] + codes] = 1
    return encoded


def decode_features(encoded: np.ndarray, metadata: dict) -> pd.DataFrame:
    """反标准化并解码类别 / Reverse scaling and decode categorical blocks by their largest value."""
    columns = {}
    for column in metadata["feature_columns"]:
        layout = metadata["encoding_layout"][column]
        if layout["kind"] == "numeric":
            columns[column] = (
                encoded[:, layout["start"]] * metadata["numeric_scale"][column]
                + metadata["numeric_mean"][column]
            )
        else:
            codes = encoded[:, layout["start"] : layout["stop"]].argmax(axis=1)
            columns[column] = np.asarray(metadata["categories"][column])[codes]
    return pd.DataFrame(columns, columns=metadata["feature_columns"])


def prepare_adult(config: dict, cache_dir: Path) -> DataBundle:
    """缓存官方 ZIP 并划分数据 / Cache the official ZIP, preserve its test split, and prepare clients."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_id = sha256(config["url"].encode()).hexdigest()[:12]
    archive_path = cache_dir / f"adult-{source_id}.zip"
    if not archive_path.exists():
        with urlopen(config["url"]) as response:
            content = response.read()
        # Parallel runs only see a complete archive, never a partially written ZIP.
        # 并发运行只会读到完整归档，不会将下载中的半个文件当作缓存。
        with NamedTemporaryFile(dir=cache_dir, suffix=".zip.tmp", delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(archive_path)
    archive_bytes = archive_path.read_bytes()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        train_bytes = archive.read("adult.data")
        test_bytes = archive.read("adult.test")
    manifest = {
        "url": config["url"],
        "archive": archive_path.name,
        "sha256": sha256(archive_bytes).hexdigest(),
        "file_sha256": {
            "adult.data": sha256(train_bytes).hexdigest(),
            "adult.test": sha256(test_bytes).hexdigest(),
        },
    }
    (cache_dir / f"adult-{source_id}.manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    official_train, official_labels = _read_adult(train_bytes)
    test_records, test_y = _read_adult(test_bytes)
    train_rows, validation_rows = _split_validation(
        official_labels,
        config["validation_fraction"],
        config["split_seed"],
    )
    train_records = official_train.iloc[train_rows].reset_index(drop=True)
    validation_records = official_train.iloc[validation_rows].reset_index(drop=True)
    train_y, validation_y = official_labels[train_rows], official_labels[validation_rows]
    clients = _partition_clients(train_y, config)
    metadata = _fit_encoding(train_records)
    metadata.update(manifest)
    metadata["split"] = {
        "config": dict(config),
        "train_source_indices": train_rows.tolist(),
        "validation_source_indices": validation_rows.tolist(),
        "client_indices": [rows.tolist() for rows in clients],
    }
    return DataBundle(
        train_x=encode_features(train_records, metadata),
        train_y=train_y,
        validation_x=encode_features(validation_records, metadata),
        validation_y=validation_y,
        test_x=encode_features(test_records, metadata),
        test_y=test_y,
        train_records=train_records,
        client_indices=clients,
        metadata=metadata,
    )
