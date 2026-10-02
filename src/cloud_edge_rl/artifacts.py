"""保存每次实验的配置、指标和状态。 / Store each experiment's records."""

from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from cloud_edge_rl.config import RunSpec


def _utc_now() -> str:
    """使用 UTC 记录跨机器时间。 / Record portable UTC timestamps."""
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """管理单次实验文件及失败记录。 / Manage one run and its failure journal."""

    def __init__(self, root: Path, spec: RunSpec) -> None:
        """按 run_id 隔离产物。 / Isolate artifacts by run ID."""
        self.spec = spec
        self.path = Path(root) / "runs" / spec.run_id
        self.started_at = ""
        self.started_clock = 0.0

    def __enter__(self) -> RunStore:
        """创建新实验，拒绝覆盖已有结果。 / Create a run without overwriting prior results."""
        self.path.mkdir(parents=True, exist_ok=False)
        self.started_at = _utc_now()
        self.started_clock = time.perf_counter()
        self.write_json("config.json", asdict(self.spec))
        self.write_json(
            "metadata.json",
            {
                "run_id": self.spec.run_id,
                "stage": self.spec.stage,
                "method": self.spec.method,
                "seed": self.spec.seed,
                "risk_version": self.spec.risk_version,
                "provenance": self.spec.provenance,
            },
        )
        self._write_status("running")
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        error_traceback: TracebackType | None,
    ) -> bool:
        """只在运行边界记录失败并继续抛出。 / Journal errors without suppressing them."""
        if error is None:
            self._write_status("completed")
        else:
            details = "".join(traceback.format_exception(error_type, error, error_traceback))
            (self.path / "error.txt").write_text(details, encoding="utf-8")
            self._write_status("failed", error=str(error))
        return False

    def _write_status(self, state: str, **details: Any) -> None:
        """记录状态与实际耗时。 / Record run state and elapsed wall time."""
        self.write_json(
            "status.json",
            {
                "state": state,
                "started_at": self.started_at,
                "updated_at": _utc_now(),
                "elapsed_seconds": time.perf_counter() - self.started_clock,
                **details,
            },
        )

    def write_json(self, name: str, value: Any) -> Path:
        """原子替换 JSON，避免半份快照。 / Atomically replace a JSON snapshot."""
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
        return path

    def append_metrics(self, row: dict[str, Any]) -> Path:
        """按首行列名追加指标。 / Append metrics using the first row's column order."""
        path = self.path / "metrics.csv"
        exists = path.exists()
        if exists:
            with path.open(encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream))
        else:
            columns = list(row)
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        return path

    def save_checkpoint(self, state: dict[str, Any], name: str = "checkpoint.pt") -> Path:
        """保存用户提供的训练状态。 / Save the training state supplied by user code."""
        import torch

        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        return path

    def load_checkpoint(self, name: str = "checkpoint.pt") -> dict[str, Any]:
        """仅加载本项目可信断点。 / Load only trusted checkpoints created by this project."""
        import torch

        # RNG 状态可含 Python 对象，因此只接受自己的文件。 / RNG state may contain Python objects.
        return torch.load(self.path / name, map_location="cpu", weights_only=False)
