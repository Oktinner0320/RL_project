"""绝对时间驱动的云边资源与开销模型。 / Cloud-edge costs driven by absolute simulated time."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoundCost:
    """同步训练轮的时延、能量和客户端耗时。 / Latency, energy, and client times for a synchronous round."""

    time_seconds: float
    energy_joules: float
    client_seconds: np.ndarray


class ResourceModel:
    """固定时槽随机轨迹，所有策略在同一绝对时间看到相同资源。 / A seeded slot trace shared across policies."""

    def __init__(self, config: dict, seed: int):
        """保留资源参数和整条轨迹的种子。 / Keep resource settings and the seed for the full trace."""
        self.config = config
        self.seed = seed

    def observe(self, elapsed_seconds: float) -> np.ndarray:
        """只返回当前时槽的上下行带宽和忙碌比例。 / Return current upload/download bandwidth and busy fraction."""
        slot = int(elapsed_seconds // self.config["interval_seconds"])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, slot]))
        count = len(self.config["bandwidth_mbps"])
        jitter = self.config["bandwidth_jitter"]
        bandwidth = np.column_stack((self.config["bandwidth_mbps"], self.config["download_mbps"]))
        bandwidth = bandwidth * rng.uniform(1 - jitter, 1 + jitter, (count, 2))
        load = rng.uniform(*self.config["load_range"], size=count)
        return np.column_stack((bandwidth, load)).astype(np.float32)

    def estimate(
        self,
        client_ids: tuple[int, ...],
        batch_size: int,
        model_bytes: int,
        elapsed_seconds: float,
        sample_counts: np.ndarray | None = None,
    ) -> RoundCost:
        """同步轮等待最慢客户端，模型通信量不随 batch 增大。 / Wait for the slowest client; model bytes are batch-independent."""
        selected = np.asarray(client_ids, dtype=int)
        resources = self.observe(elapsed_seconds)[selected]
        samples = np.full(len(selected), batch_size) if sample_counts is None else sample_counts
        bits = model_bytes * 8
        download = bits / (resources[:, 1] * 1_000_000)
        upload = bits / (resources[:, 0] * 1_000_000)
        throughput = np.asarray(self.config["samples_per_second"])[selected] * (1 - resources[:, 2])
        compute = samples / throughput
        client_seconds = download + compute + upload
        aggregate = self.config["aggregation_seconds"]
        energy = (
            download @ np.asarray(self.config["receive_watts"])[selected]
            + compute @ np.asarray(self.config["compute_watts"])[selected]
            + upload @ np.asarray(self.config["transmit_watts"])[selected]
            + aggregate * self.config["server_watts"]
        )
        return RoundCost(float(client_seconds.max() + aggregate), float(energy), client_seconds)
