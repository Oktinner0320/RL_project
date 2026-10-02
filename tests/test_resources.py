"""验证时槽一致性和物理开销公式。 / Verify slot consistency and physical cost formulas."""

import numpy as np
import pytest

from cloud_edge_rl.resources import ResourceModel


def resource_config() -> dict:
    """构造测试用资源配置。 / Build resource settings used only by tests."""
    return {
        "interval_seconds": 10,
        "bandwidth_mbps": [8, 4],
        "download_mbps": [16, 8],
        "samples_per_second": [100, 50],
        "compute_watts": [20, 30],
        "transmit_watts": [5, 6],
        "receive_watts": [3, 4],
        "bandwidth_jitter": 0,
        "load_range": [0, 0],
        "aggregation_seconds": 0.5,
        "server_watts": 40,
    }


def test_resource_trace_uses_absolute_slots():
    config = resource_config()
    config.update(bandwidth_jitter=0.2, load_range=[0.1, 0.5])
    resources = ResourceModel(config, seed=17)
    first = resources.observe(12)
    resources.observe(60)
    np.testing.assert_array_equal(first, resources.observe(19))
    np.testing.assert_array_equal(first, ResourceModel(config, seed=17).observe(12))
    assert not np.array_equal(first, resources.observe(20))


def test_round_waits_for_slowest_client_and_counts_aggregation():
    resources = ResourceModel(resource_config(), seed=17)
    cost = resources.estimate((0, 1), 10, 1_000_000, 0)
    # 第一客户端 0.5+0.1+1，第二客户端 1+0.2+2。 / Client times are download + compute + upload.
    np.testing.assert_allclose(cost.client_seconds, [1.6, 3.2])
    assert cost.time_seconds == pytest.approx(3.7)
    assert cost.energy_joules == pytest.approx(
        0.5 * 3 + 0.1 * 20 + 1 * 5 + 1 * 4 + 0.2 * 30 + 2 * 6 + 0.5 * 40
    )


def test_batch_size_changes_compute_but_not_model_communication():
    resources = ResourceModel(resource_config(), seed=17)
    small = resources.estimate((0,), 10, 1_000_000, 0)
    large = resources.estimate((0,), 20, 1_000_000, 0)
    assert large.time_seconds - small.time_seconds == pytest.approx(0.1)
    assert large.energy_joules - small.energy_joules == pytest.approx(2)


def test_actual_sample_count_controls_executed_cost():
    resources = ResourceModel(resource_config(), seed=17)
    cost = resources.estimate((0,), 20, 1_000_000, 0, sample_counts=np.asarray([5]))
    assert cost.time_seconds == pytest.approx(2.05)
