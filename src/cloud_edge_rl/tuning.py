"""显式搜索空间的候选表；不运行算法。 / Candidate tables from explicit search spaces."""

from itertools import product

import numpy as np


def sample_trials(config: dict) -> list[dict]:
    """无放回抽样候选，数值全部由 Notebook 提供。 / Sample candidates without replacement."""
    names = ("learning_rate", "risk_weight", "uncertainty_weight")
    candidates = [dict(zip(names, values)) for values in product(*(config[name] for name in names))]
    selected = np.random.default_rng(config["seed"]).choice(
        len(candidates), config["trials"], replace=False
    )
    return [{"trial": index, **candidates[position]} for index, position in enumerate(selected)]
