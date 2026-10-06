"""少量显式组件选择，保留研究者入口。 / A few explicit component choices retaining researcher entry points."""

from pathlib import Path

from .. import core
from .sources import source_root


def required_sources(config: dict, method: str, stage: str) -> list[str]:
    """从已选择方法派生源码依赖。 / Derive source dependencies from the selected methods."""
    components = config["components"]
    names = set()
    if components["backend"] == "tabular_mlp" or components["attack"] == "tabular_gia":
        names.add("tabular_gia")
    if components["risk"] == "tabm":
        names.add("tabm")
    if method == "metacs":
        names.add("metacs_fl")
    if method == "pqn":
        names.add("pqn")
    if stage == "compare_calibration":
        names.update(("conformalopt", "ocp_unlock"))
    return sorted(names)


def make_backend(spec, root: Path, data):
    """业务模型保持同一结构与优化设置。 / Keep the business model and optimizer settings shared."""
    config = {**spec.config["model"], "seed": spec.seed}
    match spec.config["components"]["backend"]:
        case "researcher":
            return core.FederatedBackend(data, config)
        case "tabular_mlp":
            from .federated import TabularFedSGD

            widths = config["hidden_sizes"]
            if len(set(widths)) != 1:
                raise ValueError("The upstream small MLP requires equal hidden widths")
            return TabularFedSGD(
                source_root(spec.config, root),
                data,
                {
                    **config,
                    "architecture": "tabular_gia_mlp",
                    "hidden_size": widths[0],
                    "hidden_layers": len(widths),
                },
            )
        case value:
            raise ValueError(f"Unknown federated implementation: {value}")


def make_risk(spec, root: Path):
    """选择用户核心或明确的表格风险对照。 / Select the researcher core or an explicit tabular-risk reference."""
    implementation = spec.config["components"]["risk"]
    if implementation == "researcher":
        return core.RiskEstimator({**spec.config["risk"], "seed": spec.seed}, spec.risk_version)
    if implementation in ("tabm", "mlp"):
        from .risk import ReferenceRiskEstimator

        return ReferenceRiskEstimator(
            source_root(spec.config, root),
            {
                **spec.config["comparisons"]["risk"],
                "seed": spec.seed,
                "device": spec.config["risk"]["device"],
                "model_kind": implementation,
            },
            spec.risk_version,
        )
    raise ValueError(f"Unknown risk implementation: {implementation}")


def make_attack(spec, root: Path, artifact_directory: Path):
    """保存目标克隆所需路径，攻击端不接收真值。 / Preserve clone paths without exposing reference records."""
    config = {**spec.config["attack"], "seed": spec.seed}
    match spec.config["components"]["attack"]:
        case "researcher":
            return core.GradientAttack(config)
        case "tabular_gia":
            from .attack import TabularGradientAttack

            return TabularGradientAttack(
                {
                    **config,
                    **spec.config["comparisons"]["attack"],
                    "source_root": str(source_root(spec.config, root)),
                    "artifact_directory": str(artifact_directory),
                }
            )
        case value:
            raise ValueError(f"Unknown attack implementation: {value}")


def train_policy(spec, root: Path, env, previous, on_checkpoint):
    """第三方 PQN 与研究者 DQN/PPO 独立路由。 / Route reference PQN separately from researcher DQN/PPO."""
    common = {
        "seed": spec.seed,
        "algorithm": spec.method,
        "risk_version": spec.risk_version,
        "warmup_actions": spec.config["execution"]["warmup_actions"],
    }
    if spec.method == "pqn":
        from .pqn import train_agent

        return train_agent(
            env, {**spec.config["comparisons"]["pqn"], **common}, previous, on_checkpoint
        )
    return core.train_agent(env, {**spec.config["rl"], **common}, previous, on_checkpoint)


def load_policy(spec, state: dict):
    """训练配置由保存的运行固定，评价不重新调参。 / Evaluate frozen training settings without retuning."""
    common = {"seed": spec.seed, "algorithm": spec.method, "risk_version": spec.risk_version}
    if spec.method == "pqn":
        from .pqn import load_agent

        return load_agent(state, {**spec.config["comparisons"]["pqn"], **common})
    return core.load_agent(state, {**spec.config["rl"], **common})
