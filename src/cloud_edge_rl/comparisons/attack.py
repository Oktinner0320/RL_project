"""官方 IG 反演的主项目适配。 / Main-project adapter for the upstream Inverting Gradients attack.

Source: scaleoutsystems/tabular-GIA, Apache-2.0; gradient-matching optimizer stays upstream.
保留官方梯度匹配优化器，原始记录只由框架的可信评价端读取。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..data import decode_features
from ..types import AttackInput
from .federated import make_model
from .sources import load_module


class TabularGradientAttack:
    """只接收更新前参数、单客户端梯度和公开编码。 / Receive only pre-update weights, single-client gradients and public encoding."""

    def __init__(self, config: dict):
        """源码与产物路径随目标克隆保留。 / Preserve source and artifact paths when the audit loop clones a target."""
        self.config = config
        self.root = Path(config["source_root"])
        if not config["labels_known"]:
            raise ValueError("The current tabular-GIA reference adapter uses known labels")

    def reconstruct(self, observed: AttackInput):
        """以梯度目标选重启，不读取真实恢复分数。 / Choose restarts by gradient objective, never reconstruction truth."""
        module = load_module(
            self.root,
            "tabular_gia",
            "leakpro/attacks/gia_attacks/invertinggradients.py",
            "leakpro.attacks.gia_attacks.invertinggradients",
        )
        from leakpro.fl_utils.data_utils import GiaTabularExtension
        from leakpro.fl_utils.gia_optimizers import MetaSGD
        from leakpro.fl_utils.gia_train import train_nostep

        config = self.config
        model_config = {**observed.model_config, "device": config["device"]}
        model = make_model(self.root, model_config)
        model.load_state_dict(observed.model_state)
        model.eval()
        names = [name for name, _ in model.named_parameters()]
        if set(names) != set(observed.gradients):
            raise ValueError("Observed gradient names do not match the recorded source model")
        device = torch.device(config["device"])
        dtype = getattr(torch, model_config["dtype"])
        gradients = [observed.gradients[name].to(device=device, dtype=dtype) for name in names]
        labels = observed.labels.to(device=device, dtype=dtype).reshape(-1, 1)
        shape = torch.zeros(
            (observed.batch_size, model_config["input_size"]), device=device, dtype=dtype
        )
        loader = DataLoader(
            TensorDataset(shape, labels), batch_size=observed.batch_size, shuffle=False
        )
        results = []
        for restart in range(config["restarts"]):
            torch.manual_seed((config["seed"] + restart) % 2**32)
            settings = module.InvertingConfig(
                tv_reg=config["tv_reg"],
                attack_lr=config["learning_rate"],
                at_iterations=config["iterations"],
                optimizer=MetaSGD(lr=model_config["learning_rate"]),
                criterion=model.criterion,
                data_extension=GiaTabularExtension(),
                epochs=config["epochs"],
                label_known=config["labels_known"],
                median_pooling=config["median_pooling"],
                top10norms=config["top10norms"],
                vectorized_client_batch_size=None,
            )
            attacker = module.InvertingGradients(
                model,
                loader,
                None,
                None,
                train_fn=train_nostep,
                configs=settings,
                observed_client_gradient=gradients,
            )
            attacker.attack_cache_folder_path = str(
                Path(config["artifact_directory"]) / f"target{config['seed']}-restart{restart}"
            )
            attacker.prepare_attack()
            last_iteration = -1
            for last_iteration, _, _ in attacker.run_attack():
                pass
            if last_iteration != config["iterations"] - 1 or not np.isfinite(attacker.best_loss):
                raise RuntimeError(
                    f"Upstream inversion did not complete: iteration={last_iteration}"
                )
            reconstruction = torch.cat([batch[0] for batch in attacker.best_reconstruction])
            results.append((float(attacker.best_loss), reconstruction.detach().cpu().numpy()))
        loss, reconstruction = min(results, key=lambda result: result[0])
        self.last_diagnostics = {
            "gradient_objective": loss,
            "iterations": config["iterations"],
            "restarts": config["restarts"],
        }
        return decode_features(reconstruction, observed.feature_metadata)
