# 云边联邦训练的风险感知强化学习框架

本项目研究：动态调度改变训练轨迹后，经验隐私风险代理是否出现系统性低估，以及不确定性和有限梯度反演审计能否提高决策可靠性。客户端选择与批量大小的联合调度是应用场景，模型效用、时间和能量是共同评价维度。

这是供研究者填写算法的工程框架。数据处理、Gymnasium 环境、资源模拟、基线调度、审计记录、实验编排和 Modal 接口由框架提供；PyTorch 模型、FedSGD、DQN/PPO、风险估计器与梯度反演由您实现。相关接口仍以 `NotImplementedError` 停止，不提供假梯度、替代收敛曲线或虚构实验结果。

## 安装与入口

在项目目录运行：

```powershell
uv sync --locked
uv run jupyter lab
```

打开 [experiment.ipynb](experiment.ipynb)。所有全局实验参数、阶段开关、运行输入和 Modal 资源都在 Notebook 的唯一 `CONFIG` 单元格中调整。Python 和 UV 版本由 `.python-version`、`pyproject.toml` 指定，依赖解析以 `uv.lock` 为准。本机使用 CPU PyTorch 做调试和分析，Modal Linux 镜像使用锁定的 CUDA wheel。

Notebook 的默认执行只检查本地运行环境并准备 Adult 数据；研究训练与云任务通过唯一配置单元格中的 `CONFIG.workflow.jobs` 阶段开关显式启用。首次数据准备需要访问官方数据源。不要在模块、第二份配置文件、命令行超参数或 Notebook 的其他单元格中建立另一套调参入口。

检查框架：

```powershell
uv run pytest
uv run ruff check .
```

测试替身只位于 `tests/`，用于验证接口、计量和隔离；测试通过不代表学习算法完成、梯度反演有效或 Modal 云训练已经验收。

## 文件与职责

| 入口 | 用途 |
|---|---|
| `experiment.ipynb` | 配置、提交、查询、下载与结果分析的唯一入口 |
| `src/cloud_edge_rl/core.py` | 您实现的全部学习和反演核心 |
| `data.py`、`metrics.py` | 官方数据处理、原字段重构评价和风险误差指标 |
| `environment.py`、`resources.py` | Gymnasium 交互、联合动作、资源计量、预算和公开状态 |
| `baselines.py` | 固定、轮询、时延贪心、风险贪心，以及共享预热函数 |
| `audit.py` | 保存实际上传、执行有限审计、隔离攻击输入与参考真值 |
| `pipeline.py`、`config.py` | 阶段编排、配置快照、源码与依赖身份 |
| `modal_runner.py`、`artifacts.py` | 云任务、Volume、运行状态和产物存储 |
| `tests/` | 框架行为和边界测试 |

表中 Python 模块均位于 `src/cloud_edge_rl/`。完整研究协议见 [RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md)，实施任务与课程映射见 [TASKS.md](docs/TASKS.md)，第三方来源见 [REFERENCES.md](docs/REFERENCES.md)。

## 您需要实现的核心接口

| 接口 | 您负责的内容 |
|---|---|
| `FederatedBackend.reset`、`step_round` | 初始化 PyTorch MLP；同一更新前模型下计算各选中客户端的平均批梯度，聚合后只更新一次全局模型 |
| `FederatedBackend.checkpoint`、`load_checkpoint`、`evaluate_test` | 模型、优化器、采样随机状态；精确恢复；最终测试集评价 |
| `Agent.act`、`checkpoint` | 从公开观察选择动作，保存策略状态 |
| `train_agent`、`load_agent` | DQN/PPO 训练、推理恢复、优化器与随机状态、经验回放和训练断点 |
| `RiskEstimator.fit`、`predict`、`checkpoint`、`load_checkpoint` | 按完整运行分组拟合风险，返回均值与不确定性，冻结并保存模型版本 |
| `GradientAttack.reconstruct` | 仅从 `AttackInput` 做单客户端梯度反演，返回 Adult 原始字段表 |

输入输出字段定义在 `types.py`。`ClientUpdate.sample_ids` 属于可信评价侧；`AttackInput` 包括更新前模型、其公开构建配置 `model_config`、单客户端批平均梯度、已知标签、批量大小和公开编码信息。构建配置必须对应源上传的架构、激活函数、损失和 dtype，不能用审计阶段当前配置替代。不要将参考记录或真实重构结果传入策略和攻击函数。

`CloudEdgeEnv` 遵循 Gymnasium 的 `reset` / `step` 接口。公开观察包含资源、进度、采样历史、候选动作收益/开销/风险以及 `risk_available`。`progress` 的顺序为验证损失、验证准确率、已用时间比例、已用轮数比例、剩余样本预算比例；完整训练轮超预算后，最后一项可以为负。需要扁平向量时可使用 Gymnasium 的 `spaces.flatten`。

## 实验闭环

1. 检查本地环境并准备数据，阅读数据划分与编码元信息。
2. 实现 MLP/FedSGD，先用固定或轮询策略采集真实上传。初始模式显式传入 `risk=None`；风险不可用并不等于零风险。该模式要求风险惩罚权重为零，保存的预测均值和不确定性为 `None`。
3. 实现梯度反演，先执行 `benchmark` 阶段对实际上传按 batch 测量攻击耗时，再在 Notebook 中将测量运行作为 `audit` 输入。对实际上传执行周期或随机审计，得到原字段重构标签。
4. 实现风险估计与不确定性，按完整运行分组拟合/校准，保存明确的风险版本。
5. 实现 DQN/PPO，冻结本阶段的风险版本并训练调度策略；所有策略共享候选状态与预热流程。
6. 在策略诱导的新轨迹上继续审计并更新风险模型，比较固定、周期、随机和主动审计的效果。
7. 使用独立测试运行和审计面板评价代理低估、真实攻击恢复率、效用及资源开销。未出现偏移或 RL 未优于基线同样是有效研究结果。

每次新的 `reset` 后，按 Notebook 配置调用 `baselines.run_warmup`。该函数逐个执行给定动作，并在终止或截断时立即停止；预热本身计入同一时间、样本、能量和审计记录。用户训练循环不得额外免费执行一轮预热。

## Modal 与恢复

Notebook 调用 `modal_runner` 提交配置快照。Modal 使用同一源码和 `uv.lock` 构建镜像，在一个逻辑训练任务中模拟三个客户端；这不是三台真实边缘机器的网络部署。L4 等 GPU 类型及 CPU、内存、超时、并发和 Volume 名均从 Notebook 读取，不在 Python 模块硬编码实验资源。

初次使用请按 [Modal 官方认证说明](https://modal.com/docs/guide) 完成本机认证。提交、等待、读取结果和下载产物分别由 `submit_runs`、`wait_runs` / `get_run`、`download_run` 提供；具体阶段入口以 Notebook 为准。Volume 保存数据缓存、配置、指标、攻击输入、审计标签和检查点，远程任务在成功或失败时提交持久化日志。

`runs/<run_id>/` 保存每次运行的输入和结果。相同源码与实验输入产生稳定身份，已有运行不会被静默覆盖。`workflow.reuse_completed` 启用时，重复执行会读取已完成摘要；相同身份的失败或未完成运行仍会报错。恢复操作在唯一配置单元格中更改 `experiment.tag`，并将旧检查点设为新运行的显式输入。

同一风险版本下的完整恢复，需要用户保存并恢复策略、优化器、随机状态、回放缓冲区及包装环境的 `snapshot()`；恢复使用 `load_snapshot()`，不要再 `reset` 或重复预热。切换风险版本时保留策略权重、清空旧版本 replay 并开始新阶段。这些算法状态迁移由 `core.py` 的实现负责，框架不会暗中替您决定。

小额云预算是实验规划目标，不是 Modal 计费硬上限。资源超时与审计预算用于限制工作量，实际计费应结合真实调用记录和 Modal 账单检查。

## 当前边界

实际环境、Notebook、测试与 Modal 验证记录见 [VERIFICATION.md](docs/VERIFICATION.md)。调参后，`validate` 和 `evaluate` 都通过 `inputs.training_run_ids` 继承对应训练快照；透明基线使用 `evaluate_baselines`，避免用当前默认参数覆盖已经选定的 trial。

代码与函数使用中英文简要说明，必要的接口检查直接报错，不静默补造输入或降级到替代算法。已完成一次真实 Modal 镜像构建、L4 运行环境检查与产物下载；这证明云端环境连通，不代表算法级训练验收。框架验收与研究结论分开记录，项目尚不能声称已完成风险校准、RL 训练、独立攻击评价或课程研究结论。
