# 云边联邦训练的风险感知强化学习框架

[中文](README.md) | [English](README.en.md)

课程论文材料见 [paper/README.md](paper/README.md)：详细中文提纲、英文 Typst 骨架、ACM 风格课程适配模板，以及编译和版面核验说明。成稿正文预算为 20 页，当前骨架按实际内容分页。

近期外部算法筛选与实测见 [reference_algorithms/REPORT.md](reference_algorithms/REPORT.md)，主项目接入结果见 [BASELINE_INTEGRATION_RESULTS.md](docs/BASELINE_INTEGRATION_RESULTS.md)。六个官方仓库、独立 UV 测试环境及原始结果均保存在 `reference_algorithms/`；当前唯一可编辑配置为 `config/experiment.json`。

本项目研究：动态调度改变训练轨迹后，经验隐私风险代理是否出现系统性低估，以及不确定性和有限梯度反演审计能否提高决策可靠性。客户端选择与批量大小的联合调度是应用场景，模型效用、时间和能量是共同评价维度。

研究者原生第一版已实现，`core.py` 保留公开入口，具体实现位于 `src/cloud_edge_rl/algorithms/`：`federated.py`（同步 FedSGD）、`attack.py`（已知标签梯度反演）、`risk.py`（MLP 风险集成）、`dqn.py`、`ppo.py` 和 `policy.py`（训练与恢复）。第三方基线仍独立位于 `comparisons/`，不会自动替换原生核心。代码可运行不等于研究性能已经验收。

## 首版交付与验收目标

首版交付“曝光一致、风险校准的联合调度原型”，以 [一版验收标准](docs/一版验收标准.md)（`first-acceptance-v1`）为准：

- Adult IID、正常资源、三个独立配对训练种子；开发验证选择并冻结一个 DQN/PPO 主方法及唯一强参考 `b`。
- 三种子均值同时满足：准确率最多下降 **1 个百分点**，模拟时间和能量增幅各不超过 **15%**，完整真实审计的曝光加权风险至少降低 **5%**；至少 **2/3** 种子任务可行且真实风险下降。
- 独立共同面板的代理曝光加权 MAE 不超过冻结参考的 **1.10 倍**；相对未经校准代理的配对消融须降低平均低估至少 **5%**，且该误差增幅不超过 **10%**。
- 保留 30 轮含 12 轮预热、每方法/种子/划分最多 3,000 次实际交互、真值隔离、完整轨迹审计和独立确认要求；不得重置已耗尽预算。

五种子、双划分、双资源场景及更严格的 95% 参考目标保留为后续研究，不作为首版全部必须通过的门槛。**当前尚未通过首版验收；管线修复和历史开发结果不等于独立正式确认。** 所有可执行参数在 `config/experiment.json` 登记。

## 安装与入口

在项目目录运行：

```powershell
uv sync --locked --extra comparisons
uv run --extra comparisons jupyter lab
```

编辑 [config/experiment.json](config/experiment.json)，再打开 [experiment.ipynb](experiment.ipynb)。Notebook 只负责加载配置、准备、提交/恢复、查询、回传和核验。Python 和 UV 版本由 `.python-version`、`pyproject.toml` 指定，依赖解析以 `uv.lock` 为准。本机使用 CPU PyTorch 做调试和分析，Modal Linux 镜像使用锁定的 CUDA wheel。

默认所有 `workflow.experiment` 操作开关关闭，Run All 只加载配置并读取已有本地状态。新执行统一通过 `Experiment` 生命周期；当前全链路为 `fresh` 七方法开发实验，自动绑定采集、攻击、审计、风险拟合、分段训练和验证依赖。正式测试仍需单独满足冻结与隔离准入。

`prepare_enabled` 生成不可变任务图；`submission_enabled` 允许提交；`resume_enabled` 显式选择恢复；`refresh_enabled` 查询云端；`collection_enabled` 和 `verification_enabled` 分别启用回传与本地核验。研究参数冻结后，只有这六个操作开关可变。`graph_admission` 指向既有预算、历史交互、源码与隔离审批的绑定产物，不是另一份算法配置。使用与状态说明见 [训练管线](docs/训练管线.md)。

检查框架：

```powershell
uv run --extra comparisons pytest
uv run --extra comparisons ruff check .
```

测试替身用于验证接口、计量和隔离；原生核心与第三方适配器另有真实算法的定向测试。测试通过不等于首版性能验收或正式研究结论成立。

## 接入的第三方基线

`CONFIG.components` 默认选择 `researcher`。`CONFIG.variants` 中的 `reference-*`、`metacs`、`pqn` 明确选择第三方组件；风险检查点记录实现与架构，评价阶段冻结训练设置。所有方法收到相同的公开资源、候选评分和最近参与窗口。

PQN 从官方 JAX 实现移植为 PyTorch，串行调用真实 Gymnasium/FedSGD 环境。MetaCS 调用官方目标与历史公平性函数，对共同 12 个动作精确枚举；该适配没有执行原版 LNS 搜索。

历史 `comparisons.workflow` 基线闭环和原调用记录保留用于兼容恢复；当前 Notebook 不再启用第二套提交、下载控制器。各算法与比较规划仍可复用。周期取样比较的是已审计子集，不能据此判断全客户端或最坏客户端风险；最终审计标签不得回流拟合、训练或校准。

2026-10-05 基线预实验的 64 步 PQN 和 2 个 OCP arms 是历史小规模可行性设置，不能作为充分调参或首版确认实验。新比较在唯一 JSON 配置登记配对种子、训练与攻击预算；OCP 原生协议要求足够多的真实审计条目，不会复制数据满足长度条件。其覆盖率仅针对已审计序列，信息揭示次数不等于节省实际攻击次数。

上述历史基线的 24 个风险拟合标签各来自两条运行的 12 次审计，均处于预热轮次；不能据此声称覆盖了完整训练阶段。固定策略留出审计包含预热后上传。风险 MAE 较低也不等于低估更少或调度更安全，应同时检查低估量和真实攻击标签。OCP 上游损失常数固定为 40，传入的 `C` 不是有效调参项；接入保持原生规则。

固定仓库仍在 `reference_algorithms/repos/`，不会整体并入包；缺少源码请按[参考目录说明](reference_algorithms/README.md)恢复。ConformalOpt/OCP 的许可记录和 PQN 移植归属见 [THIRD_PARTY.md](docs/THIRD_PARTY.md)。

## 文件与职责

| 入口 | 用途 |
|---|---|
| `config/experiment.json` | 唯一可编辑实验配置 |
| `experiment.ipynb` | 提交、恢复、查询、下载与结果查看前端 |
| `experiment.py`、`experiment_graph.py` | 统一生命周期、云端依赖图和持久化状态 |
| `experiment_admission.py`、`experiment_transfer.py` | 历史额度准入、依赖闭包整批回传 |
| `src/cloud_edge_rl/core.py`、`algorithms/` | 原生算法公开入口及 FedSGD、攻击、风险、DQN/PPO 的实际实现 |
| `data.py`、`metrics.py` | 官方数据处理、原字段重构评价和风险误差指标 |
| `environment.py`、`resources.py` | Gymnasium 交互、联合动作、资源计量、预算和公开状态 |
| `baselines.py` | 固定、轮询、时延贪心、风险贪心，以及共享预热函数 |
| `comparisons/` | 第三方基线适配、显式组件选择、真实审计校准比较与闭环编排 |
| `audit.py` | 保存实际上传、执行有限审计、隔离攻击输入与参考真值 |
| `pipeline.py`、`config.py` | 阶段编排、配置快照、源码与依赖身份 |
| `modal_runner.py`、`artifacts.py` | 云任务、Volume、运行状态和产物存储 |
| `tests/` | 框架行为和边界测试 |

表中 Python 模块均位于 `src/cloud_edge_rl/`。首版门槛见 [一版验收标准](docs/一版验收标准.md)，完整研究协议见 [RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md)，实施任务与课程映射见 [TASKS.md](docs/TASKS.md)，第三方来源见 [REFERENCES.md](docs/REFERENCES.md)。`docs/` 当前被 Git 忽略，因此本 README 同时保留正式目标摘要；本次不改变忽略规则。

## 原生核心接口与检查职责

| 接口 | 您负责的内容 |
|---|---|
| `FederatedBackend.reset`、`step_round` | 初始化 PyTorch MLP；同一更新前模型下计算各选中客户端的平均批梯度，聚合后只更新一次全局模型 |
| `FederatedBackend.checkpoint`、`load_checkpoint`、`evaluate_test` | 模型、优化器、采样随机状态；精确恢复；最终测试集评价 |
| `Agent.act`、`checkpoint` | 从公开观察选择动作，保存策略状态 |
| `train_agent`、`load_agent` | DQN/PPO 训练、推理恢复、优化器与随机状态、经验回放和训练断点 |
| `RiskEstimator.fit`、`predict`、`checkpoint`、`load_checkpoint` | 按完整运行分组拟合风险，返回均值与不确定性，冻结并保存模型版本 |
| `GradientAttack.reconstruct` | 仅从 `AttackInput` 做单客户端梯度反演，返回 Adult 原始字段表 |

输入输出字段定义在 `types.py`。`ClientUpdate.sample_ids` 属于可信评价侧；`AttackInput` 包括更新前模型、其公开构建配置 `model_config`、单客户端批平均梯度、已知标签、批量大小和公开编码信息。构建配置必须对应源上传的架构、激活函数、损失和 dtype，不能用审计阶段当前配置替代。不要将参考记录或真实重构结果传入策略和攻击函数。

`CloudEdgeEnv` 遵循 Gymnasium 的 `reset` / `step` 接口。公开观察包含资源、进度、采样历史、最近参与窗口 `participation_window`、候选动作收益/开销/风险以及 `risk_available`。参与窗口记录已经执行的客户端选择，所有策略均可见。`progress` 的顺序为验证损失、验证准确率、已用时间比例、已用轮数比例、剩余样本预算比例；完整训练轮超预算后，最后一项可以为负。需要扁平向量时可使用 Gymnasium 的 `spaces.flatten`。

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

Notebook 调用 `Experiment` 提交独立于本机连接的 Modal 任务图。Modal 使用冻结源码与 `uv.lock` 构建镜像，在一个逻辑任务中模拟三个客户端。GPU、CPU、内存、时限和 Volume 来自唯一 JSON 配置；各计算子进程有显式时限与进程树看门狗。

初次使用请按 [Modal 官方认证说明](https://modal.com/docs/guide) 完成本机认证。云端按依赖执行并核验，逐阶段提交 Volume；任务结束后统一回传完整依赖闭包，再做本地语义核验。Notebook 关闭不会触发重训，下载失败只恢复传输。

`runs/<run_id>/` 保存每次运行的输入和结果。任务图首次准备时生成并冻结非内容派生的运行 ID；重开同一控制目录沿用这些 ID。计算、云端核验、回传和本地闭合分别记录。未知提交不会自动重发，历史失败和未知消费不会清零。

恢复使用同一控制目录。已知原调用先挂接；只有确认其安全暂停或仅核验失败，才允许显式继续。已经计算的节点只重新核验，计算失败/状态不明须先对账。每次继续使用独立回传目录，避免旧清单覆盖新状态。旧实验记录保留，不能自动冒充新源码下的已完成节点。

同一风险版本下的完整恢复，需要用户保存并恢复策略、优化器、随机状态、回放缓冲区及包装环境的 `snapshot()`；恢复使用 `load_snapshot()`，不要再 `reset` 或重复预热。切换风险版本时保留策略权重、清空旧版本 replay 并开始新阶段。这些算法状态迁移由 `core.py` 的实现负责，框架不会暗中替您决定。

小额云预算是实验规划目标，不是 Modal 计费硬上限。资源超时与审计预算用于限制工作量，实际计费应结合真实调用记录和 Modal 账单检查。

## 当前边界

实际环境、Notebook、测试与 Modal 验证记录见 [VERIFICATION.md](docs/VERIFICATION.md)。调参后，`validate` 和 `evaluate` 都通过 `inputs.training_run_ids` 继承对应训练快照；透明基线使用 `evaluate_baselines`，避免用当前默认参数覆盖已经选定的 trial。

代码与函数使用中英文简要说明，必要的接口检查直接报错，不静默补造输入或降级到替代算法。2026-10-05 的历史基线闭环完成 Modal L4 的 13 阶段、26 个真实运行及 26 单元执行 Notebook；当时 145 项测试通过。首轮审计覆盖失败和随后产物 I/O 控制器超时均保留记录，显式恢复接回 19 个原调用，仅提交 7 个新任务。实际结果与原始 CSV 见 [集成实测记录](docs/BASELINE_INTEGRATION_RESULTS.md)。后续原生实现和开发检查另见 [整体验收记录](docs/NATIVE_OVERALL_ACCEPTANCE.md)，它们尚未完成首版独立确认。

该历史运行只验证小规模闭环：200 次迭代、一次重启的攻击弱于边际先验，六种调度审计的 EMR 均为零；四种贪心/MetaCS 对照在预热后选择完全相同动作，风险差异不能归因于调度改进。TabM 降低 MAE 但没有减少平均低估；SQT/LQT 的完全覆盖伴随接近 1 的风险上界，OCP 未达到 0.9 目标覆盖。该轮校准仅做事后对照，没有修改调度奖励。首版仍需按冻结后的新标准补齐独立面板、消融、三种子确认和完整实际审计。
