# 框架验收记录 / Framework verification

核验日期：2026-10-02。以下为实际执行结果，算法 TODO 未被替代或填充。

| 检查 | 结果 |
|---|---|
| UV 独立环境 | UV 0.12.11 完成 lock 与 sync；Python 3.12.14 |
| Windows 依赖 | PyTorch 2.11.0+cpu、Gymnasium 1.3.0、Modal 1.6.0 实际导入成功 |
| 完整测试 | 86 passed；Gymnasium 对开放 Box 上下界给出两条提示，没有失败 |
| 代码检查 | `ruff check .` 与 `ruff format --check .` 通过 |
| Notebook | 最终 22 个单元通过真实 Jupyter 内核执行；默认只运行本地版本检查和 Adult 准备 |
| Adult | 官方 ZIP 下载成功；训练 26,049、验证 6,512、官方测试 16,281；编码维度 121；三个客户端各 8,683 条 |
| Modal 镜像 | Python 3.12.14 Debian 镜像以同一 `uv.lock`、UV 0.12.11 完成 Linux 依赖构建 |
| Modal GPU | 实际返回 NVIDIA L4、PyTorch 2.11.0+cu128、CUDA 12.8，`cuda_available=true` |
| 持久化 | 云端运行完成，配置、运行版本、状态与摘要成功从 Volume 下载 |
| Git | 已初始化本地仓库；尚未创建提交，所有源码可供审阅 |

已运行的云端检查为 `runtime_check`，不是模型训练或攻击实验。未执行 MLP/FedSGD、DQN/PPO、风险拟合或梯度反演，也没有科学效果或最终云账单结论。

本地证据（自动生成并被 Git 忽略）：

- `output/verification/experiment.executed.ipynb`：默认流程的执行副本；修改参数仍使用根目录的唯一 Notebook。
- `output/verification/modal-check.json`：Modal 调用 ID 与运行 ID。
- `output/verification/modal-result.json`：实际 GPU 与版本返回。
- `runs/runtime_check-fixed-s7-4d4df2bf5699/`：云端检查下载的完整产物。
- `runs/prepare_data-*/`：数据来源、SHA256、训练/验证/测试划分和客户端索引。

## 已覆盖的关键行为

- 本地与 Modal 检查使用不同运行身份；本地缓存不能替代云端验证。
- Notebook 仅有一个参数单元；配置快照不受后续对象修改影响。
- 调参候选无重复，验证与最终测试继承选定 trial 的参数。
- 正常训练终止和时间上限截断分开，预热计入预算，恢复不重复 reset。
- 模型通信量不随 batch 错误倍增；风险先按客户端截断再聚合。
- 未拟合风险模型时显式缺失，不将其解释为实际零风险。
- 攻击者输入和参考记录分离；上传时保存源模型构建配置及独立张量快照。
- 同一上传在不同审计顺序下获得相同攻击随机种子；调用预算与算时预算均记录。
- 验证/测试审计标签不能进入风险拟合；风险阈值会实际改变超阈统计。
- 未实现核心直接抛出 `NotImplementedError`，失败状态和错误日志保留。

## 仍由研究者完成

按 `core.py` 的双语契约实现学习与攻击函数。真实算法接入后，需要重新核验 FedSGD 数学语义、优化器/随机状态恢复、攻击有效性、阶段切换时 replay 清理及研究假设。框架单元测试不能替代这些科学验收。
