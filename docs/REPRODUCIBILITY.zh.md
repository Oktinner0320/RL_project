# 安装与复现

[项目主页](../README.zh.md) | [English](REPRODUCIBILITY.md)

除非另有说明，所有命令都在项目根目录执行。
查看已保存结果、运行测试和重新训练的前提条件不同。

## 1. 安装锁定环境

- Python **3.12.14**；项目要求 Python 3.12。
- uv **0.12.11**，按[官方说明](https://docs.astral.sh/uv/getting-started/installation/)安装。
- 支持 Windows 或 Linux；Windows 使用 CPU PyTorch，Linux 使用锁定的 CUDA 版本。
- 可选交互展示需要 Node **22.12+** 或 **24+**，以及 npm。
- 云端执行需要已认证的 [Modal 账户](https://modal.com/docs/guide)。

```powershell
uv sync --locked --extra comparisons
uv run --locked --extra comparisons jupyter lab
```

选择项目的 `.venv` 内核。保留 `uv.lock`，不要用未锁定版本的安装替代。

## 2. 不执行云任务，仅查看本地内容

运行 `experiment.ipynb` 前，将 `config/experiment.json` 中
`workflow.experiment` 下的六个现有开关设为 `false`：

```json
{
  "prepare_enabled": false,
  "submission_enabled": false,
  "resume_enabled": false,
  "refresh_enabled": false,
  "collection_enabled": false,
  "verification_enabled": false
}
```

只修改这六个值，不要替换整个配置对象。当前提交的配置开启了其中五个，
**并非只读默认值**。全部关闭后，笔记本只读取已有本地状态，或报告
`not_prepared`，不会创建实验或启动云任务。

## 3. 查看已交付结果

[开发结果目录](../output/fresh-seven-method-rerun-20261010-v1/report-analysis-20261010-v1/)
包含六份公开文件：

| 文件 | 用途 |
|---|---|
| `final_rows.csv` | 最近报告的验证终点 |
| `evaluation_rows.csv` | 已记录的检查点评估 |
| `attack_diagnostics.csv` | 已保存的攻击诊断 |
| `analysis.json` | 描述性汇总与证据范围 |
| `diagnostics.json` | 数据限制 |
| `analysis-validation.json` | 报告数值与交互次数的交叉检查 |

无需训练即可直接读取这些文件。它们不包含完整检查点，也不能证明独立研究确认已通过。

使用可选交互回放前：

```powershell
cd frontend
npm ci
npm run build
cd ..
```

打开 `visualization.ipynb` 并运行其各部分。笔记本读取已有文件，
不会导出新数据或重新生成图片；资源缺失时会提示。
随附的历史补充数据包含 2026-10-08 的七个验证运行，共 209 轮记录，
与较新的开发摘要分开保存。

`frontend/public/data/` 下的当前快照、`figure_assets/manifest.json`
和 `figure_assets/main/*.png` 不在 Git 中。完整图片展示需另行提供这些文件。
Windows 下，Node 和 PowerShell 支持由笔记本管理的预览，默认有效期为 30 分钟。
设置 `STOP_PREVIEW=True` 并运行最后一个单元格即可停止；其他环境使用已有的本地
`FRONTEND_URL`。

## 4. 执行已获批准的训练实验

**仅凭干净克隆，不能提交当前配置的训练实验。**
克隆中缺少 `workflow.experiment.graph_admission` 指向的审批记录。
先取得已授权、与源码匹配的执行包和预算批准；修改开关或复制旧审批并不能授权新运行。

1. 安装依赖并完成 Modal 认证。
2. 恢复固定版本的参考仓库，保留来源记录：

   ```powershell
   uv run --locked --extra comparisons python reference_algorithms/download.py
   ```

   此操作下载六个公开仓库，最多三个并发；不会训练模型或安装其中的脚本。
   来源与许可见[第三方归属](THIRD_PARTY.zh.md)。
3. 检查 `config/experiment.json`：获准的实验目录、Volume、源码版本、审批路径、
   种子、资源和超时设置必须与执行包一致。
4. 仅为该获准运行开启准备和提交。打开 `experiment.ipynb`，按
   **Load → Submit → Status** 顺序执行。新提交保持恢复开关关闭。
5. 原调用完成后，执行 **Status → Collect → Verify → Results**。
   按需开启刷新、下载和核验开关。等待状态不代表完成；不要为修复下载而再次提交。

当前 `native_acceptance.fresh_benchmark` 配置登记了七种方法，
训练种子为 `8101–8103`，验证种子为 `8201–8203`；
每个学习方法与训练种子组合执行 3,000 次策略交互。
检查点位于 `500, 1000, 1500, 2000, 2500, 3000`。
执行请求为一个 H100 容器、16 个 CPU 和 65,536 MiB 内存，
并明确设置单节点和整体超时。这是开发运行，官方测试集访问和正式确认均关闭。
交互次数预算并不能保证云费用上限。

运行记录保存在 `workflow.experiment.directory` 下；运行产物包含配置、指标、
摘要及适用的检查点。完整完成要求本次执行的传输记录为 `transport_complete=true`，
且本地核验成功。已取得的分片完整，不等于整个实验完整。
当前归档运行的传输和本地核验尚未完成，公开摘要保留了这一限制。

## 5. 恢复与排查

| 情况 | 操作 |
|---|---|
| 没有本地实验 | 关闭全部开关后查看，或取得已获批准的执行包 |
| 缺少参考源码 | 使用上述固定版本下载器，不要替换为上游最新版本 |
| 缺少图片或快照 | 提供展示补充包，不要推断缺失数据 |
| 原云调用仍在运行 | 使用同一实验目录读取状态 |
| 调用已完成，但下载未完成 | 跳过 Submit，针对同一已记录调用重复 Status 和 Collect |
| 导出受阻 | 若允许，下载已提交的分片；整个实验的核验仍保持待完成 |
| 安全暂停的执行需要继续 | 先核对记录，仅在审批有效时开启提交和恢复 |
| 消耗未知，或源码与配置不匹配 | 停止，保留记录，解决差异后再执行 |

## 6. 获得测试补充包后检查代码

测试目录按项目约定不纳入 Git。运行测试前先恢复匹配版本的测试包；
干净克隆无法复现测试结果。针对原生核心的检查命令：

```powershell
uv run --locked --extra comparisons pytest src/cloud_edge_rl/algorithms/test/test_native.py src/cloud_edge_rl/algorithms/test/test_federated.py src/cloud_edge_rl/algorithms/test/test_environment_resume.py src/cloud_edge_rl/algorithms/test/test_policy_snapshots.py
```

根据 `pyproject.toml`，直接运行 `pytest` 仅搜索 `tests/`，
不覆盖模块内的测试目录。记录测试对应的提交、完整命令和退出结果。
测试通过不代表算法具有性能优势。
