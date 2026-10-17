# 消融研究

[English](README.md) | [项目主页](../README.zh.md) | [复现说明](../docs/REPRODUCIBILITY.zh.md)

此独立入口用于 N1/N3 研究，使用项目环境并读取 `config/experiment.json`，不修改主实验。
研究动机与算法依据放在论文中。

## 前置条件

- 安装带 `comparisons` 扩展的项目环境。
- 提供匹配的 `ablations/test/` 测试材料；测试不包含在 Git 中。
- 提供已获批准的 `ablations/output/N-cloud-v1/` 执行材料，包括冻结源码、风险输入和审批记录。
- 提交前完成 Modal 认证并核对预算。

纯 Git 克隆缺少这些本地记录和测试，不能将其预检表述为成功执行。
准备新目录不会授予额外的执行或费用权限。

## 使用笔记本

使用**独立的项目 `.venv` 内核**打开 [experiment.ipynb](experiment.ipynb)。
运行前检查阶段：当前提交的笔记本设置为 `PHASE="continue"`，不是 `pilot`。

| 阶段 | 范围 |
|---|---|
| `pilot` | 首个配对种子；每组训练交互 500 次 |
| `continue` | 首个配对；每组追加 700 次，达到 1,200 次后验证 |
| `replicate` | 其余配对种子；须另行批准 |

依次运行 **Locate → Open plan → Preflight → Submit/Wait → Collect/Verify/Results**。
已有记录的调用会复用；只有核验通过的阶段记录才能证明完成。
重新打开笔记本时检查同一工作目录，不通过直接调用内部 worker 绕过审批。

在项目根目录查看本地准备状态（不是云端调用状态）：

```powershell
uv run --locked --extra comparisons python -m ablations.runner status --name N-cloud-v1
```

本地预检限时 90 秒；阶段等待上限分别为 4,800 / 9,000 / 27,000 秒，
节点计算限时 1,800 秒，下载限时两小时。
如需停止选定阶段，在笔记本中调用 `session.cancel(PHASE)`。
保留失败记录及未知消耗，不盲目重试。

## 输出与限制

记录位于 `ablations/output/N-cloud-v1/phases/<phase>/`，包括状态、下载产物和 `closure.json`。
笔记本展示已有的完成记录；这些产物不包含在 Git 中。
结果使用**预测风险**，不读取官方测试集。
其他 L/Q/E/W 研究仍为定义，尚未实现。
