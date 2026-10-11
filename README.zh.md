# 云边联邦训练的风险感知强化学习项目

[English（默认）](README.md) | [中文](README.zh.md)

本课程项目在 UCI Adult 数据集上实现模拟联邦训练的强化学习调度，
包含原生 DQN/PPO、参考算法适配、实验恢复及只读结果展示。
本文只说明代码使用与交付。

## 开始使用

使用 **Python 3.12.14**、**uv 0.12.11**，在项目根目录执行：

```powershell
uv sync --locked --extra comparisons
uv run --locked --extra comparisons jupyter lab
```

**打开 `experiment.ipynb` 后，不要直接 Run All。**
当前提交的配置已开启准备、提交、云状态刷新、下载和核验，仅恢复开关关闭。
只查看本地内容时，应先将 `workflow.experiment` 的六个操作开关全部设为 `false`。
具体步骤见 [复现说明](docs/REPRODUCIBILITY.zh.md)。

| 目的 | 入口 |
|---|---|
| 查看已保存的开发结果表 | [结果目录](output/fresh-seven-method-rerun-20261010-v1/report-analysis-20261010-v1/) |
| 浏览已有图片与历史回放 | [visualization.ipynb](visualization.ipynb) |
| 执行已获批准的实验、查询或下载 | [experiment.ipynb](experiment.ipynb) |
| 使用可选 N1/N3 消融 | [消融说明](ablations/README.md) |
| 安装与复现结果 | [复现说明](docs/REPRODUCIBILITY.zh.md) |

## 文件位置

- `src/cloud_edge_rl/algorithms/`：原生训练、攻击、风险估计与 DQN/PPO。
- `src/cloud_edge_rl/`：环境、指标、实验执行、恢复和存储。
- `src/cloud_edge_rl/comparisons/`：明确选择的第三方适配器。
- `config/experiment.json`：唯一可编辑实验配置。
- `pyproject.toml`、`uv.lock`：环境要求与依赖锁定。
- `reference_algorithms/`：固定版本来源、下载器和参考运行环境。
- [frontend/](frontend/README.md)、[figure_assets/](figure_assets/)：交互展示与制图工具。

## 复现范围与限制

- Windows 使用 CPU PyTorch，Linux 使用锁定的 CUDA 版本。
  当前训练配置需要 Modal 账户认证和有效执行审批。
- Git 包含六份开发结果 CSV/JSON 和历史回放 JSON；不包含全部检查点、
  原始运行、测试文件、当前展示快照和 PNG。额外材料见复现说明。
- 公开摘要仅为开发结果，完整本地产物核验和独立研究确认尚未完成。
- 预测风险不是真实审计风险；时间和能量为模拟量，未知云费用不记零。

来源和许可见 [第三方归属](docs/THIRD_PARTY.zh.md)。保留真实 Git 历史；
老师未明确要求时，无需额外导出独立历史文件。
