# 第三方来源与归属

[项目主页](../README.zh.md) | [English](THIRD_PARTY.md)

源码下载步骤见[安装与复现](REPRODUCIBILITY.zh.md)。

固定版本和下载归档的校验值见
[catalog.json](../reference_algorithms/catalog.json) 和
[provenance/](../reference_algorithms/provenance/)。下载的仓库与主程序包分开管理，
不在 Git 中整体提交。

| 来源 | 项目中的用途 | 记录的许可信息 |
|---|---|---|
| [tabular-GIA](https://github.com/scaleoutsystems/tabular-GIA) | 模型与攻击适配器 | Apache-2.0 |
| [TabM](https://github.com/yandex-research/tabm) | 风险估计适配器 | Apache-2.0 |
| [PQN](https://github.com/mttga/purejaxql) | 已标注来源的 PyTorch 移植 | Apache-2.0 |
| [MetaCS-FL](https://github.com/alan-lira/metacs-fl) | 调度适配器，并非完整的上游 LNS 实现 | CeCILL-C 1.0 |
| [ConformalOpt](https://github.com/chrismohrii/conformalopt) | 独立研究对照 | `setup.py` 声明为 MIT；下载版本未附许可文本 |
| [OCP](https://github.com/trustml-lab/ocp-semi-bandit) | 独立研究对照 | 下载版本中未找到许可文本 |

PQN 移植随附[许可文本](../src/cloud_edge_rl/comparisons/licenses/PQN-LICENSE.txt)
和 [NOTICE](../src/cloud_edge_rl/comparisons/licenses/NOTICE.txt)。重新分发移植代码时须保留二者。
打包或重新分发 ConformalOpt/OCP 源码前，需确认相关许可；不要将其标为可自由再分发。

Python 依赖记录在 [pyproject.toml](../pyproject.toml) 和 `uv.lock` 中；
前端依赖记录在 [package.json](../frontend/package.json) 和 `package-lock.json` 中。
保留适用的上游声明。数据集取自公开的
[UCI Adult 来源](https://archive.ics.uci.edu/dataset/2/adult)。
模板和参考文献来源在论文中注明。
