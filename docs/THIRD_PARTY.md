# Third-Party Sources and Attribution

[Project home](../README.md) | [中文入口](../README.zh.md)

Source download steps are in [Installation and Reproduction](REPRODUCIBILITY.md).

Pinned versions and archive checksums are in
[catalog.json](../reference_algorithms/catalog.json) and
[provenance/](../reference_algorithms/provenance/). Downloaded repositories remain
separate from the main package and are not submitted wholesale in Git.

| Source | Included use | Recorded license |
|---|---|---|
| [tabular-GIA](https://github.com/scaleoutsystems/tabular-GIA) | Model and attack adapters | Apache-2.0 |
| [TabM](https://github.com/yandex-research/tabm) | Risk-estimation adapter | Apache-2.0 |
| [PQN](https://github.com/mttga/purejaxql) | Attributed PyTorch port | Apache-2.0 |
| [MetaCS-FL](https://github.com/alan-lira/metacs-fl) | Scheduling adapter, not the full upstream LNS implementation | CeCILL-C 1.0 |
| [ConformalOpt](https://github.com/chrismohrii/conformalopt) | Separate research comparison | MIT declared in `setup.py`; downloaded revision has no license text |
| [OCP](https://github.com/trustml-lab/ocp-semi-bandit) | Separate research comparison | No license found in the downloaded revision |

The PQN port ships its [license](../src/cloud_edge_rl/comparisons/licenses/PQN-LICENSE.txt)
and [NOTICE](../src/cloud_edge_rl/comparisons/licenses/NOTICE.txt). Keep both when
redistributing the port. Resolve ConformalOpt/OCP permissions before bundling or
redistributing their sources; do not label them freely redistributable.

Python dependencies are recorded in [pyproject.toml](../pyproject.toml) and
`uv.lock`; frontend dependencies in [package.json](../frontend/package.json) and
`package-lock.json`. Preserve applicable upstream notices. Dataset acquisition
uses the public [UCI Adult source](https://archive.ics.uci.edu/dataset/2/adult).
Template and bibliography references belong in the written report.

<details>
<summary>中文说明（可选）</summary>

固定版本和下载摘要保存在参考目录，下载的仓库不整体提交。PQN 移植必须保留许可和
NOTICE；ConformalOpt/OCP 的再发布权限仍需确认。适配器并不等同于完整复现原论文。

</details>
