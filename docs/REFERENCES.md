# 参考来源与复用边界

检索日期：**2026-10-02**。以下仓库用于理解研究协议和指导用户实现；当前框架没有复制这些项目的第三方算法代码，也没有声称已复现它们的结果。尚未为第三方仓库固定实验 commit，因此不提供虚构的 commit hash。实际复用时应记录准确 revision、引用论文并遵守许可证。

## 研究与开源代码

| 来源 | 许可证 | 本项目参考内容 | 使用边界 |
|---|---|---|---|
| [scaleoutsystems/tabular-GIA](https://github.com/scaleoutsystems/tabular-GIA) | Apache-2.0 | 表格梯度反演的威胁模型、预处理、指标与实验协议 | 研究参考，算法由用户实现；不直接把旧工程环境当成本项目依赖 |
| [eth-sri/tableak](https://github.com/eth-sri/tableak) | MIT；[LICENSE.txt](https://github.com/eth-sri/tableak/blob/main/LICENSE.txt) | TabLeak 的表格反演、类别表示与重构不确定性思路 | 原项目环境和运行假设与本项目不同，不能把旧 Conda/CPU 流程直接视为 UV/Modal 验证 |
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | MIT | 简洁 DQN/PPO 实现和训练循环组织 | 用户学习参考，不将整库列为依赖，不替用户复制核心实现 |
| [mit-han-lab/dlg](https://github.com/mit-han-lab/dlg) | MIT | Deep Leakage from Gradients 的 PyTorch 梯度匹配原理 | 原始图像示例不等同于完成表格反演，需自行实现本项目字段约束与评价 |

核心研究材料对应的 [arXiv 论文 v1](https://arxiv.org/abs/2606.00986v1) 在代码可用性说明中链接 `tabular-GIA`。研究设计仍以用户提供的《云边联邦训练强化学习_研究总结与课程要求.pdf》为背景；课程硬性约束来自用户提供的 `final_project.pdf`，两者分开引用和解释。

旧项目的 Python、Gym/Gymnasium、PyTorch 与平台依赖不作为本项目兼容性证据。本项目安装以 `pyproject.toml` 和 `uv.lock` 为唯一依赖事实，不在参考文档中维持另一组版本配置。

## 数据

- [UCI Adult 数据集](https://archive.ics.uci.edu/dataset/2/adult)，DOI：[10.24432/C5XW20](https://doi.org/10.24432/C5XW20)，许可证 **CC BY 4.0**。
- 使用官方 Adult 训练/测试文件及其原始字段语义；下载地址由 Notebook 配置。每次准备保存文件散列、来源地址和划分元信息。
- 报告和后续公开产物应注明数据来源及使用的许可；不要将本项目训练/验证划分误称为官方完整划分。

## 官方工程文档

- [UV 项目工作流](https://docs.astral.sh/uv/guides/projects/)：依赖、锁文件和项目运行。
- [Gymnasium 自定义环境](https://gymnasium.farama.org/introduction/create_custom_env/)：环境接口、空间、终止与截断。
- [PyTorch 文档](https://docs.pytorch.org/docs/stable/index.html)：张量、自动微分、训练和状态保存。
- [Modal 指南](https://modal.com/docs/guide)：认证、云函数、GPU、镜像和 Volume。

官方文档用于核对工程接口；框架测试和实际云任务记录用于证明本项目行为。链接存在或上游宣称支持某功能，都不能替代本项目的实际验收。
