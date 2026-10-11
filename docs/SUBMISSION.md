# Course Submission Checklist

[Project home](../README.md) | [Reproduction](REPRODUCIBILITY.md)

This checklist covers the course's code deliverable: organized, readable code,
clear run/reproduction instructions, and genuine version control. Mathematical
formulation, algorithm selection, tuning analysis, and discussion belong in the
report. Project-specific performance targets are not instructor grading rules.

## Git submission

- [ ] Submit the intended commit/branch with its real Git history.
- [ ] Include `src/`, entry notebooks, `config/experiment.json`, `.python-version`,
  `pyproject.toml`, `uv.lock`, and the tracked reference-source records.
- [ ] Include the English README, optional Chinese entry, these submission
  documents, module guides, and the updated `.gitignore`.
- [ ] Keep the six published CSV/JSON result files and historical playback JSON
  consistent with the report's source IDs and evidence scope.
- [ ] Check the public links and install/run instructions from the delivered files,
  not from an author-only local folder.
- [ ] Exclude credentials, virtual environments, caches, and downloaded upstream
  repositories. Retain third-party license and attribution records.

## Separate reproduction supplements

Do not describe Git alone as a complete reproduction package. If the handoff
claims the corresponding capability, include these additional materials through
the agreed course delivery channel:

| Capability | Required supplement |
|---|---|
| Reproduce test results | Matching ignored test directories and a report with commit, command, and exit result |
| Show the current figure gallery and snapshot | `figure_assets/main/`, its manifest, and `frontend/public/data/` |
| Verify complete saved experiments | Matching controller records, full run artifacts/checkpoints, and transfer/verification receipts |
| Retrain on Modal | Authorized execution package, source-matched approval, account setup, and stated resource/cost expectations |

Before supplying artifacts, review them for credentials, private records, and
redistribution restrictions. Preserve relative directory structure. Label a
missing or partially delivered capability explicitly; do not invent success.

## Final handoff checks

- [ ] Demonstrate at least one documented path available to the grader: saved
  tables, recorded playback, tests with the supplement, or an approved run.
- [ ] State that current published tables are development summaries and full local
  verification/independent research confirmation are incomplete.
- [ ] Distinguish predicted risk from audited risk, simulated resources from
  measured cloud cost, and executed code from demonstrated performance.
- [ ] Deliver the written report separately: 8–20 pages excluding references,
  1.5 line spacing, Arial 12, one-inch margins. The presentation is 15 minutes.
  Check the final report itself; the manuscript skeleton is not a finished report.

<details>
<summary>中文说明（可选）</summary>

代码交付重点是结构清晰、易读、运行与复现说明明确，以及真实 Git 历史。
算法和理论依据写在论文中，不需要在工程说明重复。提交本次文档及忽略规则时，
不要顺带公开本地研究笔记、凭据、测试或大型产物。

测试、完整图片、完整实验核验或重训如属于交付承诺，应另行提供对应补充包；
缺少的部分明确标注。论文与 15 分钟展示单独提交，不能把论文骨架当成成稿。

</details>
