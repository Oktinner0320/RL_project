# Presentation Data Format — v1

[Viewing and generation](README.md) | [Frontend](../frontend/README.md)

Presentation records are read-only projections of saved results, not experiment
configuration or a new research evaluator.

## Files

| File | Required content |
|---|---|
| `frontend/public/data/index.json` | `schema_version="results_index_v1"`, `current_snapshot`, and `snapshots` with `snapshot_id`, `generated_at`, `path` |
| Snapshot `results.json` | `schema_version="results_visualization_v1"`, `snapshot_id`, `generated_at`, `experiments`, `runs`, `audits`, `calibration`, `acceptance_reports`, `warnings`, `metric_definitions` |
| `figure_assets/manifest.json` | `schema_version="figure_manifest_v1"`, snapshot identity, figure captions/descriptions/source IDs/file paths, and skipped-family reasons |

Index paths are confined relative URLs. Figure paths are relative to
`figure_assets/`: images under `main/`, source records under `reports/figure-data/`.
The notebook reads images separately; the web app needs the snapshot index but
does not fetch the figure manifest.

## Record identifiers and fields

- Experiments retain `experiment_id`, evidence scope, sources, separate compute /
  transport / verification / acceptance status, interaction counts, and nullable
  actual cloud cost.
- Runs retain `run_id`, experiment/method/arm, phase/stage, training and evaluation
  seeds, partition, risk version/measurement, budget, summary, safe configuration,
  metrics, training rows, decisions, uploads, and warnings.
- Sources use project-relative `path`, optionally `record` or `line`, never private
  absolute paths.
- Metrics keep original keys and units: `validation_accuracy` is a fraction,
  `elapsed_seconds` simulated seconds, and `energy_joules` simulated joules.
  Training rows keep `environment_steps`; decisions keep `episode`, `round`,
  `kind`, and saved `values`. Do not relabel Q values as probabilities.
- Action maps use `action_id`, `client_ids`, and `batch_size`. Unknown maps stay
  unavailable. Uploads retain only safe IDs and recorded display metadata.
- Audit records match exact run/upload identities and retain model and scope;
  calibration records retain model/fit/panel identities and their sources.

## Data handling rules

1. Missing values remain `null` or explicitly unavailable; JSON contains no
   nonfinite numbers. Missing risk or cost is not zero.
2. Keep predicted risk separate from audited risk and partial audit coverage
   separate from whole-policy claims. Display recorded verdicts only.
3. Reject contradictory copies. Deduplicate by run and recorded row identity;
   resumed logs can contain earlier history. Do not add copied prefixes twice.
4. Compare only compatible splits, stages, risk versions, and budgets. A single
   seed's rounds do not provide independent statistical replicates.
5. Exclude credentials, raw gradients, weights, private features, sample IDs,
   packet paths, and trusted reference records from public projections.
6. Historical replay remains a separately dated supplement. It must not replace
   newer records or imply current-batch results.

<details>
<summary>中文说明（可选）</summary>

保留来源、标识、单位和缺失值；预测风险与真实审计风险分开。复制记录和恢复日志
不能重复计数。网页不读取图片清单，图片由笔记本单独展示；历史回放不替代当前结果。

</details>
