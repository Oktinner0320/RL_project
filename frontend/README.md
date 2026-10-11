# Cloud–Edge Lab

[Project home](../README.md) | [Visualization guide](../figure_assets/README.md)

Read-only TypeScript/Vite viewer for saved experiment snapshots and recorded
playback. It does not train, contact Modal, modify scientific results, or calculate
research acceptance. Interface text is English.

## Build and open

Use Node **22.12+** or **24+** with npm. From `frontend/`:

```powershell
npm ci
npm run typecheck
npm run build
```

Open root `visualization.ipynb` with the project Python kernel and run its preview
section, or serve the build through an existing local preview. The notebook does
not install dependencies or build the frontend. The build output is `dist/`.
Rebuild after updating public data.

## Required data

- `public/data/index.json` and its listed `results.json` snapshots are mandatory
  for the web app. They are generated files, excluded from Git; restore them from
  the presentation package or export available sources using the
  [figure tools](../figure_assets/README.md).
- `public/replays/native-validation-v1.json` is a tracked historical supplement:
  seven 2026-10-08 validation runs and 209 recorded rounds. It is **not** a
  replacement for the snapshot index and is not a newer-batch result.
- No snapshot: the app shows **Data unavailable**. A missing replay supplement
  leaves an otherwise valid snapshot usable.

The single screen offers run/episode selection, round playback, synchronized
curves, and an inspector for recorded actions and saved scores. The image gallery
is in the notebook, not the web app. Motion illustrates recorded cooperation;
its cadence, particles, and paths are not measured timing or live device status.

## Checks and regeneration

`npm run typecheck` and `npm run build` use tracked source. `npm run test` requires
the matching ignored `frontend/test/` supplement; a clean clone has no tests.

`node scripts/prepare-replays.mjs` regenerates the historical supplement only
when its fixed original run files are available locally. It does not download
them. Keep source identities and rebuild afterwards; do not substitute newer
runs without updating the documented scope.

Missing states, action mappings, costs, scores, and audits remain missing.
Predicted risk is not audited risk; summary-only records do not become round
traces. Playback does not establish fair cross-batch comparisons or acceptance.
The shared schema is documented in [DATA_CONTRACT.md](../figure_assets/DATA_CONTRACT.md).

<details>
<summary>中文说明（可选）</summary>

先安装前端依赖并构建，再通过根目录可视化笔记本打开预览。网页必须有快照索引，
历史回放文件不能单独替代索引。图片画廊在笔记本中；动画不表示真实网络耗时。
前端测试需要额外提供被忽略的测试目录。

</details>
