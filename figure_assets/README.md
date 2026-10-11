# Figures and Saved-Result Viewing

[Project home](../README.md) | [Data format](DATA_CONTRACT.md)

Root [visualization.ipynb](../visualization.ipynb) is the unified viewing entry.
It directly reads existing images and JSON and can embed the built frontend.
It does **not** install dependencies, export results, render figures, or build
the frontend automatically. It does not start training or contact Modal.

## View existing resources

Install the project environment as described in
[Reproducibility](../docs/REPRODUCIBILITY.md), then build the optional web viewer:

```powershell
cd frontend
npm ci
npm run build
cd ..
```

Open `visualization.ipynb` using the project `.venv` kernel. Its sections load
the snapshot, embed Cloud–Edge Lab, and browse `figure_assets/main/` through
`manifest.json`. Images are read from disk, not regenerated.

- Complete gallery: supply `manifest.json` and the corresponding `main/*.png`.
- Current web snapshot: supply `frontend/public/data/` and rebuild the frontend.
- Git alone: includes tools, guides, summary tables, and historical replay JSON,
  but not those generated images or snapshots. Missing resources are reported.
- Windows automatic preview: loopback-only, default 30-minute lease, Node and
  PowerShell required. Set `STOP_PREVIEW=True` and run the last cell to stop it.
  Kernel restart may also stop its child processes; reopen the preview if needed.
  Alternatively set `FRONTEND_URL` to an existing local preview.

## Generate presentation outputs, when needed

These optional commands read available local sources and write presentation
files only. Run them from the repository root; they do not obtain missing cloud
artifacts. Default export sources include the N1/N3 study and seven-method report.

```powershell
uv run --locked --extra comparisons python figure_assets/scripts/export_results.py --root .
uv run --locked --extra comparisons python figure_assets/scripts/render_figures.py --root . --width double
```

Export writes snapshots and an index under `frontend/public/data/`.
Rendering writes PNGs under `figure_assets/main/`, an image manifest, and source
records under `figure_assets/reports/figure-data/`. Rebuild the frontend after
exporting. Rendering updates the current image set; preserve any needed previous
presentation package before rerendering.

Use `--width single` for narrower figures. Missing evidence produces a recorded
skip, not invented plots. The separate Conceptual mechanism figure is not
generated here. Captions and descriptions in the manifest accompany the figures;
the interpretation and method justification belong in the report.

Generated files, reports, caches, and local tests remain ignored. Raw gradients,
private feature records, sample identities, credentials, and weights are not
presentation exports. Display success is not experiment verification or proof
of improved privacy.

<details>
<summary>中文说明（可选）</summary>

根目录可视化笔记本只读已有图片和 JSON，不自动导出、制图或构建网页。
需要完整展示时，另行提供图片、清单和前端快照；缺失内容会明确提示。
可选生成命令仅处理已有本地结果，不下载云产物，不执行训练。

</details>
