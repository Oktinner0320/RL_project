# Results visualization and publication assets

This isolated presentation layer reads existing results. It never starts an
experiment, changes scientific configuration, or modifies `src/`, `config/`,
`ablations/`, original notebooks, dependencies, ledgers or historical artifacts.
The central notebook is **`../visualization.ipynb` at the project root**.

## Quick start

Use the project's existing Python environment. Matplotlib, pandas, psutil and
IPython are required; no changes to the project's dependency files are made.

```powershell
cd frontend
npm ci
cd ..
# Open visualization.ipynb using the project Python kernel and run its sections.
```

The notebook reuses the current snapshot, renders the figures, builds the Vite
frontend, and embeds a loopback-only preview with a 30-minute lease. Dependencies
are not installed automatically. Set `REFRESH_DATA=True` to export updated
existing results; set `STOP_PREVIEW=True` in its final cell to stop the preview.

Equivalent bounded notebook APIs:

```python
from figure_assets.scripts.notebook_tools import (
    export_data, generate_figures, build_frontend, start_preview, stop_preview,
)
export_data(timeout_seconds=60)
generate_figures(width="double", timeout_seconds=90)
build_frontend(timeout_seconds=90)
start_preview(lease_seconds=1800)
# stop_preview()  # Stop only this receipt-bound local process.
```

## Files and shared data

- `DATA_CONTRACT.md`: the shared presentation schema, provenance and evidence rules.
- `scripts/export_results.py`: bounded, allowlisted, read-only result adapters.
- `scripts/render_figures.py`: evidence-aware Matplotlib figure suite.
- `scripts/notebook_tools.py`: bounded stages, disk-state restoration and leased server.
- `styles/acm.mplstyle`: publication typography and visual defaults.
- `main/`: the single image asset directory; PNG only, at 600 DPI for the three
  compound ablation figures and 320 DPI for the other families.
- `manifest.json`: current image index, captions, descriptions and skipped-family reasons.
- `reports/figure-data/`: provenance JSON, kept outside the image directory.
- `reports/`: export, render, subprocess and preview receipts; not scientific evidence.
- `../frontend/public/data/`: immutable public snapshots, atomically published index.

Generated assets, reports, caches and local tests are ignored. The exporter uses
literal IDs and content/field comparisons, never file hashes. Original inputs
are not copied wholesale; weights, raw gradients, private feature records,
sample identities and credentials are excluded.

## Paper figures

`width="single"` targets approximately 85 mm, and `"double"` approximately
178 mm. Actual manuscript `\columnwidth` / `\textwidth` takes precedence.
Only high-resolution PNG is produced. Re-rendering updates `main/`, not a new snapshot directory. The
white-background learned-method palette is shared with the web UI; paper baselines
are subdued gray. Markers and line styles supplement color. Assets carry source run IDs and their
development/formal scope. Include each manifest caption and textual description
with its figure; adapt the latter to ACM's `\Description{...}`.

Supported families include system/data-flow overview, learning curves, method
comparison, resource–risk tradeoffs, proxy reliability, calibration, ablation and
recorded-action/coverage analysis. Missing evidence yields an explicit skip,
not a synthetic figure. One training seed cannot justify confidence intervals;
predicted risk cannot justify claims about audited privacy risk. Displayed
baselines are exploratory unless independently frozen by an authoritative report.

### Compound n-step ablation figures

- `main/paired_ablation_1.png`: all recorded validation rounds for an exact
  N1/N3 training/evaluation seed, matched training budget and episode. Accuracy
  and exposure-weighted **predicted** risk sit above time/energy endpoint
  dumbbells. Warmup, actual exposure denominators and signed endpoint changes
  remain recorded; no best-round selection or confidence interval is inferred.
- `main/decision_atlas_1.png`: matched N1/N3 columns align client participation
  events, discrete batch ribbons and `max Q − Q(action)` heatmaps on actual
  rounds. Client sets and batch sizes come from the declared action space;
  white rings mark actual actions, not an assumed argmax. Missing or incomplete
  Q vectors stay gray, never zero. Heuristic/probability vectors are not merged.
- `main/training_diagnostics_1.png`: compatible N1/N3 histories share cumulative
  interaction axes. Copied prefixes are deduplicated by step/update identity,
  preserving all source run IDs. Faint raw traces remain behind a declared
  centered 11-record moving mean; missing segments are not bridged. The 500-step
  continuation boundary is marked. TD loss and Q magnitude do not rank policy
  performance across different n-step targets.

These use blue/vermillion plus solid/dashed lines, quiet grids, semantic labels
and a shared Q-gap scale. Both single- and double-column layouts are supported;
double column is recommended for the six-panel decision atlas. Their provenance
JSON remains under `reports/figure-data/`, outside `main/`. Re-rendering removes
only superseded PNGs named in the previous manifest; unrelated files are kept.

Design references: [Rougier et al., Ten Simple Rules for Better Figures](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833),
[Wilke, Multi-panel figures](https://clauswilke.com/dataviz/multi-panel-figures.html),
[Crameri et al., The misuse of colour in science communication](https://www.nature.com/articles/s41467-020-19160-7),
and [Weissgerber et al., Beyond Bar and Line Graphs](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002128).

## Preview safety and recovery

Only built UI assets, public snapshots and generated figures are served. There
is no general project-root file server, directory listing, cloud/training API or
execution callback. Requests cannot traverse directories or follow symlinks.
The preview binds to `127.0.0.1`, expires after at most one hour, and has its own
process-tree watchdog. A notebook/kernel restart does not create a duplicate
server or extend its existing lease. Stop uses UUID, root and PID creation-time
identity, not a broad process name or port-based kill.

Python export, rendering and build stages have explicit deadlines and durable
logs under `reports/stages/`. Failed stages remain visible. A visual-layer
success is not experiment success or first-acceptance-v1 passage.
