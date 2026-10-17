# Installation and Reproduction

[Project home](../README.md) | [中文](REPRODUCIBILITY.zh.md)

Run commands from the repository root unless a different directory is shown.
Viewing saved results, running tests, and retraining have different prerequisites.

## 1. Install the locked environment

- Python **3.12.14**; the package requires Python 3.12.
- uv **0.12.11**, installed using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).
- Windows or Linux; Windows uses CPU PyTorch and Linux the locked CUDA build.
- For the optional viewer: Node **22.12+** or **24+**, with npm.
- For cloud execution: an authenticated [Modal account](https://modal.com/docs/guide).

```powershell
uv sync --locked --extra comparisons
uv run --locked --extra comparisons jupyter lab
```

Select the project `.venv` kernel. Keep `uv.lock`; do not replace pinned versions
with an unconstrained installation.

## 2. Inspect without cloud execution

Before running `experiment.ipynb`, set these existing keys under
`workflow.experiment` in `config/experiment.json` to `false`:

```json
{
  "prepare_enabled": false,
  "submission_enabled": false,
  "resume_enabled": false,
  "refresh_enabled": false,
  "collection_enabled": false,
  "verification_enabled": false
}
```

Change only these six values, not the entire configuration object. The checked-in
configuration currently enables five of them; it is **not** a read-only default.
With all six disabled, the notebook reads existing local state or reports
`not_prepared`. It does not create an experiment or start a cloud job.

## 3. View delivered results

The [development result directory](../output/fresh-seven-method-rerun-20261010-v1/report-analysis-20261010-v1/)
contains six published files:

| File | Use |
|---|---|
| `final_rows.csv` | Latest reported validation endpoints |
| `evaluation_rows.csv` | Recorded checkpoint evaluations |
| `attack_diagnostics.csv` | Saved attack diagnostics |
| `analysis.json` | Descriptive summary and evidence scope |
| `diagnostics.json` | Data limitations |
| `analysis-validation.json` | Cross-checks of reported numbers and interaction counts |

These files can be read directly without training. They are not a complete set
of checkpoints or proof of independent research confirmation.

For optional interactive playback:

```powershell
cd frontend
npm ci
npm run build
cd ..
```

Open `visualization.ipynb` and run its sections. The notebook reads existing files;
it does not export new data or regenerate figures. It reports missing resources.
The shipped historical supplement contains seven 2026-10-08 validation runs
(209 recorded rounds), separate from the newer development summaries.

Current snapshots under `frontend/public/data/`, `figure_assets/manifest.json`,
and `figure_assets/main/*.png` are not included in Git. Provide them separately
for the full gallery. On Windows, Node and PowerShell enable a notebook-owned
preview with a default 30-minute lease. Set `STOP_PREVIEW=True` and run the last
cell to stop it; on other setups, use an existing local `FRONTEND_URL`.

## 4. Run an approved training experiment

**A clean clone alone cannot submit the configured training experiment.**
It lacks the approval record referenced by `workflow.experiment.graph_admission`.
Obtain an authorized, source-matched execution package and budget approval first;
changing switches or copying an old approval does not authorize a new run.

1. Install dependencies and authenticate Modal.
2. Restore the pinned reference repositories, retaining their provenance records:

   ```powershell
   uv run --locked --extra comparisons python reference_algorithms/download.py
   ```

   This downloads six public repositories, at most three concurrently; it does
   not train models or install their scripts. See [attribution](THIRD_PARTY.md).
3. Check `config/experiment.json`: the approved directory, Volume, source version,
   approval path, seeds, resources, and timeouts must match the execution package.
4. Enable preparation and submission only for that approved run. Open
   `experiment.ipynb`; run **Load → Submit → Status** in order. Keep resume off
   for a new submission.
5. After the original call completes, run **Status → Collect → Verify → Results**.
   Enable refresh, collection, and verification for these operations as needed.
   Pending states are not completion; do not submit again to fix a download.

The current `native_acceptance.fresh_benchmark` configuration registers seven
methods, training seeds `8101–8103`, validation seeds `8201–8203`, and 3,000
policy interactions per learned method/seed. Checkpoints are at
`500, 1000, 1500, 2000, 2500, 3000`. Execution requests one H100 container,
16 CPUs and 65,536 MiB memory, with explicit node and overall timeouts.
This is a development run; official-test access and formal confirmation are off.
Cloud charges are not guaranteed by an interaction budget.

Expected records live under `workflow.experiment.directory`; run artifacts
contain configuration, metrics, summaries, and applicable checkpoints.
Full completion requires a current-attempt `transport_complete=true` transfer
and successful local verification. A complete available shard is not a complete
experiment. The current archived run has incomplete transfer/local verification;
the published summaries retain that limitation.

## 5. Resume or troubleshoot

| Situation | Action |
|---|---|
| No local experiment | Inspect with all switches off, or obtain an approved execution package |
| Missing reference source | Use the pinned downloader above; do not substitute the latest upstream revision |
| Missing images or snapshot | Supply the presentation package; do not infer missing data |
| Original cloud call still running | Read status using the same experiment directory |
| Completed call, incomplete download | Skip Submit; repeat Status and Collect for the same recorded call |
| Export blocked | Collect already committed shards when offered; keep whole-experiment verification pending |
| Safe paused execution requiring continuation | Reconcile records first; enable submission and resume only with valid approval |
| Unknown consumption or source/config mismatch | Stop; preserve the records and resolve the mismatch before further execution |

## 6. Check code when the test supplement is available

Test directories are intentionally excluded from Git. Restore the matching test
package before running tests; a clean clone cannot reproduce test results.
For a focused native-core check:

```powershell
uv run --locked --extra comparisons pytest src/cloud_edge_rl/algorithms/test/test_native.py src/cloud_edge_rl/algorithms/test/test_federated.py src/cloud_edge_rl/algorithms/test/test_environment_resume.py src/cloud_edge_rl/algorithms/test/test_policy_snapshots.py
```

Plain `pytest` searches only `tests/` according to `pyproject.toml`; it does not
cover the module-local test directories. Record the tested commit, exact command,
and exit result. Passing tests is not evidence of algorithm superiority.
