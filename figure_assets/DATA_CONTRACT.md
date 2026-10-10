# Results visualization contract — version 1

This is a read-only presentation contract, not an experiment configuration or a new acceptance engine. Only `frontend/`, `figure_assets/`, and root `visualization.ipynb` may be created or changed by this work. Original scientific artifacts are never rewritten. No file hashes are calculated or compared.

## Shared snapshot

`frontend/public/data/index.json` has `schema_version: "results_index_v1"`, `current_snapshot: string`, and `snapshots: [{snapshot_id, generated_at, path}]`. `path` is a confined relative URL such as `<snapshot_id>/results.json`. Export a complete snapshot first, then atomically update the index. IDs are timestamps plus a random suffix, not content identities.

`results.json` contains:

- `schema_version: "results_visualization_v1"`, `snapshot_id: string`, `generated_at: string`.
- `experiments: Experiment[]`, `runs: Run[]`, `audits: Audit[]`, `calibration: Calibration[]`, `acceptance_reports: object[]`.
- `warnings: string[]`, `metric_definitions: object` (units, weighting and scope).

`Experiment` has `experiment_id`, `name`, `kind`, `evidence_scope`, `sources: Source[]`, `status: {compute, transport, verification, acceptance}`, `actual_cloud_cost_usd: number | null`, `counts: {physical_train_interactions: number | null, physical_validation_interactions: number | null, training_seed_count: number}`, `evidence_gaps: string[]`.

`Run` has `run_id`, `experiment_id`, `method`, `arm: string | null`, `phase`, `stage`, `train_seed: number | null`, `eval_seed: number | null`, `partition`, `risk_measurement`, `risk_version: string | null`, `evidence_scope`, `status: {compute, transport, verification, acceptance}`, `budget: {new_interactions: number | null, cumulative_interactions: number | null, warmup_interactions: number | null}`, `summary: object`, `config_summary: object`, `metrics: object[]`, `training: object[]`, `decisions: object[]`, `uploads: object[]`, `sources: Source[]`, `warnings: string[]`.

`Source` has only `path: string` (project-relative, never private absolute paths), optional `record: number` and optional `line: number`.

Use original metric keys: `validation_accuracy` (fraction), `validation_loss`, `elapsed_seconds` (simulated s), `energy_joules` (simulated J), `predicted_exposure_risk`, `risk_exposures`, `action_id`, `round`, `episode`, `is_warmup`, `reward`, `terminated`, `truncated`, `c0_participations`, etc. Empty CSV values are null, booleans stay booleans, and nonfinite values cannot enter JSON. Each row has `source_index` if useful for provenance. Training rows retain `environment_steps`, `td_loss`, `q_mean`, `updates`. Decision rows retain `episode`, `round`, `kind`, `values`, `margin`. Uploads expose only IDs, episode/round, action/client/batch, prediction/uncertainty, warmup and decision metadata; exclude features, packet/reference paths, sample identities and payloads.

`config_summary` includes safe display fields: `dataset`, `partition`, `num_clients`, `gamma`, `n_step`, `warmup_rounds`, `max_rounds`, and `action_space: [{action_id, client_ids: number[], batch_size: number}]`. An unknown action mapping must remain unavailable, not guessed.

`Audit` has `run_id`, `upload_id`, `trajectory_id`, `model_id: string | null`, `fit_seed: number | null`, `panel_seed: number | null`, `prediction: number | null`, `raw_prediction: number | null`, `calibrated_prediction: number | null`, `truth: number | null`, `batch_size: number | null`, `is_warmup: boolean | null`, `scope`, `source: Source`. Match labels by exact documented IDs. Partial labels are not a complete-policy risk.

`Calibration` has `group_id`, `fit_seed: number | null`, `model_id: string | null`, `scope`, `raw_u: number | null`, `cal_u: number | null`, `raw_ewmae: number | null`, `cal_ewmae: number | null`, `panel_ids: string[]`, `sources: Source[]`. Never mix different models or developer panels to imply formal passage.

## Rendering outputs

`figure_assets/generated/<snapshot_id>/manifest.json` contains `schema_version: "figure_manifest_v1"`, `snapshot_id`, `generated_at`, `figures: [{id, title, scope, caption, description, files: {pdf, svg, png, data}, source_run_ids: string[]}]`, `skipped: [{id, reason}]`. All file paths are relative to that generated directory and must be confined. Figure outputs remain under `figure_assets/`, including figure source JSON and English caption/description text. The frontend reads this manifest and files through the local read-only `/figure-assets/` endpoint, never through a project-root file server.

## Evidence rules

- Distinguish compute, transport, semantic verification and algorithm acceptance. Existing reports are displayed; never recompute an official acceptance verdict in TypeScript.
- No missing-cost or missing-risk zero imputation. Predicted risk and audited risk are different measures.
- Deduplicate copied runs by `run_id` plus content comparison of safe actual results; reject contradictory copies. A resumed training log may contain its predecessor's history; use declared cumulative interaction counters and exact row identity, not blind concatenation.
- First supported source is `ablations/output/N-cloud-v1/phases/{pilot,continue}`: each N1/N3 arm has train 500 then 700 (cumulative 1200), one validation trajectory, training seed 96101 and evaluation seed 96201. Training diagnostics in train700 already include train500 diagnostics. Total physical new training is 2400 and validation is 60, not totals summed across copied directories.
- Only compare compatible experiment/split/stage/risk-version/interaction budgets. Exploratory baseline selection is not a frozen official reference.
- Do not manufacture statistical uncertainty from rounds of one training seed. Preserve all failures and original denominators.

## Interfaces for independent implementation

- Export CLI: `python figure_assets/scripts/export_results.py --root <project> [--source ablations/output/N-cloud-v1] [--snapshot-id <id>]`; default writes snapshots to `frontend/public/data/`. `export_snapshot(root: Path, *, sources: list[Path] | None = None, snapshot_id: str | None = None) -> dict` returns `snapshot_id`, `data_path`, `index_path`, `report_path`.
- Figure CLI: `python figure_assets/scripts/render_figures.py --root <project> [--snapshot <results.json>] [--width single|double]`. `render_snapshot(root: Path, *, snapshot: Path | None = None, width: str = "double") -> dict` returns `manifest_path`, `figures`, `skipped`. Read current index when snapshot is omitted.
- Notebook helpers are primary-owned. They invoke the above APIs and serve only built frontend files plus generated figures, with loopback binding, an explicit lease/deadline, identity-bound stop, and no training/cloud callbacks.
