import { object, parseSnapshot, type Snapshot } from './model.ts';

// Supplemental frontend-owned projections do not rewrite the original snapshot or merge rounds.
export function addHistoricalReplays(snapshot: Snapshot, value: unknown): Snapshot {
  if (!object(value) || value.schema_version !== 'frontend_replays_v1' || !object(value.experiment) || !Array.isArray(value.runs) || !value.runs.length || value.runs.length > 32)
    throw new Error('Invalid historical replay envelope');
  const experiment = value.experiment;
  const supplemental = parseSnapshot({ ...snapshot, experiments: [experiment], runs: value.runs, audits: [], calibration: [], acceptance_reports: [] });
  if (supplemental.runs.some(r => r.stage !== 'validate' || r.experiment_id !== experiment.experiment_id || r.evidence_scope !== 'historical_validation_replay_only' || typeof r.replay_source_label !== 'string'))
    throw new Error('Invalid historical validation identity');
  const existing = new Set(snapshot.runs.map(r => r.run_id));
  const additions = supplemental.runs.filter(r => !existing.has(r.run_id));
  const skipped = supplemental.runs.length - additions.length;
  return {
    ...snapshot,
    runs: [...snapshot.runs, ...additions],
    experiments: additions.length && !snapshot.experiments.some(e => e.experiment_id === experiment.experiment_id) ? [...snapshot.experiments, experiment] : [...snapshot.experiments],
    warnings: [...snapshot.warnings, ...(additions.length ? [`补充 ${additions.length} 条历史验证轨迹，仅用于回放；不替换当前实验的汇总或形成跨批次比较。`] : []), ...(skipped ? [`${skipped} 条历史运行 ID 已存在，保留原始快照记录，未覆盖或拼接。`] : [])],
  };
}
