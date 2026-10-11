export type Row = Record<string, unknown>;
export interface Source { path: string; record?: number; line?: number }
export interface Run extends Row { run_id: string; experiment_id: string; method: string; arm: string | null; phase: string; stage: string; train_seed: number | null; eval_seed: number | null; partition: string; risk_measurement: string; risk_version: string | null; evidence_scope: string; status: Row; budget: Row; summary: Row; config_summary: Row; metrics: Row[]; training: Row[]; decisions: Row[]; uploads: Row[]; sources: Source[]; warnings: string[] }
export interface Snapshot { schema_version: string; snapshot_id: string; generated_at: string; experiments: Row[]; runs: Run[]; audits: Row[]; calibration: Row[]; acceptance_reports: Row[]; warnings: string[]; metric_definitions: Row }
export const object = (v: unknown): v is Row => v !== null && typeof v === 'object' && !Array.isArray(v);
export const number = (v: unknown): number | null => typeof v === 'number' && Number.isFinite(v) ? v : null;
export function confined(path: unknown): string {
  if (typeof path !== 'string' || !path || !/^[A-Za-z0-9_./-]+$/.test(path) || path.startsWith('/') || path.split('/').some(p => !p || p === '.' || p === '..')) throw new Error('Unsafe relative data path');
  return path;
}
function finite(v: unknown): void {
  if (typeof v === 'number' && !Number.isFinite(v)) throw new Error('Nonfinite snapshot value');
  if (Array.isArray(v)) v.forEach(finite); else if (object(v)) Object.values(v).forEach(finite);
}
function strings(row: Row, keys: string[]): void { for (const key of keys) if (typeof row[key] !== 'string') throw new Error(`Invalid ${key}`); }
function nullableNumbers(row: Row, keys: string[]): void { for (const key of keys) if (row[key] !== null && number(row[key]) === null) throw new Error(`Invalid numeric ${key}`); }
function sources(v: unknown): void {
  if (!Array.isArray(v)) throw new Error('Invalid sources');
  for (const s of v) { if (!object(s) || typeof s.path !== 'string' || !s.path || s.path.startsWith('/') || s.path.includes('\\') || /^[a-z]+:/i.test(s.path) || s.path.split('/').includes('..')) throw new Error('Invalid project-relative source'); for (const key of ['record','line']) if (s[key] !== undefined && (!Number.isInteger(s[key]) || Number(s[key]) < 0)) throw new Error(`Invalid source ${key}`); }
}
function numericRows(rows: Row[]): void {
  for (const row of rows) for (const [key,v] of Object.entries(row)) {
    if (['validation_accuracy','validation_loss','elapsed_seconds','energy_joules','predicted_exposure_risk','risk_exposures','round','episode','action_id','reward','environment_steps','td_loss','q_mean','updates'].includes(key) || /^c\d+_participations$/.test(key)) if (v !== null && number(v) === null) throw new Error(`Invalid numeric row ${key}`);
    if (['is_warmup','terminated','truncated'].includes(key) && v !== null && typeof v !== 'boolean') throw new Error(`Invalid boolean ${key}`);
  }
}
function actualResults(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(actualResults);
  if (!object(v)) return v;
  return Object.fromEntries(Object.keys(v).sort().filter(k=>!['sources','source','source_index'].includes(k)).map(k=>[k,actualResults(v[k])]));
}
export function parseSnapshot(v: unknown): Snapshot {
  finite(v);
  if (!object(v) || v.schema_version !== 'results_visualization_v1' || typeof v.snapshot_id !== 'string' || typeof v.generated_at !== 'string' || !object(v.metric_definitions)) throw new Error('Invalid snapshot envelope');
  for (const key of ['experiments','runs','audits','calibration','acceptance_reports']) if (!Array.isArray(v[key]) || !(v[key] as unknown[]).every(object)) throw new Error(`Invalid ${key}`);
  if (!Array.isArray(v.warnings) || !v.warnings.every(x => typeof x === 'string')) throw new Error('Invalid warnings');
  confined(v.snapshot_id);
  for (const experiment of v.experiments as Row[]) {
    strings(experiment,['experiment_id','name','kind','evidence_scope']);
    if (!object(experiment.counts) || !object(experiment.status)) throw new Error('Invalid experiment counts/status');
    nullableNumbers(experiment,['actual_cloud_cost_usd']);
    nullableNumbers(experiment.counts,['physical_train_interactions','physical_validation_interactions']);
    if (!Number.isInteger(experiment.counts.training_seed_count) || Number(experiment.counts.training_seed_count) < 0) throw new Error('Invalid training_seed_count');
    if (!Array.isArray(experiment.evidence_gaps) || !experiment.evidence_gaps.every(x=>typeof x==='string')) throw new Error('Invalid evidence gaps');
    sources(experiment.sources);
  }
  for (const audit of v.audits as Row[]) { strings(audit,['run_id','upload_id','trajectory_id','scope']); nullableNumbers(audit,['fit_seed','panel_seed','prediction','raw_prediction','calibrated_prediction','truth','batch_size']); if (audit.model_id !== null && typeof audit.model_id !== 'string') throw new Error('Invalid audit model'); if (audit.is_warmup !== null && typeof audit.is_warmup !== 'boolean') throw new Error('Invalid audit warmup'); sources([audit.source]); }
  for (const calibration of v.calibration as Row[]) { strings(calibration,['group_id','scope']); nullableNumbers(calibration,['fit_seed','raw_u','cal_u','raw_ewmae','cal_ewmae']); if (calibration.model_id !== null && typeof calibration.model_id !== 'string') throw new Error('Invalid calibration model'); if (!Array.isArray(calibration.panel_ids) || !calibration.panel_ids.every(x=>typeof x==='string')) throw new Error('Invalid calibration panels'); sources(calibration.sources); }
  const seen = new Map<string, string>();
  const runs: Run[] = [];
  for (const raw of v.runs as Row[]) {
    for (const key of ['run_id','experiment_id','method','phase','stage','partition','risk_measurement','evidence_scope']) if (typeof raw[key] !== 'string') throw new Error(`Invalid run ${key}`);
    for (const key of ['arm','risk_version']) if (raw[key] !== null && typeof raw[key] !== 'string') throw new Error(`Invalid run ${key}`);
    for (const key of ['train_seed','eval_seed']) if (raw[key] !== null && number(raw[key]) === null) throw new Error(`Invalid ${key}`);
    for (const key of ['status','budget','summary','config_summary']) if (!object(raw[key])) throw new Error(`Invalid run ${key}`);
    nullableNumbers(raw.budget as Row,['new_interactions','cumulative_interactions','warmup_interactions']);
    for (const key of ['metrics','training','decisions','uploads','sources']) if (!Array.isArray(raw[key]) || !(raw[key] as unknown[]).every(object)) throw new Error(`Invalid run ${key}`);
    if (!Array.isArray(raw.warnings) || !raw.warnings.every(x => typeof x === 'string')) throw new Error('Invalid run warnings');
    sources(raw.sources);
    numericRows(raw.metrics as Row[]); numericRows(raw.training as Row[]);
    const actions = (raw.config_summary as Row).action_space;
    if (actions !== undefined && actions !== null) {
      if (!Array.isArray(actions)) throw new Error('Invalid action space');
      const ids = new Set<number>();
      for (const a of actions) { if (!object(a) || !Number.isInteger(a.action_id) || !Array.isArray(a.client_ids) || !a.client_ids.every(id=>Number.isInteger(id) && id>=0) || !Number.isInteger(a.batch_size) || Number(a.batch_size)<=0 || ids.has(Number(a.action_id))) throw new Error('Invalid action mapping'); ids.add(Number(a.action_id)); }
    }
    const serialized = JSON.stringify(actualResults(raw));
    const id = raw.run_id as string;
    if (seen.has(id) && seen.get(id) !== serialized) throw new Error(`Contradictory run copies: ${id}`);
    if (!seen.has(id)) runs.push(raw as Run);
    seen.set(id, serialized);
  }
  return { ...v, runs } as unknown as Snapshot;
}
export function compatible(a: Run, b: Run): boolean {
  const same = (x: unknown, y: unknown) => JSON.stringify(actualResults(x ?? null)) === JSON.stringify(actualResults(y ?? null));
  return ['experiment_id','phase','stage','partition','risk_measurement','risk_version'].every(k => a[k] === b[k])
    && ['new_interactions','cumulative_interactions'].every(k => number(a.budget[k]) !== null && a.budget[k] === b.budget[k])
    && same(a.budget.warmup_interactions,b.budget.warmup_interactions)
    && ['dataset','partition','action_space','max_rounds','num_clients','checkpoint_step','training_interaction_budget'].every(k=>same(a.config_summary[k],b.config_summary[k]));
}
export function scopeWarnings(run: Run): string[] {
  const unknown = ['dataset','max_rounds','num_clients','checkpoint_step','training_interaction_budget'].filter(k=>run.config_summary[k] === null || run.config_summary[k] === undefined);
  if (!Array.isArray(run.config_summary.action_space) || !run.config_summary.action_space.length) unknown.push('action_space');
  if (number(run.budget.warmup_interactions) === null) unknown.push('warmup denominator');
  return unknown.length ? [`Unknown scientific metadata: ${unknown.join(', ')}. Matching missing fields are not established equivalence; final summaries are descriptive only.`] : [];
}
export function runLabel(run: Run): string { return `${run.method}${run.arm ? ` / ${run.arm}` : ''} · train ${run.train_seed ?? 'unknown'} / eval ${run.eval_seed ?? 'unknown'} · checkpoint ${run.config_summary.checkpoint_step ?? 'unreported'} · training budget ${run.config_summary.training_interaction_budget ?? 'unreported'}`; }
export function defaultReference(runs: Run[]): Run | undefined { const validation=runs.filter(r=>r.stage==='validate' && r.partition==='validation'); return validation.find(r=>r.arm==='N1') ?? validation.find(r=>r.summary.is_latest===true) ?? validation[0] ?? runs[0]; }
export function latestScope(runs: Run[], latestOnly: boolean): {available: boolean; runs: Run[]} { const available=runs.some(r=>typeof r.summary.is_latest==='boolean'); return {available,runs:latestOnly&&available?runs.filter(r=>r.summary.is_latest===true):runs}; }
export function effectRows(run: Run): Row[] { return run.metrics.filter(r => r.is_warmup === false); }
export function finalValue(run: Run, key: string): number | null { const rows = effectRows(run); if (rows.length) return number(rows.at(-1)?.[key]); return run.summary.is_warmup === true ? null : number(run.summary[key]); }
export interface CheckpointSeries { label: string; method: string; points: { step: number; value: number | null; run_id: string }[] }
export function checkpointSeries(runs: Run[], reference: Run, key: string): CheckpointSeries[] {
  const groups = new Map<string, { run: Run; points: CheckpointSeries['points'] }>();
  for (const run of runs) {
    const step=number(run.config_summary.checkpoint_step);
    if (step===null || run.train_seed===null || run.eval_seed===null || run.stage!=='validate' || number(run.config_summary.training_interaction_budget)!==step) continue;
    const stripped=(r: Run): Run=>({...r,config_summary:{...r.config_summary,checkpoint_step:null,training_interaction_budget:null}});
    if (!compatible(stripped(reference),stripped(run))) continue;
    const id=JSON.stringify([run.method,run.arm,run.train_seed,run.eval_seed]);
    const group=groups.get(id)??{run,points:[]}; group.points.push({step,value:finalValue(run,key),run_id:run.run_id});groups.set(id,group);
  }
  return [...groups.values()].map(g=>{ const counts=new Map<number,number>();g.points.forEach(p=>counts.set(p.step,(counts.get(p.step)??0)+1)); return {method:g.run.method,label:`${g.run.method}${g.run.arm?` / ${g.run.arm}`:''} · train ${g.run.train_seed} / eval ${g.run.eval_seed}`,points:g.points.filter(p=>counts.get(p.step)===1).sort((a,b)=>a.step-b.step)}; }).filter(g=>g.points.length>0);
}
export function actionLabel(run: Run, id: unknown): string {
  const actions = run.config_summary.action_space;
  if (!Array.isArray(actions)) return `Action ${String(id)} · mapping unavailable`;
  const a = actions.find(a => object(a) && a.action_id === id);
  if (!object(a) || !Array.isArray(a.client_ids) || number(a.batch_size) === null) return `Action ${String(id)} · mapping unavailable`;
  return `Action ${String(id)} · clients ${a.client_ids.join(', ')} · batch ${String(a.batch_size)}`;
}
