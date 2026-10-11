import { defaultReference, runLabel, number, object, type Row, type Run } from './model.ts';

export interface JointAction { action_id: number; client_ids: number[]; batch_size: number }
export interface ReplayFrame {
  row: Row;
  index: number;
  episode: number | null;
  round: number | null;
  previous: Row | null;
  action: JointAction | null;
  decisions: Row[];
  decision: Row | null;
  scoreLabel: string;
  scores: Map<number, number | null>;
  uploads: Row[];
  audits: Row[];
  warnings: string[];
}

export function jointActions(run: Run): JointAction[] {
  const space = run.config_summary.action_space;
  if (!Array.isArray(space)) return [];
  return space.filter(object).filter(a => Number.isInteger(a.action_id) && Number(a.action_id) >= 0 &&
    Array.isArray(a.client_ids) && a.client_ids.every(id => Number.isInteger(id) && id >= 0) &&
    Number.isInteger(a.batch_size) && Number(a.batch_size) > 0).map(a => ({
    action_id: Number(a.action_id), client_ids: [...a.client_ids as number[]], batch_size: Number(a.batch_size),
  }));
}

export function clientIds(run: Run): number[] {
  const ids = new Set(jointActions(run).flatMap(a => a.client_ids));
  const count = number(run.config_summary.num_clients);
  if (count !== null && Number.isInteger(count) && count >= 0 && count <= 128)
    for (let id = 0; id < count; id++) ids.add(id);
  for (const row of run.metrics) for (const key of Object.keys(row)) {
    const match = /^c(\d+)_participations$/.exec(key);
    if (match) ids.add(Number(match[1]));
  }
  return [...ids].sort((a, b) => a - b);
}

export const episodeKey = (row: Row): string => number(row.episode) === null ? 'unreported' : String(row.episode);
export function episodeKeys(run: Run): string[] { return [...new Set(run.metrics.map(episodeKey))]; }
export function decisionRuns(runs: Run[]): Run[] { return runs.filter(r => r.stage === 'validate'); }
export function hasReplay(run: Run): boolean {
  if (run.stage !== 'validate' || run.metrics.length < 2) return false;
  const actions = new Set(jointActions(run).map(a => a.action_id));
  const counts = new Map<string, number>();
  for (const row of run.metrics) if (Number.isInteger(row.episode) && Number(row.episode) >= 0 && Number.isInteger(row.round) && Number(row.round) > 0 && Number.isInteger(row.action_id) && actions.has(Number(row.action_id))) {
    const key = `${row.episode}:${row.round}`; counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const episodes = new Map<string, number>();
  for (const [key, count] of counts) if (count === 1) { const episode = key.split(':')[0]!; episodes.set(episode, (episodes.get(episode) ?? 0) + 1); }
  return [...episodes.values()].some(count => count >= 2);
}
export function replayRunLabel(run: Run): string {
  if (typeof run.replay_source_label === 'string') return `${run.method} · ${run.replay_source_label} · ${run.train_seed === null ? '' : `train ${run.train_seed} / `}eval ${run.eval_seed ?? 'Not exported'} · ${run.metrics.length} round records`;
  return `${hasReplay(run) ? 'Round-by-round' : 'Summary only'} · ${runLabel(run)}`;
}
export function preferredDecisionRun(runs: Run[], id: string): Run | undefined {
  const eligible = decisionRuns(runs);
  const playable = eligible.filter(hasReplay);
  return eligible.find(r => r.run_id === id) ?? defaultReference(playable.length ? playable : eligible);
}

const scoreLabels: Record<string, string> = {
  q_values: 'Q value · expected return', probabilities: 'Policy probability', action_probabilities: 'Policy probability',
  policy_probabilities: 'Policy probability', heuristic_scores: 'Heuristic score',
};

function roundKey(row: Row): number | null {
  const round = number(row.round), index = number(row.round_index);
  if (round !== null && index !== null && round !== index) return null;
  return index ?? round;
}

export function replayFrames(run: Run, episode: string, audits: Row[] = []): ReplayFrame[] {
  const actions = jointActions(run);
  const rows = run.metrics.filter(row => episodeKey(row) === episode).slice().sort((a, b) =>
    (number(a.round) ?? Infinity) - (number(b.round) ?? Infinity));
  const uploadCounts = new Map<string, number>();
  for (const upload of run.uploads) if (typeof upload.upload_id === 'string')
    uploadCounts.set(upload.upload_id, (uploadCounts.get(upload.upload_id) ?? 0) + 1);
  return rows.map((row, index) => {
    const ep = number(row.episode), round = number(row.round);
    const warnings: string[] = [];
    const keyed = ep !== null && round !== null;
    const duplicate = keyed && rows.filter(r => r.episode === ep && r.round === round).length > 1;
    if (!keyed) warnings.push('Missing episode / round: this record is displayed without inferring round-level joins.');
    if (duplicate) warnings.push('Duplicate metric records for this round: no decision or upload join is silently selected.');
    const matches = (r: Row) => keyed && !duplicate && r.episode === ep && roundKey(r) === round;
    const decisions = run.decisions.filter(matches);
    const decision = decisions.length === 1 ? decisions[0]! : null;
    if (decisions.length > 1) warnings.push('Multiple decision records for this round: no score vector is selected.');
    const scoreLabel = decision && typeof decision.kind === 'string' ? scoreLabels[decision.kind] : undefined;
    const scores = new Map<number, number | null>();
    if (decision && scoreLabel && Array.isArray(decision.values))
      for (const action of actions) scores.set(action.action_id, number(decision.values[action.action_id]));
    if (decision && !scoreLabel) warnings.push('Unknown decision score type: the raw record is retained without interpreting it as Q values or probabilities.');
    if (!decision) warnings.push('No unique decision vector was exported for this round; policy scores are not inferred from upload predictions.');
    if (!actions.length) warnings.push('Action mapping not exported: client sets and batch sizes are not inferred.');
    const uploads = run.uploads.filter(u => matches(u) && (u.run_id === undefined || u.run_id === run.run_id));
    if (run.uploads.some(u => u.episode === ep && (u.round === round || u.round_index === round) &&
      number(u.round) !== null && number(u.round_index) !== null && u.round !== u.round_index))
      warnings.push('Conflicting upload round and round_index: ambiguous joins were excluded.');
    const ids = new Set(uploads.filter(u => typeof u.upload_id === 'string' && uploadCounts.get(u.upload_id) === 1).map(u => u.upload_id));
    const matchedAudits = audits.filter(a => a.run_id === run.run_id && ids.has(a.upload_id));
    if (uploads.some(u => typeof u.upload_id === 'string' && uploadCounts.get(u.upload_id)! > 1))
      warnings.push('Duplicate upload IDs: corresponding audit records are not automatically joined.');
    const previousRows = keyed ? rows.filter(r => r.episode === ep && r.round === round - 1) : [];
    return {
      row, index, episode: ep, round,
      previous: !duplicate && previousRows.length === 1 ? previousRows[0]! : null,
      action: actions.find(a => a.action_id === row.action_id) ?? null,
      decisions, decision, scoreLabel: scoreLabel ?? 'Score not exported', scores, uploads, audits: matchedAudits, warnings,
    };
  });
}

export function actionRanking(frame: ReplayFrame): { action_id: number; value: number }[] {
  return [...frame.scores].filter((entry): entry is [number, number] => entry[1] !== null)
    .map(([action_id, value]) => ({ action_id, value })).sort((a, b) => b.value - a.value || a.action_id - b.action_id);
}

export function actionDistribution(run: Run): { action: JointAction; count: number }[] {
  return jointActions(run).map(action => ({ action, count: run.metrics.filter(r =>
    r.is_warmup === false && r.action_id === action.action_id).length }));
}

export function tracePoints(frames: ReplayFrame[]): { label: string; frame: ReplayFrame | null }[] {
  const points: { label: string; frame: ReplayFrame | null }[] = [];
  frames.forEach((frame, i) => {
    const previous = frames[i - 1];
    if (previous && previous.round !== null && frame.round !== null && frame.round - previous.round > 1)
      points.push({ label: 'Not recorded', frame: null });
    points.push({ label: frame.round === null ? 'Unreported' : String(frame.round), frame });
  });
  return points;
}
