// Read-only projection of existing validation artifacts. Never imports or executes the experiment framework.
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, statSync, writeFileSync, renameSync, existsSync } from 'node:fs';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const cohort = 'historical-native-acceptance-20261008';
const root = 'output/native-acceptance-20261008';
export const replaySources = [
  ['dqn', `${root}/policy-validation-dqn-s201-retry2/runs/validate-dqn-s201-1c5807309e26`],
  ['ppo', `${root}/policy-validation-ppo-s201/runs/validate-ppo-s201-b562dd2f004a`],
  ['pqn', `${root}/policy-validation-pqn-s201/runs/validate-pqn-s201-b7e57a65e40b`],
  ['metacs', `${root}/policy-validation-metacs-s201/runs/validate-metacs-s201-69431882a057`],
  ['fixed', `${root}/policy-validation-fixed-s201/runs/validate-fixed-s201-70cfbd77561e`],
  ['time_greedy', `${root}/policy-validation-time-greedy-s201/runs/validate-time_greedy-s201-4250f5953f11`],
  ['risk_greedy', `${root}/policy-validation-risk-greedy-s201/runs/validate-risk_greedy-s201-37c1842da7ea`],
];
const metricKeys = new Set(['episode', 'round', 'elapsed_seconds', 'energy_joules', 'upload_bytes', 'download_bytes', 'processed_samples', 'unique_samples', 'validation_loss', 'validation_accuracy', 'risk_version', 'risk_available', 'reward_protocol', 'is_warmup', 'risk_exposures', 'predicted_exposure_risk', 'action_id', 'loss_gain', 'round_time_seconds', 'round_energy_joules', 'predicted_risk', 'worst_predicted_risk', 'round_upload_bytes', 'round_download_bytes', 'reward', 'total_reward', 'terminated', 'truncated', 'decision_kind', 'decision_margin']);
export const uploadKeys = ['upload_id', 'episode', 'round', 'round_index', 'client_id', 'batch_size', 'predicted_risk', 'uncertainty', 'decision_kind', 'is_warmup'];
const decisionKeys = ['episode', 'round', 'kind', 'values', 'margin', 'entropy', 'state_value'];
const stringKeys = new Set(['risk_version', 'reward_protocol', 'decision_kind']);
const boolKeys = new Set(['risk_available', 'is_warmup', 'terminated', 'truncated']);
const allowedMetric = key => metricKeys.has(key) || /^c\d+_(participations|samples|unique)$/.test(key);
const pick = (row, keys) => Object.fromEntries(keys.filter(k => Object.hasOwn(row, k)).map(k => [k, row[k]]));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const assert = (condition, message) => { if (!condition) throw new Error(message); };

export function parseCSV(text) {
  const rows = []; let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') { if (quoted && text[i + 1] === '"') { field += '"'; i++; } else quoted = !quoted; }
    else if (!quoted && c === ',') { row.push(field); field = ''; }
    else if (!quoted && (c === '\n' || c === '\r')) {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); if (row.some(v => v !== '')) rows.push(row); row = []; field = '';
    } else field += c;
  }
  assert(!quoted, 'Unclosed CSV field');
  if (field || row.length) { row.push(field); rows.push(row); }
  const header = rows.shift() ?? [];
  assert(header.length && new Set(header).size === header.length, 'Invalid CSV header');
  return rows.map(values => {
    assert(values.length === header.length, 'Invalid CSV row width');
    return Object.fromEntries(header.map((key, i) => {
      if (!allowedMetric(key)) return [key, undefined];
      const raw = values[i];
      if (!raw) return [key, null];
      if (stringKeys.has(key)) return [key, raw];
      if (boolKeys.has(key)) { assert(/^(true|false)$/i.test(raw), `Invalid CSV boolean ${key}`); return [key, raw.toLowerCase() === 'true']; }
      const value = Number(raw); assert(Number.isFinite(value), `Invalid CSV number ${key}`); return [key, value];
    }).filter(([key]) => allowedMetric(key)));
  });
}

export function projectRun([method, dir], base = project) {
  const sources = [];
  const read = file => {
    const path = `${dir}/${file}`, absolute = resolve(base, path);
    assert(statSync(absolute).size <= 4 * 1024 * 1024, `Replay source exceeds 4 MiB: ${path}`);
    const bytes = readFileSync(absolute); sources.push({ path, sha256: sha(bytes), bytes: bytes.length }); return bytes.toString('utf8');
  };
  const metadata = JSON.parse(read('metadata.json')), spec = JSON.parse(read('config.json')), status = JSON.parse(read('status.json')), summary = JSON.parse(read('summary.json'));
  assert(metadata.stage === 'validate' && spec.stage === 'validate' && summary.stage === 'validate', 'Only genuine validation records can be imported');
  assert(metadata.method === method && spec.method === method && metadata.run_id === spec.run_id && metadata.run_id === summary.run_id, 'Replay identity mismatch');
  assert(status.state === 'completed' && (status.run_id === undefined || status.run_id === metadata.run_id), 'Replay source is not completed');
  assert(metadata.seed === spec.seed && metadata.risk_version === spec.risk_version && metadata.risk_version === summary.risk_version, 'Replay seed/risk identity mismatch');
  assert(status.started_at?.startsWith('2026-10-08'), 'Historical source date mismatch');
  const config = spec.config, env = config.environment, count = config.data.num_clients;
  assert(Number.isInteger(count) && count > 0 && count <= 128 && Array.isArray(env.client_sets) && Array.isArray(env.batch_sizes), 'Missing saved action configuration');
  assert(env.client_sets.every(ids => Array.isArray(ids) && ids.length && new Set(ids).size === ids.length && ids.every(id => Number.isInteger(id) && id >= 0 && id < count)) && env.batch_sizes.every(n => Number.isInteger(n) && n > 0), 'Invalid saved action configuration');
  const action_space = env.client_sets.flatMap(ids => env.batch_sizes.map(batch => ({ client_ids: [...ids], batch_size: batch }))).map((a, action_id) => ({ action_id, ...a }));
  const metrics = parseCSV(read('metrics.csv'));
  assert(metrics.length >= 2 && metrics.length <= 1000, 'Replay row limit is 2–1000');
  const keys = new Set();
  for (const row of metrics) {
    const key = `${row.episode}:${row.round}`;
    assert(Number.isInteger(row.episode) && row.episode >= 0 && Number.isInteger(row.round) && row.round > 0 && !keys.has(key), 'Invalid or duplicate replay round'); keys.add(key);
    assert(Number.isInteger(row.action_id) && action_space[row.action_id], 'Recorded action has no saved mapping');
    assert(row.risk_version === metadata.risk_version, 'Round risk version mismatch');
  }
  assert(metrics.at(-1).round === summary.round && metrics.at(-1).validation_accuracy === summary.validation_accuracy, 'Trace does not match its final summary');
  const jsonLines = text => text.trim() ? text.trim().split(/\r?\n/).map(line => JSON.parse(line)) : [];
  const uploads = jsonLines(read('uploads.jsonl')).map(row => pick(row, uploadKeys));
  assert(uploads.length <= 5000, 'Upload metadata limit exceeded');
  const uploadIds = new Set();
  for (const upload of uploads) {
    const round = upload.round_index ?? upload.round;
    const row = metrics.find(r => r.episode === upload.episode && r.round === round), action = row && action_space[row.action_id];
    assert(typeof upload.upload_id === 'string' && !uploadIds.has(upload.upload_id), 'Duplicate or missing upload ID'); uploadIds.add(upload.upload_id);
    assert(upload.round_index === undefined || upload.round === undefined || upload.round_index === upload.round, 'Conflicting upload round aliases');
    assert(action && action.client_ids.includes(upload.client_id) && action.batch_size === upload.batch_size, 'Upload does not match its recorded joint action');
  }
  const decisions = existsSync(resolve(base, dir, 'policy_decisions.jsonl')) ? jsonLines(read('policy_decisions.jsonl')).map(row => pick(row, decisionKeys)) : [];
  assert(decisions.length <= 1000, 'Decision vector limit exceeded');
  const decisionRounds = new Set();
  for (const d of decisions) {
    const key = `${d.episode}:${d.round}`;
    assert(keys.has(key) && !decisionRounds.has(key), 'Decision has no unique recorded round'); decisionRounds.add(key);
    assert(typeof d.kind === 'string' && Array.isArray(d.values) && d.values.length === action_space.length && d.values.every(Number.isFinite), 'Invalid recorded decision vector');
  }
  for (const source of sources) assert(sha(readFileSync(resolve(base, source.path))) === source.sha256, `Replay input changed during projection: ${source.path}`);
  const learned = ['dqn', 'ppo', 'pqn'].includes(method);
  return {
    run_id: metadata.run_id, experiment_id: cohort, method, arm: null, phase: 'historical-validation-20261008', stage: 'validate',
    train_seed: learned ? config.experiment.training_seed ?? null : null, eval_seed: metadata.seed, partition: 'validation',
    risk_measurement: 'predicted_exposure_risk', risk_version: metadata.risk_version, evidence_scope: 'historical_validation_replay_only',
    replay_source_label: '2026-10-08 historical validation',
    status: pick(status, ['state', 'started_at', 'updated_at', 'finished_at', 'elapsed_seconds']),
    budget: { new_interactions: null, cumulative_interactions: null, warmup_interactions: metrics.filter(r => r.is_warmup === true).length },
    summary: { run_id: summary.run_id, stage: summary.stage, ...pick(summary, Object.keys(summary).filter(allowedMetric)) },
    config_summary: {
      dataset: config.data.url, partition: config.data.partition, num_clients: count, max_rounds: env.max_rounds, action_space,
      gamma: ['dqn', 'ppo'].includes(method) ? config.rl.gamma ?? null : null, n_step: null, warmup_rounds: null,
      checkpoint_step: null, training_interaction_budget: null,
    },
    metrics, decisions, uploads, training: [], sources,
    warnings: ['Historical validation trajectory from a different cohort than the newer summaries; not used for equal-budget or same-state comparisons.', 'Training checkpoint steps and actual training budget are not verified by these validation artifacts.', ...(decisions.length ? [] : ['No policy score vector was saved for this run; replay includes only actual actions, upload metadata and feedback.'])],
  };
}

export function prepareReplays(entries = replaySources) {
  const runs = entries.map(entry => projectRun(entry));
  return {
    schema_version: 'frontend_replays_v1',
    experiment: {
      experiment_id: cohort, name: '2026-10-08 historical validation replay', kind: 'historical_validation_replay', evidence_scope: 'historical_validation_replay_only',
      actual_cloud_cost_usd: null, status: { state: 'source_runs_completed' },
      counts: { physical_train_interactions: null, physical_validation_interactions: runs.reduce((n, r) => n + r.metrics.length, 0), training_seed_count: new Set(runs.map(r => r.train_seed).filter(n => n !== null)).size },
      evidence_gaps: ['Supplementary replay only; existing snapshots are unchanged and no fair cross-cohort comparison or acceptance verdict is established.'],
      sources: runs.flatMap(r => r.sources),
    },
    runs,
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), sample = args.includes('--sample') ? args[args.indexOf('--sample') + 1] : null;
  assert(args.every((a, i) => ['--sample', '--out'].includes(a) || ['--sample', '--out'].includes(args[i - 1])), 'Unknown replay preparation argument');
  const outputArg = args.includes('--out') ? args[args.indexOf('--out') + 1] : null;
  assert(!sample || outputArg, 'Sample gate requires an isolated --out path');
  const entries = sample ? replaySources.filter(([method]) => method === sample) : replaySources;
  assert(entries.length, 'Unknown sample method');
  const output = resolve(outputArg ?? resolve(project, 'frontend/public/replays/native-validation-v1.json'));
  const rel = relative(resolve(project, 'frontend/public/replays'), output);
  const temp = relative('C:/Users/42155/AppData/Local/Temp/opencode', output);
  assert((!rel.startsWith('..') && !rel.includes(':')) || (!temp.startsWith('..') && !temp.includes(':')), 'Output must stay in frontend/public/replays or approved external temporary artifacts');
  const document = prepareReplays(entries), serialized = `${JSON.stringify(document)}\n`;
  mkdirSync(dirname(output), { recursive: true });
  if (!existsSync(output) || readFileSync(output, 'utf8') !== serialized) {
    const staging = `${output}.${process.pid}.tmp`; writeFileSync(staging, serialized, { flag: 'wx' }); renameSync(staging, output);
  }
  console.log(JSON.stringify({ output, runs: document.runs.map(r => ({ method: r.method, run_id: r.run_id, rounds: r.metrics.length, uploads: r.uploads.length, decisions: r.decisions.length })), bytes: Buffer.byteLength(serialized), source_checksums_verified: true }));
}
