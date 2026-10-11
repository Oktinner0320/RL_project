import { confined, object, parseSnapshot, type Snapshot } from './model.ts';
export interface Index { current_snapshot: string; snapshots: { snapshot_id: string; generated_at: string; path: string }[] }
async function json(url: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(url, { signal, cache: 'no-cache' });
  if (!response.ok) throw new Error(`Data request failed (${response.status}): ${url}`);
  return response.json();
}
export async function loadIndex(signal: AbortSignal): Promise<Index> {
  const v = await json(`${import.meta.env.BASE_URL}data/index.json`, signal);
  if (!object(v) || v.schema_version !== 'results_index_v1' || typeof v.current_snapshot !== 'string' || !Array.isArray(v.snapshots)) throw new Error('Malformed snapshot index');
  for (const s of v.snapshots) { if (!object(s) || typeof s.snapshot_id !== 'string' || typeof s.generated_at !== 'string') throw new Error('Malformed index entry'); confined(s.path); }
  return v as unknown as Index;
}
export async function loadSnapshot(index: Index, id: string, signal: AbortSignal): Promise<Snapshot> {
  const entry = index.snapshots.find(s => s.snapshot_id === id);
  if (!entry) throw new Error('Snapshot not listed in index');
  const result = parseSnapshot(await json(`${import.meta.env.BASE_URL}data/${confined(entry.path)}`, signal));
  if (result.snapshot_id !== id) throw new Error('Snapshot identity mismatch');
  return result;
}
