import './style.css';
import { loadIndex, loadSnapshot, type Index } from './data.ts';
import type { Snapshot } from './model.ts';
import { clearCharts } from './charts.ts';
import { el, panel, button, select } from './ui.ts';
import { mountDecisionLab } from './collaboration.ts';
const app = document.querySelector<HTMLDivElement>('#app')!;
const params = new URLSearchParams(location.search);
let data: Snapshot | undefined;
let index: Index | undefined;
let request = new AbortController();
let cleanup: (() => void) | undefined;
let runId = params.get('run') ?? '';
let episode = params.get('episode') ?? '';
let frame = Number(params.get('frame')) || 0;
let preview = params.has('preview') && Number.isInteger(Number(params.get('preview'))) ? Number(params.get('preview')) : null;
const content = el('main'); content.id = 'main';
function persist(): void {
  const q = new URLSearchParams(); q.set('page','decision');
  if(data) q.set('snapshot',data.snapshot_id); if(runId) q.set('run',runId); if(episode) q.set('episode',episode);
  q.set('frame',String(frame)); if(preview!==null) q.set('preview',String(preview));
  history.replaceState(null,'',`${location.pathname}?${q}`);
}
function stop(): void { cleanup?.(); cleanup = undefined; }
function shell(): void {
  app.replaceChildren(); const skip = el('a','Skip to main content','skip'); skip.href='#main'; app.append(skip);
  const workspace = el('div',undefined,'workspace'); const header = el('header');
  const brand = el('div',undefined,'app-brand'); brand.append(el('strong','CLOUD / EDGE'),el('span','Reinforcement learning for cloud–edge collaboration','eyebrow'));
  const controls = el('div',undefined,'controls'); if(index) controls.append(select('Experiment snapshot',index.snapshots.map(s => [s.snapshot_id,s.snapshot_id]),data?.snapshot_id ?? '',id => void load(id)));
  controls.append(button('Reload snapshot',() => void load(data?.snapshot_id))); header.append(brand,controls); workspace.append(header,content); app.append(workspace);
}
async function load(id?: string): Promise<void> {
  request.abort(); request = new AbortController(); const signal=request.signal; stop(); clearCharts(); content.replaceChildren(el('p','Loading experiment snapshot…','empty')); shell();
  try { index=await loadIndex(signal); const next=await loadSnapshot(index,id ?? params.get('snapshot') ?? index.current_snapshot,signal); if(signal.aborted)return; data=next; shell(); render(); }
  catch(e) { if(signal.aborted)return; data=undefined; shell(); const p=panel('Data unavailable','No synthetic results are substituted for experiment records.'); p.append(el('p',String(e),'error'),button('Retry snapshot',() => { params.delete('snapshot'); void load(); })); content.replaceChildren(p); }
}
function render(): void {
  stop(); clearCharts(); content.replaceChildren(); if(!data)return;
  content.append(el('p',`${data.snapshot_id} · ${data.generated_at}`,'eyebrow'),el('h1','Cloud–Edge Lab'));
  cleanup=mountDecisionLab(content,data,{run:runId,episode,frame,preview},s=>{runId=s.run;episode=s.episode;frame=s.frame;preview=s.preview;persist();},render);
  persist();
}
window.addEventListener('pagehide',()=>{stop();request.abort();clearCharts();});
void load();
