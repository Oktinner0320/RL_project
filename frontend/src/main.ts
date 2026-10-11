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
  app.replaceChildren(); const skip = el('a','跳到主内容','skip'); skip.href='#main'; app.append(skip);
  const workspace = el('div',undefined,'workspace'); const header = el('header');
  const brand = el('div',undefined,'app-brand'); brand.append(el('strong','CLOUD / EDGE'),el('span','强化学习驱动的云边协同','eyebrow'));
  const controls = el('div',undefined,'controls'); if(index) controls.append(select('实验快照',index.snapshots.map(s => [s.snapshot_id,s.snapshot_id]),data?.snapshot_id ?? '',id => void load(id)));
  controls.append(button('重新读取快照',() => void load(data?.snapshot_id))); header.append(brand,controls); workspace.append(header,content); app.append(workspace);
}
async function load(id?: string): Promise<void> {
  request.abort(); request = new AbortController(); const signal=request.signal; stop(); clearCharts(); content.replaceChildren(el('p','读取实验快照…','empty')); shell();
  try { index=await loadIndex(signal); const next=await loadSnapshot(index,id ?? params.get('snapshot') ?? index.current_snapshot,signal); if(signal.aborted)return; data=next; shell(); render(); }
  catch(e) { if(signal.aborted)return; data=undefined; shell(); const p=panel('数据不可用','未使用模拟结果替代真实实验记录。'); p.append(el('p',String(e),'error'),button('重试当前快照',() => { params.delete('snapshot'); void load(); })); content.replaceChildren(p); }
}
function render(): void {
  stop(); clearCharts(); content.replaceChildren(); if(!data)return;
  content.append(el('p',`${data.snapshot_id} · ${data.generated_at}`,'eyebrow'),el('h1','云边交互台'));
  cleanup=mountDecisionLab(content,data,{run:runId,episode,frame,preview},s=>{runId=s.run;episode=s.episode;frame=s.frame;preview=s.preview;persist();},render);
  persist();
}
window.addEventListener('pagehide',()=>{stop();request.abort();clearCharts();});
void load();
