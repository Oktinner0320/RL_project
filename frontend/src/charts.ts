import * as echarts from 'echarts/core';
import { LineChart, BarChart, HeatmapChart, type LineSeriesOption } from 'echarts/charts';
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, VisualMapComponent, AriaComponent, type GridComponentOption, type TooltipComponentOption, type LegendComponentOption, type DataZoomComponentOption, type VisualMapComponentOption } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { BarSeriesOption, HeatmapSeriesOption } from 'echarts/charts';
echarts.use([LineChart,BarChart,HeatmapChart,GridComponent,TooltipComponent,LegendComponent,DataZoomComponent,VisualMapComponent,AriaComponent,CanvasRenderer]);
type Option = echarts.ComposeOption<LineSeriesOption | BarSeriesOption | HeatmapSeriesOption | GridComponentOption | TooltipComponentOption | LegendComponentOption | DataZoomComponentOption | VisualMapComponentOption>;
import { effectRows, number, checkpointSeries, type Run, type Row } from './model.ts';
import { tracePoints, type ReplayFrame } from './replay.ts';
import { el, panel } from './ui.ts';
const palette: Record<string,string> = {dqn:'#0072B2',ppo:'#E69F00',pqn:'#009E73',fixed:'#666666',metacs:'#CC79A7',time_greedy:'#56B4E9',risk_greedy:'#D55E00',n1:'#0072B2',n3:'#E69F00'};
export function color(method: string): string { return palette[method.toLowerCase()] ?? '#607D8B'; }
const active: echarts.ECharts[] = [];
const observer = new ResizeObserver(() => active.forEach(c => c.resize()));
export function clearCharts(): void { observer.disconnect(); active.splice(0).forEach(c => c.dispose()); echarts.disconnect('effects'); echarts.disconnect('decision-traces'); }
export function chart(parent: HTMLElement, title: string, option: Option, linked = false): echarts.ECharts {
  const p = panel(title); const host = el('div',undefined,'chart'); host.role = 'img'; host.setAttribute('aria-label',title); p.append(host); parent.append(p);
  const c = echarts.init(host); active.push(c); observer.observe(host); if (linked) c.group = 'effects';
  c.setOption({ animation: false, aria: { enabled: true }, textStyle: { fontFamily: 'Inter, Segoe UI, sans-serif', color:'#405269' }, tooltip: { trigger:'axis', renderMode:'richText' }, grid:{left:64,right:24,top:56,bottom:65}, ...option });
  return c;
}
export function replayTraces(parent: HTMLElement, frames: ReplayFrame[], run: Run, onSelect: (index: number) => void): (index: number) => void {
  const points = tracePoints(frames);
  const views: { chart: echarts.ECharts; key: string }[] = [];
  const metrics = [
    ['validation_accuracy', '验证准确率 · fraction'], ['reward', '即时奖励 · reward'],
    ['round_time_seconds', '本轮模拟时间 · s'], ['round_energy_joules', '本轮模拟能量 · J'],
    ['predicted_exposure_risk', '预测暴露风险 · 非审计真值'],
  ] as const;
  for (const [key, title] of metrics) {
    if (!frames.some(f => number(f.row[key]) !== null)) {
      const missing = panel(title); missing.append(el('p', '该指标未导出。缺失不表示零，其他预测不替代暴露风险。', 'empty')); parent.append(missing); continue;
    }
    const c = chart(parent, title, {
      grid: { left: 58, right: 18, top: 18, bottom: 58 },
      xAxis: { type: 'category', name: 'Round', data: points.map(p => p.label), boundaryGap: false },
      yAxis: { type: 'value', scale: true },
      dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 5 }],
      series: [
        { id: 'observed', name: title, type: 'line', showSymbol: true, showAllSymbol: true, symbolSize: 7, connectNulls: false,
          lineStyle: { width: 2, type: run.arm === 'N3' ? 'dashed' : 'solid' }, itemStyle: { color: color(run.arm ?? run.method) },
          data: points.map(p => p.frame ? number(p.frame.row[key]) : null) },
        { id: 'selected-frame', name: '当前轮', type: 'line', showSymbol: true, showAllSymbol: true, symbolSize: 12,
          itemStyle: { color: '#182f46', borderColor: '#fff', borderWidth: 2 }, data: points.map(() => null), z: 5 },
      ],
    });
    c.group = 'decision-traces';
    c.on('click', params => {
      const point = points[Number(params.dataIndex)];
      if (params.componentType === 'series' && point?.frame) onSelect(point.frame.index);
    });
    views.push({ chart: c, key });
  }
  if (views.length) echarts.connect('decision-traces');
  return index => {
    for (const view of views) view.chart.setOption({ series: [{ id: 'selected-frame', data: points.map(p =>
      p.frame?.index === index ? number(p.frame.row[view.key]) : null) }] });
  };
}
export function line(parent: HTMLElement, title: string, runs: Run[], key: string, training: boolean, smooth: boolean): void {
  if (!runs.some(r => (training?r.training:effectRows(r)).some(row => number(row[key]) !== null))) { const p=panel(title);p.append(el('p',training?'Training diagnostics were not exported for these validation observations. No training trace is fabricated.':'No logged non-warmup round measurements. Final-summary observations are shown separately, not as trajectories.','empty'));parent.append(p);return; }
  const series: LineSeriesOption[] = runs.flatMap(run => {
    const sourceRows = training ? run.training : effectRows(run);
    const episodes = training ? ['training'] : [...new Set(sourceRows.map(row=>String(row.episode??'unreported')))];
    return episodes.map(episode=>{
    const rows = training ? sourceRows : sourceRows.filter(row=>String(row.episode??'unreported')===episode);
    const points = rows.map(r => [number(r[training ? 'environment_steps' : 'round']), number(r[key])]);
    const smoothed = points.map((p,i) => { const window = points.slice(Math.max(0,i-4),i+1).map(v => v[1]).filter((v): v is number => v !== null && v !== undefined); return [p[0], p[1] === null ? null : window.reduce((a,b) => a+b,0)/window.length]; });
    return { name: `${run.method}${run.arm ? ` / ${run.arm}` : ''} · train ${run.train_seed??'unknown'} / eval ${run.eval_seed??'unknown'}${training?'':` · episode ${episode}`} · ${run.run_id}`, type:'line' as const, showSymbol:false, connectNulls:false, lineStyle:{width:2,type:run.arm==='N3'?'dashed':'solid'}, itemStyle:{color:color(run.arm??run.method)}, data:(smooth ? smoothed : points) as [number, number | null][] };
    });
  });
  chart(parent,title,{ legend:{type:'scroll',top:0}, xAxis:{type:'value',name:training?'Environment steps':'Round',nameLocation:'middle',nameGap:27}, yAxis:{type:'value',scale:true}, dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:8}], series }, !training);
  if (!training) echarts.connect('effects');
  const p = parent.lastElementChild; if (p && !runs.some(r => (training?r.training:effectRows(r)).some(row => number(row[key]) !== null))) p.append(el('p','No measured values available. Missing values are not zero.', 'empty'));
}
export function checkpoints(parent: HTMLElement, title: string, runs: Run[], reference: Run, key: string): void {
  const groups=checkpointSeries(runs,reference,key);
  if (!groups.length) { const p=panel(title);p.append(el('p','No identified checkpoint trajectory available for this cohort.','empty'));parent.append(p);return; }
  chart(parent,title,{legend:{type:'scroll',top:0},xAxis:{type:'value',name:'Checkpoint training interactions',nameLocation:'middle',nameGap:30},yAxis:{type:'value',scale:true},dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:8}],series:groups.map(g=>({name:g.label,type:'line' as const,connectNulls:false,showSymbol:true,itemStyle:{color:color(g.method)},data:g.points.map(p=>({value:[p.step,p.value],name:p.run_id}))}))});
}
export function heatmap(parent: HTMLElement, title: string, values: Row[], labels: string[]): void {
  const data: number[][] = [];
  values.forEach((r,x) => { if (Array.isArray(r.values)) r.values.forEach((v,y) => { const n = number(v); if (n !== null) data.push([x,y,n]); }); });
  if (!data.length) { const p = panel(title); p.append(el('p','Q-value vectors are unavailable for this episode.', 'empty')); parent.append(p); return; }
  const nums = data.map(v => v[2]!);
  chart(parent,title,{tooltip:{trigger:'item',renderMode:'richText'},grid:{left:110,right:40,bottom:80},xAxis:{type:'category',data:values.map(r => String(r.round ?? 'Unknown'))},yAxis:{type:'category',data:labels},visualMap:{min:Math.min(...nums),max:Math.max(...nums),orient:'horizontal',left:'center',bottom:0,inRange:{color:['#ecf3f5','#75b5bb','#164762']}},series:[{type:'heatmap',data}]});
}
