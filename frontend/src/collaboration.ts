import { number, runLabel, type Run, type Snapshot, type Row } from './model.ts';
import { jointActions, clientIds, episodeKeys, preferredDecisionRun, decisionRuns, replayFrames, actionRanking, type ReplayFrame, type JointAction } from './replay.ts';
import { replayTraces, color } from './charts.ts';
import { el, button, select, panel, records, format, table } from './ui.ts';

export interface LabState { run: string; episode: string; frame: number; preview: number | null }
type Inspect = 'state' | 'agent' | 'action' | 'cloud' | 'reward' | 'upload' | 'download' | `client:${number}`;
const NS = 'http://www.w3.org/2000/svg';
const value = (v: unknown): string => v === null || v === undefined ? '未导出' : format(v);
const clients = (action: JointAction): string => action.client_ids.map(id => `C${id}`).join(' + ');
const actionName = (action: JointAction): string => `A${action.action_id} · ${clients(action)} · batch ${action.batch_size}`;
function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, text?: string): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, String(val));
  if (text !== undefined) node.textContent = text;
  return node;
}
function svgButton(node: SVGElement, label: string, key: string, onClick: () => void): void {
  node.setAttribute('role', 'button'); node.setAttribute('tabindex', '0'); node.setAttribute('aria-label', label);
  node.dataset.focusKey = key;
  node.addEventListener('click', onClick);
  node.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } });
}
function preserveFocus(host: HTMLElement, update: () => void): void {
  const active = document.activeElement;
  const key = active instanceof Element && host.contains(active) ? active.getAttribute('data-focus-key') : null;
  update();
  if (key) Array.from(host.querySelectorAll<HTMLElement | SVGElement>('[data-focus-key]')).find(e => e.dataset.focusKey === key)?.focus({ preventScroll: true });
}
function stat(parent: HTMLElement, label: string, val: unknown, note?: string): void {
  const block = el('div', undefined, 'lab-stat');
  block.append(el('span', label), el('strong', value(val)));
  if (note) block.append(el('small', note));
  parent.append(block);
}
function status(row: Row): string {
  if (row.is_warmup === true) return '预热 · 非学习策略评分';
  if (row.is_warmup !== false) return '预热身份未导出';
  return '非预热记录';
}

function drawGraph(host: HTMLElement, run: Run, frame: ReplayFrame, preview: JointAction | null, playing: boolean, inspect: (target: Inspect) => void): void {
  preserveFocus(host, () => {
    host.replaceChildren();
    const ids = clientIds(run);
    const width = Math.max(760, ids.length * 190 + 50);
    const graph = svg('svg', { viewBox: `0 0 ${width} 410`, class: `collaboration-graph${playing ? ' is-playing' : ''}`, 'aria-label': 'RL 调度器选择客户端与批量大小，边端本地计算并上传，云端聚合产生奖励反馈' });
    const defs = svg('defs');
    for (const [name, fill] of [['up', '#168776'], ['down', '#5988b1'], ['preview', '#b67625']] as const) {
      const marker = svg('marker', { id: `arrow-${name}`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' });
      marker.append(svg('path', { d: 'M 0 0 L 10 5 L 0 10 Z', fill })); defs.append(marker);
    }
    graph.append(defs, svg('rect', { x: 12, y: 10, width: width - 24, height: 128, rx: 16, class: 'cloud-zone' }),
      svg('text', { x: 30, y: 34, class: 'svg-zone-label' }, 'CLOUD / 云端决策与聚合'));
    const boxes: { x: number; title: string; sub: string; target: Inspect }[] = [
      { x: width * .08, title: `${run.method.toUpperCase()}${run.arm ? ` / ${run.arm}` : ''} 调度器`, sub: `${frame.scoreLabel} · γ ${value(run.config_summary.gamma)}`, target: 'agent' },
      { x: width * .38, title: frame.action ? `联合动作 A${frame.action.action_id}` : '动作映射缺失', sub: frame.action ? `${clients(frame.action)} · batch ${frame.action.batch_size}` : `记录 action_id ${value(frame.row.action_id)}`, target: 'action' },
      { x: width * .68, title: '全局模型聚合', sub: `准确率 ${value(frame.row.validation_accuracy)}`, target: 'cloud' },
    ];
    for (const box of boxes) {
      const g = svg('g', { class: 'cloud-node' });
      g.append(svg('rect', { x: box.x, y: 49, width: width * .245, height: 68, rx: 10 }),
        svg('text', { x: box.x + 14, y: 77, class: 'svg-title' }, box.title),
        svg('text', { x: box.x + 14, y: 99, class: 'svg-note' }, box.sub));
      svgButton(g, `检查${box.title}`, box.target, () => inspect(box.target)); graph.append(g);
    }
    graph.append(svg('text', { x: width * .35, y: 88, class: 'svg-arrow' }, '→'), svg('text', { x: width * .65, y: 88, class: 'svg-arrow' }, '→'));
    const trunk = width * .5;
    if (ids.length) {
      const firstX = width / (ids.length + 1), lastX = firstX * ids.length;
      graph.append(svg('path', { d: `M${trunk} 138 L${trunk} 180 M${firstX - 9} 180 L${lastX + 22} 180`, class: 'bus-link' }));
      const down = svg('g', { class: 'link-summary' });
      down.append(svg('rect', { x: 20, y: 154, width: 213, height: 33, rx: 7 }),
        svg('text', { x: 32, y: 175, class: 'svg-note' }, `↓ 模型下发 · ${value(frame.row.round_download_bytes)} B`));
      svgButton(down, '检查本轮模型下发总量', 'download', () => inspect('download'));
      const up = svg('g', { class: 'link-summary' });
      up.append(svg('rect', { x: width - 238, y: 154, width: 218, height: 33, rx: 7 }),
        svg('text', { x: width - 226, y: 175, class: 'svg-note' }, `↑ 梯度上传 · ${value(frame.row.round_upload_bytes)} B`));
      svgButton(up, '检查本轮梯度上传总量和安全元数据', 'upload', () => inspect('upload'));
      graph.append(down, up);
    }
    ids.forEach((id, i) => {
      const x = (width / (ids.length + 1)) * (i + 1);
      const actual = frame.action?.client_ids.includes(id) === true;
      const explored = preview?.client_ids.includes(id) === true;
      for (const [offset, direction, start, end] of [[-9, 'down', 180, 285], [9, 'up', 285, 180]] as const) {
        const path = svg('path', { d: `M${x + offset} ${start} L${x + offset} ${end}`, class: `client-link ${direction}${actual ? ' selected' : ''}`, 'marker-end': actual ? `url(#arrow-${direction})` : '' });
        const hit = svg('path', { d: `M${x + offset} 185 L${x + offset} 280`, class: 'link-hit' });
        svgButton(hit, `C${id} ${direction === 'up' ? '上传' : '下发'}链路 · ${actual ? '实际动作参与' : '未在实际动作中参与'}`, `link-${id}-${direction}`, () => inspect(direction === 'up' ? 'upload' : 'download'));
        graph.append(path, hit);
      }
      if (explored) graph.append(svg('path', { d: `M${x + 22} 180 L${x + 22} 281`, class: 'preview-link', 'marker-end': 'url(#arrow-preview)' }));
      const node = svg('g', { class: `edge-node${actual ? ' actual' : ''}`, 'data-client': id, 'data-actual': String(actual), 'data-preview': String(explored) });
      node.append(svg('rect', { x: x - 83, y: 288, width: 166, height: 91, rx: 12 }),
        svg('text', { x: x - 65, y: 314, class: 'svg-title' }, `C${id} / 边端客户端`),
        svg('text', { x: x - 65, y: 337, class: 'svg-note' }, actual ? `实际参与 · batch ${frame.action!.batch_size}` : frame.action ? '本轮未选中' : '参与动作未知'),
        svg('text', { x: x - 65, y: 360, class: 'svg-note' }, `累计参与 ${value(frame.row[`c${id}_participations`])} 次`));
      if (explored) node.append(svg('rect', { x: x - 88, y: 283, width: 176, height: 101, rx: 14, class: 'preview-outline' }));
      svgButton(node, `检查客户端 C${id} 的本轮上传和参与记录`, `client-${id}`, () => inspect(`client:${id}`)); graph.append(node);
      const upload = frame.uploads.filter(u => u.client_id === id);
      graph.append(svg('rect', { x: x - 66, y: 205, width: 132, height: 57, rx: 7, class: 'link-caption', 'pointer-events': 'none' }),
        svg('text', { x, y: 223, 'text-anchor': 'middle', class: 'svg-note', 'pointer-events': 'none' }, actual ? '本地计算' : frame.action ? '未参与' : '参与状态未知'),
        svg('text', { x, y: 248, 'text-anchor': 'middle', class: 'svg-note', 'pointer-events': 'none' }, upload.length ? `${upload.length} 条上传记录` : '无本轮上传记录'));
    });
    if (!ids.length) graph.append(svg('text', { x: width / 2, y: 265, 'text-anchor': 'middle', class: 'svg-note' }, '客户端身份未导出，不构造假想拓扑'));
    graph.append(svg('text', { x: width / 2, y: 403, 'text-anchor': 'middle', class: 'svg-footnote' }, '边端保留本地数据，只展示安全上传元数据 · 链路为机制示意，不代表真实时序'));
    host.append(graph);
  });
}

export function mountDecisionLab(host: HTMLElement, snapshot: Snapshot, initial: LabState, onChange: (state: LabState) => void, onReset: () => void): () => void {
  const run = preferredDecisionRun(snapshot.runs, initial.run);
  if (!run) { host.append(el('p', '该快照未导出验证运行，无法进行冻结策略回放。', 'empty')); return () => {}; }
  const state: LabState = { ...initial, run: run.run_id };
  const episodes = episodeKeys(run);
  if (!episodes.includes(state.episode)) state.episode = episodes[0] ?? 'unreported';
  const frames = replayFrames(run, state.episode, snapshot.audits);
  state.frame = Math.min(Math.max(0, Math.floor(state.frame)), Math.max(0, frames.length - 1));
  const actions = jointActions(run);
  if (!actions.some(a => a.action_id === state.preview)) state.preview = null;
  let timer: ReturnType<typeof setInterval> | undefined;
  let disposed = false;
  let inspectTarget: Inspect = 'agent';
  const notify = () => onChange({ ...state });
  const current = () => frames[state.frame];
  const preview = () => actions.find(a => a.action_id === state.preview) ?? null;

  const identity = el('div', undefined, 'lab-identity');
  identity.append(el('span', 'FROZEN POLICY / 冻结策略回放', 'mode-tag'), el('span', '实验记录 · 客户端模拟 · 不执行训练', 'muted'));
  host.append(identity, el('p', '不是看一张结果图，而是进入一次决策：检查记录、探索联合动作，再看真实执行和奖励。', 'lead'));
  const controls = panel('选择一段真实决策', '验证回放不更新策略；参考方法可能不是强化学习。切换记录不意味着同状态公平比较。');
  controls.classList.add('lab-controls');
  const runs = decisionRuns(snapshot.runs);
  const methodKey = (r: Run) => `${r.method}${r.arm ? ` / ${r.arm}` : ''}`;
  const methods = [...new Set(runs.map(methodKey))];
  controls.append(select('策略 / 方法', methods.map(m => [m, m]), methodKey(run), method => {
    const next = preferredDecisionRun(runs.filter(r => methodKey(r) === method), '');
    if (!next) return;
    pause(); onChange({ run: next.run_id, episode: '', frame: 0, preview: null }); onReset();
  }), select('运行 / 种子 / 检查点', runs.filter(r => methodKey(r) === methodKey(run)).map(r => [r.run_id, runLabel(r)]), run.run_id, id => {
    pause(); onChange({ run: id, episode: '', frame: 0, preview: null }); onReset();
  }), select('Episode', episodes.map(e => [e, e === 'unreported' ? '未导出 · 汇总记录' : `Episode ${e}`]), state.episode, ep => {
    pause(); onChange({ ...state, episode: ep, frame: 0, preview: null }); onReset();
  }));
  const runIdentity = el('details', undefined, 'run-identity');
  runIdentity.append(el('summary', `train ${value(run.train_seed)} / eval ${value(run.eval_seed)} · 训练预算 ${value(run.config_summary.training_interaction_budget)} · ${run.partition}`),
    el('code', run.run_id), el('p', `风险版本：${value(run.risk_version)} · evidence_scope：${run.evidence_scope}`, 'muted'));
  controls.append(runIdentity); host.append(controls);

  const loop = el('div', undefined, 'rl-loop');
  const stages: [Inspect, string, string][] = [['state', '01 / STATE', '状态与记录边界'], ['agent', '02 / AGENT', '策略如何评分'], ['action', '03 / ACTION', '客户端 × batch'], ['cloud', '04 / ENVIRONMENT', '云边协同执行'], ['reward', '05 / REWARD', '奖励与下一轮反馈']];
  for (const [target, label, text] of stages) {
    const b = button('', () => openInspector(target)); b.append(el('small', label), el('span', text)); b.dataset.focusKey = `loop-${target}`; loop.append(b);
  }
  host.append(loop, el('p', '↺ 环境反馈 → 下一轮状态；验证时没有策略更新。点击任一环节查看来源。', 'loop-caption'));

  const timeline = el('section', undefined, 'lab-timeline');
  const previous = button('← 上一条', () => go(state.frame - 1));
  const play = button('播放记录', () => {
    if (timer !== undefined) { pause(); update(); return; }
    if (state.frame >= frames.length - 1) state.frame = 0;
    timer = setInterval(() => {
      if (state.frame >= frames.length - 1) { pause(); update(); return; }
      state.frame++; state.preview = null; update();
      if (state.frame >= frames.length - 1) { pause(); update(); }
    }, 1000);
    update();
  });
  const next = button('下一条 →', () => go(state.frame + 1));
  const roundLabel = el('output', undefined, 'round-output'); roundLabel.setAttribute('aria-live', 'off');
  const slider = el('input'); slider.type = 'range'; slider.min = '0'; slider.max = String(Math.max(0, frames.length - 1)); slider.step = '1'; slider.value = String(state.frame); slider.setAttribute('aria-label', '按记录索引选择回放轮次'); slider.oninput = () => go(Number(slider.value));
  const timelineButtons = el('div', undefined, 'controls'); timelineButtons.append(previous, play, next, roundLabel);
  timeline.append(timelineButtons, slider, el('small', '按实际记录逐条回放，不补齐未记录轮次；播放速度不是实际计算或通信耗时。'));
  host.append(timeline);

  const grid = el('div', undefined, 'lab-grid');
  const statePanel = panel('State / 状态', '不以当前轮结果冒充决策前状态。'); statePanel.classList.add('state-panel');
  const stateBody = el('div'); statePanel.append(stateBody, button('检查状态来源', () => openInspector('state')));
  const environment = panel('Action → Environment', '点击云端、客户端或链路，检查当前轮记录。'); environment.classList.add('environment-panel');
  const graphHost = el('div', undefined, 'graph-host'); environment.append(graphHost, el('p', '窄屏可横向滚动查看全部云边节点；也可按 Tab 选择节点并按 Enter 查看记录。', 'graph-scroll-note'));
  const legend = el('div', undefined, 'graph-legend'); legend.append(el('span', '实线 / 实际动作', 'legend-actual'), el('span', '虚线 / 候选预览', 'legend-preview'), el('span', '灰色 / 未参与或未知', 'legend-idle')); environment.append(legend);
  const feedback = panel('Reward / 真实反馈', '只显示实际动作执行后的记录。'); feedback.classList.add('reward-panel');
  const feedbackBody = el('div'); feedback.append(feedbackBody, button('查看奖励与证据', () => openInspector('reward')));
  grid.append(statePanel, environment, feedback); host.append(grid);

  const decision = panel('联合动作实验区', '点击一个候选动作，预览客户端与通信路径。评分是策略估计，不是隐私风险，也不是替代动作已执行的结果。');
  decision.classList.add('decision-panel');
  const decisionHeader = el('div', undefined, 'decision-heading');
  const scoreHeading = el('div'); const actualLabel = el('strong'); const scoreNote = el('p', undefined, 'muted');
  scoreHeading.append(actualLabel, scoreNote);
  const resetPreview = button('清除候选预览', () => { state.preview = null; update(); });
  decisionHeader.append(scoreHeading, resetPreview); decision.append(decisionHeader);
  const decisionGrid = el('div', undefined, 'decision-grid');
  const matrixHost = el('div', undefined, 'action-matrix-host');
  const matrixButtons = new Map<number, { button: HTMLButtonElement; score: HTMLElement; badge: HTMLElement }>();
  if (actions.length) {
    const batches = [...new Set(actions.map(a => a.batch_size))].sort((a, b) => a - b);
    const groups = [...new Set(actions.map(a => clients(a)))];
    const wrap = el('div', undefined, 'table-wrap'); const matrix = el('table', undefined, 'action-matrix');
    const thead = el('thead'); const heading = el('tr'); heading.append(el('th', '客户端集合 × batch'));
    for (const batch of batches) { const th = el('th', `batch ${batch}`); th.scope = 'col'; heading.append(th); } thead.append(heading); matrix.append(thead);
    const tbody = el('tbody');
    for (const group of groups) {
      const tr = el('tr'); const th = el('th', group); th.scope = 'row'; tr.append(th);
      for (const batch of batches) {
        const td = el('td'); const matches = actions.filter(a => clients(a) === group && a.batch_size === batch);
        if (!matches.length) td.append(el('span', '无此动作', 'muted'));
        for (const action of matches) {
          const b = button('', () => { pause(); state.preview = action.action_id; update(); }); b.classList.add('action-cell'); b.dataset.action = String(action.action_id); b.dataset.focusKey = `action-${action.action_id}`;
          const score = el('strong'); const badge = el('small'); b.append(el('span', `A${action.action_id}`), score, badge); td.append(b); matrixButtons.set(action.action_id, { button: b, score, badge });
        }
        tr.append(td);
      }
      tbody.append(tr);
    }
    matrix.append(tbody); wrap.append(matrix); matrixHost.append(wrap, el('p', '窄屏可横向滚动动作矩阵，查看全部 batch 选项。', 'matrix-scroll-note'));
  } else matrixHost.append(el('p', '该运行没有动作映射。不能从 action_id 推测 batch 或客户端。', 'empty'));
  const rankingHost = el('div', undefined, 'action-ranking'); decisionGrid.append(matrixHost, rankingHost); decision.append(decisionGrid);
  const previewSummary = el('div', undefined, 'preview-summary'); previewSummary.setAttribute('aria-live', 'polite'); decision.append(previewSummary);
  const warnings = el('div', undefined, 'lab-warnings'); decision.append(warnings); host.append(decision);

  const tracesHeader = el('div', undefined, 'section-heading'); tracesHeader.append(el('h2', '这一次决策之后，发生了什么？'), el('p', '同一 episode 的已记录结果 · 点击曲线数据点同步回放；键盘可操作上方时间轴。预热记录保留但明确标注。', 'muted'));
  const traces = el('div', undefined, 'chart-grid lab-traces'); host.append(tracesHeader, traces);
  const syncCharts = replayTraces(traces, frames, run, go);
  const provenance = el('details', undefined, 'provenance'); provenance.append(el('summary', '记录来源与交互边界'),
    el('p', '主画布由 SVG 与真实快照记录生成，不读取论文图片。模拟时间和能量不是设备在线计量。Q 值不构成因果解释；未执行的候选动作没有替代准确率、能耗或审计真值。'), records('运行来源', run.sources), records('快照 / 运行警告', [...snapshot.warnings, ...run.warnings])); host.append(provenance);

  const dialog = el('dialog', undefined, 'inspector'); dialog.setAttribute('aria-labelledby', 'inspector-title');
  const dialogHead = el('div', undefined, 'inspector-heading'); const dialogTitle = el('h2'); dialogTitle.id = 'inspector-title';
  const close = button('关闭 ×', () => dialog.close()); close.setAttribute('aria-label', '关闭证据详情'); dialogHead.append(dialogTitle, close);
  const dialogBody = el('div'); dialog.append(dialogHead, dialogBody); host.append(dialog);

  function pause(): void { if (timer !== undefined) clearInterval(timer); timer = undefined; play.textContent = '播放记录'; }
  function go(index: number): void { if (disposed || !frames.length) return; pause(); state.frame = Math.min(Math.max(0, index), frames.length - 1); state.preview = null; update(); }
  function openInspector(target: Inspect): void { pause(); inspectTarget = target; update(); const frame = current(); if (!frame) return; fillInspector(frame); if (!dialog.open) dialog.showModal(); }
  function fillInspector(frame: ReplayFrame): void {
    const id = inspectTarget.startsWith('client:') ? Number(inspectTarget.slice(7)) : null;
    dialogTitle.textContent = id !== null ? `C${id} / 上传与参与证据` : ({ state: 'State / 决策前信息边界', agent: 'Agent / 策略评分', action: 'Action / 实际与候选', cloud: 'Environment / 联邦更新', reward: 'Reward / 记录与风险证据', upload: '梯度上传 / 本轮总量', download: '模型下发 / 本轮总量' } as Record<string, string>)[inspectTarget] ?? '记录来源';
    dialogBody.replaceChildren(el('p', `run ${run!.run_id} · episode ${value(frame.episode)} · round ${value(frame.round)}`, 'muted'));
    if (inspectTarget === 'state') dialogBody.append(el('p', '完整历史 observation（资源、参与窗口、候选成本）未导出。下列上一轮反馈仅是参考记录，不等同于策略当时看到的完整状态。', 'empty'), records('上一轮已记录反馈', frame.previous ? [frame.previous] : []));
    else if (inspectTarget === 'agent') dialogBody.append(el('p', `${frame.scoreLabel}；按 action_id 关联向量。选择动作不一定等于最高评分，不能据此推断探索机制。`, 'muted'), records('本轮原始决策', frame.decisions));
    else if (inspectTarget === 'action') dialogBody.append(records('实际动作 / 候选预览', [{ actual: frame.action, preview: preview(), preview_executed: false }]), el('p', '候选预览不改变真实记录；没有计算反事实结果。', 'empty'));
    else if (inspectTarget === 'download') dialogBody.append(el('p', `本轮总下发 ${value(frame.row.round_download_bytes)} B；没有单客户端通信量，不能平均分摊。`, 'empty'));
    else if (inspectTarget === 'upload' || id !== null) {
      const uploads = id === null ? frame.uploads : frame.uploads.filter(u => u.client_id === id);
      dialogBody.append(el('p', `本轮总上传 ${value(frame.row.round_upload_bytes)} B。客户端只展示安全元数据；不展示样本内容或梯度载荷。`, 'muted'),
        table(['上传 ID', '客户端', 'batch', '预测风险', '不确定性', '决策类型'], uploads.map(u => [u.upload_id, u.client_id, u.batch_size, u.predicted_risk, u.uncertainty, u.decision_kind])), records('本轮上传原始元数据', uploads));
      if (id !== null) dialogBody.append(table(['累计参与次数', '累计样本计数', '累计 unique 计数'], [[frame.row[`c${id}_participations`], frame.row[`c${id}_samples`], frame.row[`c${id}_unique`]]]));
    }
    if (inspectTarget === 'reward' || inspectTarget === 'cloud') dialogBody.append(el('p', '奖励仅使用记录 reward。分项与历史权重未导出，不依据当前配置重新计算。预测风险、不确定性和审计真值分开解释。', 'empty'),
      table(['奖励', '协议', '准确率', '损失', '预测暴露风险', '上传预测均值'], [[frame.row.reward, frame.row.reward_protocol, frame.row.validation_accuracy, frame.row.validation_loss, frame.row.predicted_exposure_risk, frame.row.predicted_risk]]));
    dialogBody.append(el('h3', '实际审计标签 / 独立证据'), el('p', frame.audits.length ? `${frame.audits.length} 条精确匹配的审计记录；不同 scope / model 保持分开，不能推出完整策略安全。` : '本轮无可精确匹配的审计标签。预测不替代真值；没有浏览器生成的通过结论。', 'muted'), records('按 run_id + 唯一 upload_id 匹配', id === null ? frame.audits : frame.audits.filter(a => frame.uploads.some(u => u.client_id === id && u.upload_id === a.upload_id))),
      records('当前指标与来源', [{ observation: frame.row, sources: run!.sources }]), records('关联限制', frame.warnings));
  }

  function update(): void {
    if (disposed) return;
    const frame = current();
    previous.disabled = !frames.length || state.frame === 0; next.disabled = !frames.length || state.frame === frames.length - 1;
    play.disabled = frames.length < 2; slider.disabled = !frames.length; slider.value = String(state.frame);
    play.textContent = timer !== undefined ? '暂停回放' : '播放记录';
    resetPreview.disabled = state.preview === null;
    if (!frame) { stateBody.replaceChildren(el('p', '该运行没有逐轮指标记录。', 'empty')); graphHost.replaceChildren(el('p', '没有回放帧，不构造云边执行状态。', 'empty')); feedbackBody.replaceChildren(); notify(); return; }
    roundLabel.textContent = `Episode ${value(frame.episode)} · Round ${value(frame.round)} · ${state.frame + 1} / ${frames.length} · ${status(frame.row)}`;
    slider.setAttribute('aria-valuetext', roundLabel.textContent);
    stateBody.replaceChildren(el('span', '历史 observation 未导出', 'missing-tag'), el('p', '资源状态 / 参与窗口 / 候选成本', 'muted'));
    if (frame.previous) {
      stateBody.append(el('h3', `上一轮 ${value(frame.previous.round)} 的反馈参考`));
      stat(stateBody, '上一轮准确率', frame.previous.validation_accuracy);
      stat(stateBody, '上一轮累计模拟时间 · s', frame.previous.elapsed_seconds);
      stat(stateBody, '上一轮累计模拟能量 · J', frame.previous.energy_joules);
    } else stateBody.append(el('p', '无唯一相邻上一轮记录，不能重建决策前状态。', 'empty'));
    stateBody.append(el('p', `当前记录：round ${value(frame.round)} / 上限 ${value(run!.config_summary.max_rounds)}`, 'muted'));
    drawGraph(graphHost, run!, frame, preview(), timer !== undefined, openInspector);
    feedbackBody.replaceChildren();
    stat(feedbackBody, '即时奖励 rₜ', frame.row.reward, '记录总值 · 不拆造分项');
    stat(feedbackBody, '验证准确率', frame.row.validation_accuracy);
    stat(feedbackBody, '本轮模拟时间 · s', frame.row.round_time_seconds);
    stat(feedbackBody, '本轮模拟能量 · J', frame.row.round_energy_joules);
    stat(feedbackBody, '预测暴露风险', frame.row.predicted_exposure_risk, '非真实审计风险');
    feedbackBody.append(el('span', frame.audits.length ? `${frame.audits.length} 条精确匹配审计记录` : '本轮审计标签缺失', 'audit-tag'));
    actualLabel.textContent = frame.action ? `实际执行：${actionName(frame.action)}` : `实际 action_id ${value(frame.row.action_id)} · 映射未导出`;
    scoreNote.textContent = `${frame.scoreLabel} · 记录决策间隔 ${value(frame.decision?.margin)} · ${status(frame.row)}`;
    const ranked = actionRanking(frame); const low = ranked.at(-1)?.value ?? 0; const high = ranked[0]?.value ?? 0;
    for (const action of actions) {
      const cell = matrixButtons.get(action.action_id)!; const score = frame.scores.get(action.action_id) ?? null;
      const actual = frame.action?.action_id === action.action_id; const explored = state.preview === action.action_id;
      cell.button.classList.toggle('actual', actual); cell.button.classList.toggle('preview', explored);
      cell.button.setAttribute('aria-pressed', String(explored));
      cell.button.setAttribute('aria-label', `${actionName(action)}，${frame.scoreLabel} ${value(score)}${actual ? '，实际执行' : ''}${explored ? '，候选预览' : ''}`);
      cell.score.textContent = value(score); cell.badge.textContent = [actual ? '实际' : '', explored ? '预览' : ''].filter(Boolean).join(' / ') || '探索路径';
      cell.button.style.setProperty('--score-strength', String(score === null ? 0 : high === low ? .1 : .04 + (score - low) / (high - low) * .14));
    }
    preserveFocus(rankingHost, () => {
      rankingHost.replaceChildren(el('h3', '当前状态的候选排名'));
      if (!ranked.length) rankingHost.append(el('p', '评分未导出，仍可预览映射路径，但不能比较策略偏好。', 'empty'));
      for (const [i, entry] of ranked.slice(0, 3).entries()) {
        const action = actions.find(a => a.action_id === entry.action_id)!;
        const b = button('', () => { pause(); state.preview = entry.action_id; update(); }); b.classList.add('ranking-row'); b.dataset.focusKey = `rank-${entry.action_id}`;
        b.append(el('span', `${i + 1}`), el('span', `A${entry.action_id} · ${clients(action)}\nbatch ${action.batch_size}`), el('strong', value(entry.value))); rankingHost.append(b);
      }
      rankingHost.append(el('p', '排名按当前记录的评分排序；不跨方法、轮次比较绝对 Q 值。', 'muted'));
    });
    const explored = preview(); previewSummary.replaceChildren();
    if (explored) {
      const actualScore = frame.action ? frame.scores.get(frame.action.action_id) ?? null : null;
      const previewScore = frame.scores.get(explored.action_id) ?? null;
      previewSummary.append(el('strong', `候选预览：${actionName(explored)}`), el('span', `${frame.scoreLabel} ${value(previewScore)} · 与实际动作评分差 ${value(previewScore !== null && actualScore !== null ? previewScore - actualScore : null)}`),
        el('small', '未执行 · 不生成替代准确率、时间、能量或风险结果。右侧反馈仍属于实际动作。'));
    } else previewSummary.append(el('span', '点击矩阵或候选排名，探索另一条云边协同路径；实线实际动作保持不变。'));
    warnings.replaceChildren(); frame.warnings.forEach(w => warnings.append(el('p', w, 'muted')));
    syncCharts(state.frame); if (dialog.open) fillInspector(frame); notify();
  }
  update();
  return () => { disposed = true; pause(); if (dialog.open) dialog.close(); dialog.remove(); };
}
