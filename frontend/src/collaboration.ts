import { number, type Run, type Snapshot, type Row } from './model.ts';
import { jointActions, episodeKeys, preferredDecisionRun, decisionRuns, replayRunLabel, hasReplay, replayFrames, actionRanking, type ReplayFrame, type JointAction } from './replay.ts';
import { graphGeometry, buildParticleScene, mountParticles, phaseLabels, type Phase, type Point } from './particles.ts';
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

function mountGraph(host: HTMLElement, run: Run, inspect: (target: Inspect) => void, onPhase: (phase: Phase) => void) {
  const geometry = graphGeometry(run), { width, height } = geometry;
  const scene = el('div', undefined, 'graph-scene');
  const graph = svg('svg', { viewBox: `0 0 ${width} ${height}`, class: 'collaboration-graph', 'aria-label': 'RL 调度器选择客户端与批量大小，边端本地计算并上传，云端聚合产生奖励反馈' });
  const pathOf = (points: Point[]) => points.map((p, i) => `${i ? 'L' : 'M'}${p.x} ${p.y}`).join(' ');
  const centered = (x: number, y: number, cls: string, text = '') => svg('text', { x, y, class: cls, 'text-anchor': 'middle' }, text);
  const defs = svg('defs');
  for (const [name, fill] of [['up', '#65d4b7'], ['down', '#7abcf0'], ['preview', '#efb766']] as const) {
    const marker = svg('marker', { id: `arrow-${name}`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' });
    marker.append(svg('path', { d: 'M 0 0 L 10 5 L 0 10 Z', fill })); defs.append(marker);
    const gradient = svg('radialGradient', { id: `glow-${name}` });
    gradient.append(svg('stop', { offset: '0%', 'stop-color': fill, 'stop-opacity': .27 }), svg('stop', { offset: '100%', 'stop-color': fill, 'stop-opacity': 0 })); defs.append(gradient);
  }
  graph.append(defs, svg('rect', { x: 12, y: 10, width: width - 24, height: 204, rx: 20, class: 'cloud-zone' }),
    svg('text', { x: 32, y: 37, class: 'svg-zone-label' }, 'CLOUD / 策略与全局模型'));
  const node = (point: Point, cls: string, glyph: string) => {
    const group = svg('g', { class: cls });
    group.append(svg('circle', { cx: point.x, cy: point.y, r: 64, fill: 'url(#glow-down)', class: 'node-halo' }),
      svg('circle', { cx: point.x, cy: point.y, r: 43, class: 'node-orbit' }),
      svg('circle', { cx: point.x, cy: point.y, r: 30, class: 'node-core' }), centered(point.x, point.y + 7, 'node-glyph', glyph));
    return group;
  };
  const cloudNodes = [
    { point: geometry.agent, target: 'agent' as const, glyph: 'π', label: `${run.method.toUpperCase()}${run.arm ? ` / ${run.arm}` : ''} 调度器` },
    { point: geometry.action, target: 'action' as const, glyph: 'aₜ', label: '联合动作' },
    { point: geometry.cloud, target: 'cloud' as const, glyph: 'θ', label: '全局模型聚合' },
  ].map(spec => {
    const group = node(spec.point, `cloud-node node-${spec.target}`, spec.glyph);
    const title = centered(spec.point.x, 173, 'svg-title', spec.label), sub = centered(spec.point.x, 193, 'svg-note'); group.append(title, sub);
    svgButton(group, `检查${spec.label}`, spec.target, () => inspect(spec.target)); graph.append(group);
    return { ...spec, group, title, sub };
  });
  graph.append(svg('path', { d: pathOf([{ x: geometry.agent.x + 48, y: 105 }, { x: geometry.action.x - 48, y: 105 }]), class: 'policy-link', 'marker-end': 'url(#arrow-down)' }));
  const edges = geometry.edges.map(edge => {
    const paths = (['down', 'up'] as const).map(direction => {
      const path = svg('path', { d: pathOf(edge[direction]), class: `client-link ${direction}` });
      const hit = svg('path', { d: pathOf(edge[direction]), class: 'link-hit' });
      svgButton(hit, `检查 C${edge.id} ${direction === 'up' ? '上传' : '下发'}链路`, `link-${edge.id}-${direction}`, () => inspect(direction === 'up' ? 'upload' : 'download')); graph.append(path, hit);
      return { direction, path };
    });
    const group = node(edge.center, 'edge-node', `C${edge.id}`); group.dataset.client = String(edge.id);
    const title = centered(edge.center.x, 391, 'svg-title'), participation = centered(edge.center.x, 413, 'svg-note'), upload = centered(edge.center.x, 435, 'svg-note'); group.append(title, participation, upload);
    svgButton(group, `检查客户端 C${edge.id} 的本轮上传和参与记录`, `client-${edge.id}`, () => inspect(`client:${edge.id}`)); graph.append(group);
    return { ...edge, group, title, participation, upload, paths };
  });
  const down = svg('g', { class: 'link-summary' }), up = svg('g', { class: 'link-summary' });
  const downText = centered(width * .25, 474, 'svg-note'), upText = centered(width * .75, 474, 'svg-note');
  down.append(downText); up.append(upText);
  svgButton(down, '检查本轮模型下发总量', 'download', () => inspect('download'));
  svgButton(up, '检查本轮梯度上传总量和安全元数据', 'upload', () => inspect('upload'));
  graph.append(down, up, centered(width / 2, 502, 'svg-footnote', '机制示意 · 安全上传元数据'));
  if (!edges.length) graph.append(centered(width / 2, 330, 'svg-note', '客户端身份未导出'));
  scene.append(graph); host.append(scene);
  const particles = mountParticles(scene, onPhase);
  return {
    update(frame: ReplayFrame, preview: JointAction | null, playing: boolean) {
      cloudNodes[0]!.sub.textContent = `${frame.scoreLabel}${number(run.config_summary.gamma) === null ? '' : ` · γ ${value(run.config_summary.gamma)}`}`;
      cloudNodes[1]!.title.textContent = frame.action ? `联合动作 A${frame.action.action_id}` : '动作映射缺失';
      cloudNodes[1]!.sub.textContent = frame.action ? `${clients(frame.action)} · batch ${frame.action.batch_size}` : `action_id ${value(frame.row.action_id)}`;
      cloudNodes[2]!.sub.textContent = `准确率 ${value(frame.row.validation_accuracy)}`;
      downText.textContent = `↓ 模型下发 · ${value(frame.row.round_download_bytes)} B`;
      upText.textContent = `↑ 梯度上传 · ${value(frame.row.round_upload_bytes)} B`;
      graph.querySelectorAll('.preview-link, .preview-outline').forEach(n => n.remove());
      for (const edge of edges) {
        const actual = frame.action?.client_ids.includes(edge.id) === true, explored = preview?.client_ids.includes(edge.id) === true;
        edge.group.classList.toggle('actual', actual); edge.group.dataset.actual = String(actual); edge.group.dataset.preview = String(explored);
        edge.group.querySelector('.node-halo')!.setAttribute('fill', actual ? 'url(#glow-up)' : 'url(#glow-down)');
        edge.title.textContent = actual ? `实际参与 · batch ${frame.action!.batch_size}` : frame.action ? '本轮未选中' : '参与动作未知';
        edge.participation.textContent = `累计参与 ${value(frame.row[`c${edge.id}_participations`])} 次`;
        const uploads = frame.uploads.filter(u => u.client_id === edge.id);
        edge.upload.textContent = uploads.length ? `${uploads.length} 条上传记录` : '无本轮上传记录';
        for (const { path, direction } of edge.paths) { path.classList.toggle('selected', actual); path.setAttribute('marker-end', actual ? `url(#arrow-${direction})` : ''); }
        if (explored) {
          graph.insertBefore(svg('path', { d: pathOf(edge.preview), class: 'preview-link', 'marker-end': 'url(#arrow-preview)', 'pointer-events': 'none' }), edge.group);
          edge.group.append(svg('circle', { cx: edge.center.x, cy: edge.center.y, r: 51, class: 'preview-outline', 'pointer-events': 'none' }));
        }
      }
      particles.update(buildParticleScene(run, frame), playing, String(frame.index));
    },
    highlight(clientIds: number[]) { for (const edge of edges) edge.group.classList.toggle('hover-linked', clientIds.includes(edge.id)); },
    setEnabled: particles.setEnabled,
    dispose() { particles.dispose(); scene.remove(); },
  };
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
  identity.append(el('span', 'POLICY REPLAY / 策略回放', 'mode-tag'));
  identity.append(el('span', typeof run.replay_source_label === 'string' ? `${run.replay_source_label} · ${frames.length} 轮` : hasReplay(run) ? '逐轮验证记录' : '仅汇总记录', `mode-tag${run.replay_source_label ? ' historical-tag' : ''}`));
  host.append(identity);
  const controls = panel('决策回放');
  controls.classList.add('lab-controls');
  const runs = decisionRuns(snapshot.runs);
  const methodKey = (r: Run) => `${r.method}${r.arm ? ` / ${r.arm}` : ''}`;
  const methods = [...new Set(runs.map(methodKey))];
  controls.append(select('策略 / 方法', methods.map(m => [m, m]), methodKey(run), method => {
    const next = preferredDecisionRun(runs.filter(r => methodKey(r) === method), '');
    if (!next) return;
    pause(); onChange({ run: next.run_id, episode: '', frame: 0, preview: null }); onReset();
  }), select('运行 / 种子 / 检查点', runs.filter(r => methodKey(r) === methodKey(run)).sort((a, b) => Number(hasReplay(b)) - Number(hasReplay(a))).map(r => [r.run_id, replayRunLabel(r)]), run.run_id, id => {
    pause(); onChange({ run: id, episode: '', frame: 0, preview: null }); onReset();
  }), select('Episode', episodes.map(e => [e, e === 'unreported' ? '未导出 · 汇总记录' : `Episode ${e}`]), state.episode, ep => {
    pause(); onChange({ ...state, episode: ep, frame: 0, preview: null }); onReset();
  }));
  const runIdentity = el('details', undefined, 'run-identity');
  runIdentity.append(el('summary', `train ${value(run.train_seed)} / eval ${value(run.eval_seed)} · 训练预算 ${value(run.config_summary.training_interaction_budget)} · ${run.partition}`),
    el('code', run.run_id), el('p', `风险版本：${value(run.risk_version)} · evidence_scope：${run.evidence_scope}`, 'muted'));
  controls.append(runIdentity); host.append(controls);

  const loop = el('div', undefined, 'rl-loop');
  const stages: [Inspect, string, string][] = [['state', '01 / STATE', '环境状态'], ['agent', '02 / AGENT', '策略评分'], ['action', '03 / ACTION', '客户端 × batch'], ['cloud', '04 / ENVIRONMENT', '云边协同执行'], ['reward', '05 / REWARD', '奖励反馈']];
  for (const [target, label, text] of stages) {
    const b = button('', () => openInspector(target)); b.append(el('small', label), el('span', text)); b.dataset.focusKey = `loop-${target}`; loop.append(b);
  }
  host.append(loop, el('p', '环境状态 → 策略决策 → 联合调度 → 云边执行 → 奖励反馈', 'loop-caption'));

  const timeline = el('section', undefined, 'lab-timeline');
  const previous = button('← 上一条', () => go(state.frame - 1));
  const play = button('播放记录', () => {
    if (timer !== undefined) { pause(); update(); return; }
    if (state.frame >= frames.length - 1) state.frame = 0;
    state.preview = null;
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
  timeline.append(timelineButtons, slider, el('small', '轮次回放 · 1 秒 / 条'));
  host.append(timeline);

  const grid = el('div', undefined, 'lab-grid');
  const statePanel = panel('State / 状态', '上一轮反馈与当前轮次'); statePanel.classList.add('state-panel');
  const stateBody = el('div'); statePanel.append(stateBody, button('检查状态来源', () => openInspector('state')));
  const environment = panel('Action → Environment', '点击云端、客户端或链路，检查当前轮记录。'); environment.classList.add('environment-panel');
  const graphHost = el('div', undefined, 'graph-host'); environment.append(graphHost, el('p', '窄屏可横向滚动查看全部云边节点；也可按 Tab 选择节点并按 Enter 查看记录。', 'graph-scroll-note'));
  const legend = el('div', undefined, 'graph-legend'); legend.append(el('span', '实线 / 实际动作', 'legend-actual'), el('span', '虚线 / 候选预览', 'legend-preview'), el('span', '灰色 / 未参与或未知', 'legend-idle')); environment.append(legend);
  const graphStatus = el('div', undefined, 'graph-status'); const phaseLabel = el('span', '静态回放', 'phase-label');
  let motionEnabled = true;
  const motionButton = button('关闭动效', () => { motionEnabled = !motionEnabled; motionButton.textContent = motionEnabled ? '关闭动效' : '开启动效'; motionButton.setAttribute('aria-pressed', String(motionEnabled)); graphView.setEnabled(motionEnabled); });
  motionButton.setAttribute('aria-pressed', 'true'); graphStatus.append(phaseLabel, motionButton); environment.append(graphStatus);
  const graphView = mountGraph(graphHost, run, openInspector, phase => {
    if (disposed) return;
    phaseLabel.textContent = phaseLabels[phase];
    const target = phase === 'agent' ? 'agent' : phase === 'reward' ? 'reward' : phase === 'static' ? '' : 'cloud';
    loop.querySelectorAll('button').forEach(b => b.classList.toggle('phase-active', b.dataset.focusKey === `loop-${target}`));
    feedback.classList.toggle('phase-active', phase === 'reward');
  });
  const feedback = panel('Reward / 反馈', '当前轮执行结果'); feedback.classList.add('reward-panel');
  const feedbackBody = el('div'); feedback.append(feedbackBody, button('查看奖励与证据', () => openInspector('reward')));
  grid.append(statePanel, environment, feedback); host.append(grid);

  const decision = panel('联合动作', '选择客户端组合与批量大小，预览协同路径。');
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
           b.addEventListener('pointerenter', () => graphView.highlight(action.client_ids)); b.addEventListener('pointerleave', () => graphView.highlight([]));
           b.addEventListener('focus', () => graphView.highlight(action.client_ids)); b.addEventListener('blur', () => graphView.highlight([]));
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

  const tracesHeader = el('div', undefined, 'section-heading'); tracesHeader.append(el('h2', '轮次反馈'), el('p', '点击曲线数据点，查看对应轮次。', 'muted'));
  const traces = el('div', undefined, 'chart-grid lab-traces'); host.append(tracesHeader, traces);
  const syncCharts = replayTraces(traces, frames, run, go);
  const provenance = el('details', undefined, 'provenance'); provenance.append(el('summary', '记录来源与交互边界'),
    el('p', '时间与能量为模拟记录。候选预览仅展示路径与策略评分，反馈指标对应实际执行动作。'), records('运行来源', run.sources), records('快照 / 运行警告', [...snapshot.warnings, ...run.warnings])); host.append(provenance);

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
    if (!frame) { graphView.dispose(); stateBody.replaceChildren(el('p', '该运行没有逐轮指标记录。', 'empty')); graphHost.replaceChildren(el('p', '没有回放帧，不构造云边执行状态。', 'empty')); feedbackBody.replaceChildren(); notify(); return; }
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
    graphView.update(frame, preview(), timer !== undefined);
    feedbackBody.replaceChildren();
    stat(feedbackBody, '即时奖励 rₜ', frame.row.reward);
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
      rankingHost.append(el('p', '当前轮策略评分 · Top 3', 'muted'));
    });
    const explored = preview(); previewSummary.replaceChildren();
    if (explored) {
      const actualScore = frame.action ? frame.scores.get(frame.action.action_id) ?? null : null;
      const previewScore = frame.scores.get(explored.action_id) ?? null;
      previewSummary.append(el('strong', `候选预览：${actionName(explored)}`), el('span', `${frame.scoreLabel} ${value(previewScore)} · 与实际动作评分差 ${value(previewScore !== null && actualScore !== null ? previewScore - actualScore : null)}`),
        el('small', '未执行的候选路径 · 反馈面板显示实际动作结果。'));
    } else previewSummary.append(el('span', '选择候选动作预览路径 · 实线为实际动作，虚线为候选路径。'));
    warnings.replaceChildren(); frame.warnings.forEach(w => warnings.append(el('p', w, 'muted')));
    syncCharts(state.frame); if (dialog.open) fillInspector(frame); notify();
  }
  const suspend = () => { if (document.hidden && timer !== undefined) { pause(); update(); } };
  document.addEventListener('visibilitychange', suspend);
  update();
  return () => { disposed = true; pause(); document.removeEventListener('visibilitychange', suspend); graphView.dispose(); if (dialog.open) dialog.close(); dialog.remove(); };
}
