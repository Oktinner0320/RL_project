import { number, type Run, type Snapshot, type Row } from './model.ts';
import { jointActions, episodeKeys, preferredDecisionRun, decisionRuns, replayRunLabel, hasReplay, replayFrames, actionRanking, type ReplayFrame, type JointAction } from './replay.ts';
import { graphGeometry, buildParticleScene, mountParticles, phaseLabels, type Phase, type Point } from './particles.ts';
import { replayTraces, color } from './charts.ts';
import { el, button, select, panel, records, format, table } from './ui.ts';

export interface LabState { run: string; episode: string; frame: number; preview: number | null }
type Inspect = 'state' | 'agent' | 'action' | 'cloud' | 'reward' | 'upload' | 'download' | `client:${number}`;
const NS = 'http://www.w3.org/2000/svg';
const value = (v: unknown): string => v === null || v === undefined ? 'Not exported' : format(v);
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
  if (row.is_warmup === true) return 'Warmup · not learned-policy scores';
  if (row.is_warmup !== false) return 'Warmup status not exported';
  return 'Post-warmup record';
}

function mountGraph(host: HTMLElement, run: Run, inspect: (target: Inspect) => void, onPhase: (phase: Phase) => void) {
  const geometry = graphGeometry(run), { width, height } = geometry;
  const scene = el('div', undefined, 'graph-scene');
  const graph = svg('svg', { viewBox: `0 0 ${width} ${height}`, class: 'collaboration-graph', 'aria-label': 'The RL scheduler selects clients and batch size; edge clients compute and upload, and cloud aggregation produces reward feedback' });
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
    svg('text', { x: 32, y: 37, class: 'svg-zone-label' }, 'CLOUD / POLICY & GLOBAL MODEL'));
  const node = (point: Point, cls: string, glyph: string) => {
    const group = svg('g', { class: cls });
    group.append(svg('circle', { cx: point.x, cy: point.y, r: 64, fill: 'url(#glow-down)', class: 'node-halo' }),
      svg('circle', { cx: point.x, cy: point.y, r: 43, class: 'node-orbit' }),
      svg('circle', { cx: point.x, cy: point.y, r: 30, class: 'node-core' }), centered(point.x, point.y + 7, 'node-glyph', glyph));
    return group;
  };
  const cloudNodes = [
    { point: geometry.agent, target: 'agent' as const, glyph: 'π', label: `${run.method.toUpperCase()}${run.arm ? ` / ${run.arm}` : ''} scheduler` },
    { point: geometry.action, target: 'action' as const, glyph: 'aₜ', label: 'Joint action' },
    { point: geometry.cloud, target: 'cloud' as const, glyph: 'θ', label: 'Global aggregation' },
  ].map(spec => {
    const group = node(spec.point, `cloud-node node-${spec.target}`, spec.glyph);
    const title = centered(spec.point.x, 173, 'svg-title', spec.label), sub = centered(spec.point.x, 193, 'svg-note'); group.append(title, sub);
    svgButton(group, `Inspect ${spec.label}`, spec.target, () => inspect(spec.target)); graph.append(group);
    return { ...spec, group, title, sub };
  });
  graph.append(svg('path', { d: pathOf([{ x: geometry.agent.x + 48, y: 105 }, { x: geometry.action.x - 48, y: 105 }]), class: 'policy-link', 'marker-end': 'url(#arrow-down)' }));
  const edges = geometry.edges.map(edge => {
    const paths = (['down', 'up'] as const).map(direction => {
      const path = svg('path', { d: pathOf(edge[direction]), class: `client-link ${direction}` });
      const hit = svg('path', { d: pathOf(edge[direction]), class: 'link-hit' });
      svgButton(hit, `Inspect C${edge.id} ${direction === 'up' ? 'upload' : 'download'} link`, `link-${edge.id}-${direction}`, () => inspect(direction === 'up' ? 'upload' : 'download')); graph.append(path, hit);
      return { direction, path };
    });
    const group = node(edge.center, 'edge-node', `C${edge.id}`); group.dataset.client = String(edge.id);
    const title = centered(edge.center.x, 391, 'svg-title'), participation = centered(edge.center.x, 413, 'svg-note'), upload = centered(edge.center.x, 435, 'svg-note'); group.append(title, participation, upload);
    svgButton(group, `Inspect client C${edge.id} upload and participation records for this round`, `client-${edge.id}`, () => inspect(`client:${edge.id}`)); graph.append(group);
    return { ...edge, group, title, participation, upload, paths };
  });
  const down = svg('g', { class: 'link-summary' }), up = svg('g', { class: 'link-summary' });
  const downText = centered(width * .25, 474, 'svg-note'), upText = centered(width * .75, 474, 'svg-note');
  down.append(downText); up.append(upText);
  svgButton(down, 'Inspect total model download for this round', 'download', () => inspect('download'));
  svgButton(up, 'Inspect total gradient upload and safe metadata for this round', 'upload', () => inspect('upload'));
  graph.append(down, up, centered(width / 2, 502, 'svg-footnote', 'Mechanism illustration · safe upload metadata'));
  if (!edges.length) graph.append(centered(width / 2, 330, 'svg-note', 'Client identities not exported'));
  scene.append(graph); host.append(scene);
  const particles = mountParticles(scene, onPhase);
  return {
    update(frame: ReplayFrame, preview: JointAction | null, playing: boolean) {
      cloudNodes[0]!.sub.textContent = `${frame.scoreLabel}${number(run.config_summary.gamma) === null ? '' : ` · γ ${value(run.config_summary.gamma)}`}`;
      cloudNodes[1]!.title.textContent = frame.action ? `Joint action A${frame.action.action_id}` : 'Action mapping missing';
      cloudNodes[1]!.sub.textContent = frame.action ? `${clients(frame.action)} · batch ${frame.action.batch_size}` : `action_id ${value(frame.row.action_id)}`;
      cloudNodes[2]!.sub.textContent = `Accuracy ${value(frame.row.validation_accuracy)}`;
      downText.textContent = `↓ Model download · ${value(frame.row.round_download_bytes)} B`;
      upText.textContent = `↑ Gradient upload · ${value(frame.row.round_upload_bytes)} B`;
      graph.querySelectorAll('.preview-link, .preview-outline').forEach(n => n.remove());
      for (const edge of edges) {
        const actual = frame.action?.client_ids.includes(edge.id) === true, explored = preview?.client_ids.includes(edge.id) === true;
        edge.group.classList.toggle('actual', actual); edge.group.dataset.actual = String(actual); edge.group.dataset.preview = String(explored);
        edge.group.querySelector('.node-halo')!.setAttribute('fill', actual ? 'url(#glow-up)' : 'url(#glow-down)');
        edge.title.textContent = actual ? `Selected · batch ${frame.action!.batch_size}` : frame.action ? 'Not selected this round' : 'Participation unknown';
        edge.participation.textContent = `Total participations: ${value(frame.row[`c${edge.id}_participations`])}`;
        const uploads = frame.uploads.filter(u => u.client_id === edge.id);
        edge.upload.textContent = uploads.length ? `${uploads.length} upload records` : 'No uploads recorded';
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
  if (!run) { host.append(el('p', 'No validation runs were exported in this snapshot; frozen-policy replay is unavailable.', 'empty')); return () => {}; }
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
  identity.append(el('span', 'POLICY REPLAY', 'mode-tag'));
  identity.append(el('span', typeof run.replay_source_label === 'string' ? `${run.replay_source_label} · ${frames.length} rounds` : hasReplay(run) ? 'Round-by-round validation' : 'Summary only', `mode-tag${run.replay_source_label ? ' historical-tag' : ''}`));
  host.append(identity);
  const controls = panel('Decision replay');
  controls.classList.add('lab-controls');
  const runs = decisionRuns(snapshot.runs);
  const methodKey = (r: Run) => `${r.method}${r.arm ? ` / ${r.arm}` : ''}`;
  const methods = [...new Set(runs.map(methodKey))];
  controls.append(select('Policy / method', methods.map(m => [m, m]), methodKey(run), method => {
    const next = preferredDecisionRun(runs.filter(r => methodKey(r) === method), '');
    if (!next) return;
    pause(); onChange({ run: next.run_id, episode: '', frame: 0, preview: null }); onReset();
  }), select('Run / seed / checkpoint', runs.filter(r => methodKey(r) === methodKey(run)).sort((a, b) => Number(hasReplay(b)) - Number(hasReplay(a))).map(r => [r.run_id, replayRunLabel(r)]), run.run_id, id => {
    pause(); onChange({ run: id, episode: '', frame: 0, preview: null }); onReset();
  }), select('Episode', episodes.map(e => [e, e === 'unreported' ? 'Not exported · summary record' : `Episode ${e}`]), state.episode, ep => {
    pause(); onChange({ ...state, episode: ep, frame: 0, preview: null }); onReset();
  }));
  const runIdentity = el('details', undefined, 'run-identity');
  runIdentity.append(el('summary', `train ${value(run.train_seed)} / eval ${value(run.eval_seed)} · training budget ${value(run.config_summary.training_interaction_budget)} · ${run.partition}`),
    el('code', run.run_id), el('p', `Risk version: ${value(run.risk_version)} · evidence_scope: ${run.evidence_scope}`, 'muted'));
  controls.append(runIdentity); host.append(controls);

  const loop = el('div', undefined, 'rl-loop');
  const stages: [Inspect, string, string][] = [['state', '01 / STATE', 'Environment state'], ['agent', '02 / AGENT', 'Policy scoring'], ['action', '03 / ACTION', 'Clients × batch'], ['cloud', '04 / ENVIRONMENT', 'Cloud–edge execution'], ['reward', '05 / REWARD', 'Reward feedback']];
  for (const [target, label, text] of stages) {
    const b = button('', () => openInspector(target)); b.append(el('small', label), el('span', text)); b.dataset.focusKey = `loop-${target}`; loop.append(b);
  }
  host.append(loop, el('p', 'Environment state → Policy decision → Joint scheduling → Cloud–edge execution → Reward feedback', 'loop-caption'));

  const timeline = el('section', undefined, 'lab-timeline');
  const previous = button('← Previous', () => go(state.frame - 1));
  const play = button('Play records', () => {
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
  const next = button('Next →', () => go(state.frame + 1));
  const roundLabel = el('output', undefined, 'round-output'); roundLabel.setAttribute('aria-live', 'off');
  const slider = el('input'); slider.type = 'range'; slider.min = '0'; slider.max = String(Math.max(0, frames.length - 1)); slider.step = '1'; slider.value = String(state.frame); slider.setAttribute('aria-label', 'Select replay round by record index'); slider.oninput = () => go(Number(slider.value));
  const timelineButtons = el('div', undefined, 'controls'); timelineButtons.append(previous, play, next, roundLabel);
  timeline.append(timelineButtons, slider, el('small', 'Round replay · 1 second / record'));
  host.append(timeline);

  const grid = el('div', undefined, 'lab-grid');
  const statePanel = panel('State', 'Previous feedback and current round'); statePanel.classList.add('state-panel');
  const stateBody = el('div'); statePanel.append(stateBody, button('Inspect state source', () => openInspector('state')));
  const environment = panel('Action → Environment', 'Select a cloud node, client or link to inspect this round.'); environment.classList.add('environment-panel');
  const graphHost = el('div', undefined, 'graph-host'); environment.append(graphHost, el('p', 'On narrow screens, scroll horizontally to see all nodes, or use Tab and Enter to inspect them.', 'graph-scroll-note'));
  const legend = el('div', undefined, 'graph-legend'); legend.append(el('span', 'Solid / actual action', 'legend-actual'), el('span', 'Dashed / candidate preview', 'legend-preview'), el('span', 'Gray / inactive or unknown', 'legend-idle')); environment.append(legend);
  const graphStatus = el('div', undefined, 'graph-status'); const phaseLabel = el('span', 'Static replay', 'phase-label');
  let motionEnabled = true;
  const motionButton = button('Disable motion', () => { motionEnabled = !motionEnabled; motionButton.textContent = motionEnabled ? 'Disable motion' : 'Enable motion'; motionButton.setAttribute('aria-pressed', String(motionEnabled)); graphView.setEnabled(motionEnabled); });
  motionButton.setAttribute('aria-pressed', 'true'); graphStatus.append(phaseLabel, motionButton); environment.append(graphStatus);
  const graphView = mountGraph(graphHost, run, openInspector, phase => {
    if (disposed) return;
    phaseLabel.textContent = phaseLabels[phase];
    const target = phase === 'agent' ? 'agent' : phase === 'reward' ? 'reward' : phase === 'static' ? '' : 'cloud';
    loop.querySelectorAll('button').forEach(b => b.classList.toggle('phase-active', b.dataset.focusKey === `loop-${target}`));
    feedback.classList.toggle('phase-active', phase === 'reward');
  });
  const feedback = panel('Reward', 'Recorded results for this round'); feedback.classList.add('reward-panel');
  const feedbackBody = el('div'); feedback.append(feedbackBody, button('Inspect reward and evidence', () => openInspector('reward')));
  grid.append(statePanel, environment, feedback); host.append(grid);

  const decision = panel('Joint action', 'Select a client set and batch size to preview the collaboration paths.');
  decision.classList.add('decision-panel');
  const decisionHeader = el('div', undefined, 'decision-heading');
  const scoreHeading = el('div'); const actualLabel = el('strong'); const scoreNote = el('p', undefined, 'muted');
  scoreHeading.append(actualLabel, scoreNote);
  const resetPreview = button('Clear candidate preview', () => { state.preview = null; update(); });
  decisionHeader.append(scoreHeading, resetPreview); decision.append(decisionHeader);
  const decisionGrid = el('div', undefined, 'decision-grid');
  const matrixHost = el('div', undefined, 'action-matrix-host');
  const matrixButtons = new Map<number, { button: HTMLButtonElement; score: HTMLElement; badge: HTMLElement }>();
  if (actions.length) {
    const batches = [...new Set(actions.map(a => a.batch_size))].sort((a, b) => a - b);
    const groups = [...new Set(actions.map(a => clients(a)))];
    const wrap = el('div', undefined, 'table-wrap'); const matrix = el('table', undefined, 'action-matrix');
    const thead = el('thead'); const heading = el('tr'); heading.append(el('th', 'Client set × batch'));
    for (const batch of batches) { const th = el('th', `batch ${batch}`); th.scope = 'col'; heading.append(th); } thead.append(heading); matrix.append(thead);
    const tbody = el('tbody');
    for (const group of groups) {
      const tr = el('tr'); const th = el('th', group); th.scope = 'row'; tr.append(th);
      for (const batch of batches) {
        const td = el('td'); const matches = actions.filter(a => clients(a) === group && a.batch_size === batch);
        if (!matches.length) td.append(el('span', 'No such action', 'muted'));
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
    matrix.append(tbody); wrap.append(matrix); matrixHost.append(wrap, el('p', 'On narrow screens, scroll the action matrix horizontally to see all batch options.', 'matrix-scroll-note'));
  } else matrixHost.append(el('p', 'This run has no action mapping. Clients and batch sizes cannot be inferred from action_id.', 'empty'));
  const rankingHost = el('div', undefined, 'action-ranking'); decisionGrid.append(matrixHost, rankingHost); decision.append(decisionGrid);
  const previewSummary = el('div', undefined, 'preview-summary'); previewSummary.setAttribute('aria-live', 'polite'); decision.append(previewSummary);
  const warnings = el('div', undefined, 'lab-warnings'); decision.append(warnings); host.append(decision);

  const tracesHeader = el('div', undefined, 'section-heading'); tracesHeader.append(el('h2', 'Round feedback'), el('p', 'Select a chart point to inspect its round.', 'muted'));
  const traces = el('div', undefined, 'chart-grid lab-traces'); host.append(tracesHeader, traces);
  const syncCharts = replayTraces(traces, frames, run, go);
  const provenance = el('details', undefined, 'provenance'); provenance.append(el('summary', 'Record sources and interaction limits'),
    el('p', 'Time and energy are simulated. Candidate previews show paths and policy scores only; feedback reflects the actual executed action.'), records('Run sources', run.sources), records('Snapshot / run warnings', [...snapshot.warnings, ...run.warnings])); host.append(provenance);

  const dialog = el('dialog', undefined, 'inspector'); dialog.setAttribute('aria-labelledby', 'inspector-title');
  const dialogHead = el('div', undefined, 'inspector-heading'); const dialogTitle = el('h2'); dialogTitle.id = 'inspector-title';
  const close = button('Close ×', () => dialog.close()); close.setAttribute('aria-label', 'Close evidence details'); dialogHead.append(dialogTitle, close);
  const dialogBody = el('div'); dialog.append(dialogHead, dialogBody); host.append(dialog);

  function pause(): void { if (timer !== undefined) clearInterval(timer); timer = undefined; play.textContent = 'Play records'; }
  function go(index: number): void { if (disposed || !frames.length) return; pause(); state.frame = Math.min(Math.max(0, index), frames.length - 1); state.preview = null; update(); }
  function openInspector(target: Inspect): void { pause(); inspectTarget = target; update(); const frame = current(); if (!frame) return; fillInspector(frame); if (!dialog.open) dialog.showModal(); }
  function fillInspector(frame: ReplayFrame): void {
    const id = inspectTarget.startsWith('client:') ? Number(inspectTarget.slice(7)) : null;
    dialogTitle.textContent = id !== null ? `C${id} / Upload and participation evidence` : ({ state: 'State / Pre-decision information limits', agent: 'Agent / Policy scores', action: 'Action / Actual and candidate', cloud: 'Environment / Federated update', reward: 'Reward / Records and risk evidence', upload: 'Gradient upload / Round total', download: 'Model download / Round total' } as Record<string, string>)[inspectTarget] ?? 'Record sources';
    dialogBody.replaceChildren(el('p', `run ${run!.run_id} · episode ${value(frame.episode)} · round ${value(frame.round)}`, 'muted'));
    if (inspectTarget === 'state') dialogBody.append(el('p', 'The full historical observation (resources, participation window and candidate costs) was not exported. Previous-round feedback below is a reference, not the full state observed by the policy.', 'empty'), records('Recorded previous-round feedback', frame.previous ? [frame.previous] : []));
    else if (inspectTarget === 'agent') dialogBody.append(el('p', `${frame.scoreLabel}; vectors are joined by action_id. The selected action need not have the highest score; the exploration mechanism cannot be inferred from this alone.`, 'muted'), records('Raw decision for this round', frame.decisions));
    else if (inspectTarget === 'action') dialogBody.append(records('Actual action / candidate preview', [{ actual: frame.action, preview: preview(), preview_executed: false }]), el('p', 'Candidate previews do not change recorded results; no counterfactual outcomes are computed.', 'empty'));
    else if (inspectTarget === 'download') dialogBody.append(el('p', `Total round download: ${value(frame.row.round_download_bytes)} B. Per-client traffic was not recorded and is not allocated evenly.`, 'empty'));
    else if (inspectTarget === 'upload' || id !== null) {
      const uploads = id === null ? frame.uploads : frame.uploads.filter(u => u.client_id === id);
      dialogBody.append(el('p', `Total round upload: ${value(frame.row.round_upload_bytes)} B. Only safe client metadata is shown; sample contents and gradient payloads are excluded.`, 'muted'),
        table(['Upload ID', 'Client', 'Batch', 'Predicted risk', 'Uncertainty', 'Decision type'], uploads.map(u => [u.upload_id, u.client_id, u.batch_size, u.predicted_risk, u.uncertainty, u.decision_kind])), records('Raw upload metadata for this round', uploads));
      if (id !== null) dialogBody.append(table(['Total participations', 'Total sample count', 'Total unique count'], [[frame.row[`c${id}_participations`], frame.row[`c${id}_samples`], frame.row[`c${id}_unique`]]]));
    }
    if (inspectTarget === 'reward' || inspectTarget === 'cloud') dialogBody.append(el('p', 'Only the recorded reward is used. Components and historical weights were not exported; reward is not recomputed from current configuration. Predicted risk, uncertainty and audit ground truth are distinct.', 'empty'),
      table(['Reward', 'Protocol', 'Accuracy', 'Loss', 'Predicted exposure risk', 'Mean upload prediction'], [[frame.row.reward, frame.row.reward_protocol, frame.row.validation_accuracy, frame.row.validation_loss, frame.row.predicted_exposure_risk, frame.row.predicted_risk]]));
    dialogBody.append(el('h3', 'Audit labels / Independent evidence'), el('p', frame.audits.length ? `${frame.audits.length} exactly matched audit records; different scopes and models remain separate and do not establish whole-policy safety.` : 'No exactly matched audit labels for this round. Predictions do not replace ground truth; no pass verdict is generated by the browser.', 'muted'), records('Matched by run_id + unique upload_id', id === null ? frame.audits : frame.audits.filter(a => frame.uploads.some(u => u.client_id === id && u.upload_id === a.upload_id))),
      records('Current metrics and sources', [{ observation: frame.row, sources: run!.sources }]), records('Join limitations', frame.warnings));
  }

  function update(): void {
    if (disposed) return;
    const frame = current();
    previous.disabled = !frames.length || state.frame === 0; next.disabled = !frames.length || state.frame === frames.length - 1;
    play.disabled = frames.length < 2; slider.disabled = !frames.length; slider.value = String(state.frame);
    play.textContent = timer !== undefined ? 'Pause replay' : 'Play records';
    resetPreview.disabled = state.preview === null;
    if (!frame) { graphView.dispose(); stateBody.replaceChildren(el('p', 'This run has no round-by-round metric records.', 'empty')); graphHost.replaceChildren(el('p', 'No replay frames are available; cloud–edge execution states are not fabricated.', 'empty')); feedbackBody.replaceChildren(); notify(); return; }
    roundLabel.textContent = `Episode ${value(frame.episode)} · Round ${value(frame.round)} · ${state.frame + 1} / ${frames.length} · ${status(frame.row)}`;
    slider.setAttribute('aria-valuetext', roundLabel.textContent);
    stateBody.replaceChildren(el('span', 'Historical observation not exported', 'missing-tag'), el('p', 'Resources / participation window / candidate costs', 'muted'));
    if (frame.previous) {
      stateBody.append(el('h3', `Reference feedback from round ${value(frame.previous.round)}`));
      stat(stateBody, 'Previous accuracy', frame.previous.validation_accuracy);
      stat(stateBody, 'Previous cumulative time · s', frame.previous.elapsed_seconds);
      stat(stateBody, 'Previous cumulative energy · J', frame.previous.energy_joules);
    } else stateBody.append(el('p', 'No unique adjacent previous round is available; the pre-decision state cannot be reconstructed.', 'empty'));
    stateBody.append(el('p', `Current record: round ${value(frame.round)} / limit ${value(run!.config_summary.max_rounds)}`, 'muted'));
    graphView.update(frame, preview(), timer !== undefined);
    feedbackBody.replaceChildren();
    stat(feedbackBody, 'Immediate reward rₜ', frame.row.reward);
    stat(feedbackBody, 'Validation accuracy', frame.row.validation_accuracy);
    stat(feedbackBody, 'Simulated round time · s', frame.row.round_time_seconds);
    stat(feedbackBody, 'Simulated round energy · J', frame.row.round_energy_joules);
    stat(feedbackBody, 'Predicted exposure risk', frame.row.predicted_exposure_risk, 'Not audited risk');
    feedbackBody.append(el('span', frame.audits.length ? `${frame.audits.length} exactly matched audit records` : 'Audit labels missing for this round', 'audit-tag'));
    actualLabel.textContent = frame.action ? `Executed: ${actionName(frame.action)}` : `Actual action_id ${value(frame.row.action_id)} · mapping not exported`;
    scoreNote.textContent = `${frame.scoreLabel} · recorded decision margin ${value(frame.decision?.margin)} · ${status(frame.row)}`;
    const ranked = actionRanking(frame); const low = ranked.at(-1)?.value ?? 0; const high = ranked[0]?.value ?? 0;
    for (const action of actions) {
      const cell = matrixButtons.get(action.action_id)!; const score = frame.scores.get(action.action_id) ?? null;
      const actual = frame.action?.action_id === action.action_id; const explored = state.preview === action.action_id;
      cell.button.classList.toggle('actual', actual); cell.button.classList.toggle('preview', explored);
      cell.button.setAttribute('aria-pressed', String(explored));
      cell.button.setAttribute('aria-label', `${actionName(action)}, ${frame.scoreLabel} ${value(score)}${actual ? ', executed' : ''}${explored ? ', candidate preview' : ''}`);
      cell.score.textContent = value(score); cell.badge.textContent = [actual ? 'Actual' : '', explored ? 'Preview' : ''].filter(Boolean).join(' / ') || 'Explore path';
      cell.button.style.setProperty('--score-strength', String(score === null ? 0 : high === low ? .1 : .04 + (score - low) / (high - low) * .14));
    }
    preserveFocus(rankingHost, () => {
      rankingHost.replaceChildren(el('h3', 'Candidate ranking for this round'));
      if (!ranked.length) rankingHost.append(el('p', 'Scores were not exported. Mapped paths can be previewed, but policy preferences cannot be compared.', 'empty'));
      for (const [i, entry] of ranked.slice(0, 3).entries()) {
        const action = actions.find(a => a.action_id === entry.action_id)!;
        const b = button('', () => { pause(); state.preview = entry.action_id; update(); }); b.classList.add('ranking-row'); b.dataset.focusKey = `rank-${entry.action_id}`;
        b.append(el('span', `${i + 1}`), el('span', `A${entry.action_id} · ${clients(action)}\nbatch ${action.batch_size}`), el('strong', value(entry.value))); rankingHost.append(b);
      }
      rankingHost.append(el('p', 'Policy scores for this round · Top 3', 'muted'));
    });
    const explored = preview(); previewSummary.replaceChildren();
    if (explored) {
      const actualScore = frame.action ? frame.scores.get(frame.action.action_id) ?? null : null;
      const previewScore = frame.scores.get(explored.action_id) ?? null;
      previewSummary.append(el('strong', `Candidate preview: ${actionName(explored)}`), el('span', `${frame.scoreLabel} ${value(previewScore)} · score difference from actual ${value(previewScore !== null && actualScore !== null ? previewScore - actualScore : null)}`),
        el('small', 'Unexecuted candidate path · feedback shows actual action results.'));
    } else previewSummary.append(el('span', 'Select a candidate to preview its paths · solid = actual; dashed = candidate.'));
    warnings.replaceChildren(); frame.warnings.forEach(w => warnings.append(el('p', w, 'muted')));
    syncCharts(state.frame); if (dialog.open) fillInspector(frame); notify();
  }
  const suspend = () => { if (document.hidden && timer !== undefined) { pause(); update(); } };
  document.addEventListener('visibilitychange', suspend);
  update();
  return () => { disposed = true; pause(); document.removeEventListener('visibilitychange', suspend); graphView.dispose(); if (dialog.open) dialog.close(); dialog.remove(); };
}
