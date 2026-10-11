import type { Container, Engine, Particle } from '@tsparticles/engine';
import { clientIds, type ReplayFrame } from './replay.ts';
import { number, type Run } from './model.ts';

export interface Point { x: number; y: number }
export type Phase = 'agent' | 'down' | 'local' | 'up' | 'aggregate' | 'reward' | 'static';
export interface Track { phase: Phase; color: string; points: Point[]; radius?: number; client?: number }
export interface Scene { width: number; height: number; tracks: Track[] }
export const phaseWindows = { agent: [0, 160], down: [160, 380], local: [380, 560], up: [560, 780], aggregate: [780, 900], reward: [900, 1000] } as const;
export const phaseLabels: Record<Phase, string> = { agent: '策略评分', down: '模型下发', local: '边端计算', up: '梯度汇聚', aggregate: '模型聚合', reward: '奖励反馈', static: '静态回放' };
const blue = '#7abcf0', teal = '#65d4b7';

// SVG and Canvas use the same coordinates; neither topology nor motion uses random placement.
export function graphGeometry(run: Run) {
  const ids = clientIds(run), width = Math.max(880, ids.length * 215 + 80);
  const agent = { x: width * .17, y: 105 }, action = { x: width * .5, y: 105 }, cloud = { x: width * .83, y: 105 };
  const edges = ids.map((id, i) => {
    const x = width / (ids.length + 1) * (i + 1);
    return { id, center: { x, y: 328 },
      down: [{ x: action.x - 10, y: 145 }, { x: action.x - 10, y: 226 }, { x: x - 10, y: 226 }, { x: x - 10, y: 280 }],
      up: [{ x: x + 10, y: 280 }, { x: x + 10, y: 246 }, { x: cloud.x + 10, y: 246 }, { x: cloud.x + 10, y: 145 }],
      preview: [{ x: action.x + 12, y: 145 }, { x: action.x + 12, y: 262 }, { x: x + 22, y: 262 }, { x: x + 22, y: 281 }],
    };
  });
  return { width, height: 510, agent, action, cloud, edges };
}

export function buildParticleScene(run: Run, frame: ReplayFrame): Scene {
  const g = graphGeometry(run), tracks: Track[] = [];
  // Large/unknown topologies remain readable static SVGs rather than partial animated evidence.
  if (!frame.action || !g.edges.length || g.edges.length > 8) return { ...g, tracks };
  tracks.push({ phase: 'agent', color: blue, points: [{ x: g.agent.x + 48, y: 105 }, { x: g.action.x - 48, y: 105 }] });
  for (const edge of g.edges) {
    if (!frame.action.client_ids.includes(edge.id)) continue;
    if ((number(frame.row.round_download_bytes) ?? 0) > 0) tracks.push({ phase: 'down', color: blue, points: edge.down, client: edge.id });
    tracks.push({ phase: 'local', color: teal, points: [edge.center], radius: 43, client: edge.id });
    if (frame.uploads.some(u => u.client_id === edge.id)) tracks.push({ phase: 'up', color: teal, points: edge.up, client: edge.id });
  }
  if (tracks.some(t => t.phase === 'up')) {
    tracks.push({ phase: 'aggregate', color: teal, points: [g.cloud], radius: 44 });
    if (number(frame.row.reward) !== null) tracks.push({ phase: 'reward', color: teal, points: [g.cloud], radius: 55 });
  }
  return { width: g.width, height: g.height, tracks };
}

export function phaseAt(ms: number): Phase {
  for (const [phase, [start, end]] of Object.entries(phaseWindows)) if (ms >= start && ms < end) return phase as Phase;
  return 'static';
}

export function alongPath(points: Point[], fraction: number): Point {
  const first = points[0] ?? { x: 0, y: 0 };
  if (points.length < 2) return { ...first };
  const lengths = points.slice(1).map((p, i) => Math.hypot(p.x - points[i]!.x, p.y - points[i]!.y));
  let distance = Math.max(0, Math.min(1, fraction)) * lengths.reduce((sum, n) => sum + n, 0);
  for (const [i, length] of lengths.entries()) {
    if (distance <= length && length > 0) {
      const a = points[i]!, b = points[i + 1]!, ratio = distance / length;
      return { x: a.x + (b.x - a.x) * ratio, y: a.y + (b.y - a.y) * ratio };
    }
    distance -= length;
  }
  return { ...points.at(-1)! };
}

export function sampleTrack(track: Track, offset: number, elapsed: number, animated: boolean): Point & { opacity: number } {
  const window = track.phase === 'static' ? [0, 1000] : phaseWindows[track.phase];
  const progress = (elapsed - window[0]!) / (window[1]! - window[0]!);
  const t = animated ? (progress + offset) % 1 : offset;
  const center = track.points[0] ?? { x: 0, y: 0 };
  const point = track.radius ? { x: center.x + Math.cos(t * Math.PI * 2) * track.radius, y: center.y + Math.sin(t * Math.PI * 2) * track.radius } : alongPath(track.points, t);
  return { ...point, opacity: animated ? progress >= 0 && progress < 1 ? .85 * Math.min(1, progress * 8, (1 - progress) * 8) : 0 : .32 };
}

export function particleBudget(scene: Scene, mobile: boolean): { track: Track; offset: number }[] {
  const perTrack = mobile ? 3 : 6, limit = mobile ? 48 : 96;
  // Fair allocation across paths; visual counts are unrelated to samples/batch/bytes.
  return scene.tracks.flatMap(track => Array.from({ length: Math.min(perTrack, Math.floor(limit / Math.max(1, scene.tracks.length))) }, (_, i) => ({ track, offset: i / perTrack }))).slice(0, limit);
}

interface Runtime { scene: Scene; specs: Map<Particle, { track: Track; offset: number }>; animated: boolean; epoch: number; onPhase: (phase: Phase) => void }
const runtimes = new WeakMap<Container, Runtime>();
let enginePromise: Promise<Engine> | undefined;
let nextId = 0;
function paintParticle(container: Container, particle: Particle, runtime: Runtime): void {
  const spec = runtime.specs.get(particle);
  if (!spec) return;
  const elapsed = Math.min(999, Math.max(0, performance.now() - runtime.epoch));
  const pose = sampleTrack(spec.track, spec.offset, elapsed, runtime.animated);
  particle.position.x = pose.x / runtime.scene.width * container.canvas.size.width;
  particle.position.y = pose.y / runtime.scene.height * container.canvas.size.height;
  if (particle.opacity) particle.opacity.value = pose.opacity;
  if (runtime.animated) runtime.onPhase(phaseAt(elapsed));
}
function loadEngine(): Promise<Engine> {
  return enginePromise ??= Promise.all([import('@tsparticles/engine'), import('@tsparticles/basic')]).then(async ([{ tsParticles }, { loadBasic }]) => {
    await loadBasic(tsParticles);
    await tsParticles.pluginManager.register(e => {
      e.pluginManager.addParticleUpdater('rl-record-path', container => Promise.resolve({
        init: () => {}, isEnabled: () => runtimes.has(container),
        update: (particle: Particle) => { const runtime = runtimes.get(container); if (runtime) paintParticle(container, particle, runtime); },
      }));
    });
    return tsParticles;
  }).catch(error => { enginePromise = undefined; throw error; });
}

export interface ParticleController { update(scene: Scene, playing: boolean, key: string): void; setEnabled(enabled: boolean): void; dispose(): void }
export function mountParticles(host: HTMLElement, onPhase: (phase: Phase) => void): ParticleController {
  const canvas = document.createElement('canvas'); canvas.className = 'particle-canvas'; canvas.setAttribute('aria-hidden', 'true'); canvas.id = `rl-particles-${++nextId}`; host.append(canvas);
  const motion = matchMedia('(prefers-reduced-motion: reduce)'), mobile = matchMedia('(max-width: 700px)');
  const runtime: Runtime = { scene: { width: 880, height: 510, tracks: [] }, specs: new Map(), animated: false, epoch: performance.now(), onPhase };
  let container: Container | undefined, disposed = false, enabled = true, playing = false, visible = true, key = '', lastPhase: Phase = 'static';
  runtime.onPhase = phase => { if (phase === lastPhase) return; lastPhase = phase; host.dataset.phase = phase; onPhase(phase); };
  function settle(): void {
    if (disposed || !container) return;
    container.pause();
    runtime.animated = enabled && playing && !motion.matches && !document.hidden && visible && runtime.scene.tracks.length > 0;
    host.dataset.motion = !enabled ? 'off' : runtime.animated ? 'playing' : 'static';
    if (!runtime.animated) { lastPhase = 'static'; host.dataset.phase = 'static'; onPhase('static'); }
    container.canvas.resize();
    for (const particle of runtime.specs.keys()) paintParticle(container, particle, runtime);
    container.canvas.render.clear();
    container.canvas.render.drawParticles({ value: 0, factor: 0 });
    if (runtime.animated) container.play();
  }
  function populate(): void {
    if (!container || disposed) return;
    container.pause(); container.particles.clear(); runtime.specs.clear();
    for (const spec of particleBudget(runtime.scene, mobile.matches)) {
      const particle = container.particles.addParticle({ x: 1, y: 1 }, {
        paint: { fill: { enable: true, color: { value: spec.track.color } } },
        size: { value: spec.track.radius ? 1.7 : 2.4 }, opacity: { value: .7 }, move: { enable: false },
      });
      if (particle) runtime.specs.set(particle, spec);
    }
    host.dataset.particles = String(container.particles.count);
    container.fpsLimit = mobile.matches ? 30 : 60;
    settle();
  }
  const resize = new ResizeObserver(() => settle()); resize.observe(host);
  const intersection = new IntersectionObserver(entries => { visible = entries[0]?.isIntersecting ?? true; settle(); }); intersection.observe(host);
  motion.addEventListener('change', settle); mobile.addEventListener('change', populate); document.addEventListener('visibilitychange', settle);
  host.dataset.motion = 'loading';
  void loadEngine().then(async engine => {
    if (disposed) return;
    const loaded = await engine.load({ id: canvas.id, element: canvas, options: {
      autoPlay: false, fullScreen: false, fpsLimit: 60, detectRetina: true,
      pauseOnBlur: false, pauseOnOutsideViewport: false, resize: { enable: false },
      particles: { number: { value: 0, density: { enable: false } }, move: { enable: false }, shape: { type: 'circle' }, opacity: { value: .7 } },
    } });
    if (disposed) { loaded?.destroy(); return; }
    if (!loaded) throw new Error('Particle container unavailable');
    container = loaded; runtimes.set(container, runtime); container.canvas.setPointerEvents('none'); populate();
  }).catch(() => {
    if (disposed) return;
    container?.destroy(); canvas.remove(); host.dataset.motion = 'unavailable'; host.dataset.particles = '0'; onPhase('static');
  });
  return {
    update(scene, nextPlaying, nextKey) {
      if (disposed) return;
      const changed = nextKey !== key;
      if (changed || nextPlaying && !playing) runtime.epoch = performance.now();
      key = nextKey; playing = nextPlaying; runtime.scene = scene;
      if (changed) populate(); else settle();
    },
    setEnabled(value) { enabled = value; settle(); },
    dispose() {
      if (disposed) return; disposed = true;
      resize.disconnect(); intersection.disconnect(); motion.removeEventListener('change', settle); mobile.removeEventListener('change', populate); document.removeEventListener('visibilitychange', settle);
      if (container) { runtimes.delete(container); container.destroy(); }
      runtime.specs.clear(); canvas.remove();
    },
  };
}
