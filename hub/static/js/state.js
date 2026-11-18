// Fleet state: SSE connection to the hub, per-node rolling buffers, and the
// derived numbers every view shows (ported from the old single-host dashboard so
// the same caveats about vLLM aggregate vs per-request rates still apply).
import { isNum, api, isEngineBackend } from './util.js';

export const BUFFER_LEN = 120; // 2 minutes at 1 s

const listeners = new Set();
export const state = {
  nodes: [],          // [{id, name, color, online, error, last_seen, latency_ms, snapshot}]
  models: {},         // nodeId -> models[]
  buffers: {},        // nodeId -> {t:[], tps:[], ...}
  connected: false,
  serverSkew: 0,
  lastUpdate: 0,
};

export const onChange = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
const emit = () => listeners.forEach((fn) => { try { fn(state); } catch (e) { console.error(e); } });

export const nodeById = (id) => state.nodes.find((n) => n.id === id);
export const now = () => Date.now() / 1000 + state.serverSkew;

function pushBuffer(node) {
  const s = node.snapshot;
  if (!s || !node.online) return;
  const b = state.buffers[node.id] ||= { t: [], tps: [], ingest: [], util: [], vram: [], power: [], temp: [], fan: [], queue: [], ctx: [] };
  const t = Math.round(s.collected_at || Date.now() / 1000);
  if (b.t.length && b.t[b.t.length - 1] >= t) return; // same sample as last tick
  const d = derive(s);
  const g = d.gpu;
  const add = (k, v) => { b[k].push(isNum(v) ? v : null); if (b[k].length > BUFFER_LEN) b[k].shift(); };
  add('t', t);
  add('tps', d.genLive);
  add('ingest', d.ingestLive);
  add('util', g.util);
  add('vram', g.mem_total > 0 ? (g.mem_used / g.mem_total) * 100 : null);
  add('power', g.power);
  add('temp', g.temp);
  add('fan', g.fan);
  add('queue', s.model_stats?.requests_waiting ?? 0);
  add('ctx', s.context_info?.ctx_pct ?? 0);
}

function applyFleet(payload) {
  state.serverSkew = (payload.server_time || Date.now() / 1000) - Date.now() / 1000;
  state.nodes = payload.nodes || [];
  for (const n of state.nodes) {
    // family="infra" stacks (e.g. the shared proxy fleet) are not models you switch to.
    if (n.snapshot?.models) state.models[n.id] = n.snapshot.models.filter((m) => m.family !== 'infra');
    pushBuffer(n);
  }
  for (const id of Object.keys(state.models)) {
    if (!state.nodes.some((n) => n.id === id)) { delete state.models[id]; delete state.buffers[id]; }
  }
  state.lastUpdate = Date.now();
  emit();
}

let es = null;
let pollTimer = null;
export function connect() {
  if (es) es.close();
  clearInterval(pollTimer);
  es = new EventSource('/api/stream');
  es.addEventListener('fleet', (ev) => {
    state.connected = true;
    try { applyFleet(JSON.parse(ev.data)); } catch (e) { console.error(e); }
  });
  es.onerror = () => {
    state.connected = false;
    emit();
    // EventSource retries on its own; poll meanwhile so the page stays live.
    if (!pollTimer) {
      pollTimer = setInterval(async () => {
        if (state.connected) { clearInterval(pollTimer); pollTimer = null; return; }
        try { applyFleet(await api('/api/fleet')); } catch { /* still down */ }
      }, 2000);
    }
  };
}
export const refreshNow = async () => { try { applyFleet(await api('/api/fleet')); } catch { /* ignore */ } };

// ------------------------------------------------------------------ derived

/** The numbers the UI shows for one node snapshot. */
export function derive(s) {
  const stats = s?.model_stats || {};
  const live = s?.live_throughput || {};
  const ci = s?.context_info || {};
  const gpus = s?.gpu_stats || [];
  const gpu = gpus[0] || {};
  const active = s?.active || null;
  const engine = isEngineBackend(active);

  // Generation: live first, then last completed, then live average.
  const liveTps = stats.live_tps || 0;
  const lastTps = stats.last_completed_tps || stats.live_average_tps || 0;
  const gen = liveTps > 0 ? liveTps : (lastTps > 0 ? lastTps : 0);
  let genSub = 'tokens / sec';
  if (engine) {
    // Aggregate across all running streams vs per-request average are different units.
    const run = stats.requests_running || 0;
    const perStream = liveTps > 0 && run > 1 ? liveTps / run : null;
    const avgTxt = stats.average_rate_tps
      ? (perStream ? `${perStream.toFixed(1)}/stream now` : `${stats.average_rate_tps.toFixed(1)}/req avg`)
      : null;
    // 0 tok/s with a request running is prefill, not idle.
    const st = liveTps >= 0.5 ? 'aggregate' : (run > 0 ? 'prefilling' : 'idle');
    genSub = avgTxt ? `${st} · ${avgTxt}` : (liveTps >= 0.5 ? 'tokens / sec' : st);
  } else if (liveTps <= 0 && lastTps > 0) {
    genSub = 'last completed';
  }

  // Ingest: last completed, live while ingesting, best, live.
  let ingest = 0;
  let ingestSub = 'tokens / sec';
  const lastIngest = stats.last_ingest_tps;
  const liveIngest = stats.last_live_ingest_tps;
  const bestIngest = stats.best_ingest_tps;
  const ingesting = live.state !== 'idle' && liveIngest > 0;
  if (lastIngest > 0) ingest = lastIngest;
  else if (ingesting) ingest = liveIngest;
  else if (bestIngest > 0) { ingest = bestIngest; ingestSub = 'best seen'; }
  else if (liveIngest > 0) ingest = liveIngest;
  if (engine && !(ingest > 0)) {
    const liveIng = live.ingest_tps ?? stats.ingest_tps;
    if (liveIng > 0) ingest = liveIng;
    else if (stats.avg_prefill_rate_tps > 0) {
      ingest = stats.avg_prefill_rate_tps;
      // Effective, not raw: prompt_tokens_total counts prefix-cache hits.
      ingestSub = 'effective avg · incl. cache hits';
    }
  }

  // Context
  const nCtx = ci.n_ctx, nPast = ci.n_past;
  const ctxIsAvg = ci.n_past_is_average === true;
  const ctxLive = ci.n_past_live === true;
  let ctxLabel = ctxLive ? 'Context · live' : (ctxIsAvg ? 'Context / request' : 'Context');
  let ctxPct = nCtx > 0 && nPast > 0 ? Math.min(100, (nPast / nCtx) * 100) : 0;
  let ctxSub;
  if (nCtx != null && nPast > 0) {
    const share = `${((nPast / nCtx) * 100).toFixed(1)}% of window`;
    const tgt = ci.n_past_target;
    const filling = ctxLive && tgt > nPast + 1 ? `filling ${Math.round((100 * nPast) / tgt)}% of ${(tgt / 1024).toFixed(1)}k · ` : '';
    ctxSub = ctxIsAvg ? `avg per request · ${share}` : filling + share;
  } else if (nCtx != null) {
    ctxSub = ctxIsAvg ? 'no completed requests yet' : 'window idle';
  } else {
    ctxSub = 'window used';
  }

  return {
    active, engine, stats, live, ci, gpu, gpus,
    gen, genLive: liveTps, genSub,
    ingest, ingestSub, ingestLive: ingesting ? liveIngest : (engine ? (live.ingest_tps || 0) : 0),
    nCtx, nPast, ctxPct, ctxLabel, ctxSub,
    healthy: !!active?.healthy,
  };
}

/** Overall node state for pills: offline | switching | healthy | starting | idle */
export function nodeStatus(node) {
  if (!node.enabled) return { key: 'disabled', label: 'Disabled', cls: '' };
  if (!node.online) return { key: 'offline', label: 'Offline', cls: 'bad' };
  const s = node.snapshot || {};
  if (s.switch_in_progress) return { key: 'switching', label: 'Switching', cls: 'warn', pulse: true };
  if (s.active?.healthy) return { key: 'healthy', label: 'Healthy', cls: 'ok' };
  if (s.active) return { key: 'starting', label: 'Starting', cls: 'warn', pulse: true };
  return { key: 'idle', label: 'No model', cls: '' };
}

// Switch progress: map the agent's free-text messages onto five steps.
export const SWITCH_STEPS = ['Images', 'Stop current', 'Start', 'Load', 'Ready'];
export function switchPhase(s) {
  const msg = (s?.last_message || '').toLowerCase();
  const m = msg.match(/loading[^:]*:\s*(\d+)%/);
  if (m) return { step: 3, pct: parseInt(m[1], 10) };
  if (/loading/.test(msg)) return { step: 3 };
  if (/build|pull|image/.test(msg)) return { step: 0 };
  if (/stopping|switching to/.test(msg)) return { step: 1 };
  if (/starting/.test(msg)) return { step: 2 };
  return { step: 2 };
}
