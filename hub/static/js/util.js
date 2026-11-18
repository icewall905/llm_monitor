// Small shared helpers: escaping, formatting, DOM patching, API calls, icons.

export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Replace innerHTML only when it changed, so 1 s updates don't reset hover/scroll. */
export function setHTML(el, html) {
  if (!el) return;
  if (el.__html !== html) { el.innerHTML = html; el.__html = html; }
}
export function setText(el, text) {
  if (!el) return;
  const t = String(text ?? '');
  if (el.textContent !== t) el.textContent = t;
}
/** Fill every [data-k] under root from a map of key -> text. */
export function bind(root, map) {
  for (const [k, v] of Object.entries(map)) {
    root.querySelectorAll(`[data-k="${k}"]`).forEach((el) => setText(el, v));
  }
}

// ---------------------------------------------------------------- formatting
export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
export const fixed = (v, d = 1, dash = '—') => (isNum(v) ? v.toFixed(d) : dash);
export function fmtTps(v) {
  if (!isNum(v)) return '—';
  return v >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(1);
}
export function fmtTok(n) {
  if (!isNum(n)) return '—';
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k`;
  return `${Math.round(n)}`;
}
export function fmtCtx(v) {
  const n = parseInt(v, 10);
  if (!n) return '';
  return n >= 1024 ? `${Math.round(n / 1024)}k` : `${n}`;
}
export function fmtSec(v) {
  if (!isNum(v)) return '—';
  return v < 1 ? `${(v * 1000).toFixed(v < 0.1 ? 1 : 0)} ms` : `${v.toFixed(v < 10 ? 2 : 1)} s`;
}
export const fmtGB = (b) => (isNum(b) ? `${(b / 1073741824).toFixed(1)} GB` : '—');
export const pct = (f, d = 1) => (isNum(f) ? `${(f * 100).toFixed(d)}%` : '—');
export function fmtLatency(ms) {
  if (!isNum(ms)) return '—';
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
}
export function fmtElapsed(sec) {
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m}m ${String(sec % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}
export function ago(tsSec) {
  if (!tsSec) return 'never';
  const s = Math.max(0, Date.now() / 1000 - tsSec);
  if (s < 5) return 'just now';
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
export function parseTs(v) {
  if (v == null) return null;
  if (isNum(v)) return v > 1e12 ? v : v * 1000;
  const t = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(v) ? v : `${v}Z`);
  return Number.isNaN(t) ? null : t;
}
export function fmtTime(v, withDate = false) {
  const ms = parseTs(v);
  if (ms == null) return '—';
  const d = new Date(ms);
  const today = new Date().toDateString() === d.toDateString();
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  if (!withDate && today) return time;
  return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`;
}

export const BACKEND_LABEL = { llamacpp: 'llama.cpp', vllm: 'vLLM', strata: 'Strata', tabbyapi: 'TabbyAPI', beellama: 'BeeLlama' };
export const backendOf = (m) => (m?.backend || (m?.family === 'vllm' ? 'vllm' : 'llamacpp'));
export const isEngineBackend = (m) => ['vllm', 'strata', 'tabbyapi'].includes(backendOf(m)) || m?.family === 'vllm';
export const backendBadge = (m) => {
  const b = backendOf(m);
  return `<span class="backend ${esc(b)}">${esc(BACKEND_LABEL[b] || b)}</span>`;
};

export function modelTags(m, { compact = false } = {}) {
  const t = [];
  if (m.params) t.push(`<span class="tag">${esc(m.params)}</span>`);
  if (m.ctx_size) t.push(`<span class="tag">${esc(fmtCtx(m.ctx_size))} ctx</span>`);
  if (!compact && m.max_seqs && isEngineBackend(m)) t.push(`<span class="tag">${esc(m.max_seqs)} slots</span>`);
  if (m.quant) t.push(`<span class="tag quant" title="${esc(m.quant)}">${esc(m.quant)}</span>`);
  if (m.thinking) t.push('<span class="tag cot">COT</span>');
  if (m.vision) t.push('<span class="tag vision">VISION</span>');
  return t.join('');
}

// ---------------------------------------------------------------- sparkline
export function sparkline(values, color, { height = 30, fill = true, max = null } = {}) {
  const v = values.map((x) => (isNum(x) ? x : 0));
  const n = v.length;
  if (n < 2) return `<svg class="spark" viewBox="0 0 100 ${height}" preserveAspectRatio="none"></svg>`;
  const hi = Math.max(max ?? 0, ...v, 1e-9);
  const pts = v.map((y, i) => [(i / (n - 1)) * 100, height - 2 - (y / hi) * (height - 4)]);
  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join('');
  const area = `${line}L100,${height}L0,${height}Z`;
  const id = `g${Math.random().toString(36).slice(2, 8)}`;
  return `<svg class="spark" viewBox="0 0 100 ${height}" preserveAspectRatio="none" aria-hidden="true">
    <defs><linearGradient id="${id}" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0" stop-color="${color}" stop-opacity=".28"/><stop offset="1" stop-color="${color}" stop-opacity="0"/>
    </linearGradient></defs>
    ${fill ? `<path d="${area}" fill="url(#${id})"/>` : ''}
    <path d="${line}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>
  </svg>`;
}

export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- API
export async function api(path, { method = 'GET', body } = {}) {
  const res = await fetch(path, {
    method,
    cache: 'no-store',
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch { /* empty */ }
  if (!res.ok) {
    const err = new Error(data.error || `HTTP ${res.status}`);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

export function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key) ?? 'null');
    localStorage.setItem(key, JSON.stringify(value));
  } catch { /* private mode */ }
  return value;
}

// ---------------------------------------------------------------- icons (lucide-style, inline)
const P = {
  fleet: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  node: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
  models: '<path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
  requests: '<path d="M3 6h18M3 12h18M3 18h12"/>',
  events: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/>',
  play: '<path d="m6 4 14 8-14 8V4Z"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
  restart: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
  swap: '<path d="M16 3h5v5"/><path d="M4 20 21 3"/><path d="M21 16v5h-5"/><path d="m15 15 6 6"/><path d="M4 4l5 5"/>',
  bolt: '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8Z"/>',
  close: '<path d="M18 6 6 18M6 6l12 12"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  logo: '<path d="M4 17l5-6 4 4 7-9" stroke-width="2.6"/>',
  refresh: '<path d="M21 12a9 9 0 0 1-15.4 6.4L3 16"/><path d="M3 12a9 9 0 0 1 15.4-6.4L21 8"/><path d="M21 3v5h-5M3 21v-5h5"/>',
  gauge: '<path d="M12 14l4-4"/><path d="M3.3 19a10 10 0 1 1 17.4 0"/>',
  plug: '<path d="M12 22v-5M9 8V2M15 8V2M18 8v5a6 6 0 0 1-12 0V8Z"/>',
  trash: '<path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
  key: '<circle cx="7.5" cy="15.5" r="4.5"/><path d="m21 2-9.6 9.6M15.5 7.5l3 3L22 7l-3-3"/>',
};
export const icon = (name, attrs = '') =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" ${attrs}>${P[name] || ''}</svg>`;
