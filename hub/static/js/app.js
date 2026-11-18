// App shell: rail, top bar, hash router, keyboard shortcuts, theme.
import { esc, icon, setHTML, fmtTps } from './util.js';
import { state, connect, onChange, derive, nodeStatus } from './state.js';
import { openPalette } from './palette.js';
import { fleetView } from './views/fleet.js';
import { nodeView } from './views/node.js';
import { modelsView } from './views/models.js';
import { requestsView } from './views/requests.js';
import { eventsView } from './views/events.js';
import { settingsView } from './views/settings.js';

const main = document.getElementById('main');
const rail = document.getElementById('rail');
const topbar = document.getElementById('topbar');

let view = null;
let route = { name: '', arg: null, params: new URLSearchParams() };
let lastNodeId = null;
let ready = false;

// ------------------------------------------------------------------ theme
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('theme', t); } catch { /* ignore */ }
  document.querySelector('meta[name=theme-color]')?.setAttribute('content', t === 'light' ? '#f4f5f8' : '#0a0c10');
  renderTop();
  view?.retheme?.();
}
window.addEventListener('set-theme', (e) => setTheme(e.detail));

// ------------------------------------------------------------------ router
function parseHash() {
  const h = location.hash.replace(/^#\/?/, '');
  const [path, qs] = h.split('?');
  const parts = path.split('/').filter(Boolean);
  return { name: parts[0] || '', arg: parts[1] ? decodeURIComponent(parts[1]) : null, params: new URLSearchParams(qs || '') };
}

function navigate() {
  if (!ready) return;
  let r = parseHash();
  // A single-host setup opens straight into that host.
  if (r.name === '' && state.nodes.length === 1) r = { name: 'node', arg: state.nodes[0].id, params: r.params };
  if (r.name === '') r.name = 'fleet';
  if (r.name === 'node' && !r.arg) r.arg = lastNodeId || state.nodes[0]?.id;
  if (r.name === 'node') lastNodeId = r.arg;
  const same = r.name === route.name && r.arg === route.arg && r.params.toString() === route.params.toString();
  route = r;
  if (!same || !view) {
    view?.destroy?.();
    main.scrollTop = 0;
    const views = { fleet: fleetView, node: nodeView, models: modelsView, requests: requestsView, events: eventsView, settings: settingsView };
    const fn = views[r.name] || fleetView;
    view = r.name === 'node' ? fn(main, r.arg) : fn(main, r.params);
  }
  renderRail();
  renderTop();
}
window.addEventListener('hashchange', navigate);

// ------------------------------------------------------------------ chrome
function renderRail() {
  const errors = state.nodes.reduce((a, n) => a + (n.snapshot?.log_error_count || 0), 0);
  const nodeTarget = lastNodeId || state.nodes[0]?.id;
  const item = (name, label, href, ic, badge = '') => `<a class="rail-btn ${route.name === name ? 'active' : ''}" href="${href}" aria-label="${label}">
      ${icon(ic)}<span class="rail-label">${label}</span>${badge}</a>`;
  setHTML(rail, `
    <a class="logo" href="#/fleet" aria-label="Home"><svg viewBox="0 0 64 64" aria-hidden="true"><rect x="10" y="14" width="44" height="36" rx="9" fill="#0f1a2a" stroke="#5ec2ff" stroke-width="4"/><circle cx="25" cy="31" r="5" fill="#5ec2ff"/><circle cx="39" cy="31" r="5" fill="#5ec2ff"/><rect x="24" y="41" width="16" height="3" rx="1.5" fill="#6fe58d"/><rect x="30" y="6" width="4" height="8" rx="2" fill="#5ec2ff"/><circle cx="32" cy="5" r="3" fill="#6fe58d"/></svg></a>
    ${item('fleet', 'Fleet', '#/fleet', 'fleet')}
    ${nodeTarget ? item('node', 'Host', `#/node/${encodeURIComponent(nodeTarget)}`, 'node') : ''}
    ${item('models', 'Models', '#/models', 'models')}
    ${item('requests', 'Requests', '#/requests', 'requests')}
    ${item('events', 'Events', '#/events', 'events', errors ? `<span class="rail-badge">${errors > 99 ? '99+' : errors}</span>` : '')}
    <div class="rail-bottom">${item('settings', 'Settings', '#/settings', 'settings')}</div>`);
}

function renderTop() {
  const titles = { fleet: 'Fleet', node: 'Host', models: 'Models', requests: 'Requests', events: 'Events', settings: 'Settings' };
  const pills = state.nodes.length > 1 || route.name === 'node'
    ? state.nodes.map((n) => {
      const st = nodeStatus(n);
      const d = n.online ? derive(n.snapshot) : null;
      const on = route.name === 'node' && route.arg === n.id;
      return `<a class="node-pill ${on ? 'active' : ''}" href="#/node/${encodeURIComponent(n.id)}" title="${esc(n.snapshot?.active?.label || st.label)}">
        <span class="dot ${st.cls}${st.pulse ? ' pulse' : ''}"></span>${esc(n.name)}${d && d.genLive > 0 ? `<span class="tps">${fmtTps(d.genLive)}</span>` : ''}</a>`;
    }).join('') : '';
  const dark = document.documentElement.dataset.theme !== 'light';
  setHTML(topbar, `
    <div class="crumbs"><span>${titles[route.name] || 'Fleet'}</span></div>
    <div class="node-pills">${pills}</div>
    <span class="spacer"></span>
    <div class="top-actions">
      <span class="conn" title="${state.connected ? 'Live updates connected' : 'Reconnecting to hub…'}"><span class="dot ${state.connected ? 'ok' : 'warn pulse'}"></span><span class="lbl">${state.connected ? 'Live' : 'Reconnecting'}</span></span>
      <button class="kbd-search" data-palette>${icon('search', 'width="15" height="15"')}<span class="lbl">Switch model or jump…</span><span class="k">⌘K</span></button>
      <button class="icon-btn" data-theme-toggle aria-label="Toggle theme">${icon(dark ? 'sun' : 'moon')}</button>
    </div>`);
}
topbar.addEventListener('click', (e) => {
  if (e.target.closest('[data-palette]')) openPalette({ nodeId: route.name === 'node' ? route.arg : null });
  if (e.target.closest('[data-theme-toggle]')) setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
});

document.addEventListener('keydown', (e) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName);
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette({ nodeId: route.name === 'node' ? route.arg : null }); return; }
  if (typing || e.metaKey || e.ctrlKey || e.altKey || document.querySelector('.overlay')) return;
  const go = { f: '#/fleet', m: '#/models', r: '#/requests', e: '#/events', s: '#/settings' }[e.key];
  if (go) location.hash = go;
  if (e.key === '/') { e.preventDefault(); openPalette(); }
  if (/^[1-9]$/.test(e.key) && state.nodes[+e.key - 1]) location.hash = `#/node/${encodeURIComponent(state.nodes[+e.key - 1].id)}`;
});

// ------------------------------------------------------------------ live updates
let rafPending = false;
onChange(() => {
  if (!ready) { ready = true; navigate(); return; }
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    if (document.hidden) return;
    view?.update?.(state);
    renderRail();
    renderTop();
    const n = route.name === 'node' ? state.nodes.find((x) => x.id === route.arg) : null;
    const tps = n?.online ? derive(n.snapshot).genLive : 0;
    document.title = n ? `${tps > 0 ? `${fmtTps(tps)} t/s · ` : ''}${n.name} — LLM Fleet` : 'LLM Fleet';
  });
});

renderTop();
main.innerHTML = '<div class="page"><div class="kpis">' + '<div class="card kpi skeleton"></div>'.repeat(6) + '</div></div>';
connect();
