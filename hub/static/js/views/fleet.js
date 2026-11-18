// Fleet overview: one card per GPU host.
import { esc, icon, setHTML, isNum, fmtTps, fmtTok, sparkline, cssVar, ago, backendBadge } from '../util.js';
import { state, derive, nodeStatus, switchPhase } from '../state.js';

function gpuBlock(g) {
  const vp = g.mem_total > 0 ? (g.mem_used / g.mem_total) * 100 : 0;
  return `<div class="gpu-mini">
    <span class="gname" title="${esc(g.name)}">GPU ${esc(g.index)} · ${esc((g.name || '').replace(/^NVIDIA\s+(GeForce\s+)?/, ''))}</span>
    <span class="gm"><b>${isNum(g.util) ? g.util.toFixed(0) : '—'}</b>%</span>
    <span class="gm"><b>${isNum(g.mem_used) ? (g.mem_used / 1024).toFixed(1) : '—'}</b>GB</span>
    <span class="gm" style="${g.temp >= 85 ? 'color:var(--bad)' : ''}"><b>${isNum(g.temp) ? g.temp.toFixed(0) : '—'}</b>°C</span>
    <span class="gm"><b>${isNum(g.power) ? g.power.toFixed(0) : '—'}</b>W</span>
    <div class="meters">
      <div class="meter" title="GPU utilization"><i style="--c:var(--m-util);width:${g.util || 0}%"></i></div>
      <div class="meter" title="VRAM"><i style="--c:var(--m-vram);width:${vp.toFixed(1)}%"></i></div>
    </div>
  </div>`;
}

function nodeCard(n) {
  const st = nodeStatus(n);
  const s = n.snapshot || {};
  const head = `<div class="nc-head">
      <span class="nc-swatch" style="background:${esc(n.color || 'var(--accent)')}"></span>
      <span class="nc-name">${esc(n.name)}</span>
      <span class="pill ${st.cls}"><span class="dot ${st.cls}${st.pulse ? ' pulse' : ''}"></span>${st.label}</span>
      <span class="nc-meta">${n.online ? `${n.latency_ms ?? '—'} ms` : ''}</span>
    </div>`;
  if (!n.online) {
    return `<article class="card node-card offline" data-node="${esc(n.id)}">${head}
      <div class="nc-offline"><b>${esc(n.error || 'Not reachable')}</b><span>Last seen ${esc(ago(n.last_seen))} · ${esc(n.url || '')}</span></div></article>`;
  }
  const d = derive(s);
  const a = d.active;
  const buf = state.buffers[n.id];
  let switching = '';
  if (s.switch_in_progress) {
    const ph = switchPhase(s);
    switching = `<div class="nc-switch"><span class="spinner"></span><span>${esc(s.last_message || 'Switching…')}${ph.pct != null ? '' : ''}</span></div>`;
  }
  const engine = d.engine;
  const third = engine
    ? `<div><div class="eyebrow">Streams</div><div class="mid num">${isNum(d.stats.requests_running) ? Math.round(d.stats.requests_running) : '—'}<span class="muted" style="font-size:12px">${d.stats.requests_waiting > 0 ? ` +${Math.round(d.stats.requests_waiting)} queued` : ''}</span></div></div>`
    : `<div><div class="eyebrow">Context</div><div class="mid num">${d.nPast > 0 ? fmtTok(d.nPast) : '0'}<span class="muted" style="font-size:12px">${d.nCtx ? ` / ${fmtTok(d.nCtx)}` : ''}</span></div></div>`;
  return `<article class="card node-card" data-node="${esc(n.id)}">${head}
    <div class="nc-model">
      <div class="nc-model-row">${a ? backendBadge(a) : ''}<span class="nc-model-name" title="${esc(a?.label || '')}">${esc(a ? a.label : 'No model running')}</span></div>
    </div>
    ${switching}
    <div class="nc-perf">
      <div><div class="eyebrow">Generation</div><div class="big">${fmtTps(d.gen)}<small>t/s</small></div><div class="muted" style="font-size:11.5px;margin-top:3px">${esc(d.genSub)}</div></div>
      <div><div class="eyebrow">Ingest</div><div class="mid num">${fmtTps(d.ingest)}</div></div>
      ${third}
    </div>
    <div class="nc-spark">${buf ? sparkline(buf.tps.slice(-90), cssVar('--m-gen'), { height: 38 }) : ''}</div>
    ${engine || d.nCtx ? `<div style="padding:8px 16px 0"><div class="meter" title="${esc(d.ctxLabel)}: ${d.ctxPct.toFixed(1)}%"><i style="--c:var(--m-ctx);width:${d.ctxPct.toFixed(1)}%"></i></div></div>` : ''}
    <div class="nc-gpus">${(d.gpus || []).map(gpuBlock).join('') || '<span class="muted">No GPU data</span>'}</div>
  </article>`;
}

export function fleetView(main) {
  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Fleet</h1><p data-sub></p></div></div>
    <section class="fleet-summary" data-summary></section>
    <section class="fleet-grid" data-grid></section>
  </div>`;
  const grid = main.querySelector('[data-grid]');
  grid.addEventListener('click', (e) => {
    const c = e.target.closest('[data-node]');
    if (c) location.hash = `#/node/${encodeURIComponent(c.dataset.node)}`;
    if (e.target.closest('[data-add]')) location.hash = '#/settings';
  });

  function update() {
    const nodes = state.nodes;
    const online = nodes.filter((n) => n.online);
    let totalTps = 0, gpus = 0, vramUsed = 0, vramTotal = 0, power = 0, running = 0;
    for (const n of online) {
      const d = derive(n.snapshot);
      totalTps += d.genLive || 0;
      if (n.snapshot?.active?.healthy) running += 1;
      for (const g of d.gpus) { gpus += 1; vramUsed += g.mem_used || 0; vramTotal += g.mem_total || 0; power += g.power || 0; }
    }
    setHTML(main.querySelector('[data-sub]'), esc(`${online.length} of ${nodes.length} host${nodes.length === 1 ? '' : 's'} online · ${gpus} GPU${gpus === 1 ? '' : 's'}`));
    setHTML(main.querySelector('[data-summary]'), `
      <div class="card sum-tile"><div class="eyebrow">Models serving</div><div class="v">${running}<small>/ ${nodes.length}</small></div></div>
      <div class="card sum-tile"><div class="eyebrow">Generation now</div><div class="v" style="color:var(--m-gen)">${fmtTps(totalTps)}<small>t/s</small></div></div>
      <div class="card sum-tile"><div class="eyebrow">VRAM in use</div><div class="v">${(vramUsed / 1024).toFixed(0)}<small>/ ${(vramTotal / 1024).toFixed(0)} GB</small></div></div>
      <div class="card sum-tile"><div class="eyebrow">Power draw</div><div class="v">${power.toFixed(0)}<small>W</small></div></div>`);
    setHTML(grid, nodes.map(nodeCard).join('') + `<button class="card add-node-card" data-add>${icon('plus', 'width="22" height="22"')}<span>Add a host</span></button>`);
    if (!nodes.length) {
      setHTML(grid, `<div class="card empty" style="grid-column:1/-1">${icon('plug')}<h3>No hosts yet</h3><p>Run the agent on a GPU host, then add its URL and token in Settings.</p><a class="btn primary" href="#/settings">Add a host</a></div>`);
    }
  }
  update();
  return { update, destroy() {} };
}
