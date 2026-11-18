// Node view: everything about one GPU host — the old single-host dashboard, redesigned.
import {
  esc, icon, bind, setHTML, setText, isNum, fmtTps, fmtTok, fmtSec, fmtGB, pct, fmtElapsed,
  fmtTime, sparkline, cssVar, api, store, modelTags, backendBadge, ago,
} from '../util.js';
import { state, nodeById, derive, nodeStatus, switchPhase, SWITCH_STEPS } from '../state.js';
import { TimeChart, legendHTML } from '../charts.js';
import { restartModel, stopAll, clearSwitch, runBenchmark } from '../actions.js';
import { openPalette } from '../palette.js';

const RANGES = [['live', 'Live'], ['5m', '5m'], ['15m', '15m'], ['1h', '1h'], ['24h', '24h'], ['7d', '7d'], ['30d', '30d']];
const STUCK_SEC = 180;

const PERF_SERIES = [
  { key: 'tps', label: 'Generation', color: '--m-gen', scale: 'tps', fmt: (v) => `${v.toFixed(1)} t/s`, width: 2, fill: true },
  { key: 'ingest', label: 'Ingest', color: '--m-ingest', scale: 'ing', fmt: (v) => `${Math.round(v)} t/s`, hidden: true },
  { key: 'util', label: 'GPU util', color: '--m-util', scale: 'pct', fmt: (v) => `${v.toFixed(0)}%` },
  { key: 'vram', label: 'VRAM', color: '--m-vram', scale: 'pct', fmt: (v) => `${v.toFixed(1)}%` },
  { key: 'ctx', label: 'Context', color: '--m-ctx', scale: 'pct', fmt: (v) => `${v.toFixed(1)}%` },
  { key: 'queue', label: 'Queued', color: '--m-streams', scale: 'q', fmt: (v) => `${Math.round(v)}`, step: true },
];
const PERF_SCALES = {
  pct: { range: [0, 100], side: 3, show: true, grid: true, fmt: (v) => `${v}%` },
  tps: { side: 1, show: true, fmt: (v) => `${v}`, minMax: 10 },
  ing: { show: false, minMax: 10 },
  q: { show: false, minMax: 4 },
};
const THERMAL_SERIES = [
  { key: 'power', label: 'Power', color: '--m-power', scale: 'w', fmt: (v) => `${v.toFixed(0)} W`, width: 2, fill: true },
  { key: 'temp', label: 'Temp', color: '--m-temp', scale: 'pct', fmt: (v) => `${v.toFixed(0)} °C` },
  { key: 'fan', label: 'Fan', color: '--m-fan', scale: 'pct', fmt: (v) => `${v.toFixed(0)}%` },
];
const THERMAL_SCALES = {
  pct: { range: [0, 100], side: 3, show: true, grid: true, fmt: (v) => `${v}` },
  w: { side: 1, show: true, fmt: (v) => `${v}W`, minMax: 100 },
};
const HIST_MAP = { tps: 'tps_max', ingest: 'ingest_max', util: 'util', vram: 'vram_pct', ctx: 'ctx_pct_max', queue: 'waiting_max', power: 'power', temp: 'temp', fan: 'fan' };

const tile = (k, label, color, { meter = false, spark = false, hidden = false } = {}) => `
  <div class="card kpi" data-tile="${k}" style="--c:var(${color})" ${hidden ? 'hidden' : ''}>
    <div class="kpi-label"><span class="dot"></span><span data-k="${k}-label">${esc(label)}</span></div>
    <div class="kpi-value"><span data-k="${k}">—</span><small data-k="${k}-unit"></small></div>
    <div class="kpi-sub" data-k="${k}-sub"></div>
    <div class="kpi-foot">${meter ? `<div class="meter"><i data-meter="${k}"></i></div>` : ''}${spark ? `<div data-spark="${k}"></div>` : ''}</div>
  </div>`;

const kv = (k, label, wide = false) => `<div class="kv ${wide ? 'wide' : ''}"><div class="k">${esc(label)}</div><div class="v" data-k="${k}">—</div></div>`;

export function nodeView(main, nodeId) {
  let range = store('chartRange') || 'live';
  let perf = null;
  let thermal = null;
  let histTimer = null;
  let lastHist = null;

  main.innerHTML = `
  <div class="page" data-node-page>
    <section class="card hero">
      <div class="hero-main">
        <div class="hero-status">
          <span class="pill" data-status><span class="dot"></span><span data-k="status"></span></span>
          <span data-backend></span>
          <span class="muted" style="font-size:12px" data-k="hero-meta"></span>
        </div>
        <h1 class="hero-title" data-k="model">—</h1>
        <div class="tags" data-tags></div>
      </div>
      <div class="hero-actions">
        <button class="btn primary" data-act="switch">${icon('swap')}Switch model</button>
        <button class="btn" data-act="restart">${icon('restart')}Restart</button>
        <button class="btn danger" data-act="stop">${icon('stop')}Stop all</button>
      </div>
    </section>
    <div data-switch-host></div>

    <section class="kpis">
      ${tile('gen', 'Generation', '--m-gen', { spark: true })}
      ${tile('ingest', 'Ingest', '--m-ingest', { spark: true })}
      ${tile('ctx', 'Context', '--m-ctx', { meter: true })}
      ${tile('pool', 'KV pool', '--m-pool', { meter: true, hidden: true })}
      ${tile('streams', 'Streams', '--m-streams', { meter: true, hidden: true })}
      ${tile('cache', 'Prefix cache', '--m-cache', { meter: true, hidden: true })}
      ${tile('util', 'GPU utilization', '--m-util', { meter: true })}
      ${tile('vram', 'VRAM', '--m-vram', { meter: true })}
      ${tile('temp', 'Temperature', '--m-temp', { meter: true })}
    </section>

    <section class="node-grid">
      <div class="node-col">
        <div class="card chart-card">
          <div class="card-head">
            <div><div class="card-title">Performance</div><div class="card-sub" data-k="perf-sub">Generation speed, GPU and context</div></div>
            <div class="seg" data-ranges>${RANGES.map(([k, l]) => `<button data-range="${k}">${l}</button>`).join('')}</div>
          </div>
          <div class="legend" data-legend="perf"></div>
          <div class="chart-wrap" data-chart="perf"><div class="chart-note" data-note="perf"></div></div>
        </div>
        <div class="card chart-card">
          <div class="card-head"><div><div class="card-title">Power &amp; thermals</div><div class="card-sub" data-k="power-sub"></div></div></div>
          <div class="legend" data-legend="thermal"></div>
          <div class="chart-wrap short" data-chart="thermal"><div class="chart-note" data-note="thermal"></div></div>
        </div>
        <div class="card">
          <div class="card-head"><div class="card-title">GPU processes</div><span class="card-sub" data-k="proc-count"></span></div>
          <div class="table-wrap" style="padding:6px 4px 4px">
            <table class="tbl"><thead><tr><th>PID</th><th>Process</th><th>VRAM</th><th class="hide-sm">CPU</th><th class="hide-sm">RAM</th><th>GPU</th></tr></thead><tbody data-procs></tbody></table>
          </div>
        </div>
      </div>

      <div class="node-col side">
        <div class="card" data-engine hidden>
          <div class="card-head"><div class="card-title">Engine</div><span class="card-sub" data-k="eng-sub"></span></div>
          <div class="card-body">
            <div class="eyebrow section-label">Latency</div>
            <div class="kv-list">${kv('ttft', 'Time to first token')}${kv('queue', 'Queue wait')}${kv('itl', 'Inter-token')}${kv('e2e', 'End-to-end')}</div>
            <div class="eyebrow section-label">Batch &amp; tokens</div>
            <div class="kv-list">${kv('batch', 'Tokens / step')}${kv('preempt', 'Preemptions')}${kv('avgprompt', 'Avg prompt')}${kv('avggen', 'Avg generated')}</div>
            <div class="eyebrow section-label">Cache &amp; decoding</div>
            <div class="kv-list">${kv('tiers', 'Prefix hit GPU / host')}${kv('spec', 'Speculative decode')}${kv('offload', 'KV offload read / write')}${kv('loads', 'KV tier promotions')}${kv('finish', 'Finish reasons', true)}</div>
            <div class="eng-note" data-eng-note hidden></div>
          </div>
        </div>

        <div class="card">
          <div class="card-head"><div class="card-title">Model &amp; GPU</div></div>
          <div class="card-body">
            <div class="kv-list">
              ${kv('avgrate', 'Avg rate')}${kv('runs', 'Completed runs')}
              ${kv('clocks', 'Clocks gfx / mem')}${kv('power', 'Power draw')}
              ${kv('gpuname', 'GPU', true)}
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <div class="card-title">Benchmark</div>
            <span class="spacer"></span>
            <button class="btn sm" data-bench="balanced">${icon('bolt')}Quick</button>
            <button class="btn sm" data-bench="full">Full</button>
          </div>
          <div class="card-body">
            <div class="bench-head">
              <div class="bench-num"><div class="eyebrow">Prefill</div><div class="v"><span data-k="b-pre">—</span><small>t/s</small></div></div>
              <div class="bench-num"><div class="eyebrow">Generation</div><div class="v"><span data-k="b-gen">—</span><small>t/s</small></div></div>
            </div>
            <div class="bench-last" data-k="b-last"></div>
            <div class="bench-list" data-bench-list></div>
          </div>
        </div>
      </div>
    </section>
  </div>`;

  const root = main.querySelector('[data-node-page]');
  const q = (sel) => root.querySelector(sel);

  // ---- actions
  q('[data-act=switch]').onclick = () => openPalette({ nodeId, mode: 'models' });
  q('[data-act=restart]').onclick = () => restartModel(nodeId);
  q('[data-act=stop]').onclick = () => stopAll(nodeId);
  root.querySelectorAll('[data-bench]').forEach((b) => { b.onclick = () => runBenchmark(nodeId, b.dataset.bench); });
  root.addEventListener('click', (e) => {
    if (e.target.closest('[data-clear-switch]')) clearSwitch(nodeId);
  });

  // ---- charts
  function cloneScales(sc) { return Object.fromEntries(Object.entries(sc).map(([k, v]) => [k, { ...v }])); }
  function buildChartsSafe() {
    perf?.destroy(); thermal?.destroy();
    perf = new TimeChart(q('[data-chart=perf]'), { series: PERF_SERIES, scales: cloneScales(PERF_SCALES) });
    thermal = new TimeChart(q('[data-chart=thermal]'), { series: THERMAL_SERIES, scales: cloneScales(THERMAL_SCALES) });
    renderLegends();
    drawCharts(true);
  }
  function renderLegends() {
    setHTML(q('[data-legend=perf]'), legendHTML(PERF_SERIES, perf.hidden));
    setHTML(q('[data-legend=thermal]'), legendHTML(THERMAL_SERIES, thermal.hidden));
  }
  root.querySelectorAll('[data-legend]').forEach((lg) => lg.addEventListener('click', (e) => {
    const b = e.target.closest('[data-series]');
    if (!b) return;
    (lg.dataset.legend === 'perf' ? perf : thermal).toggle(b.dataset.series);
    renderLegends();
  }));

  function setRange(r) {
    range = r;
    store('chartRange', r);
    root.querySelectorAll('[data-range]').forEach((b) => b.classList.toggle('active', b.dataset.range === r));
    clearInterval(histTimer);
    lastHist = null;
    if (r === 'live') { setNote(''); drawCharts(true); return; }
    setNote('Loading…');
    fetchHistory();
    histTimer = setInterval(fetchHistory, ['5m', '15m'].includes(r) ? 15000 : 60000);
  }
  root.querySelectorAll('[data-range]').forEach((b) => { b.onclick = () => setRange(b.dataset.range); });

  function setNote(t) {
    setText(q('[data-note=perf]'), t);
    setText(q('[data-note=thermal]'), t);
  }

  async function fetchHistory() {
    const r = range;
    try {
      const d = await api(`/api/nodes/${encodeURIComponent(nodeId)}/history?range=${encodeURIComponent(r)}`);
      if (r !== range) return;
      if (d.error && !(d.points || []).length) { setNote(d.error); lastHist = null; drawCharts(true); return; }
      lastHist = d.points || [];
      setNote(lastHist.length ? '' : 'No samples in this range yet');
      drawCharts(true);
    } catch (e) {
      if (r === range) setNote(`History unavailable: ${e.message}`);
    }
  }

  function drawCharts(force = false) {
    if (!perf) return;
    if (range === 'live') {
      const b = state.buffers[nodeId];
      if (!b) return;
      perf.setData(b.t.slice(), { tps: b.tps.slice(), ingest: b.ingest.slice(), util: b.util.slice(), vram: b.vram.slice(), ctx: b.ctx.slice(), queue: b.queue.slice() });
      thermal.setData(b.t.slice(), { power: b.power.slice(), temp: b.temp.slice(), fan: b.fan.slice() });
    } else if (force && lastHist) {
      const t = lastHist.map((p) => p.t);
      const col = (k) => lastHist.map((p) => (isNum(p[HIST_MAP[k]]) ? p[HIST_MAP[k]] : null));
      perf.setData(t, Object.fromEntries(PERF_SERIES.map((s) => [s.key, col(s.key)])));
      thermal.setData(t, Object.fromEntries(THERMAL_SERIES.map((s) => [s.key, col(s.key)])));
    }
  }

  // ---- per-tick update
  function update() {
    const node = nodeById(nodeId);
    if (!node) {
      main.innerHTML = `<div class="page"><div class="card empty"><h3>Unknown host</h3><p>No host with id “${esc(nodeId)}”.</p><a class="btn" href="#/">Back to fleet</a></div></div>`;
      return;
    }
    const s = node.snapshot || {};
    const st = nodeStatus(node);
    const d = derive(s);
    const a = d.active;
    const g = d.gpu;
    const stats = d.stats;
    const ci = d.ci;

    // hero
    const pill = q('[data-status]');
    pill.className = `pill ${st.cls}`;
    pill.querySelector('.dot').className = `dot ${st.cls}${st.pulse ? ' pulse' : ''}`;
    setText(pill.querySelector('[data-k=status]'), st.label);
    setHTML(q('[data-backend]'), a ? backendBadge(a) : '');
    bind(root, {
      model: !node.online ? `${node.name} is unreachable` : (a ? a.label : 'No model running'),
      'hero-meta': node.online
        ? `${node.name} · ${s.gpu_stats?.length || 0} GPU · agent ${s.agent_version || ''} · ${node.latency_ms ?? '—'} ms`
        : `last seen ${ago(node.last_seen)} · ${node.error || ''}`,
    });
    setHTML(q('[data-tags]'), a ? modelTags(a) : '');
    const busy = !!s.switch_in_progress;
    q('[data-act=restart]').disabled = !node.online || busy || !a;
    q('[data-act=stop]').disabled = !node.online || busy;
    q('[data-act=switch]').disabled = !node.online || busy;

    renderSwitch(node, s);

    // tiles
    const engine = d.engine;
    bind(root, {
      gen: fmtTps(d.gen), 'gen-unit': 't/s', 'gen-sub': d.genSub,
      ingest: fmtTps(d.ingest), 'ingest-unit': 't/s', 'ingest-sub': d.ingestSub,
      'ctx-label': d.ctxLabel,
      ctx: d.nCtx != null ? (d.nPast > 0 ? fmtTok(d.nPast) : '0') : '—',
      'ctx-unit': d.nCtx != null ? `/ ${fmtTok(d.nCtx)}` : '',
      'ctx-sub': d.ctxSub,
      util: isNum(g.util) ? `${g.util.toFixed(0)}` : '—', 'util-unit': '%',
      'util-sub': isNum(g.util) ? (g.util > 5 ? 'compute active' : 'idle') : '',
      vram: isNum(g.mem_used) ? (g.mem_used / 1024).toFixed(1) : '—', 'vram-unit': 'GB',
      'vram-sub': g.mem_total > 0 ? `of ${(g.mem_total / 1024).toFixed(1)} GB · ${((g.mem_used / g.mem_total) * 100).toFixed(0)}%` : '',
      temp: isNum(g.temp) ? `${g.temp.toFixed(0)}` : '—', 'temp-unit': '°C',
      'temp-sub': `fan ${isNum(g.fan) ? g.fan.toFixed(0) : '—'}% · ${isNum(g.power) ? g.power.toFixed(0) : '—'} W`,
    });
    meter('ctx', d.ctxPct, d.ctxPct > 90 ? 'bad' : d.ctxPct > 70 ? 'warn' : '');
    meter('util', g.util || 0);
    meter('vram', g.mem_total > 0 ? (g.mem_used / g.mem_total) * 100 : 0);
    meter('temp', isNum(g.temp) ? Math.max(0, Math.min(100, ((g.temp - 30) / 65) * 100)) : 0, g.temp >= 85 ? 'bad' : g.temp >= 78 ? 'warn' : '');
    const buf = state.buffers[nodeId];
    if (buf) {
      setHTML(q('[data-spark=gen]'), sparkline(buf.tps.slice(-60), cssVar('--m-gen')));
      setHTML(q('[data-spark=ingest]'), sparkline(buf.ingest.slice(-60), cssVar('--m-ingest')));
    }

    ['pool', 'streams', 'cache'].forEach((k) => { q(`[data-tile=${k}]`).hidden = !engine; });
    q('.kpis').classList.toggle('n9', engine);
    q('[data-engine]').hidden = !engine;
    if (engine) updateEngine(d, s);

    // model & gpu
    bind(root, {
      avgrate: stats.average_rate_tps ? `${stats.average_rate_tps.toFixed(2)} t/s` : '—',
      runs: (stats.completed_count || 0).toLocaleString(),
      clocks: `${g.clock_gfx ?? '—'} / ${g.clock_mem ?? '—'} MHz`,
      power: `${isNum(g.power) ? g.power.toFixed(0) : '—'} / ${isNum(g.power_limit) ? g.power_limit.toFixed(0) : '—'} W`,
      gpuname: (d.gpus || []).map((x) => `${x.index}: ${x.name}`).join(' · ') || '—',
      'power-sub': isNum(g.power_limit) ? `limit ${g.power_limit.toFixed(0)} W` : '',
    });
    root.querySelectorAll('.kv .k').forEach((el) => {
      const v = el.nextElementSibling?.dataset.k;
      if (v === 'avgrate') setText(el, engine ? 'Avg rate / request' : 'Avg rate');
      if (v === 'runs') setText(el, engine ? 'Requests served' : 'Completed runs');
    });
    if (thermal && isNum(g.power_limit)) thermal.scalesDef.w.minMax = g.power_limit;

    updateBench(s, node);
    updateProcs(s);
    if (range === 'live') drawCharts();
  }

  function meter(k, p, alert = '') {
    const el = q(`[data-meter=${k}]`);
    if (el) el.style.width = `${Math.max(0, Math.min(100, p || 0)).toFixed(1)}%`;
    const t = q(`[data-tile=${k}]`);
    if (t) {
      t.classList.toggle('alert-warn', alert === 'warn');
      t.classList.toggle('alert-bad', alert === 'bad');
    }
  }

  function renderSwitch(node, s) {
    const host = q('[data-switch-host]');
    if (!node.online || !s.switch_in_progress) host.__key = null; // force card rebuild next switch
    if (!node.online) {
      setHTML(host, `<div class="notice bad">${icon('plug', 'width="16" height="16"')}<span>Agent unreachable: ${esc(node.error || 'no response')}. Showing nothing until it reconnects.</span></div>`);
      return;
    }
    if (!s.switch_in_progress && s.active && !s.active.healthy) {
      // Container is up but not healthy yet (e.g. started outside this hub): show the
      // newest loading line from the log watcher if there is one.
      const evs = s.log_events || [];
      const load = [...evs].reverse().find((e) => e.category === 'loading' || /loading/i.test(e.message || ''));
      setHTML(host, `<div class="notice"><span class="spinner" style="color:var(--warn)"></span><span>Waiting for <b>${esc(s.active.label)}</b> to report healthy${load ? ` · ${esc(load.message)}` : ''}</span></div>`);
      return;
    }
    if (!s.switch_in_progress) {
      const msg = s.last_message;
      const recent = s.last_completed_at && (Date.now() - Date.parse(s.last_completed_at)) < 120000;
      setHTML(host, msg && recent ? `<div class="notice">${esc(msg)}</div>` : '');
      return;
    }
    const started = s.last_started_at ? Date.parse(s.last_started_at) : NaN;
    const elapsed = Number.isNaN(started) ? 0 : (Date.now() - started) / 1000;
    const stuck = elapsed >= STUCK_SEC;
    const ph = switchPhase(s);
    const target = (state.models[nodeId] || []).find((m) => m.key === s.last_requested_model);
    const steps = SWITCH_STEPS.map((label, i) => {
      const cls = i < ph.step ? 'done' : i === ph.step ? `now${ph.pct == null ? ' indet' : ''}` : '';
      return `<div class="step ${cls}" style="--p:${ph.pct ?? 0}%"><i></i><span>${label}${i === ph.step && ph.pct != null ? ` ${ph.pct}%` : ''}</span></div>`;
    }).join('');
    // Elapsed ticks every second; rebuild the whole card only when the structure changes.
    const key = `${ph.step}|${ph.pct}|${stuck}|${s.last_message}`;
    if (host.__key !== key) {
      host.__key = key;
      host.innerHTML = `
        <div class="card switch-card ${stuck ? 'stuck' : ''}">
          <div class="switch-top">
            <span class="spinner" style="color:var(--accent)"></span>
            <span class="msg">${esc(s.last_message || 'Switching…')}</span>
            ${target ? `<span class="muted" style="font-size:12px">→ ${esc(target.label)}</span>` : ''}
            <span class="elapsed" data-elapsed></span>
            <button class="btn sm ${stuck ? 'warn solid' : 'ghost'}" data-clear-switch title="Unlock the UI. Does not stop the container.">Force clear</button>
          </div>
          <div class="stepper">${steps}</div>
        </div>`;
      host.__html = null;
    }
    setText(host.querySelector('[data-elapsed]'), fmtElapsed(elapsed));
  }

  function updateEngine(d, s) {
    const stats = d.stats, ci = d.ci, a = d.active || {};
    // KV pool: resident tokens out of real capacity (spans all requests; not context usage).
    const used = ci.pool_used, tot = ci.pool_tokens;
    const poolPct = used != null && tot ? (used / tot) * 100 : 0;
    const conc = ci.pool_max_concurrency;
    // Streams vs slot cap from LLM_META max_seqs.
    const run = stats.requests_running, wait = stats.requests_waiting;
    const maxSeqs = parseInt(a.max_seqs || '0', 10) || null;
    const wr = stats.waiting_by_reason || {};
    let waitTxt = 'concurrent requests';
    if (wait > 0) {
      const why = wr.capacity > 0 && wr.deferred > 0 ? 'capacity + deferred' : wr.capacity > 0 ? 'waiting on capacity' : wr.deferred > 0 ? 'deferred' : 'queued';
      waitTxt = `${Math.round(wait)} ${why}`;
    } else if (run != null) waitTxt = run > 0 ? 'no queue' : 'idle';
    const avgQ = stats.avg_queue_time_sec;
    if (avgQ != null && run != null && (run > 0 || wait > 0)) waitTxt += ` · avg wait ${avgQ < 10 ? avgQ.toFixed(1) : Math.round(avgQ)}s`;
    const frac = stats.prompt_cached_frac;

    bind(root, {
      pool: used != null && tot ? fmtTok(used) : '—', 'pool-unit': tot ? `/ ${fmtTok(tot)}` : '',
      'pool-sub': used != null && tot ? `${poolPct.toFixed(1)}% resident${conc ? ` · ${conc.toFixed(2)}× capacity` : ''}` : 'tokens resident',
      streams: run == null ? '—' : `${Math.round(run)}`, 'streams-unit': maxSeqs ? `/ ${maxSeqs}` : '',
      'streams-sub': waitTxt,
      cache: frac == null ? '—' : (frac * 100).toFixed(1), 'cache-unit': frac == null ? '' : '%',
      'cache-sub': stats.prefix_hit_rate == null && stats.external_prefix_hit_rate == null
        ? 'prompt tokens reused' : `GPU ${pct(stats.prefix_hit_rate)} · host ${pct(stats.external_prefix_hit_rate)}`,
    });
    meter('pool', poolPct, poolPct > 90 ? 'bad' : poolPct > 75 ? 'warn' : '');
    meter('streams', run != null && maxSeqs ? (run / maxSeqs) * 100 : 0, wait > 0 ? 'warn' : '');
    meter('cache', frac == null ? 0 : frac * 100);

    const ttft = stats.avg_ttft_sec, queue = stats.avg_queue_time_sec;
    const qShare = ttft && queue != null && ttft > 0 ? queue / ttft : null;
    const pre = stats.preemptions;
    let spec = 'off', specCls = '';
    if (stats.spec_present) {
      const ar = stats.spec_accept_rate, al = stats.spec_accept_len;
      spec = ar == null ? 'on' : `${pct(ar)} accept${al ? ` · ${al.toFixed(2)}×` : ''}`;
      specCls = ar == null ? '' : ar < 0.05 ? 'bad' : ar < 0.2 ? 'warn' : 'good';
    }
    const loads = stats.offload_load_count, tierQ = stats.offload_tier_queries;
    const sbr = stats.success_by_reason || {};
    const finish = Object.entries(sbr).filter(([, v]) => v > 0).sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k} ${Math.round(v)}`).join(' · ');
    const vals = {
      ttft: [fmtSec(ttft), ttft == null ? '' : ttft > 10 ? 'bad' : ttft > 3 ? 'warn' : 'good'],
      queue: [fmtSec(queue), qShare == null ? '' : qShare > 0.5 ? 'bad' : qShare > 0.2 ? 'warn' : 'good'],
      itl: [fmtSec(stats.avg_itl_sec ?? stats.avg_tpot_sec), ''],
      e2e: [fmtSec(stats.avg_e2e_sec), ''],
      batch: [stats.avg_iter_tokens == null ? '—' : stats.avg_iter_tokens.toFixed(1), ''],
      preempt: [pre == null ? '—' : Math.round(pre).toLocaleString(), pre == null ? '' : pre > 0 ? 'warn' : 'good'],
      avgprompt: [`${fmtTok(stats.avg_prompt_per_request)} tok`, ''],
      avggen: [`${fmtTok(stats.avg_gen_per_request)} tok`, ''],
      tiers: [`${pct(stats.prefix_hit_rate)} / ${pct(stats.external_prefix_hit_rate)}`, ''],
      spec: [spec, specCls],
      offload: [stats.offload_read_bytes == null && stats.offload_store_bytes == null ? '—' : `${fmtGB(stats.offload_read_bytes)} / ${fmtGB(stats.offload_store_bytes)}`, ''],
      loads: [loads == null ? (tierQ == null ? '—' : `0 of ${Math.round(tierQ).toLocaleString()}`) : `${Math.round(loads).toLocaleString()}${tierQ ? ` of ${Math.round(tierQ).toLocaleString()}` : ''}`,
        loads == null ? (tierQ ? 'warn' : '') : 'good'],
      finish: [finish || '—', ''],
    };
    for (const [k, [txt, cls]] of Object.entries(vals)) {
      const el = q(`.kv [data-k="${k}"]`);
      if (!el) continue;
      setText(el, txt);
      el.className = `v ${cls}`;
    }
    setText(q('[data-k=eng-sub]'), `${s.active?.backend || 'vllm'} metrics${isNum(stats.gauge_age_sec) && stats.gauge_age_sec > 3 ? ` · ${Math.round(stats.gauge_age_sec)}s old` : ''}`);

    // Plain-language bottleneck note.
    const notes = [];
    if (wait > 0 && wr.capacity > 0 && maxSeqs && run != null && run < maxSeqs) {
      notes.push(`${Math.round(wait)} request(s) waiting on capacity with ${maxSeqs - Math.round(run)} slot(s) free — the prefill token budget is the limit, not max_num_seqs.`);
    }
    if (qShare != null && qShare > 0.5) notes.push(`${pct(qShare)} of time-to-first-token is queue wait.`);
    if (pre > 0) notes.push(`${Math.round(pre)} preemption(s): the KV pool is being exhausted.`);
    if (stats.spec_present && stats.spec_accept_rate != null && stats.spec_accept_rate < 0.05) notes.push('Speculative decoding is accepting almost nothing — pure overhead right now.');
    const noteEl = q('[data-eng-note]');
    noteEl.hidden = !notes.length;
    setText(noteEl, notes.join(' '));
  }

  function updateBench(s, node) {
    const bench = s.benchmark || {};
    const canRun = node.online && !!s.active?.healthy && !s.switch_in_progress && !bench.in_progress;
    root.querySelectorAll('[data-bench]').forEach((b) => {
      b.disabled = !canRun;
      const running = bench.in_progress && (bench.profile || 'balanced') === b.dataset.bench;
      const label = b.dataset.bench === 'full' ? 'Full' : 'Quick';
      setHTML(b, running ? `<span class="spinner"></span>Running` : (b.dataset.bench === 'full' ? label : `${icon('bolt')}${label}`));
    });
    const r = bench.last_result || {};
    let last = '';
    if (bench.in_progress && bench.started_at) last = `${(bench.profile || 'balanced') === 'full' ? 'Full' : 'Quick'} run in progress since ${fmtTime(bench.started_at)}`;
    else if (bench.last_error) last = `Last run failed: ${bench.last_error}`;
    else if (r.completed_at) last = `${r.profile === 'full' ? 'Full' : 'Quick'} · ${r.model_label || ''} · ${fmtTime(r.completed_at, true)}`;
    else last = 'No run yet on this agent.';
    bind(root, {
      'b-pre': r.prefill_tps ? r.prefill_tps.toFixed(1) : '—',
      'b-gen': r.gen_tps ? r.gen_tps.toFixed(1) : '—',
      'b-last': last,
    });
    const rows = (bench.history || []).slice().reverse().slice(0, 10);
    setHTML(q('[data-bench-list]'), rows.length ? rows.map((x) => {
      const prof = x.profile === 'full' ? 'full' : 'quick';
      const head = `<span class="bp ${prof}">${prof.toUpperCase()}</span><span class="bm" title="${esc(x.model_label || x.model_key)}">${esc(x.model_label || x.model_key || 'unknown')}</span><span class="bt">${esc(fmtTime(x.completed_at || x.started_at, true))}</span>`;
      if (!x.success) return `<div class="bench-row fail">${head}<div class="bv">${esc(x.error || 'failed')}</div></div>`;
      const acc = typeof x.draft_acceptance === 'number' ? `<span>draft <b>${(x.draft_acceptance * 100).toFixed(0)}%</b></span>` : '';
      return `<div class="bench-row">${head}<div class="bv"><span>prefill <b>${Number(x.prefill_tps || 0).toFixed(1)}</b></span><span>gen <b>${Number(x.gen_tps || 0).toFixed(1)}</b></span>${acc}</div></div>`;
    }).join('') : '');
  }

  function updateProcs(s) {
    const procs = s.gpu_procs || [];
    setText(q('[data-k=proc-count]'), procs.length ? `${procs.length} running` : '');
    setHTML(q('[data-procs]'), procs.length ? procs.map((p) => `<tr>
      <td class="num muted">${esc(p.pid)}</td>
      <td><span class="trunc" title="${esc(p.name)}" style="font-weight:600">${esc(p.name)}</span></td>
      <td class="num">${p.vram_mb != null ? (p.vram_mb / 1024).toFixed(1) + ' GB' : '—'}</td>
      <td class="num hide-sm">${p.cpu_pct != null ? p.cpu_pct.toFixed(1) + '%' : '—'}</td>
      <td class="num hide-sm">${p.ram_mb != null ? (p.ram_mb / 1024).toFixed(1) + ' GB' : '—'}</td>
      <td class="num">${p.gpu_pct != null ? p.gpu_pct.toFixed(0) + '%' : '—'}</td></tr>`).join('')
      : '<tr><td colspan="6" class="muted" style="text-align:center;padding:18px">No compute processes</td></tr>');
  }

  // init
  if (typeof uPlot !== 'undefined') buildChartsSafe();
  else setNote('Chart library failed to load');
  root.querySelectorAll('[data-range]').forEach((b) => b.classList.toggle('active', b.dataset.range === range));
  setRange(range);
  update();

  return {
    update,
    retheme: () => { if (typeof uPlot !== 'undefined') buildChartsSafe(); },
    destroy: () => { clearInterval(histTimer); perf?.destroy(); thermal?.destroy(); },
  };
}
