// Request log across all hosts, with a detail side sheet.
import { esc, icon, setHTML, api, fmtLatency, fmtTime, store } from '../util.js';
import { state, nodeById } from '../state.js';
import { openSheet } from '../ui.js';

export function requestsView(main) {
  const f = { node: 'all', endpoint: '', q: '', live: store('reqLive') ?? true, inference: store('reqInference') ?? true };
  let timer = null;
  let busy = false;
  let sheet = null;
  let endpoints = [];

  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Requests</h1><p data-sub>Logged by the access proxies on each host</p></div></div>
    <div class="toolbar">
      <div class="input-wrap">${icon('search')}<input class="input" data-q placeholder="Filter IP, user agent, model, path…" autocomplete="off"></div>
      <select class="select" data-node></select>
      <select class="select" data-ep></select>
      <label class="check" title="Hide /status, /props, /v1/models and other monitoring calls"><input type="checkbox" data-inf ${f.inference ? 'checked' : ''}> Inference only</label>
      <label class="check"><input type="checkbox" data-live ${f.live ? 'checked' : ''}> Live</label>
      <button class="icon-btn" data-refresh title="Refresh">${icon('refresh')}</button>
    </div>
    <div class="card">
      <div class="table-wrap" style="max-height:calc(100dvh - 230px)">
        <table class="tbl req-table">
          <thead><tr><th>Time</th><th data-col-node>Host</th><th class="hide-sm">Endpoint</th><th class="hide-sm">Origin</th><th>Model</th><th class="hide-sm">Path</th><th>Status</th><th>Latency</th><th class="hide-sm">Tokens</th><th class="hide-sm">Prompt</th></tr></thead>
          <tbody data-rows><tr><td colspan="10"><div class="skeleton" style="height:16px"></div></td></tr></tbody>
        </table>
      </div>
    </div>
  </div>`;
  const $q = (s) => main.querySelector(s);
  let deb = null;
  $q('[data-q]').addEventListener('input', (e) => { clearTimeout(deb); deb = setTimeout(() => { f.q = e.target.value.trim(); load(); }, 300); });
  $q('[data-node]').addEventListener('change', (e) => { f.node = e.target.value; loadEndpoints(); load(); });
  $q('[data-ep]').addEventListener('change', (e) => { f.endpoint = e.target.value; load(); });
  $q('[data-live]').addEventListener('change', (e) => { f.live = e.target.checked; store('reqLive', f.live); });
  $q('[data-inf]').addEventListener('change', (e) => { f.inference = e.target.checked; store('reqInference', f.inference); load(); });
  $q('[data-refresh]').onclick = () => load();
  $q('[data-rows]').addEventListener('click', (e) => {
    const tr = e.target.closest('[data-id]');
    if (tr) openDetail(tr.dataset.node, tr.dataset.id);
  });

  function renderSelects() {
    const nodes = state.nodes;
    setHTML($q('[data-node]'), `<option value="all">All hosts</option>${nodes.map((n) => `<option value="${esc(n.id)}" ${f.node === n.id ? 'selected' : ''}>${esc(n.name)}</option>`).join('')}`);
    $q('[data-node]').hidden = nodes.length < 2;
    $q('[data-col-node]').hidden = nodes.length < 2;
    setHTML($q('[data-ep]'), `<option value="">All endpoints</option>${endpoints.map((e) => `<option value="${esc(e.endpoint)}" ${f.endpoint === e.endpoint ? 'selected' : ''}>:${esc(e.endpoint)} (${e.count})</option>`).join('')}`);
  }

  async function loadEndpoints() {
    const targets = state.nodes.filter((n) => n.online && (f.node === 'all' || n.id === f.node));
    const seen = new Map();
    await Promise.all(targets.map(async (n) => {
      try {
        const d = await api(`/api/nodes/${encodeURIComponent(n.id)}/requests/endpoints`);
        for (const e of d.endpoints || []) seen.set(e.endpoint, (seen.get(e.endpoint) || 0) + e.count);
      } catch { /* host may lack a request log */ }
    }));
    endpoints = [...seen.entries()].map(([endpoint, count]) => ({ endpoint, count })).sort((a, b) => a.endpoint.localeCompare(b.endpoint));
    renderSelects();
  }

  async function load() {
    if (busy) return;
    busy = true;
    const p = new URLSearchParams({ limit: '300', node: f.node });
    if (f.endpoint) p.set('endpoint', f.endpoint);
    if (f.q) p.set('q', f.q);
    if (f.inference) p.set('only', 'inference');
    try {
      const d = await api(`/api/requests?${p}`);
      render(d);
    } catch (e) {
      setHTML($q('[data-rows]'), `<tr><td colspan="10" class="muted" style="padding:20px">Error: ${esc(e.message)}</td></tr>`);
    } finally { busy = false; }
  }

  function render(d) {
    const rows = d.rows || [];
    const multi = state.nodes.length > 1;
    const errs = Object.entries(d.errors || {}).map(([id, e]) => `${nodeById(id)?.name || id}: ${e}`);
    setHTML($q('[data-sub]'), esc(`${rows.length} most recent${errs.length ? ` · ${errs.join(' · ')}` : ''}`));
    if (!rows.length) {
      setHTML($q('[data-rows]'), `<tr><td colspan="10"><div class="empty">${icon('requests')}<h3>No requests</h3><p>${f.q || f.endpoint ? 'Nothing matches these filters.' : 'Send a request through a host\'s access proxy and it shows up here.'}</p></div></td></tr>`);
      return;
    }
    setHTML($q('[data-rows]'), rows.map((r) => {
      const n = nodeById(r.node);
      const ok = r.status && r.status < 400;
      const tok = r.prompt_tokens != null || r.completion_tokens != null ? `${r.prompt_tokens ?? '?'} → ${r.completion_tokens ?? '?'}` : '—';
      return `<tr class="clickable" data-id="${esc(r.id)}" data-node="${esc(r.node)}">
        <td class="t">${esc(fmtTime(r.ts))}</td>
        ${multi ? `<td><span class="node-tag" style="display:inline-flex;gap:6px;align-items:center"><span class="nc-swatch" style="background:${esc(n?.color || 'var(--muted)')};width:8px;height:8px"></span>${esc(n?.name || r.node)}</span></td>` : '<td hidden></td>'}
        <td class="hide-sm"><span class="ep">:${esc(r.endpoint || '?')}</span></td>
        <td class="hide-sm"><span class="ip">${esc(r.client_ip || '—')}</span><span class="ua" title="${esc(r.user_agent || '')}">${esc(r.user_agent || '')}</span></td>
        <td><span class="trunc" style="max-width:180px" title="${esc(r.model || '')}">${esc(r.model || '—')}</span></td>
        <td class="hide-sm"><span class="mono" style="font-size:12px">${esc(r.path || '—')}</span>${r.stream ? ' <span title="streamed" style="color:var(--accent)">↯</span>' : ''}</td>
        <td><span class="status-code ${ok ? '' : 'bad'}">${esc(r.status ?? '—')}</span></td>
        <td class="num">${esc(fmtLatency(r.latency_ms))}</td>
        <td class="num hide-sm">${esc(tok)}</td>
        <td class="hide-sm"><span class="trunc muted" style="max-width:320px" title="${esc(r.prompt_snippet || '')}">${esc(r.prompt_snippet || '')}</span></td>
      </tr>`;
    }).join(''));
  }

  async function openDetail(nodeId, id) {
    sheet = openSheet({ title: `Request #${id}`, onClose: () => { sheet = null; } });
    sheet.body.innerHTML = '<div class="skeleton" style="height:120px"></div>';
    try {
      const d = await api(`/api/nodes/${encodeURIComponent(nodeId)}/requests/detail?id=${encodeURIComponent(id)}`);
      if (!d.row) { sheet.body.innerHTML = `<div class="empty">Not found${d.error ? `: ${esc(d.error)}` : ''}</div>`; return; }
      sheet.title.textContent = `${d.row.method || ''} ${d.row.path || ''} · #${id}`;
      sheet.body.innerHTML = renderDetail(d.row, nodeById(nodeId));
    } catch (e) {
      sheet.body.innerHTML = `<div class="empty">Error: ${esc(e.message)}</div>`;
    }
  }

  renderSelects();
  loadEndpoints();
  load();
  timer = setInterval(() => { if (f.live && !sheet && !document.hidden) load(); }, 3000);
  let lastNodeCount = state.nodes.length;
  return {
    update() { if (state.nodes.length !== lastNodeCount) { lastNodeCount = state.nodes.length; renderSelects(); } },
    destroy() { clearInterval(timer); },
  };
}

// ------------------------------------------------------------------ detail

function contentHTML(content) {
  if (typeof content === 'string') return esc(content);
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (!part || typeof part !== 'object') return esc(String(part));
      if (part.type === 'text') return esc(part.text || '');
      if (part.type === 'image_url') {
        const u = part.image_url?.url || '';
        return `<span class="rd-img">image · ${esc(u.length > 70 ? `${u.slice(0, 70)}…` : u)}</span>\n`;
      }
      return esc(JSON.stringify(part));
    }).join('');
  }
  if (content == null) return '<span class="muted">(empty)</span>';
  return esc(JSON.stringify(content, null, 2));
}

function message(role, content, reasoning, extra = '') {
  const r = ['system', 'user', 'assistant', 'tool'].includes(role) ? role : 'system';
  return `<div class="msg"><div class="msg-role ${r}">${esc(role || 'message')}${extra}</div>
    ${reasoning ? `<details><summary>Reasoning · ${reasoning.length.toLocaleString()} chars</summary><div class="msg-content muted">${esc(reasoning)}</div></details>` : ''}
    <div class="msg-content">${contentHTML(content) || '<span class="muted">(no content)</span>'}</div></div>`;
}

function renderDetail(r, node) {
  const meta = [
    ['Host', esc(node?.name || '—')],
    ['Endpoint', `:${esc(r.endpoint || '?')}`],
    ['Time', esc(fmtTime(r.ts_iso || r.ts, true))],
    ['Status', esc(r.status ?? '—')],
    ['Latency', esc(fmtLatency(r.latency_ms))],
    ['Stream', r.stream ? 'yes' : 'no'],
    ['Tokens p / c / total', esc(`${r.prompt_tokens ?? '?'} / ${r.completion_tokens ?? '?'} / ${r.total_tokens ?? '?'}`)],
    ['Origin IP', esc(r.client_ip || '—')],
    ['Method', esc(r.method || '—')],
    ['Path', esc(r.path || '—')],
    ['User agent', esc(r.user_agent || '—')],
  ];
  let html = `<div class="rd-meta">${meta.map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`).join('')}</div>`;
  if (r.error) html += `<div class="notice bad">${esc(r.error)}</div>`;

  let req = null;
  try { req = r.request_body ? JSON.parse(r.request_body) : null; } catch { /* truncated */ }
  html += '<div class="rd-title eyebrow">Request</div>';
  if (req) {
    const keys = ['model', 'temperature', 'top_p', 'top_k', 'max_tokens', 'presence_penalty', 'frequency_penalty', 'stream'];
    const params = keys.filter((k) => req[k] !== undefined).map((k) => `<span><b>${k}</b> ${esc(JSON.stringify(req[k]))}</span>`).join('');
    if (params) html += `<div class="rd-params">${params}</div>`;
    if (Array.isArray(req.messages)) html += req.messages.map((m) => message(m.role, m.content, m.reasoning_content)).join('');
    else if (typeof req.prompt === 'string') html += message('prompt', req.prompt);
  } else if (r.request_body) {
    html += `<p class="muted">Request body too large to parse (${r.request_body.length.toLocaleString()} chars, likely truncated). See raw JSON below.</p>`;
  } else {
    html += '<p class="muted">No request body stored (only chat, completions and embeddings bodies are captured).</p>';
  }

  let resp = null;
  try { resp = r.response_body ? JSON.parse(r.response_body) : null; } catch { /* truncated */ }
  html += `<div class="rd-title eyebrow">Response${resp?._reconstructed_from_stream ? ' <span class="pill" style="height:20px">reconstructed from stream</span>' : ''}</div>`;
  if (resp && Array.isArray(resp.choices) && resp.choices.length) {
    const ch = resp.choices[0];
    const msg = ch.message || ch.delta || {};
    html += message('assistant', msg.content, msg.reasoning_content);
    const bits = [];
    if (ch.finish_reason) bits.push(`<span><b>finish</b> ${esc(ch.finish_reason)}</span>`);
    if (resp.usage) Object.entries(resp.usage).forEach(([k, v]) => { if (typeof v !== 'object') bits.push(`<span><b>${esc(k)}</b> ${esc(v)}</span>`); });
    if (bits.length) html += `<div class="rd-params">${bits.join('')}</div>`;
  } else if (resp) {
    html += '<p class="muted">Non-chat response — see raw JSON.</p>';
  } else if (r.response_body) {
    html += `<p class="muted">Response body too large to parse (${r.response_body.length.toLocaleString()} chars). See raw JSON below.</p>`;
  } else {
    html += '<p class="muted">No response body stored.</p>';
  }

  if (r.request_body || r.response_body) {
    html += '<details class="rd-raw"><summary>Raw JSON</summary>';
    if (r.request_body) html += `<div class="eyebrow" style="margin-top:10px">request_body</div><pre>${esc(r.request_body)}</pre>`;
    if (r.response_body) html += `<div class="eyebrow" style="margin-top:10px">response_body</div><pre>${esc(r.response_body)}</pre>`;
    html += '</details>';
  }
  return html;
}
