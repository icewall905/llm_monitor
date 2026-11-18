// Model library: every stack on every host, grouped by family, with one-click switch.
import { esc, icon, setHTML, store, modelTags, backendBadge, fmtTok, BACKEND_LABEL, backendOf } from '../util.js';
import { state, nodeById } from '../state.js';
import { switchModel } from '../actions.js';

const FAMILY_ORDER = ['vllm', 'muse-glimmer', 'gemma', 'gptoss', 'glm', 'mistral', 'nemotron', 'qwen', 'deepseek', 'other'];
const FAMILY_LABELS = {
  vllm: 'vLLM engines', qwen: 'Qwen', nemotron: 'Nemotron', gptoss: 'GPT-OSS', glm: 'GLM', mistral: 'Mistral',
  gemma: 'Gemma', deepseek: 'DeepSeek', 'muse-glimmer': 'Muse Glimmer', other: 'Other',
};

export function modelsView(main, params) {
  const f = {
    q: '',
    node: params.get('node') || 'all',
    backend: 'all',
    cot: false,
    vision: false,
  };
  const collapsed = store('familyCollapsed') || {};

  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Models</h1><p data-sub></p></div></div>
    <div class="toolbar">
      <div class="input-wrap">${icon('search')}<input class="input" data-q placeholder="Search name, family, quant…" autocomplete="off"></div>
      <div class="chips" data-chips></div>
    </div>
    <div data-list></div>
  </div>`;
  const qInput = main.querySelector('[data-q]');
  qInput.addEventListener('input', () => { f.q = qInput.value.toLowerCase().trim(); update(); });
  main.querySelector('[data-chips]').addEventListener('click', (e) => {
    const c = e.target.closest('[data-f]');
    if (!c) return;
    const [k, v] = c.dataset.f.split(':');
    if (k === 'cot' || k === 'vision') f[k] = !f[k];
    else f[k] = f[k] === v ? 'all' : v;
    update();
  });
  const list = main.querySelector('[data-list]');
  list.addEventListener('click', (e) => {
    const sw = e.target.closest('[data-switch]');
    if (sw) {
      const [nid, key] = sw.dataset.switch.split('|');
      const m = (state.models[nid] || []).find((x) => x.key === key);
      if (m) switchModel(nid, m);
      return;
    }
    const h = e.target.closest('[data-fam]');
    if (h) {
      collapsed[h.dataset.fam] = !collapsed[h.dataset.fam];
      store('familyCollapsed', collapsed);
      update();
    }
  });

  function rows() {
    const out = [];
    for (const n of state.nodes) {
      for (const m of state.models[n.id] || []) out.push({ n, m });
    }
    return out;
  }

  function update() {
    const all = rows();
    const nodes = state.nodes;
    const backends = [...new Set(all.map(({ m }) => backendOf(m)))].sort();
    const chip = (key, label, on, count) => `<button class="chip ${on ? 'on' : ''}" data-f="${key}">${esc(label)}${count != null ? `<span class="count">${count}</span>` : ''}</button>`;
    let chips = '';
    if (nodes.length > 1) chips += nodes.map((n) => chip(`node:${n.id}`, n.name, f.node === n.id, (state.models[n.id] || []).length)).join('');
    chips += backends.map((b) => chip(`backend:${b}`, BACKEND_LABEL[b] || b, f.backend === b, all.filter(({ m }) => backendOf(m) === b).length)).join('');
    chips += chip('cot:1', 'Reasoning', f.cot) + chip('vision:1', 'Vision', f.vision);
    setHTML(main.querySelector('[data-chips]'), chips);

    const shown = all.filter(({ n, m }) => {
      if (f.node !== 'all' && n.id !== f.node) return false;
      if (f.backend !== 'all' && backendOf(m) !== f.backend) return false;
      if (f.cot && !m.thinking) return false;
      if (f.vision && !m.vision) return false;
      if (f.q) {
        const hay = `${m.label} ${m.family} ${m.quant} ${m.params} ${m.key} ${n.name}`.toLowerCase();
        if (!f.q.split(/\s+/).every((t) => hay.includes(t))) return false;
      }
      return true;
    });
    setHTML(main.querySelector('[data-sub]'), esc(`${shown.length} of ${all.length} stacks across ${nodes.length} host${nodes.length === 1 ? '' : 's'}`));

    const groups = {};
    for (const r of shown) (groups[r.m.family || 'other'] ||= []).push(r);
    const order = FAMILY_ORDER.concat(Object.keys(groups).filter((x) => !FAMILY_ORDER.includes(x)).sort());
    const multi = nodes.length > 1;
    let html = '';
    for (const fam of order) {
      const g = groups[fam];
      if (!g) continue;
      html += `<section class="fam-group ${collapsed[fam] ? 'collapsed' : ''}">
        <div class="fam-head" data-fam="${esc(fam)}">${icon('chevron', 'class="chev"')}<span class="eyebrow">${esc(FAMILY_LABELS[fam] || fam)}</span><span class="faint num" style="font-size:11px">${g.length}</span></div>
        <div class="model-grid">${g.map(({ n, m }) => card(n, m, multi)).join('')}</div>
      </section>`;
    }
    setHTML(list, html || `<div class="card empty">${icon('models')}<h3>No models match</h3><p>${all.length ? 'Try clearing a filter.' : 'No host has reported any stacks yet.'}</p></div>`);
  }

  function card(n, m, multi) {
    const s = n.snapshot || {};
    const active = s.active?.key === m.key;
    const target = s.switch_in_progress && s.last_requested_model === m.key;
    const stats = [];
    if (m.bench && (m.bench.gen_tps || m.bench.prefill_tps)) {
      const when = m.bench.ts ? new Date(m.bench.ts * 1000).toLocaleString() : '';
      const t = esc(`Last ${(m.bench.profile || '').toUpperCase()} benchmark ${when}`);
      if (m.bench.gen_tps) stats.push(`<span class="s-gen" title="${t}">gen <b>${Number(m.bench.gen_tps).toFixed(1)}</b></span>`);
      if (m.bench.prefill_tps) stats.push(`<span class="s-pre" title="${t}">ingest <b>${Math.round(m.bench.prefill_tps)}</b></span>`);
    }
    if (m.kv_pool?.pool_tokens) {
      const conc = m.kv_pool.max_concurrency;
      stats.push(`<span class="s-kv" title="${esc(`Last KV pool ${m.kv_pool.pool_tokens.toLocaleString()} tokens${conc ? `, ${Number(conc).toFixed(1)}× full windows` : ''}`)}">kv <b>${fmtTok(m.kv_pool.pool_tokens)}</b></span>`);
    }
    if (active) {
      const ci = s.context_info || {};
      if (!ci.n_past_is_average && ci.n_ctx > 0 && ci.n_past > 0) stats.push(`<span class="s-ctx">ctx <b>${fmtTok(ci.n_past)}</b></span>`);
    }
    const busy = s.switch_in_progress || !n.online;
    let btn;
    if (active) btn = `<span class="pill ${s.active?.healthy ? 'ok' : 'warn'}" style="margin-left:auto"><span class="dot ${s.active?.healthy ? 'ok' : 'warn pulse'}"></span>${s.active?.healthy ? 'Running' : 'Starting'}</span>`;
    else if (target) btn = '<span class="pill accent" style="margin-left:auto"><span class="spinner"></span>Switching</span>';
    else btn = `<button class="btn sm" data-switch="${esc(n.id)}|${esc(m.key)}" ${busy ? 'disabled' : ''}>${icon('play')}Switch</button>`;
    return `<article class="card model-card ${active ? 'active' : ''} ${target ? 'target' : ''}">
      <div class="mc-top"><div class="mc-name">${esc(m.label)}</div></div>
      <div class="tags">${backendBadge(m)}${modelTags(m, { compact: true })}</div>
      ${stats.length ? `<div class="mc-stats">${stats.join('')}</div>` : ''}
      <div class="mc-foot">${multi ? `<span class="node-tag"><span class="nc-swatch" style="background:${esc(n.color)};width:8px;height:8px"></span>${esc(n.name)}</span>` : ''}${btn}</div>
    </article>`;
  }

  update();
  return { update, destroy() {} };
}
