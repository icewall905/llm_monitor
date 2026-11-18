// ⌘K command palette: switch a model on any host, or jump to a view.
import { esc, icon, backendBadge, fmtCtx } from './util.js';
import { mountRaw } from './ui.js';
import { state, nodeById } from './state.js';
import { switchModel } from './actions.js';

let open = null;

export function openPalette({ nodeId = null, mode = 'all' } = {}) {
  if (open) open.close();
  const o = mountRaw('palette-overlay', `
    <div class="palette" role="dialog" aria-modal="true" aria-label="Command palette">
      <div class="palette-input">${icon('search')}<input placeholder="${mode === 'models' ? 'Switch to model…' : 'Search models, hosts, views…'}" autocomplete="off" spellcheck="false"><kbd>esc</kbd></div>
      <div class="palette-list" role="listbox"></div>
      <div class="palette-foot"><span><kbd>↑</kbd> <kbd>↓</kbd> navigate</span><span><kbd>↵</kbd> select</span>${nodeId ? `<span>switching on <b>${esc(nodeById(nodeId)?.name || nodeId)}</b></span>` : ''}</div>
    </div>`, () => { open = null; });
  open = o;
  const input = o.el.querySelector('input');
  const list = o.el.querySelector('.palette-list');
  let items = [];
  let sel = 0;

  function allItems() {
    const out = [];
    const nodes = state.nodes.filter((n) => !nodeId || n.id === nodeId);
    for (const n of nodes) {
      const s = n.snapshot || {};
      for (const m of state.models[n.id] || []) {
        const active = s.active?.key === m.key;
        out.push({
          group: nodeId ? 'Models' : `Models on ${n.name}`,
          text: `${m.label} ${m.family || ''} ${m.quant || ''} ${m.backend || ''} ${n.name}`.toLowerCase(),
          html: `${backendBadge(m)}<span class="pi-main">${esc(m.label)}</span><span class="pi-sub">${active ? '<span class="pill ok" style="height:20px">running</span>' : esc([m.params, fmtCtx(m.ctx_size) && fmtCtx(m.ctx_size) + ' ctx'].filter(Boolean).join(' · '))}</span>`,
          run: () => switchModel(n.id, m),
          disabled: !n.online,
        });
      }
    }
    if (mode !== 'models') {
      out.push({ group: 'Go to', text: 'fleet overview home', html: `${icon('fleet', 'width="16" height="16"')}<span class="pi-main">Fleet overview</span>`, run: () => { location.hash = '#/'; } });
      for (const n of state.nodes) {
        out.push({ group: 'Go to', text: `host ${n.name}`.toLowerCase(), html: `<span class="dot ${n.online ? 'ok' : 'bad'}"></span><span class="pi-main">${esc(n.name)}</span><span class="pi-sub">host</span>`, run: () => { location.hash = `#/node/${encodeURIComponent(n.id)}`; } });
      }
      [['models', 'Model library'], ['requests', 'Request log'], ['events', 'Events'], ['settings', 'Settings']].forEach(([k, l]) => {
        out.push({ group: 'Go to', text: l.toLowerCase(), html: `${icon(k, 'width="16" height="16"')}<span class="pi-main">${l}</span>`, run: () => { location.hash = `#/${k}`; } });
      });
    }
    return out;
  }

  function render() {
    const qv = input.value.toLowerCase().trim();
    const terms = qv.split(/\s+/).filter(Boolean);
    items = allItems().filter((it) => terms.every((t) => it.text.includes(t)));
    sel = Math.min(sel, Math.max(0, items.length - 1));
    let html = '';
    let group = null;
    items.forEach((it, i) => {
      if (it.group !== group) { group = it.group; html += `<div class="palette-group eyebrow">${esc(group)}</div>`; }
      html += `<div class="palette-item ${i === sel ? 'sel' : ''}" data-i="${i}" role="option" style="${it.disabled ? 'opacity:.45' : ''}">${it.html}</div>`;
    });
    list.innerHTML = html || '<div class="empty">No matches</div>';
    list.querySelector('.sel')?.scrollIntoView({ block: 'nearest' });
  }

  function choose(i) {
    const it = items[i];
    if (!it || it.disabled) return;
    o.close();
    it.run();
  }

  input.addEventListener('input', () => { sel = 0; render(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); sel = Math.min(items.length - 1, sel + 1); render(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); sel = Math.max(0, sel - 1); render(); }
    if (e.key === 'Enter') { e.preventDefault(); choose(sel); }
  });
  list.addEventListener('mousemove', (e) => {
    const it = e.target.closest('[data-i]');
    if (it && +it.dataset.i !== sel) { sel = +it.dataset.i; list.querySelectorAll('.palette-item').forEach((x) => x.classList.toggle('sel', +x.dataset.i === sel)); }
  });
  list.addEventListener('click', (e) => { const it = e.target.closest('[data-i]'); if (it) choose(+it.dataset.i); });
  render();
  input.focus();
}
