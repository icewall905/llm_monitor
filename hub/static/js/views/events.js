// Event timeline from every agent's log watcher (switches, loads, OOMs, heartbeats…).
import { esc, icon, setHTML, fmtTime, parseTs } from '../util.js';
import { state } from '../state.js';

export function eventsView(main) {
  const f = { sev: 'all', node: 'all', hideHeartbeat: true };
  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Events</h1><p data-sub></p></div></div>
    <div class="toolbar">
      <div class="seg" data-sev>${[['all', 'All'], ['error', 'Errors'], ['warning', 'Warnings'], ['info', 'Info']].map(([k, l]) => `<button data-v="${k}" class="${k === 'all' ? 'active' : ''}">${l}</button>`).join('')}</div>
      <select class="select" data-node></select>
      <label class="check"><input type="checkbox" data-hb checked> Hide heartbeats</label>
    </div>
    <div class="card"><div class="card-body timeline" data-list></div></div>
  </div>`;
  main.querySelector('[data-sev]').addEventListener('click', (e) => {
    const b = e.target.closest('[data-v]');
    if (!b) return;
    f.sev = b.dataset.v;
    main.querySelectorAll('[data-sev] button').forEach((x) => x.classList.toggle('active', x === b));
    update();
  });
  main.querySelector('[data-node]').addEventListener('change', (e) => { f.node = e.target.value; update(); });
  main.querySelector('[data-hb]').addEventListener('change', (e) => { f.hideHeartbeat = e.target.checked; update(); });

  function update() {
    const sel = main.querySelector('[data-node]');
    setHTML(sel, `<option value="all">All hosts</option>${state.nodes.map((n) => `<option value="${esc(n.id)}" ${f.node === n.id ? 'selected' : ''}>${esc(n.name)}</option>`).join('')}`);
    sel.hidden = state.nodes.length < 2;
    const multi = state.nodes.length > 1;
    const evs = [];
    let errors = 0;
    for (const n of state.nodes) {
      if (f.node !== 'all' && f.node !== n.id) continue;
      errors += n.snapshot?.log_error_count || 0;
      for (const e of n.snapshot?.log_events || []) evs.push({ ...e, node: n });
    }
    evs.sort((a, b) => (parseTs(b.ts) || 0) - (parseTs(a.ts) || 0));
    const shown = evs.filter((e) => (f.sev === 'all' || e.severity === f.sev) && !(f.hideHeartbeat && e.category === 'heartbeat'));
    setHTML(main.querySelector('[data-sub]'), esc(`${shown.length} events · ${errors} error${errors === 1 ? '' : 's'} since agent start`));
    setHTML(main.querySelector('[data-list]'), shown.length ? shown.map((e) => `<div class="ev">
      <span class="when">${esc(fmtTime(e.ts))}</span>
      <span class="sev ${esc(e.severity)}"></span>
      <span class="ev-msg">${multi ? `<span class="node-tag">${esc(e.node.name)}</span>` : ''}${e.category ? `<span class="cat">${esc(e.category)}</span>` : ''}${esc(e.message)}</span>
    </div>`).join('') : `<div class="empty">${icon('events')}<h3>Nothing here</h3><p>No events match.</p></div>`);
  }
  update();
  return { update, destroy() {} };
}
