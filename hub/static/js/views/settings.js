// Settings: manage hosts (agents) and each host's stack folders.
import { esc, icon, setHTML, api, ago } from '../util.js';
import { state, nodeById, refreshNow } from '../state.js';
import { openDialog, confirmDialog, toast } from '../ui.js';

const COLORS = ['#5b8def', '#22c55e', '#a78bfa', '#f59e0b', '#14b8a6', '#ef4444', '#ec4899', '#06b6d4'];

export function settingsView(main) {
  let pathNode = null;

  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Settings</h1><p>Hosts, stack folders and appearance</p></div></div>
    <div class="settings-grid">
      <section class="card">
        <div class="card-head" style="padding-bottom:12px"><div><div class="card-title">Hosts</div><div class="card-sub">Each GPU host runs one agent. The hub polls it with a bearer token that never reaches the browser.</div></div>
          <span class="spacer"></span><button class="btn primary sm" data-add>${icon('plus')}Add host</button></div>
        <div data-nodes style="border-top:1px solid var(--border)"></div>
      </section>
      <section class="card">
        <div class="card-head" style="padding-bottom:12px;flex-wrap:wrap">
          <div><div class="card-title">Stack folders</div><div class="card-sub">Paths are as seen inside the host's agent container. Saved on the agent.</div></div>
          <span class="spacer"></span><select class="select" data-path-node></select>
        </div>
        <div data-paths style="border-top:1px solid var(--border)"></div>
      </section>
      <section class="card">
        <div class="card-head"><div class="card-title">Appearance</div></div>
        <div class="card-body"><div class="seg" data-theme>${['dark', 'light'].map((t) => `<button data-v="${t}">${t[0].toUpperCase() + t.slice(1)}</button>`).join('')}</div></div>
      </section>
      <section class="card">
        <div class="card-head"><div class="card-title">Install an agent on another host</div></div>
        <div class="card-body" style="font-size:13px;color:var(--text-2)">
          <p style="margin-top:0">Copy <span class="mono">agent/compose.example.yml</span> from this repo next to the host's LLM stacks, set <span class="mono">AGENT_TOKEN</span> and the stack paths, then <span class="mono">docker compose up -d</span>. Add it here with <span class="mono">http://&lt;host&gt;:9360</span> and the same token.</p>
          <button class="btn sm" data-gen-token>${icon('key')}Generate a token</button> <span class="mono" data-token style="font-size:12px;margin-left:8px;user-select:all"></span>
        </div>
      </section>
    </div>
  </div>`;
  const $q = (s) => main.querySelector(s);

  // theme
  const syncTheme = () => main.querySelectorAll('[data-theme] button').forEach((b) => b.classList.toggle('active', b.dataset.v === document.documentElement.dataset.theme));
  $q('[data-theme]').addEventListener('click', (e) => {
    const b = e.target.closest('[data-v]');
    if (b) { window.dispatchEvent(new CustomEvent('set-theme', { detail: b.dataset.v })); syncTheme(); }
  });
  syncTheme();

  $q('[data-gen-token]').onclick = async () => {
    try { $q('[data-token]').textContent = (await api('/api/token', { method: 'POST', body: {} })).token; } catch (e) { toast(e.message, 'bad'); }
  };
  $q('[data-add]').onclick = () => nodeForm();
  $q('[data-nodes]').addEventListener('click', async (e) => {
    const ed = e.target.closest('[data-edit]');
    const rm = e.target.closest('[data-remove]');
    if (ed) nodeForm(nodeById(ed.dataset.edit));
    if (rm) {
      const n = nodeById(rm.dataset.remove);
      if (await confirmDialog({ title: 'Remove host', body: `<p>Remove <b>${esc(n?.name)}</b> from the hub? The agent keeps running; you can add it again later.</p>`, confirm: 'Remove', tone: 'danger' })) {
        try { await api(`/api/nodes/${encodeURIComponent(rm.dataset.remove)}`, { method: 'DELETE' }); toast('Host removed', 'ok'); refreshNow(); } catch (err) { toast(err.message, 'bad'); }
      }
    }
  });
  $q('[data-path-node]').addEventListener('change', (e) => { pathNode = e.target.value; loadPaths(); });

  function renderNodes() {
    setHTML($q('[data-nodes]'), state.nodes.length ? state.nodes.map((n) => `<div class="node-row">
      <span class="nc-swatch" style="background:${esc(n.color)};width:12px;height:12px"></span>
      <div style="min-width:0">
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><b>${esc(n.name)}</b>
          <span class="pill ${n.online ? 'ok' : 'bad'}"><span class="dot ${n.online ? 'ok' : 'bad'}"></span>${n.online ? `online · ${n.latency_ms ?? '—'} ms` : 'offline'}</span>
          ${n.enabled ? '' : '<span class="pill">disabled</span>'}
          ${n.has_token ? '' : '<span class="pill warn">no token</span>'}</div>
        <div class="url">${esc(n.url)}${n.online ? '' : ` · ${esc(n.error || '')} · last seen ${esc(ago(n.last_seen))}`}</div>
      </div>
      <div class="actions" style="display:flex;gap:6px">
        <button class="btn sm" data-edit="${esc(n.id)}">${icon('edit')}Edit</button>
        <button class="btn sm danger" data-remove="${esc(n.id)}">${icon('trash')}</button>
      </div></div>`).join('') : `<div class="empty">${icon('plug')}<h3>No hosts</h3><p>Add the first agent to start monitoring.</p></div>`);
    const sel = $q('[data-path-node]');
    const online = state.nodes.filter((n) => n.online);
    if (!pathNode || !online.some((n) => n.id === pathNode)) {
      pathNode = online[0]?.id || null;
      if (pathNode) loadPaths(); else setHTML($q('[data-paths]'), '<div class="empty">No host online.</div>');
    }
    setHTML(sel, online.map((n) => `<option value="${esc(n.id)}" ${n.id === pathNode ? 'selected' : ''}>${esc(n.name)}</option>`).join(''));
    sel.hidden = online.length < 2;
  }

  // ---- add / edit host
  function nodeForm(n = null) {
    let color = n?.color || COLORS[state.nodes.length % COLORS.length];
    const dlg = openDialog({
      title: n ? `Edit ${n.name}` : 'Add host',
      wide: true,
      body: `<form class="form" data-form autocomplete="off">
        <div class="form-row"><label>Name</label><input class="input" name="name" required value="${esc(n?.name || '')}" placeholder="e.g. rtx3090-box"></div>
        <div class="form-row"><label>Agent URL</label><input class="input mono" name="url" required value="${esc(n?.url || '')}" placeholder="http://10.0.10.5:9360"><div class="help">Reachable from the hub container. Default agent port is 9360.</div></div>
        <div class="form-row"><label>Token</label><input class="input mono" name="token" type="password" placeholder="${n?.has_token ? 'Leave empty to keep the current token' : 'AGENT_TOKEN from the agent'}"></div>
        <div class="form-row"><label>Colour</label><div class="swatches">${COLORS.map((c) => `<button type="button" data-c="${c}" style="--c:${c}" class="${c === color ? 'on' : ''}"></button>`).join('')}</div></div>
        ${n ? `<label class="check"><input type="checkbox" name="enabled" ${n.enabled ? 'checked' : ''}> Poll this host</label>` : ''}
        <div class="form-actions" style="padding:8px 0 16px">
          <span class="test-result" data-test-out></span>
          <button type="button" class="btn" data-test>${icon('plug')}Test</button>
          <button type="button" class="btn ghost" data-cancel>Cancel</button>
          <button type="submit" class="btn primary">${n ? 'Save' : 'Add host'}</button>
        </div>
      </form>`,
    });
    const form = dlg.el.querySelector('[data-form]');
    form.elements.name.focus();
    dlg.el.querySelector('.swatches').addEventListener('click', (e) => {
      const b = e.target.closest('[data-c]');
      if (!b) return;
      color = b.dataset.c;
      dlg.el.querySelectorAll('.swatches button').forEach((x) => x.classList.toggle('on', x === b));
    });
    dlg.el.querySelector('[data-cancel]').onclick = () => dlg.close();
    const out = dlg.el.querySelector('[data-test-out]');
    dlg.el.querySelector('[data-test]').onclick = async () => {
      out.className = 'test-result'; out.innerHTML = '<span class="spinner" style="display:inline-block;vertical-align:-2px"></span> Testing…';
      try {
        const r = await api('/api/nodes/test', { method: 'POST', body: { url: form.url.value.trim(), token: form.token.value.trim(), id: n?.id } });
        if (r.ok) {
          out.className = 'test-result ok';
          out.textContent = `Connected to “${r.info.name}” · ${r.info.gpus?.length || 0} GPU · ${r.info.model_count} stacks · ${r.latency_ms} ms`;
          if (!form.elements.name.value) form.elements.name.value = r.info.name;
        } else { out.className = 'test-result bad'; out.textContent = r.error || 'Failed'; }
      } catch (e) { out.className = 'test-result bad'; out.textContent = e.message; }
    };
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const body = { name: form.elements.name.value.trim(), url: form.url.value.trim(), color };
      if (form.token.value.trim()) body.token = form.token.value.trim();
      if (n) body.enabled = form.enabled.checked;
      try {
        if (n) await api(`/api/nodes/${encodeURIComponent(n.id)}`, { method: 'PUT', body });
        else await api('/api/nodes', { method: 'POST', body });
        dlg.close();
        toast(n ? 'Host updated' : 'Host added', 'ok');
        setTimeout(refreshNow, 600);
      } catch (err) { out.className = 'test-result bad'; out.textContent = err.message; }
    });
  }

  // ---- stack folders (per host, stored on the agent)
  const inspectTimers = {};
  async function loadPaths() {
    const host = $q('[data-paths]');
    if (!pathNode) return;
    host.innerHTML = '<div class="card-body"><div class="skeleton" style="height:80px"></div></div>';
    try {
      const d = await api(`/api/nodes/${encodeURIComponent(pathNode)}/settings`);
      host.innerHTML = (d.settings || []).map((s) => `<div class="path-row" data-key="${esc(s.key)}">
        <div class="path-head"><b style="font-size:13px">${esc(s.label)}</b><span class="pill">${esc(s.source)}</span>${s.env_value && s.overridden ? `<span class="faint" style="font-size:11.5px">overrides ${esc(s.env_name)}=${esc(s.env_value)}</span>` : ''}</div>
        <div class="help muted" style="font-size:12px">${esc(s.help)}</div>
        <input class="input mono" value="${esc(s.value)}" data-path="${esc(s.key)}" data-scans="${s.scans ? 1 : 0}">
        <div class="path-check" data-check="${esc(s.key)}">${checkText(s, s.scans)}</div>
      </div>`).join('') + `<div class="form-actions" style="padding:14px 16px">
        <span class="test-result ${d.writable ? '' : 'bad'}">${d.writable ? `Saved to ${esc(d.settings_path)} on the agent` : `Settings file ${esc(d.settings_path)} is not writable on the agent`}</span>
        <button class="btn primary" data-save-paths ${d.writable ? '' : 'disabled'}>Save folders</button></div>`;
      host.querySelectorAll('[data-path]').forEach((inp) => inp.addEventListener('input', () => {
        clearTimeout(inspectTimers[inp.dataset.path]);
        inspectTimers[inp.dataset.path] = setTimeout(() => inspect(inp), 350);
      }));
      host.querySelector('[data-save-paths]')?.addEventListener('click', savePaths);
    } catch (e) {
      host.innerHTML = `<div class="empty">Could not load settings: ${esc(e.message)}</div>`;
    }
  }
  function checkText(i, scans) {
    if (!i.exists) return '<span class="bad">Not found inside the agent container — is it bind-mounted?</span>';
    if (!i.is_dir) return '<span class="bad">Not a folder</span>';
    if (!i.readable) return '<span class="bad">Not readable</span>';
    return `<span class="ok">✓ ${scans ? `${i.stack_count} stack${i.stack_count === 1 ? '' : 's'} in ${i.yml_count} yml file${i.yml_count === 1 ? '' : 's'}` : 'folder exists'}</span>`;
  }
  async function inspect(inp) {
    const el = main.querySelector(`[data-check="${inp.dataset.path}"]`);
    const path = inp.value.trim();
    if (!path.startsWith('/')) { el.innerHTML = '<span class="bad">Must be an absolute path</span>'; return; }
    try {
      const i = await api(`/api/nodes/${encodeURIComponent(pathNode)}/settings/inspect?path=${encodeURIComponent(path)}`);
      el.className = 'path-check';
      el.innerHTML = checkText(i, inp.dataset.scans === '1');
    } catch (e) { el.innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
  }
  async function savePaths() {
    const settings = {};
    main.querySelectorAll('[data-path]').forEach((inp) => { settings[inp.dataset.path] = inp.value.trim(); });
    try {
      const r = await api(`/api/nodes/${encodeURIComponent(pathNode)}/settings`, { method: 'POST', body: { settings } });
      toast(r.message || 'Saved', 'ok');
      loadPaths();
    } catch (e) { toast(e.message, 'bad'); }
  }

  renderNodes();
  let sig = '';
  return {
    update() {
      const s = state.nodes.map((n) => `${n.id}${n.name}${n.online}${n.url}${n.enabled}${n.color}${Math.round((n.latency_ms || 0) / 10)}`).join('|');
      if (s !== sig) { sig = s; renderNodes(); }
    },
    destroy() {},
  };
}
