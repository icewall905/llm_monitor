// Control actions shared by the node view, model library and command palette.
import { api, esc, backendBadge } from './util.js';
import { confirmDialog, toast } from './ui.js';
import { nodeById, refreshNow } from './state.js';

const post = (nodeId, action, body = {}) => api(`/api/nodes/${encodeURIComponent(nodeId)}/${action}`, { method: 'POST', body });

export async function switchModel(nodeId, model) {
  const node = nodeById(nodeId);
  const s = node?.snapshot || {};
  if (!node?.online) { toast(`${node?.name || nodeId} is offline`, 'bad'); return; }
  if (s.switch_in_progress) { toast('A switch is already running on this host', 'bad'); return; }
  if (s.active?.key === model.key && s.active?.healthy) { toast(`${model.label} is already running`); return; }
  const stopping = s.active && s.active.key !== model.key
    ? `<p>This stops <b>${esc(s.active.label)}</b> first. One model runs per GPU host.</p>` : '';
  const ok = await confirmDialog({
    title: `Switch ${node.name}`,
    body: `<p style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">${backendBadge(model)} <b>${esc(model.label)}</b></p>${stopping}`,
    confirm: 'Switch model',
  });
  if (!ok) return;
  try {
    const r = await post(nodeId, 'switch', { model: model.key });
    toast(r.message || 'Switch started', 'ok');
  } catch (e) { toast(e.message, 'bad'); }
  refreshNow();
}

export async function restartModel(nodeId) {
  const node = nodeById(nodeId);
  const label = node?.snapshot?.active?.label || 'the active model';
  const ok = await confirmDialog({
    title: 'Restart model',
    body: `<p>Restart <b>${esc(label)}</b> on ${esc(node?.name || nodeId)}? Requests in flight will fail.</p>`,
    confirm: 'Restart', tone: 'warn',
  });
  if (!ok) return;
  try { toast((await post(nodeId, 'restart')).message || 'Restarting', 'ok'); } catch (e) { toast(e.message, 'bad'); }
  refreshNow();
}

export async function stopAll(nodeId) {
  const node = nodeById(nodeId);
  const ok = await confirmDialog({
    title: 'Stop all models',
    body: `<p>Stop every model stack on <b>${esc(node?.name || nodeId)}</b>? The GPU will be freed and no model will serve requests.</p>`,
    confirm: 'Stop all', tone: 'danger',
  });
  if (!ok) return;
  try { toast((await post(nodeId, 'stop')).message || 'Stopping', 'ok'); } catch (e) { toast(e.message, 'bad'); }
  refreshNow();
}

export async function clearSwitch(nodeId) {
  try { toast((await post(nodeId, 'switch/clear')).message || 'Cleared', 'ok'); } catch (e) { toast(e.message, 'bad'); }
  refreshNow();
}

export async function runBenchmark(nodeId, profile) {
  try { toast((await post(nodeId, 'benchmark', { profile })).message || 'Benchmark started', 'ok'); } catch (e) { toast(e.message, 'bad'); }
  refreshNow();
}
