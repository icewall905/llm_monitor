// uPlot wrapper: themed, responsive time-series with a hover tooltip and
// toggleable series. Series colours come from CSS metric tokens.
import { cssVar, esc, isNum } from './util.js';

const fmtAxisTime = (ts, span) => {
  const d = new Date(ts * 1000);
  if (span > 2 * 86400) return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  if (span < 900) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
};

/**
 * series: [{key, label, color: '--m-gen', scale: 'pct'|'tps'|..., unit, fmt, step, hidden}]
 * scales: {pct: {range: [0,100], side: 0|1, show: true, label}}
 */
export class TimeChart {
  constructor(host, { series, scales }) {
    this.host = host;
    this.seriesDef = series;
    this.scalesDef = scales;
    this.hidden = new Set(series.filter((s) => s.hidden).map((s) => s.key));
    this.data = [[], ...series.map(() => [])];
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    host.appendChild(this.tip);
    this.build();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
  }

  build() {
    if (this.u) this.u.destroy();
    const grid = cssVar('--border');
    const axisColor = cssVar('--faint');
    const axes = [{
      stroke: axisColor, grid: { show: false }, ticks: { show: false }, size: 28, gap: 6,
      values: (u, splits) => {
        const span = (u.scales.x.max ?? 0) - (u.scales.x.min ?? 0);
        return splits.map((v) => fmtAxisTime(v, span));
      },
      font: '11px JetBrains Mono, monospace',
    }];
    for (const [name, sc] of Object.entries(this.scalesDef)) {
      if (!sc.show) continue;
      axes.push({
        scale: name, side: sc.side ?? 3, stroke: axisColor, size: 44, gap: 4,
        grid: { show: sc.grid ?? false, stroke: grid, width: 1 }, ticks: { show: false },
        values: (u, splits) => splits.map((v) => (sc.fmt ? sc.fmt(v) : v)),
        font: '11px JetBrains Mono, monospace',
      });
    }
    const scales = { x: { time: true } };
    for (const [name, sc] of Object.entries(this.scalesDef)) {
      scales[name] = sc.range ? { range: sc.range } : { range: (u, min, max) => [0, Math.max((isNum(max) ? max : 0) * 1.15, sc.minMax || 1)] };
    }
    const series = [{}].concat(this.seriesDef.map((s) => {
      const color = cssVar(s.color) || s.color;
      return {
        label: s.label, scale: s.scale, stroke: color, width: s.width ?? 1.8, show: !this.hidden.has(s.key),
        fill: s.fill ? `${color}22` : undefined, spanGaps: false,
        paths: s.step ? uPlot.paths.stepped({ align: 1 }) : undefined,
        points: { show: false },
      };
    }));
    const { width, height } = this.size();
    this.u = new uPlot({
      width, height, series, axes, scales,
      legend: { show: false },
      cursor: { points: { size: 7, fill: cssVar('--surface') }, drag: { x: false, y: false } },
      padding: [10, 6, 0, 0],
      hooks: { setCursor: [(u) => this.onCursor(u)] },
    }, this.data, this.host);
    this.host.appendChild(this.tip);
  }

  size() {
    const r = this.host.getBoundingClientRect();
    return { width: Math.max(200, Math.floor(r.width)), height: Math.max(120, Math.floor(r.height)) };
  }
  resize() { if (this.u) this.u.setSize(this.size()); }

  setData(ts, cols) {
    this.data = [ts, ...this.seriesDef.map((s) => cols[s.key] || ts.map(() => null))];
    this.u.setData(this.data);
  }

  setScaleMax(name, max) {
    const sc = this.scalesDef[name];
    if (sc && sc.minMax !== max) { sc.minMax = max; }
  }

  toggle(key) {
    const i = this.seriesDef.findIndex((s) => s.key === key);
    if (i < 0) return;
    if (this.hidden.has(key)) this.hidden.delete(key); else this.hidden.add(key);
    this.u.setSeries(i + 1, { show: !this.hidden.has(key) });
  }

  onCursor(u) {
    const idx = u.cursor.idx;
    if (idx == null || u.cursor.left < 0) { this.tip.style.display = 'none'; return; }
    const ts = u.data[0][idx];
    let rows = '';
    this.seriesDef.forEach((s, i) => {
      if (this.hidden.has(s.key)) return;
      const v = u.data[i + 1][idx];
      const txt = isNum(v) ? (s.fmt ? s.fmt(v) : v.toFixed(1)) : '—';
      rows += `<div class="tr" style="--c:${cssVar(s.color)}"><i></i><span>${esc(s.label)}</span>${esc(txt)}</div>`;
    });
    this.tip.innerHTML = `<div class="tt">${new Date(ts * 1000).toLocaleString([], { hour12: false })}</div>${rows}`;
    this.tip.style.display = 'block';
    const w = this.tip.offsetWidth;
    const hostW = this.host.clientWidth;
    let left = u.cursor.left + 18;
    if (left + w > hostW) left = u.cursor.left - w - 12;
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = '8px';
  }

  destroy() {
    this.ro.disconnect();
    this.u?.destroy();
    this.tip.remove();
  }
}

export function legendHTML(series, hidden) {
  return series.map((s) => `<button data-series="${s.key}" class="${hidden.has(s.key) ? 'off' : ''}" style="--c:var(${s.color})"><i></i>${esc(s.label)}</button>`).join('');
}
