// Overlays: confirm dialog, side sheet, toasts.
import { esc, icon } from './util.js';

const stack = [];
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && stack.length) { e.preventDefault(); stack[stack.length - 1].close(); }
});

function mountOverlay(cls, inner, onClose) {
  const el = document.createElement('div');
  el.className = `overlay ${cls}`;
  el.innerHTML = inner;
  document.body.appendChild(el);
  const entry = {
    el,
    close: (val) => {
      const i = stack.indexOf(entry);
      if (i >= 0) stack.splice(i, 1);
      el.remove();
      onClose?.(val);
    },
  };
  stack.push(entry);
  el.addEventListener('mousedown', (e) => { if (e.target === el) entry.close(); });
  return entry;
}

/** Promise<boolean>. tone: 'primary' | 'danger' | 'warn' */
export function confirmDialog({ title, body, confirm = 'Confirm', tone = 'primary' }) {
  return new Promise((resolve) => {
    const btnCls = tone === 'danger' ? 'btn danger solid' : tone === 'warn' ? 'btn warn solid' : 'btn primary';
    const o = mountOverlay('', `
      <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="dlg-t">
        <div class="dialog-head"><h2 id="dlg-t">${esc(title)}</h2></div>
        <div class="dialog-body">${body}</div>
        <div class="dialog-foot">
          <button class="btn ghost" data-a="no">Cancel</button>
          <button class="${btnCls}" data-a="yes">${esc(confirm)}</button>
        </div>
      </div>`, (v) => resolve(!!v));
    o.el.querySelector('[data-a=no]').onclick = () => o.close(false);
    const yes = o.el.querySelector('[data-a=yes]');
    yes.onclick = () => o.close(true);
    yes.focus();
  });
}

/** Generic dialog with custom body; returns {el, close}. */
export function openDialog({ title, body, wide = false, onClose }) {
  const o = mountOverlay('', `
    <div class="dialog ${wide ? 'wide' : ''}" role="dialog" aria-modal="true">
      <div class="dialog-head"><h2>${esc(title)}</h2></div>
      <div class="dialog-body">${body}</div>
    </div>`, onClose);
  return o;
}

export function openSheet({ title, onClose }) {
  const o = mountOverlay('sheet-overlay', `
    <aside class="sheet" role="dialog" aria-modal="true">
      <div class="sheet-head">
        <h2 data-sheet-title>${esc(title)}</h2>
        <span class="spacer"></span>
        <button class="icon-btn" data-close aria-label="Close">${icon('close')}</button>
      </div>
      <div class="sheet-body" data-sheet-body></div>
    </aside>`, onClose);
  o.el.querySelector('[data-close]').onclick = () => o.close();
  o.body = o.el.querySelector('[data-sheet-body]');
  o.title = o.el.querySelector('[data-sheet-title]');
  return o;
}

export function mountRaw(cls, inner, onClose) { return mountOverlay(cls, inner, onClose); }

export function toast(message, tone = '') {
  const host = document.getElementById('toasts');
  const el = document.createElement('div');
  el.className = `toast ${tone}`;
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .3s'; }, 4200);
  setTimeout(() => el.remove(), 4600);
}
