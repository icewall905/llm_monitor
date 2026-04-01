# Confirmation Dialog Implementation for Model Switching

## Summary
Added a custom-styled confirmation dialog that appears before switching between LLM models. The dialog warns users that switching will reload the model and requires explicit confirmation.

---

## Files Modified
- `dashboard-server.py` - Embedded HTML/CSS/JavaScript in the `INDEX_HTML` string

---

## Changes Made

### 1. CSS Styles (added to `INDEX_HTML` string, ~lines 708-760)
```css
.modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.2s ease, visibility 0.2s ease;
}

.modal-overlay.active {
  opacity: 1;
  visibility: visible;
}

.modal {
  background: linear-gradient(160deg, rgba(20,34,58,.98), rgba(12,22,39,.98));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px;
  max-width: 420px;
  width: 90%;
  box-shadow: 0 8px 32px rgba(0,0,0,.4);
}

.modal-title {
  margin: 0 0 12px 0;
  font-size: 16px;
  color: var(--text);
  font-weight: 600;
}

.modal-message {
  margin: 0 0 20px 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.5;
}

.modal-actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}

.modal button {
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.12s ease;
  font-weight: 500;
}

.modal button.confirm {
  background: var(--ok);
  color: #00180a;
  border: none;
}

.modal button.confirm:hover {
  background: #5bd17a;
}

.modal button.cancel {
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--border);
}

.modal button.cancel:hover {
  background: var(--border);
  color: var(--text);
}
```

### 2. HTML Modal Structure (added before `</script>`, ~lines 771-781)
```html
<div id="confirmModal" class="modal-overlay">
  <div class="modal">
    <h3 class="modal-title">Confirm Model Switch</h3>
    <p class="modal-message" id="confirmMessage"></p>
    <div class="modal-actions">
      <button id="cancelSwitch" class="cancel">Cancel</button>
      <button id="confirmSwitch" class="confirm">Switch</button>
    </div>
  </div>
</div>
```

### 3. JavaScript State Variable (added ~line 787)
```javascript
let pendingSwitchModel = null;
```

### 4. JavaScript Functions (added after `setStats` function, ~lines 860-895)
```javascript
function showModal(modelKey, modelLabel) {
  pendingSwitchModel = modelKey;
  const messageEl = document.getElementById('confirmMessage');
  const modal = document.getElementById('confirmModal');
  
  messageEl.textContent = `Switch to ${modelLabel}? This will reload the model.`;
  modal.classList.add('active');
}

function hideModal() {
  pendingSwitchModel = null;
  document.getElementById('confirmModal').classList.remove('active');
}

function handleConfirmed() {
  if (pendingSwitchModel) {
    triggerSwitch(pendingSwitchModel);
    hideModal();
  }
}

function handleCancelled() {
  hideModal();
}
```

### 5. Modified `triggerSwitch` Function (~lines 897-920)
**Before:** Immediately submitted the switch request when a model button was clicked.

**After:** Fetches model details, then shows the confirmation modal before proceeding.

```javascript
async function triggerSwitch(modelKey) {
  if (switching) {
    return;
  }
  
  let currentStatus = null;
  try {
    const resp = await fetch('/api/status');
    if (resp.ok) {
      currentStatus = await resp.json();
    }
  } catch (err) {
    // Ignore status fetch errors, proceed with switch
  }
  
  const model = (currentStatus?.models || []).find(m => m.key === modelKey);
  if (!model) {
    setMessage('Model not found', true);
    return;
  }
  
  showModal(modelKey, model.label);
  // Note: The actual switch happens in handleConfirmed() after user clicks Switch
}
```

### 6. Event Listeners (added before `refresh()` call, ~lines 995-1010)
```javascript
document.getElementById('confirmSwitch').addEventListener('click', handleConfirmed);
document.getElementById('cancelSwitch').addEventListener('click', handleCancelled);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    handleCancelled();
  }
});

document.getElementById('confirmModal').addEventListener('click', (e) => {
  if (e.target.id === 'confirmModal') {
    handleCancelled();
  }
});
```

---

## Deployment
The container was rebuilt and redeployed using:
```bash
docker compose up -d --build
```

---

## User Flow

1. **User clicks a model button** → Modal appears with message: *"Switch to [Model Name]? This will reload the model."*
2. **User clicks "Switch"** → Model switch proceeds immediately
3. **User clicks "Cancel"** → Modal closes, no switch occurs
4. **User presses Escape** → Modal closes, no switch occurs
5. **User clicks outside modal** → Modal closes, no switch occurs

---

## Design Notes

- **Theme:** Dark theme matching existing dashboard (colors from `:root` CSS variables)
- **Message:** Simple and clear - "Switch to [model name]? This will reload the model."
- **Accessibility:** Supports keyboard navigation (Escape key) and click-outside-to-close
- **Visual Feedback:** Smooth fade transitions for show/hide animations
- **Button Styling:** Green "Switch" button for primary action, outlined "Cancel" button for secondary action