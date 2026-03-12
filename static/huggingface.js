// ═══════════════════════════════════════════════════════════
//  HuggingFace Model ID Autocomplete
// ═══════════════════════════════════════════════════════════
(function () {
  const HF_API = 'https://huggingface.co/api/models';
  const DEBOUNCE_MS = 300;
  const MAX_RESULTS = 8;

  let dropdown = null;
  let debounceTimer = null;
  let activeIdx = -1;
  let lastQuery = '';
  let currentResults = [];

  // ── Init ──────────────────────────────────────────────────
  function init() {
    const input = document.getElementById('hfModelId');
    if (!input) return;

    // Fixed-position dropdown — appended to body to escape overflow clipping
    dropdown = document.createElement('div');
    dropdown.id = 'hf-dropdown';
    dropdown.className = 'hf-dropdown';
    document.body.appendChild(dropdown);

    input.addEventListener('input', onInput);
    input.addEventListener('keydown', onKeyDown);
    input.addEventListener('focus', () => {
      if (currentResults.length && input.value.trim() === lastQuery) showDropdown();
    });

    document.addEventListener('click', e => {
      if (!e.target.closest('#hfModelId') && !e.target.closest('#hf-dropdown')) {
        hideDropdown();
      }
    });
  }

  // ── Input handler ─────────────────────────────────────────
  function onInput(e) {
    const q = e.target.value.trim();
    clearTimeout(debounceTimer);

    if (q.length < 2) {
      lastQuery = q;
      currentResults = [];
      hideDropdown();
      return;
    }

    if (q === lastQuery && currentResults.length) {
      showDropdown();
      return;
    }

    debounceTimer = setTimeout(() => fetchSuggestions(q), DEBOUNCE_MS);
  }

  // ── Fetch suggestions from HF API ─────────────────────────
  async function fetchSuggestions(query) {
    lastQuery = query;
    try {
      const url = `${HF_API}?search=${encodeURIComponent(query)}&limit=${MAX_RESULTS}&sort=downloads&direction=-1&filter=text-generation`;
      const res = await fetch(url);
      if (!res.ok) return;
      const models = await res.json();
      currentResults = models;
      renderDropdown(models, query);
    } catch (_) {
      // Autocomplete is best-effort — don't surface network errors
    }
  }

  // ── Render dropdown items ─────────────────────────────────
  function renderDropdown(models, query) {
    activeIdx = -1;

    if (!models.length) {
      hideDropdown();
      return;
    }

    dropdown.innerHTML = models.map((m, i) => {
      const dl = m.downloads ? formatNum(m.downloads) : '';
      const tag = m.pipeline_tag ?? '';
      return `<div class="hf-item" data-idx="${i}" data-id="${escHtml(m.modelId ?? m.id)}">
        <span class="hf-item-id">${highlightMatch(escHtml(m.modelId ?? m.id), query)}</span>
        <span class="hf-item-meta">${tag ? `<span class="hf-tag">${escHtml(tag)}</span>` : ''}${dl ? `<span class="hf-dl">${dl}↓</span>` : ''}</span>
      </div>`;
    }).join('');

    dropdown.querySelectorAll('.hf-item').forEach(el => {
      el.addEventListener('mouseenter', () => setActive(+el.dataset.idx));
      el.addEventListener('click', () => selectItem(el.dataset.id));
    });

    positionDropdown();
    dropdown.style.display = 'block';
  }

  // ── Position relative to input ────────────────────────────
  function positionDropdown() {
    const input = document.getElementById('hfModelId');
    if (!input) return;
    const r = input.getBoundingClientRect();
    const desiredWidth = Math.round(r.width * 2.5);
    const maxWidth = window.innerWidth - r.left - 8;
    const width = Math.min(desiredWidth, maxWidth);
    dropdown.style.top   = (r.bottom + 2) + 'px';
    dropdown.style.left  = r.left + 'px';
    dropdown.style.width = width + 'px';
  }

  function showDropdown() {
    if (!dropdown || !currentResults.length) return;
    positionDropdown();
    dropdown.style.display = 'block';
  }

  function hideDropdown() {
    if (dropdown) dropdown.style.display = 'none';
    activeIdx = -1;
  }

  // ── Keyboard navigation ───────────────────────────────────
  function onKeyDown(e) {
    if (!dropdown || dropdown.style.display === 'none') return;
    const items = dropdown.querySelectorAll('.hf-item');
    if (!items.length) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive(Math.min(activeIdx + 1, items.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive(Math.max(activeIdx - 1, 0));
    } else if (e.key === 'Enter' && activeIdx >= 0) {
      e.preventDefault();
      selectItem(items[activeIdx].dataset.id);
    } else if (e.key === 'Escape') {
      hideDropdown();
    }
  }

  function setActive(idx) {
    const items = dropdown.querySelectorAll('.hf-item');
    items.forEach(el => el.classList.remove('hf-active'));
    activeIdx = idx;
    if (idx >= 0 && idx < items.length) {
      items[idx].classList.add('hf-active');
      items[idx].scrollIntoView({ block: 'nearest' });
    }
  }

  // ── Select a suggestion ───────────────────────────────────
  function selectItem(modelId) {
    const input = document.getElementById('hfModelId');
    if (input) input.value = modelId;
    lastQuery = modelId;
    currentResults = [];
    hideDropdown();
    // Trigger the existing fetch function in app.js
    if (typeof fetchHFConfig === 'function') fetchHFConfig();
  }

  // ── Helpers ───────────────────────────────────────────────
  function escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function highlightMatch(html, query) {
    // Highlight the last segment of the query (after last /)
    const term = query.split('/').pop().trim();
    if (!term) return html;
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return html.replace(new RegExp(`(${escaped})`, 'gi'), '<mark class="hf-hl">$1</mark>');
  }

  function formatNum(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M ';
    if (n >= 1e3) return Math.round(n / 1e3) + 'K ';
    return n + ' ';
  }

  // ── Reposition on scroll/resize ──────────────────────────
  window.addEventListener('scroll', () => {
    if (dropdown && dropdown.style.display !== 'none') positionDropdown();
  }, { passive: true });

  document.addEventListener('DOMContentLoaded', init);
})();
