// ═══════════════════════════════════════════════════════════
//  HuggingFace Model ID Autocomplete
// ═══════════════════════════════════════════════════════════
(function () {
  const HF_API = '/api/hf-proxy/api/models';
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
      const url = `${HF_API}?search=${encodeURIComponent(query)}&limit=${MAX_RESULTS}&sort=downloads&direction=-1`;
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

// ═══════════════════════════════════════════════════════════
//  HF API FETCHERS
// ═══════════════════════════════════════════════════════════

async function fetchHFConfig() {
  const modelId = document.getElementById('hfModelId').value.trim();
  if (!modelId) return;

  const statusEl = document.getElementById('fetchStatus');
  const fetchBtn = document.getElementById('fetchBtn');

  statusEl.className = 'fetch-status loading';
  statusEl.textContent = '⟳ Fetching config.json…';
  fetchBtn.disabled = true;

  try {
    const url = `/api/hf-proxy/${modelId}/resolve/main/config.json`;
    const hfToken = document.getElementById('hfToken')?.value.trim();
    const headers = hfToken ? { 'Authorization': `Bearer ${hfToken}` } : {};
    const res = await fetch(url, { headers });
    if (!res.ok) {
      if (res.status === 401) throw new Error('HTTP 401 — Unauthorized. Add HF Token in Settings');
      if (res.status === 403) throw new Error('403_FORBIDDEN');
      throw new Error(`HTTP ${res.status}`);
    }
    const rawCfg = await res.json();
    const subCfg = rawCfg.text_config || rawCfg.language_config || {};
    // Merge so that top-level properties (like vocab_size/context) are preserved 
    // even if the model structures its hidden dimensions under text_config.
    const cfg = { ...rawCfg, ...subCfg };

    // Map common HF config keys → our fields
    const mappings = {
      layers: cfg.num_hidden_layers ?? cfg.n_layer ?? cfg.num_layers ?? null,
      hidden: cfg.hidden_size ?? cfg.d_model ?? cfg.n_embd ?? null,
      ffn: cfg.intermediate_size ?? cfg.ffn_dim ?? cfg.inner_dim ?? null,
      heads: cfg.num_attention_heads ?? cfg.n_head ?? null,
      kvHeads: cfg.num_key_value_heads ?? cfg.num_attention_heads ?? cfg.n_head ?? null,
      context: cfg.max_position_embeddings ?? cfg.n_positions ?? cfg.max_seq_len ?? null,
      vocab: cfg.vocab_size ?? null,
      nExperts: cfg.n_routed_experts ?? cfg.num_experts ?? cfg.num_local_experts ?? null,
      nActive: cfg.num_experts_per_tok ?? cfg.n_activated_experts ?? cfg.experts_per_token ?? null,
      moeIntermediateSize: cfg.moe_intermediate_size ?? null,
    };

    // --- FALLBACK TIER 2: Network parsing for missing physical specs ---
    if (!mappings.vocab) {
      statusEl.textContent = '⟳ Inferring vocab limit (Tier 2)…';
      const realVocab = await fetchSafetensorsVocab(modelId, headers);
      if (realVocab) mappings.vocab = realVocab;
    }
    
    const ctxEl = document.getElementById('mContext');
    ctxEl.classList.remove('error-field'); // reset
    ctxEl.placeholder = '4096'; // default placeholder

    let finalStatusHtml = `✓ Loaded ${cfg.model_type ?? modelId}`;

    if (!mappings.context) {
      statusEl.textContent = '⟳ Inferring context limit (Tier 2)…';
      const realContext = await fetchTokenizerContext(modelId, headers);
      if (realContext) {
        mappings.context = realContext;
      } else {
        ctxEl.classList.add('error-field');
        ctxEl.placeholder = '⚠ Enter length!';
        finalStatusHtml += ` <span style="color:var(--err); font-weight:bold; margin-left:0.5rem">[⚠ Unknown Context Len - enter manually!]</span>`;
      }
    }

    // Compute params safely with potential fallbacks loaded
    mappings.params = inferParams(cfg, mappings.vocab);

    // Infer precision — quantization_config takes priority over torch_dtype
    let precisionFromConfig = null;
    const qConfig = cfg.quantization_config;
    if (qConfig) {
      const bits = qConfig.bits ?? qConfig.w_bit ?? qConfig.num_bits;
      const quantType = (qConfig.quant_type ?? qConfig.quant_method ?? '').toLowerCase();
      if (qConfig.load_in_4bit || bits === 4) precisionFromConfig = 'int4';
      else if (qConfig.load_in_8bit || bits === 8) precisionFromConfig = 'int8';
      else if (quantType.includes('fp8')) precisionFromConfig = 'fp8';
      else if (quantType.includes('fp4')) precisionFromConfig = 'fp4';
    }

    if (precisionFromConfig) {
      // modules_to_not_convert lists modules kept in full precision (typically BF16).
      // If it's non-empty on a MoE model, use mixed-precision: expert weights at
      // quantized precision, attention + embed at BF16.
      const notConverted = qConfig?.modules_to_not_convert ?? [];
      const isMoEModel = mappings.nExperts !== null;
      if (isMoEModel && notConverted.length > 0) {
        setExpertPrecision(precisionFromConfig);
        setOtherPrecision('bf16');
      } else {
        setPrecision(precisionFromConfig);
      }
    } else {
      const dtype = cfg.torch_dtype ?? cfg.dtype;
      if (dtype) {
        const dtypeMap = {
          float32: 'fp32', float16: 'fp16', bfloat16: 'bf16',
          int8: 'int8', int4: 'int4',
          float8_e4m3fn: 'fp8', float8_e5m2: 'fp8',
        };
        const mapped = dtypeMap[dtype];
        if (mapped) setPrecision(mapped);
      }
    }

    // Fill fields
    setFieldVal('mParams', mappings.params ? (mappings.params / 1e9).toFixed(2) : '');
    setFieldVal('mLayers', mappings.layers);
    setFieldVal('mHidden', mappings.hidden);
    setFieldVal('mFFN', mappings.ffn);
    setFieldVal('mHeads', mappings.heads);
    setFieldVal('mKVHeads', mappings.kvHeads);
    setFieldVal('mContext', mappings.context);
    setFieldVal('bOutput', 256);
    setFieldVal('mVocab', mappings.vocab);

    // MoE expert structure — populate new first-class fields
    setFieldVal('mNExperts', mappings.nExperts);
    setFieldVal('mNActive',  mappings.nActive);
    setFieldVal('mMoeFFN',   mappings.moeIntermediateSize);
    // Clear legacy activeParams field (new fields supersede it)
    document.getElementById('mActiveParams').value = '';

    statusEl.className = 'fetch-status ok';
    statusEl.innerHTML = typeof finalStatusHtml !== 'undefined' ? finalStatusHtml : `✓ Loaded ${cfg.model_type ?? modelId}`;

    recalculate();
  } catch (e) {
    statusEl.className = 'fetch-status err';
    if (e.message === '403_FORBIDDEN') {
      // safe html injection using text interpolation where needed
      const safeId = modelId.replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      statusEl.innerHTML = `✗ HTTP 403 — Forbidden. <a href="https://huggingface.co/${safeId}" target="_blank" style="color:inherit;text-decoration:underline;text-underline-offset:2px;">Accept the license here</a>`;
    } else {
      statusEl.textContent = e.message.includes('401') ? `✗ ${e.message}` : `✗ ${e.message} — check model ID`;
    }
  } finally {
    fetchBtn.disabled = false;
  }
}

async function fetchSafetensorsVocab(modelId, headers) {
  try {
    let targetFile = 'model.safetensors';
    
    // Check if sharded
    const idxRes = await fetch(`/api/hf-proxy/${modelId}/resolve/main/model.safetensors.index.json`, { headers });
    if (idxRes.ok) {
        const idx = await idxRes.json();
        if (idx.weight_map) {
            for (const [k, v] of Object.entries(idx.weight_map)) {
                if (k.includes('embed_tokens') || k.includes('wte') || k.includes('word_embeddings')) {
                    targetFile = v;
                    break;
                }
            }
        }
    }

    // Range Fetch 1: 8 byte INT64 for JSON Header Size
    const stUrl = `/api/hf-proxy/${modelId}/resolve/main/${targetFile}`;
    const rangeRes = await fetch(stUrl, {
      headers: { ...headers, 'Range': 'bytes=0-7' }
    });
    if (rangeRes.status !== 206 && rangeRes.status !== 200) return null;
    
    const buffer = await rangeRes.arrayBuffer();
    const dataView = new DataView(buffer);
    const headerLen = Number(dataView.getBigUint64(0, true));
    if (headerLen <= 0 || headerLen > 100000000) return null; 
    
    // Range Fetch 2: Download exact minimal Header JSON
    const jsonRes = await fetch(stUrl, {
      headers: { ...headers, 'Range': `bytes=8-${8 + headerLen - 1}` }
    });
    if (jsonRes.status !== 206 && jsonRes.status !== 200) return null;
    
    const jsonText = await jsonRes.text();
    const stMeta = JSON.parse(jsonText);
    
    // Parse pure mathematical embedding shape map
    for (const key of Object.keys(stMeta)) {
      if (key === '__metadata__') continue;
      if (key.includes('embed_tokens') || key.includes('wte') || key.includes('word_embeddings')) {
          if (stMeta[key].shape) return stMeta[key].shape[0]; 
      }
    }
  } catch (e) {
    console.warn("Safetensors vocab fallback failed", e);
  }
  return null;
}

async function fetchTokenizerContext(modelId, headers) {
  try {
    const res = await fetch(`/api/hf-proxy/${modelId}/resolve/main/tokenizer_config.json`, { headers });
    if (!res.ok) return null;
    const txt = await res.json();
    const len = txt.model_max_length || txt.max_position_embeddings || null;
    // Tokenizers sometimes define 1e30 as "unbounded". Ignore these dummy values.
    if (len && len < 10000000) return len;
  } catch (e) {
    console.warn("Tokenizer fallback failed", e);
  }
  return null;
}

